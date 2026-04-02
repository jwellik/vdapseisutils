"""Unified ObsPy client facade: auto-detection, explicit ``client_type=``, and URI schemes."""

from __future__ import annotations

import re
from pathlib import Path

from obspy.clients.earthworm import Client as EarthwormClient
from obspy.clients.fdsn import Client as FDSNClient
from obspy.clients.filesystem.sds import Client as SDSClient
from obspy.clients.seedlink import Client as SeedlinkClient

_HOST_PORT_RE = re.compile(r"^[^:\s]+:\d+$")

_FETCH_OPTION_KEYS = frozenset(
    {
        "max_download",
        "fill_value",
        "create_empty_trace",
        "empty_samp_rate",
        "verbose",
    }
)


def _split_waveform_kwargs(kwargs):
    """Split VClient waveform options for :mod:`_fetch` vs passthrough to ObsPy client."""
    kw = dict(kwargs)
    fetch_kw = {}
    for key in _FETCH_OPTION_KEYS:
        if key in kw:
            fetch_kw[key] = kw.pop(key)
    return fetch_kw, kw


def _split_host_port(serverport: str) -> tuple[str, int]:
    host, _, port_s = serverport.rpartition(":")
    if not host or not port_s.isdigit():
        raise ValueError(f"Expected host:port, got {serverport!r}")
    return host, int(port_s)


def _create_client_from_uri(uri: str, *, timeout: int) -> object:
    """Build an ObsPy client from a URI (``scheme://…``)."""
    lower = uri.lower()
    if lower.startswith("fdsnws://"):
        server = uri.split("fdsnws://", 1)[1]
        return FDSNClient(server)

    ew_prefixes = (
        "waveserver://",
        "earthworm://",
        "winston://",
        "wws://",
    )
    for p in ew_prefixes:
        if lower.startswith(p):
            rest = uri.split("://", 1)[1]
            host, port = _split_host_port(rest)
            return EarthwormClient(host, port)

    if lower.startswith("seedlink://"):
        rest = uri.split("seedlink://", 1)[1]
        host, port = _split_host_port(rest)
        return SeedlinkClient(host, port, timeout=timeout)

    if lower.startswith("neic://"):
        from obspy.clients.neic import Client as NEICClient

        rest = uri.split("neic://", 1)[1]
        host, port = _split_host_port(rest)
        return NEICClient(host, port)

    if lower.startswith("sds://"):
        sdspath = uri.split("sds://", 1)[1]
        return SDSClient(sdspath)

    raise ValueError(f"Unsupported client URI scheme: {uri!r}")


def _normalize_plain_client_string(s: str) -> str:
    """Apply legacy DataSource rules when no ``://`` is present."""
    if "://" in s:
        return s
    if "." not in s:
        return "fdsnws://" + s
    return "waveserver://" + s


class VClient:
    """
    Extended client facade: auto-detect ObsPy client type, optional ``client_type=``,
    or URI-style connection strings (legacy ``DataSource`` syntax).

    The underlying ObsPy client is stored on ``self._client``; use :attr:`client`
    for public access.

    **Construction styles**

    1. **Heuristic / explicit type (original ``VClient`` style)** — pass the same
       arguments you would pass to the underlying ObsPy client, optionally forcing
       the implementation with ``client_type=``:

       .. code-block:: python

          VClient("/path/to/sds_root")              # existing directory → SDS
          VClient("IRIS")                           # short FDSN name → FDSN
          VClient("service.iris.edu")               # FDSN-style host → FDSN
          VClient("127.0.0.1", port=16022)        # earthworm ports / kwargs → Earthworm
          VClient("127.0.0.1", port=18000)        # seedlink ports / kwargs → SeedLink
          VClient("my.host", client_type="fdsn")  # force FDSN

    2. **URI strings (legacy ``DataSource.__create_client``)** — a single positional
       string containing a scheme selects the client directly:

       .. code-block:: python

          VClient("fdsnws://IRIS")
          VClient("sds:///data/sds")
          VClient("waveserver://127.0.0.1:16022")
          VClient("earthworm://127.0.0.1:16022")
          VClient("winston://127.0.0.1:16022")
          VClient("wws://127.0.0.1:16022")
          VClient("seedlink://127.0.0.1:18000")
          VClient("neic://127.0.0.1:16017")

       For a single string **without** ``://``, the legacy normalization matches
       short FDSN names (no dot) as ``fdsnws://…`` and ``host:port``-style strings
       (with a dot) as ``waveserver://…``, except when the path is an existing
       directory (SDS) or port-based kwargs select Earthworm / SeedLink.

    Parameters
    ----------
    *args
        Passed to the underlying ObsPy client (after URI normalization if used).
    client_type : str, optional
        Force ``"fdsn"``, ``"sds"``, ``"earthworm"``, or ``"seedlink"``.
    timeout : int, optional
        Default timeout for SeedLink clients created from ``seedlink://`` URIs
        (default 60 seconds). Also forwarded if passed in ``kwargs``.
    **kwargs
        Passed through to the underlying client constructor.
    """

    def __init__(self, *args, client_type=None, timeout=60, **kwargs):
        self.client_type = client_type
        self._client = None
        self.use_bulk = True
        self._uri_source: str | None = None

        if client_type:
            self._client = self._create_client_by_type(client_type, *args, **kwargs)
        elif (
            len(args) == 1
            and isinstance(args[0], str)
            and "://" in args[0]
        ):
            self._uri_source = args[0]
            to = kwargs.pop("timeout", timeout)
            self._client = _create_client_from_uri(args[0], timeout=to)
        elif (
            len(args) == 1
            and isinstance(args[0], str)
            and "://" not in args[0]
            and not self._is_filesystem_path(args[0])
            and not self._is_earthworm_client(*args, **kwargs)
            and not self._is_seedlink_client(*args, **kwargs)
            and _HOST_PORT_RE.match(args[0].strip())
        ):
            host, port = _split_host_port(args[0].strip())
            if port in (16022, 16023, 16024):
                print(f"Detected Earthworm client: {args[0]}")
                self._client = EarthwormClient(host, port, **kwargs)
            elif port in (18000, 18001, 18002):
                print(f"Detected SeedLink client: {args[0]}")
                sl_to = kwargs.pop("timeout", timeout)
                self._client = SeedlinkClient(host, port, timeout=sl_to, **kwargs)
            else:
                normalized = _normalize_plain_client_string(args[0].strip())
                self._uri_source = normalized
                to = kwargs.pop("timeout", timeout)
                self._client = _create_client_from_uri(normalized, timeout=to)
        else:
            self._client = self._auto_detect_client(*args, **kwargs)

        ctype = self.get_client_type()
        if ctype == "EarthwormClient":
            self.use_bulk = False
        else:
            self.use_bulk = True

    def _auto_detect_client(self, *args, **kwargs):
        if not args:
            return FDSNClient("IRIS", **kwargs)

        first_arg = args[0]

        if isinstance(first_arg, str):
            if self._is_filesystem_path(first_arg):
                print(f"Detected SDS filesystem: {first_arg}")
                return SDSClient(first_arg, **kwargs)

            if self._is_earthworm_client(*args, **kwargs):
                print(f"Detected Earthworm client: {first_arg}")
                return EarthwormClient(*args, **kwargs)

            if self._is_seedlink_client(*args, **kwargs):
                print(f"Detected SeedLink client: {first_arg}")
                return SeedlinkClient(*args, **kwargs)

            print(f"Detected FDSN client: {first_arg}")
            return FDSNClient(first_arg, **kwargs)

        return FDSNClient(*args, **kwargs)

    def _create_client_by_type(self, client_type, *args, **kwargs):
        client_map = {
            "fdsn": FDSNClient,
            "sds": SDSClient,
            "earthworm": EarthwormClient,
            "seedlink": SeedlinkClient,
        }

        key = client_type.lower()
        if key not in client_map:
            raise ValueError(
                f"Unknown client type: {client_type}. Valid types: {list(client_map.keys())}"
            )

        client_class = client_map[key]
        print(f"Creating {client_type.upper()} client")
        return client_class(*args, **kwargs)

    def _is_filesystem_path(self, path_str):
        path = Path(path_str)
        return path.exists() and path.is_dir()

    def _is_earthworm_client(self, *args, **kwargs):
        earthworm_ports = [16022, 16023, 16024]
        if "port" in kwargs and kwargs["port"] in earthworm_ports:
            return True
        earthworm_params = ["timeout", "heartbeat_host", "heartbeat_port"]
        if any(param in kwargs for param in earthworm_params):
            return True
        return False

    def _is_seedlink_client(self, *args, **kwargs):
        seedlink_ports = [18000, 18001, 18002]
        if "port" in kwargs and kwargs["port"] in seedlink_ports:
            return True
        seedlink_params = ["autoconnect", "recover"]
        if any(param in kwargs for param in seedlink_params):
            return True
        return False

    def __getattr__(self, name):
        if self._client is None:
            raise AttributeError("Client not initialized")
        return getattr(self._client, name)

    def get_stations(self, *args, **kwargs):
        from vdapseisutils.utils.obspyutils.inventory import VInventory

        inventory = self._client.get_stations(*args, **kwargs)
        return VInventory(inventory)

    def get_events(self, *args, **kwargs):
        from vdapseisutils.utils.obspyutils.catalog import VCatalog

        catalog = self._client.get_events(*args, **kwargs)
        return VCatalog(catalog)

    def get_waveforms(self, *args, **kwargs):
        """
        Download waveforms via :func:`~vdapseisutils.obspy_ext.client._fetch.get_waveforms_from_client`
        (chunking, dtype handling, empty-trace behavior). Returns a :class:`~vdapseisutils.utils.obspyutils.stream.core.VStream`.
        """
        from obspy import UTCDateTime

        from vdapseisutils.obspy_ext.client._fetch import get_waveforms_from_client
        from vdapseisutils.utils.obspyutils.stream.core import VStream

        fetch_kw, kw = _split_waveform_kwargs(kwargs)

        id_ = kw.pop("id", None) or kw.pop("waveformID", None)
        args_list = list(args)

        if id_ is None and (
            len(args_list) > 0
            and isinstance(args_list[0], str)
            and ("." in args_list[0])
            and len(args_list) < 5
        ):
            id_ = args_list[0]
            args_list = args_list[1:]

        if "starttime" in kw:
            kw["starttime"] = UTCDateTime(kw["starttime"])
        if "endtime" in kw:
            kw["endtime"] = UTCDateTime(kw["endtime"])

        nslc_list = None
        t1 = t2 = None

        if isinstance(id_, list):
            nslc_list = [str(x) for x in id_]
            t1 = kw.pop("starttime", None)
            t2 = kw.pop("endtime", None)
        elif isinstance(id_, str):
            nslc_list = [id_]
            t1 = kw.pop("starttime", None)
            t2 = kw.pop("endtime", None)
        elif len(args_list) >= 6:
            net, sta, loc, cha, t1, t2 = args_list[:6]
            t1 = UTCDateTime(t1)
            t2 = UTCDateTime(t2)
            loc = loc if loc is not None else ""
            nslc_list = [f"{net}.{sta}.{loc}.{cha}"]
            args_list = args_list[6:]
        elif (
            all(k in kw for k in ("network", "station", "channel"))
            and "starttime" in kw
            and "endtime" in kw
        ):
            net = kw.pop("network")
            sta = kw.pop("station")
            if "location" in kw:
                loc = kw.pop("location")
            elif "location_code" in kw:
                loc = kw.pop("location_code")
            else:
                loc = "*"
            cha = kw.pop("channel")
            t1 = UTCDateTime(kw.pop("starttime"))
            t2 = UTCDateTime(kw.pop("endtime"))
            nslc_list = [f"{net}.{sta}.{loc}.{cha}"]

        if nslc_list is not None and t1 is not None and t2 is not None:
            st = get_waveforms_from_client(
                self._client,
                nslc_list,
                t1,
                t2,
                **fetch_kw,
                **kw,
            )
            return VStream(st)

        new_args = list(args_list)
        if len(new_args) >= 5:
            new_args[4] = UTCDateTime(new_args[4])
        if len(new_args) >= 6:
            new_args[5] = UTCDateTime(new_args[5])
        return VStream(self._client.get_waveforms(*new_args, **kw))

    def get_waveforms_bulk(self, bulk, **kwargs):
        """
        Bulk download via :func:`~vdapseisutils.obspy_ext.client._fetch.get_waveforms_bulk_from_client`.
        Returns :class:`~vdapseisutils.utils.obspyutils.stream.core.VStream`.
        """
        from vdapseisutils.obspy_ext.client._fetch import get_waveforms_bulk_from_client
        from vdapseisutils.utils.obspyutils.stream.core import VStream

        fetch_kw, client_kw = _split_waveform_kwargs(kwargs)
        st = get_waveforms_bulk_from_client(
            self._client,
            bulk,
            **fetch_kw,
            **client_kw,
        )
        return VStream(st)

    def get_waveforms_from_inventory(self, inventory, **kwargs):
        from vdapseisutils.utils.obspyutils.inventory import VInventory
        from vdapseisutils.utils.obspyutils.stream.core import VStream

        st = VInventory.get_waveforms_from_inventory(
            self._client, inventory, **kwargs
        )
        return VStream(st)

    def __repr__(self):
        if self._client:
            client_name = self._client.__class__.__name__
            return f"VClient({client_name}: {repr(self._client)})"
        return "VClient(uninitialized)"

    @property
    def client(self):
        return self._client

    def get_client_type(self):
        if self._client:
            return self._client.__class__.__name__
        return None


DataSource = VClient
