"""Unified ObsPy client facade: auto-detection, explicit ``client_type=``, and URI schemes."""

from __future__ import annotations

import re
from pathlib import Path

from obspy.clients.earthworm import Client as EarthwormClient
from obspy.clients.fdsn import Client as FDSNClient
from obspy.clients.filesystem.sds import Client as SDSClient
from obspy.clients.seedlink import Client as SeedlinkClient

_HOST_PORT_RE = re.compile(r"^[^:\s]+:\d+$")


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
        from obspy import UTCDateTime
        from obspy.core.stream import Stream

        from vdapseisutils.obspy_ext import VStreamID

        id_ = kwargs.pop("id", None) or kwargs.pop("waveformID", None)
        if id_ is None and len(args) > 0 and isinstance(args[0], str) and ("." in args[0]) and len(args) < 5:
            id_ = args[0]
            args = args[1:]
        if "starttime" in kwargs:
            kwargs["starttime"] = UTCDateTime(kwargs["starttime"])
        if "endtime" in kwargs:
            kwargs["endtime"] = UTCDateTime(kwargs["endtime"])
        if isinstance(id_, list) and self.use_bulk:
            bulk_args = []
            for idstr in id_:
                wid = VStreamID(idstr)
                net, sta, loc, cha = wid.network, wid.station, wid.location, wid.channel
                t1 = kwargs.get("starttime", None)
                t2 = kwargs.get("endtime", None)
                t1 = UTCDateTime(t1) if t1 is not None else None
                t2 = UTCDateTime(t2) if t2 is not None else None
                bulk_args.append((net, sta, loc, cha, t1, t2))
            return self._client.get_waveforms_bulk(bulk_args, **kwargs)
        if isinstance(id_, list):
            streams = []
            for idstr in id_:
                wid = VStreamID(idstr)
                net, sta, loc, cha = wid.network, wid.station, wid.location, wid.channel
                t1 = kwargs.get("starttime", None)
                t2 = kwargs.get("endtime", None)
                t1 = UTCDateTime(t1) if t1 is not None else None
                t2 = UTCDateTime(t2) if t2 is not None else None
                streams.append(self._client.get_waveforms(net, sta, loc, cha, t1, t2, **kwargs))
            merged = Stream()
            for st in streams:
                merged += st
            return merged
        if isinstance(id_, str):
            wid = VStreamID(id_)
            net, sta, loc, cha = wid.network, wid.station, wid.location, wid.channel
            t1 = kwargs.get("starttime", None)
            t2 = kwargs.get("endtime", None)
            t1 = UTCDateTime(t1) if t1 is not None else None
            t2 = UTCDateTime(t2) if t2 is not None else None
            return self._client.get_waveforms(net, sta, loc, cha, t1, t2, **kwargs)

        new_args = list(args)
        if len(new_args) >= 5:
            new_args[4] = UTCDateTime(new_args[4])
        if len(new_args) >= 6:
            new_args[5] = UTCDateTime(new_args[5])
        return self._client.get_waveforms(*new_args, **kwargs)

    def get_waveforms_bulk(self, *args, **kwargs):
        return self._client.get_waveforms_bulk(*args, **kwargs)

    def get_waveforms_from_inventory(self, inventory, **kwargs):
        from vdapseisutils.utils.obspyutils.inventory import VInventory

        return VInventory.get_waveforms_from_inventory(self._client, inventory, **kwargs)

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
