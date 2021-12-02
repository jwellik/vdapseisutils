import obspy

class datasource:
    
    def __init__( self, datasource_str, args=[] ):
        
    if ds_type == 'fdsn':
        from obspy.clients.fdsn import Client # https://docs.obspy.org/packages/obspy.clients.fdsn.html
        self.source = Client(datasource[1], *args)
        self.type='fdsn'

    if ds_type == 'ew':
        from obspy.clients.earthworm import Client # https://docs.obspy.org/packages/obspy.clients.earthworm.html#module-obspy.clients.earthworm
        self.source = Client(datasource[1], datasource[2], *args)
        self.type='ew'
        
    #if ds_type == 'sds':
    #    from obspy.clients.waveform.sds import Client # https://docs.obspy.org/packages/autogen/obspy.clients.filesystem.sds.Client.html#obspy.clients.filesystem.sds.Client
    #    ds = Client(datasource[1], *args)

    #if ds_type == 'iris':
    #    from obspy.clients.iris import Client # https://docs.obspy.org/packages/autogen/obspy.clients.iris.client.Client.html#obspy.clients.iris.client.Client

    #if ds_type == 'neic':
    #   from obspy.clients.neic import Client #https://docs.obspy.org/packages/autogen/obspy.clients.neic.client.Client.html#obspy.clients.neic.client.Client

    if ds_type == 'seedlink':
        from obspy.clients.seedlink import Client # https://docs.obspy.org/packages/autogen/obspy.clients.seedlink.basic_client.Client.html#obspy.clients.seedlink.basic_client.Client
        self.source = Client(datasource[1], datasource[2], *args)
        self.type='fdsn'
        
    if ds_type == 'files':
        self.source = datasource[1]
        self.type='files'
        
    else:
        from obspy.clients.fdsn import Client
        self.source = Client(datasource[0], *args)
        self.type='fdsn'

        
        