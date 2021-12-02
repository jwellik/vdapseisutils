class CCMatrix():

    def __init__( self, nparray, ccmin=0.0, colorbar='plasma'):
    
        self.matrix    = nparray
        self.ccmin     = ccmin
        self.colorbar  = colorbar
    
    """Return histogram of CCMatrix"""
    def histogram( self, histogram_kwargs ):
        pass