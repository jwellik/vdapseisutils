from obspy import Stream, Trace, UTCDateTime

class Clipboard(self, st, *kwargs):
    
    st = Stream()
    plot_type = 'wg' # ['wg'], 'w', 'g', 's', 'm'|'o'
    


    
class Spectrogram(self, st, *kwargs):
    
    st = Stream()
    
    cmap = 'plasma'
    axis_type = 'normal' # ['normal'] | 'annotate'
    
    #min_freq = 0.10
    #max_freq = 25.0
    freq_scale  = 'auto' # ['auto'], 'manual'
    freq_range  = (0.1,25) # Hz
    
    power_range = (0.0,120) # db
    
    window_size = 2   # seconds
    nfft        = 0   # points
    overlap     = .86 # percent (0.0-1.0)
    
    def plot(sef):
        ax = []
        return ax
    
    
    
def Waveform(self, st, *kwargs):
    
    st = Stream()
    
    amp_scale = 'auto' # ['auto'], 'manual'
    amp_range = (-1000,1000)
    
    color     = 'black'
    axis_type = 'normal' # ['normal'] | 'annotate'

    def plot():
        ax = []
        return ax
    

# def swarmclipboard(
#                     st,
#                     style='wg',
#                     wargs={'color' : 'k'},
#                     gargs={'ylim':[0,10]},
#                 ):


#     if style is not in ['wg', 'w', 'g']: style='wg'; print('{} plot style not undertsood. Using "wg"')
#     if stye is 'wg':
#         wgratio=[1,3]
#         nsubplots=2
#     elif style is 'g':
#         wgratio=[1,0]
#         nsubplots=1
#     elif style is 'w':
#         wgratio=[1,0]
#         nsubplots=1
    
    
#     nstreams   = len(st)
#     figheight  = 0.5+2.5*nstreams
#     left=0.05; right=0.95
#     top=0.05; bottom=0.05; space=0.03
#     wgratio=[1,3]
#     axheight = (1.0-top-bottom-space*(nstreams-1))/nstreams

#     fig = plt.figure(figsize=(12,figheight), constrained_layout=False)

#     for n in range(len(st)):

#         tr=st[n]

#         # Create gridspec
#         #axbottom = bottom+axheight*n+space*(n); axtop    = axbottom+axheight # bottom to top
#         axtop     = 1-top-axheight*n-space*n; axbottom  = axtop-axheight     # top to bottom
#         wg = fig.add_gridspec(
#                     nrows=wgratio[1]+1, ncols=1, left=left, right=right,
#                     #nrows=2, ncols=1, left=left, right=right,
#                     bottom=axbottom, top=axtop,
#                     wspace=0.00, hspace=0.00
#                     )
        
#         # Add first plot
#         w = fig.add_subplot(wg[0, :])
#         w.plot(tr.times("matplotlib"), tr.data, color='k')
#         w.set_xlim([tr.times("matplotlib")[0],tr.times("matplotlib")[-1]])
#         w.yaxis.set_ticks_position('right')
        
#         # Add second plot
#         if style is 'wg':
#             g = fig.add_subplot(wg[1:, :])
#             g.yaxis.set_ticks_position('right')
#             g.text(
#                 -0.05,0.67, getNSLCstr(tr), transform=g.transAxes, rotation='vertical', # 0.67 can be defined dynamically
#                 horizontalalignment='right', verticalalignment='center', fontsize=12,
#                 )
#             spectrogram(tr, axes=g, cmap='plasma', ylim=gylim)


#     # Set xaxis ticks and labels
#     # ??? settaxis( fig, n=2, minticks=3, maxticks=7 ) # n is how many plots per stream
#     locator = mdates.AutoDateLocator(minticks=3, maxticks=7)                           # Dynamic choice of xticks
#     formatter = mdates.ConciseDateFormatter(locator)                                   # Automatic date formatter
#     if len(fig.get_axes())>2:                                                          # if len>2 bc 1 stream will produce two axes
#         fig.get_axes()[0] = fig.get_axes()[0].xaxis.tick_top()                         # Put axis on top for top plot
#         fig.get_axes()[0] = fig.get_axes()[0].xaxis.set_major_locator(locator)         # Format top axis
#         fig.get_axes()[0] = fig.get_axes()[0].xaxis.set_major_formatter(formatter)
#         for n in range(1,len(fig.get_axes())):                                         # Remove middle axes
#             fig.get_axes()[n] = fig.get_axes()[n].set_xticks([])
#     fig.get_axes()[-1] = fig.get_axes()[-1].xaxis.set_major_locator(locator)           # Format bottom axis (bottom axis is always [-1])
#     fig.get_axes()[-1] = fig.get_axes()[-1].xaxis.set_major_formatter(formatter)

    
#     return fig
    