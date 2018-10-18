
# Customise the matplotlib
import matplotlib as mpl
mpl.rcParams['image.interpolation'] = 'nearest';
mpl.rcParams['axes.grid'] = True;
mpl.rcParams['legend.framealpha'] = 0.5;
mpl.rcParams['image.origin'] = 'lower';
mpl.rcParams['grid.linewidth'] = 0.15;

from . import setup;

def arrays (nf1,nf2,naxes1,naxes2):
    '''
    Create an array of figure with subplots
    '''
    axes = [[None]*nf1]*nf2;
    figs = [[None]*nf1]*nf2;

    for c in range (nf2):
        for f in range (nf1):
            print ('Create figure %i %i'%(c,f));
            fig,ax = plt.subplots (naxes1,naxes2, sharex=True);
            axes[c][f] = ax.flatten();
            figs[c][f] = fig;

    return figs, axes;
            

def scale (ax, scale):
    '''
    Write a scale
    '''
    scale = '%.1e'%(scale);
    ax.text (0.02, 0.92, scale, transform=ax.transAxes,
             horizontalalignment='left',
             verticalalignment='top',fontsize=5);

def compact (axes):
    '''
    Set compact mode
    '''
    n = len(axes.flatten());

    for i in range (n):
        ax = axes.flatten()[i];
        ax.ticklabel_format (axis='both', style='plain');
        ax.tick_params (axis='both', which='both', labelsize=5);

def base_name (axes, bstart=None, tstart=None, names=None):
    '''
    Set the names of the bases (or triplet) in axes of plot
    bstart: label only bases bstart:bstart+len(axes))
    tstart: label only bases tstart:tstart+len(axes))
    '''
    n = len(axes.flatten());

    # Get the names
    if names is None:
        if bstart is not None:
            names = setup.base_name ()[bstart:bstart+n];        
        elif tstart is not None:
            names = setup.triplet_name ()[tstart:tstart+n];
        else:
            names = setup.base_name () if n == 15 else setup.triplet_name ();

    for i in range (n):
        ax = axes.flatten()[i];
        ax.text (0.99, 0.95, names[i],
                transform = ax.transAxes,
                horizontalalignment = 'right',
                verticalalignment = 'top',fontsize=7,
                bbox={'facecolor':'white', 'alpha':0.9,
                      'pad':1, 'edgecolor':'none'});
