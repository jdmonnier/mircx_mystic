
# Customise the matplotlib
import matplotlib as mpl
mpl.rcParams['image.interpolation'] = 'nearest';
mpl.rcParams['axes.grid'] = True;
mpl.rcParams['legend.framealpha'] = 0.5;

from . import setup;

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

def base_name (axes, bstart=None, tstart=None):
    '''
    Set the names
    '''
    n = len(axes.flatten());

    # Get the names
    if bstart is not None:
        names = setup.base_name ()[bstart:bstart+n];
    elif tstart is not None:
        setup.triplet_name ()[tstart:tstart+n];
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
