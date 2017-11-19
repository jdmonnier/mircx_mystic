
# Customise the matplotlib
import matplotlib as mpl
mpl.rcParams['image.interpolation'] = 'nearest';
mpl.rcParams['axes.grid'] = True;
mpl.rcParams['legend.framealpha'] = 0.5;

from . import setup;

def close_style (ax, scale=None):
    '''
    Set the style for the plots
    '''
    # Write the scale
    if scale is not None:
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

def base_name (axes):
    '''
    Set the names
    '''
    n = len(axes.flatten());
    names = setup.base_name () if n == 15 else setup.triplet_name ();
    
    for i in range (n):
        ax = axes.flatten()[i];
        ax.text (0.98, 0.92, names[i],
                transform = ax.transAxes,
                horizontalalignment = 'right',
                verticalalignment = 'top',fontsize=7);
