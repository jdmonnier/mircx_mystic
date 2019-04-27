import numpy as np;
import os;

import matplotlib.pyplot as plt;
import matplotlib.colors as mcolors;

from astropy.stats import sigma_clipped_stats;
from astropy.io import fits as pyfits;
from astropy.modeling import models, fitting;

from skimage.feature import register_translation;

from scipy import fftpack;
from scipy.signal import medfilt;
from scipy.ndimage.interpolation import shift as subpix_shift;
from scipy.ndimage import gaussian_filter, uniform_filter, median_filter;
from scipy.optimize import least_squares, curve_fit;
from scipy.ndimage.morphology import binary_closing, binary_opening;
from scipy.ndimage.morphology import binary_dilation, binary_erosion;

from . import log, files, headers, setup, oifits, signal, plot, qc;
from .headers import HM, HMQ, HMP, HMW, rep_nan;

def compute_trends (hdrs, output='output_trends', filetype='SELECTION',
                    interactive=False, ncoher=10, nscan=64):
    '''
    Display a summary of the entire night as waterfalls,
    based on the RTS intermediate products
    '''

    elog = log.trace ('compute_trends');

    # Check inputs
    headers.check_input (hdrs,  required=1);

    # Prepare outputs arrays
    nfiles = len (hdrs);
    all_base_scan = np.zero (nfiles, nscan, 15);
    all_beam_scan = np.zero (nfiles, nscan, 6);
    
    # Loop on files to compute trend per base
    for ih,h in enumerate(hdrs):

        # Load data
        f = h['ORIGNAME'];
        log.info ('Load RTS file %s'%f);
        
        base_dft = pyfits.getdata (f, 'BASE_DFT_IMAG').astype(float) * 1.j + \
                   pyfits.getdata (f, 'BASE_DFT_REAL').astype(float);
        
        # Dimensions
        nr,nf,ny,nb = base_dft.shape;
        log.info ('Data size: '+str(base_dft.shape));

        # Do coherent integration
        log.info ('Coherent integration over %i frames'%ncoher);
        base_dft = signal.uniform_filter_cpx (base_dft,(0,ncoher,0,0),mode='constant');
        
        # Compute FFT over the lbd direction, thus OPD-scan
        log.info ('Compute 2d FFT (nscan=%i)'%nscan);
        base_scan = np.fft.fftshift (np.fft.fft (base_dft, n=nscan, axis=2), axes=2);

        # Compute mean scan over ramp and frame
        # results is of shape (nscan,nb)
        base_scan = np.abs (base_scan).mean (axis=(0,1));

        # Set data for base
        all_base_scan[ih,:,:] = base_scan;

    # Compute trend per beams
    log.info ('Compute the trend per beams');
    
    for ib,beams in enumerate (setup.base_beam ()):
        all_beam_scan[:,:,beams[0]] += all_base_scan[:,:,ib];
        all_beam_scan[:,:,beams[1]] += all_base_scan[:,:,ib];

    # Make nice plots
    log.info ('Plots');

    # ...

    # Start interactive session
    log.info ('Interactive session');

    # ...

    # Start interactive session
    log.info ('Save selection');

    # ...

    # File
    log.info ('Create file');

    # First HDU
    hdulist[0].header['FILETYPE'] = filetype;

    # ...
    
    # Write file
    files.write (hdulist, output+'.fits');
            
    plt.close("all");
    return hdulist;
    
    
    
