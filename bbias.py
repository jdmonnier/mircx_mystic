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


def compute_bbias_coeff (hdrs, bkgs, fgs, output='output_bbias', filetype='BBIAS_COEFF'):
    '''
    Compute the BBIAS_COEFF
    '''
    elog = log.trace ('compute_bbias_coeff');

    # Check inputs
    headers.check_input (hdrs, required=1);
    headers.check_input (bkgs, required=1);
    headers.check_input (fgs, required=1);

    # Build a header for the product
    hdr = hdrs[0].copy();

    # Loop on DATA_RTS files
    for ih,h in enumerate(hdrs):
        # filename
        f = h['ORIGNAME'];
        
        # Load file
        log.info ('Load DATA_RTS file %i over %i (%s)'%(ih+1,len(hdrs),f));
        # ....

    # Loop on BACKGROUND_RTS files
    for ih,h in enumerate(bkgs):
        # filename
        f = h['ORIGNAME'];
        
        # Load file
        log.info ('Load BACKGROUND_RTS file %i over %i (%s)'%(ih+1,len(bkgs),f));
        # ....

    # Loop on BACKGROUND_RTS files
    for ih,h in enumerate(fgs):
        # filename
        f = h['ORIGNAME'];
        
        # Load file
        log.info ('Load FOREGROUND_RTS file %i over %i (%s)'%(ih+1,len(fgs),f));
        # ....
        
    # Hardcod the size (ramps, frame-in-ramp, nchannel, nbaselines)
    # FIXME: read from data
    nr,nf,ny,nb = 100,80,9,15;
    
    # Outputs
    log.info ('FIXME: do the computation');
    C0 = np.zeros (ny);
    C1 = np.ones (ny);
    C2 = np.ones (ny);

    # Figures
    log.info ('Figures');
    
    fig,axes = plt.subplots (5,3, sharex=True);
    fig.suptitle (headers.summary (hdr));
    # ...
    files.write (fig,output+'_empty.png');
    
    # File
    log.info ('Create file');

    # First HDU
    hdu0 = pyfits.PrimaryHDU ([]);
    hdu0.header = hdr;
    hdu0.header['FILETYPE'] = filetype;

    # Other HDU
    hdu1 = pyfits.ImageHDU (C0);
    hdu1.header['EXTNAME'] = ('C0','Coefficient 1');
    
    hdu2 = pyfits.ImageHDU (C1);
    hdu2.header['EXTNAME'] = ('C1','Coefficient 2');
    hdu2.header['BUNIT'] = 'adu';
    
    hdu3 = pyfits.ImageHDU (C2);
    hdu3.header['EXTNAME'] = ('C2','Coefficient 3');
    hdu3.header['BUNIT'] = 'adu';

    # Write file
    hdulist = pyfits.HDUList ([hdu0,hdu1,hdu2,hdu3]);
    files.write (hdulist, output+'.fits');
    
    plt.close ("all");
    return hdulist;

    
