import numpy as np;
import os;

import matplotlib.pyplot as plt;
import matplotlib.colors as mcolors;
from matplotlib.colors import LogNorm;

from astropy.io import fits as pyfits;

from scipy.ndimage import gaussian_filter, uniform_filter, median_filter;

from . import log, files, headers, setup, oifits, signal, plot, qc;
from .headers import HM, HMQ, HMP, HMW, rep_nan;

def compute_selection (hdrs, output='output_trends', filetype='SELECTION',
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
    all_base_scan = np.zeros ((nfiles, nscan, 15));
    all_beam_scan = np.zeros ((nfiles, nscan, 6));
    
    # Loop on files to compute trend per base
    for ih,h in enumerate(hdrs):

        # Load data
        f = h['ORIGNAME'];
        log.info ('Load RTS file %s (%i over %i)'%(f,ih,nfiles));
        
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

        # Set data for base - with normalization
        all_base_scan[ih,:,:] = (base_scan - np.min(base_scan)) / (np.max(base_scan)-np.min(base_scan));

    # Compute trend per beams
    log.info ('Compute the scan per beams');
    
    for ib,beams in enumerate (setup.base_beam ()):
        all_beam_scan[:,:,beams[0]] += all_base_scan[:,:,ib];
        all_beam_scan[:,:,beams[1]] += all_base_scan[:,:,ib];

    # Make plots
    log.info ('Plots');

    # Waterfall per base
    fig,axes = plt.subplots (5,3, sharex=True);
    plot.base_name (axes);
    plot.compact (axes);
    for b in range (15):
        axes.flatten()[b].imshow (all_base_scan[:,:,b].T,aspect='auto');
    files.write (fig,output+'_base_trend.png');
    plt.close();

    # Waterfall per beam
    fig,axes = plt.subplots (6,1, sharex=True);
    plot.base_name (axes);
    plot.compact (axes);
    for b in range (6):
        axes.flatten()[b].imshow (all_beam_scan[:,:,b].T,aspect='auto');

    # Start interactive session
    log.info ('Interactive session: Click and drag over range to exclude');
    def onclick(event):
        if event.dblclick:
            xx1,xx2 = event.inaxes.get_xlim();
            event.inaxes.axvspan(xx1,xx2,facecolor='black');
        else:
            global x1;
            x1 = event.xdata;

            ## reset if click off axis
            if x1 is None:
                plt.close();
                fig,axes = plt.subplots (6,1, sharex=True);
                plot.base_name (axes);
                plot.compact (axes);
                for b in range (6):
                    axes.flatten()[b].imshow (all_beam_scan[:,:,b].T,aspect='auto');
                cid = fig.canvas.mpl_connect('button_press_event', onclick);
                cid2 = fig.canvas.mpl_connect('button_release_event', onrelease);
                plt.show();

    def onrelease(event):
        x2 = event.xdata;
        if x1 is not None:
            event.inaxes.axvspan(x1,x2,facecolor='black');
            plt.draw();

    cid = fig.canvas.mpl_connect('button_press_event', onclick);
    cid2 = fig.canvas.mpl_connect('button_release_event', onrelease);
    plt.show();

    # End interactive session
    files.write (fig,output+'_beam_trend.png');
    fig.canvas.mpl_disconnect(cid);
    fig.canvas.mpl_disconnect(cid2);
    log.info ('Save selection');

    # File
    log.info ('Create file');

    # First HDU
    hdu0 = pyfits.PrimaryHDU ([]);
    hdu0.header = hdrs[0];
    hdu0.header['FILETYPE'] = filetype;

    # Save trends if we want to re-rerun a selection
    # without having to re-load everthing maybe

    hdu1 = pyfits.ImageHDU (all_beam_scan);
    hdu1.header['EXTNAME'] = ('ALL_BEAM_SCAN',);

    hdu2 = pyfits.ImageHDU (all_base_scan);
    hdu2.header['EXTNAME'] = ('ALL_BASE_SCAN',);

    # We should save the selection somwhere if any,
    # and the filenames probably
    
    # ...
    
    # Write file
    hdulist = pyfits.HDUList ([hdu0,hdu1,hdu2]);
    files.write (hdulist, output+'.fits');
            
    plt.close("all");
    return hdulist;
    
    
    
