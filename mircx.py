import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import sigma_clipped_stats
from astropy.io import fits as pyfits
from scipy.fftpack import fft, ifft

from . import log, files

def check_hdrs_input (hdrs, required=1):
    ''' Check the input when provided as hdrs'''

    # Ensure a list
    if type(hdrs) is not list:
        hdrs = [hdrs];

    # Check inputs are headers
    hdrs = [h for h in hdrs if type(h) is pyfits.header.Header or \
            type(h) is pyfits.hdu.compressed.CompImageHeader];

    if len(hdrs) < required:
        raise ValueError ('Missing mandatory input');
    

def remove_background (cube, hdr):
    ''' Remove the background from a cube(r,f,xy), in-place'''

    # Load background
    log.info ('Load background %s'%hdr['ORIGNAME']);
    hdulist  = pyfits.open(hdr['ORIGNAME']);
    bkg_data = hdulist[0].data;
    hdulist.close ();
    
    # Remove background
    log.info ('Remove background');
    cube -= bkg_data[None,:,:,:];

def crop_fringe_window (cube, hdr):
    ''' Extract fringe window from a cube(r,f,xy)'''
    log.info ('Extract the fringe window');
    
    # Load window
    sx = hdr['HIERARCH MIRC QC FRINGE_WIN STARTX'];
    nx = hdr['HIERARCH MIRC QC FRINGE_WIN NX'];
    sy = hdr['HIERARCH MIRC QC FRINGE_WIN STARTY'];
    ny = hdr['HIERARCH MIRC QC FRINGE_WIN NY'];

    # Crop the fringe window
    output = cube[:,:,sy:sy+ny,sx:sx+nx]

    return output;

def crop_empty_window (cube, hdr):
    ''' Extract empty window from a cube(r,f,xy)'''
    log.info ('Extract the empty window');
    
    # Load window
    sx = hdr['HIERARCH MIRC QC EMPTY_WIN STARTX'];
    nx = hdr['HIERARCH MIRC QC EMPTY_WIN NX'];
    sy = hdr['HIERARCH MIRC QC EMPTY_WIN STARTY'];
    ny = hdr['HIERARCH MIRC QC EMPTY_WIN NY'];

    # Crop the fringe window
    output = cube[:,:,sy:sy+ny,sx:sx+nx]

    return output;
    
def compute_background (hdrs,output='output_bkg'):
    '''
    Compute BACKGROUND_REDUCED file from a sequence of
    BACKGROUND. The output file had the mean and rms over
    all frames, written as ramp.
    '''
    elog = log.trace ('compute_background');

    # Check inputs
    check_hdrs_input (hdrs, required=1);
    
    # Load files
    hdr,cube = files.load_raw (hdrs, coaddRamp=True);

    # Background mean
    log.info ('Compute mean over ramps');
    bkg_mean = np.mean (cube, axis=0);
    
    # Add QC parameters
    nf,nx,ny = bkg_mean.shape;
    d = 10;

    # Select which one to plot
    idf = int(nf/2);
    idx = int(nx/2);
    idy = int(ny/2);
    
    (mean,med,std) = sigma_clipped_stats (bkg_mean[idf,idx-d:idx+d,idy-d:idy+d]);
    hdr.set ('HIERARCH MIRC QC MEAN MED',med,'[adu]');
    hdr.set ('HIERARCH MIRC QC MEAN STD',std,'[adu]');

    # Create output HDU
    hdu1 = pyfits.PrimaryHDU (bkg_mean);
    hdu1.header = hdr;

    # Update header
    hdu1.header['BZERO'] = 0;
    hdu1.header['BUNIT'] = 'ADU';
    hdu1.header['FILETYPE'] = 'BACKGROUND_REDUCED';
    for i,h in enumerate(hdrs):
        hdu1.header['HIERARCH MIRC PRO RAW%i'%i] = h['ORIGNAME'];

    # Write output file
    hdulist = pyfits.HDUList ([hdu1]);
    files.write (hdulist, output+'.fits');

    # Figures
    fig,(ax1,ax2) = plt.subplots (2,1);
    ax1.imshow (bkg_mean[idf,:,:], vmin=med-5*std, vmax=med+5*std);
    ax2.imshow (bkg_mean[idf,:,:], vmin=med-20*std, vmax=med+20*std);
    fig.savefig (output+'_mean.png');

    fig,ax = plt.subplots();
    ax.hist (bkg_mean[idf,:,:].flatten(),bins=med+std*np.linspace(-10,10,50));
    ax.set_xlabel ("Value at frame nf/2 (adu)");
    ax.set_ylabel ("Number of pixels");
    fig.savefig (output+'_histo.png');

    fig,ax = plt.subplots();
    ax.plot (np.median (bkg_mean,axis=(1,2)));
    ax.set_xlabel ("Frame");
    ax.set_ylabel ("Median of pixels (adu)");
    fig.savefig (output+'_ramp.png');

    plt.close("all");
    return hdulist;

def compute_pixmap (hdrs,bkg,output='output_pixmap'):
    '''
    Find the location of the fringe on the detector.
    The output file contains a binary (0/1) image.
    '''
    elog = log.trace ('compute_pixmap');
    
    # Check inputs
    check_hdrs_input (hdrs, required=1);
    check_hdrs_input (bkg, required=1);
    
    # Load files
    hdr,cube = files.load_raw (hdrs, coaddRamp=True);

    # Remove background
    remove_background (cube, bkg[0]);

    # Compute the sum
    log.info ('Compute mean over ramps and frames');
    fmean = np.mean (cube, axis=(0,1));

    # Figures
    fig,ax = plt.subplots();
    ax.imshow (fmean);
    fig.savefig (output+'_sum.png');
    
    # Get spectral limits of profile
    fcut = np.median (fmean,axis=1);
    fcut /= np.max (fcut);

    idy_s = np.argmax(fcut>0.25);
    idy_e = len(fcut) - np.argmax(fcut[::-1]>0.25);
    
    log.info ('Found limit in spectral direction: %i:%i'%(idy_s,idy_e));
    fmeancut = fmean[idy_s:idy_e,:];

    # Get spatial limits of profile
    fcut = np.median (fmeancut,axis=0);
    fcut /= np.max (fcut);
    
    idx_s = np.argmax(fcut>0.25);
    idx_e = len(fcut) - np.argmax(fcut[::-1]>0.25);
    
    log.info ('Found limit in spatial direction: %i:%i'%(idx_s,idx_e));
    fmeancut = fmeancut[:,idx_s:idx_e];

    # Figures
    fig,ax = plt.subplots();
    ax.imshow (fmeancut);
    fig.savefig (output+'_cut.png');

    # Compute the pix map as binary
    pixmap = np.zeros(fmean.shape);
    pixmap[idy_s:idy_e,idx_s:idx_e] = 1;

    # Add QC parameters
    hdr.set ('HIERARCH MIRC QC FRINGE_WIN STARTX',idx_s,'[pix]');
    hdr.set ('HIERARCH MIRC QC FRINGE_WIN NX',idx_e-idx_s,'[pix]');
    hdr.set ('HIERARCH MIRC QC FRINGE_WIN STARTY',idy_s,'[pix]');
    hdr.set ('HIERARCH MIRC QC FRINGE_WIN NY',idy_e-idy_s,'[pix]');

    # Add QC parameters
    hdr.set ('HIERARCH MIRC QC EMPTY_WIN STARTX',200,'[pix]');
    hdr.set ('HIERARCH MIRC QC EMPTY_WIN NX',50,'[pix]');
    hdr.set ('HIERARCH MIRC QC EMPTY_WIN STARTY',idy_e+10,'[pix]');
    hdr.set ('HIERARCH MIRC QC EMPTY_WIN NY',10,'[pix]');

    # Check background subtraction in empty region
    empty = crop_empty_window (cube, hdr);
    empty = np.mean (empty, axis=(0,1));
    (mean,med,std) = sigma_clipped_stats (empty);
    hdr.set ('HIERARCH MIRC QC EMPTY MED',med,'[adu]');
    hdr.set ('HIERARCH MIRC QC EMPTY MEAN',mean,'[adu]');
    hdr.set ('HIERARCH MIRC QC EMPTY STD',std,'[adu]');
    
    # Create output HDU
    hdu1 = pyfits.PrimaryHDU (fmeancut);
    hdu1.header = hdr;

    # Update header
    hdu1.header['BZERO'] = 0;
    hdu1.header['FILETYPE'] = 'PIXMAP';

    # Set files
    for i,h in enumerate(hdrs):
        hdu1.header['HIERARCH MIRC PRO RAW%i'%i] = h['ORIGNAME'];
    hdu1.header['HIERARCH MIRC PRO BACKGROUND'] = bkg[0]['ORIGNAME'];

    # Write output file
    hdulist = pyfits.HDUList (hdu1);
    files.write (hdulist, output+'.fits');
        
    plt.close("all");
    return fmean;

def compute_preproc (hdrs,bkg,pmap,output='output_pixmap'):
    '''
    Compute preproc file
    '''
    elog = log.trace ('compute_preproc');

    # Check inputs
    check_hdrs_input (hdrs, required=1);
    check_hdrs_input (bkg, required=1);
    check_hdrs_input (pmap, required=1);

    # Load files
    hdr,cube = files.load_raw (hdrs);

    # Remove background
    remove_background (cube, bkg[0]);

    # Check background subtraction in empty region
    empty = crop_empty_window (cube, hdr);
    empty = np.mean (empty, axis=(0,1));
    (mean,med,std) = sigma_clipped_stats (empty);
    hdr.set ('HIERARCH MIRC QC EMPTY MED',med,'[adu]');
    hdr.set ('HIERARCH MIRC QC EMPTY MEAN',mean,'[adu]');
    hdr.set ('HIERARCH MIRC QC EMPTY STD',std,'[adu]');
    
    # Crop fringe part
    fringe = crop_fringe_window (cube, pmap[0]);

    # Create output HDU
    log.info ('Create file');
    hdu1 = pyfits.PrimaryHDU (fringe);
    hdu1.header = hdr;

    # Update header
    hdu1.header['BZERO'] = 0;
    hdu1.header['BUNIT'] = 'ADU';
    hdu1.header['FILETYPE'] = 'PREPROC';

    # Set files
    for i,h in enumerate(hdrs):
        hdu1.header['HIERARCH MIRC PRO RAW%i'%i] = h['ORIGNAME'];
    hdu1.header['HIERARCH MIRC PRO BACKGROUND'] = bkg[0]['ORIGNAME'];
    hdu1.header['HIERARCH MIRC PRO PIXMAP'] = pmap[0]['ORIGNAME'];

    # Write output file
    hdulist = pyfits.HDUList (hdu1);
    files.write (hdulist, output+'.fits');
    
    plt.close("all");
    return fringe;
    
def compute_snr (hdrs,output=None,overwrite=True):

    nr,nf,nx,ny = fringe.shape;
    ny2 = int(ny/2);

    # Check inputs
    check_hdrs_input (hdrs, required=1);
    
    # Compute fft
    log.info ('Compute FFT');
    fringe_ft = fft (fringe, axis=-1);

    # Compute integrated PSD (no coherent integration)
    log.info ('Compute PSD');
    mean_psd = np.mean (np.abs(fringe_ft)**2, (0,1));

    # Figures
    (mean,med,std) = sigma_clipped_stats (mean_psd);
    fig,ax = plt.subplots(3,1);
    ax[0].imshow (mean_psd[:,0:ny2],vmin=med-5*std,vmax=med+5*std);
    ax[1].plot (mean_psd[:,0:ny2].T);
    ax[2].plot (mean_psd[:,0:ny2].T); ax[2].set_ylim (med-3*std, med+3*std);
    fig.savefig (output+'_psd.png');

    # Compute cross-spectra
    log.info ('Compute CSP');
    csd = 0.5 * fringe_ft[:,0:nf-3:4] * np.conj (fringe_ft[:,2:nf-1:4]) + \
          0.5 * fringe_ft[:,1:nf-2:4] * np.conj (fringe_ft[:,3:nf-0:4]);
    mean_psd = np.mean (csd, (0,1)).real;

    # Figures
    (mean,med,std) = sigma_clipped_stats (mean_psd);
    fig,ax = plt.subplots(3,1);
    ax[0].imshow (mean_psd[:,0:ny2],vmin=med-5*std,vmax=med+5*std);
    ax[1].plot (mean_psd[:,0:ny2].T);
    ax[2].plot (mean_psd[:,0:ny2].T); ax[2].set_ylim (med-3*std, med+3*std);
    fig.savefig (output+'_csd.png');
    
    
    # Create output HDU
    hdu1 = pyfits.PrimaryHDU (fringe);
    hdu1.header = hdr;

    # Update header
    hdu1.header['BZERO'] = 0;
    hdu1.header['BUNIT'] = 'ADU';
    hdu1.header['FILETYPE'] = 'SNR';

    # Set files
    for i,h in enumerate(hdrs):
        hdu1.header['HIERARCH MIRC PRO RAW%i'%i] = h['ORIGNAME'];
    hdu1.header['HIERARCH MIRC PRO BACKGROUND'] = bkg['ORIGNAME'];

    # Write output file
    hdulist = pyfits.HDUList (hdu1);
    files.write (hdulist, output+'.fits');
        
    plt.close("all");
    return fringe;
