import numpy as np
import os
import matplotlib.pyplot as plt
from astropy.stats import sigma_clipped_stats
from astropy.io import fits as pyfits
from scipy.fftpack import fft, ifft

from . import log, files

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
    # Load window
    log.info ('Load window %s'%hdr['ORIGNAME']);
    hdulist  = pyfits.open(hdr['ORIGNAME']);
    sx = hdulist[0].header['HIERARCH MIRC QC FRINGE_WIN STARTX'];
    nx = hdulist[0].header['HIERARCH MIRC QC FRINGE_WIN NX'];
    sy = hdulist[0].header['HIERARCH MIRC QC FRINGE_WIN STARTY'];
    ny = hdulist[0].header['HIERARCH MIRC QC FRINGE_WIN NY'];
    hdulist.close ();

    # Crop the fringe window
    log.info ('Extract the fringe window');
    output = cube[:,:,sy:sy+ny,sx:sx+nx]

    return output;
    
def compute_background (hdrs,output=None,overwrite=True):
    '''
    Compute BACKGROUND_REDUCED file from a sequence of
    BACKGROUND. The output file had the mean and rms over
    all frames, written as ramp.
    '''
    elog = log.trace ('compute_background');
    
    # Default output
    if output is None:
        output = files.calib_output (hdrs[0])+'_bkg';

    # Check
    if os.path.exists (output+'.fits') and overwrite is False:
        log.info ('Product already exists');
        return 1;
    
    # Load files
    hdr,cube = files.load_raw (hdrs, coaddRamp=True);

    # Background mean and rms
    log.info ('Compute mean');
    bkg_mean = cube[0,:,:,:];
    
    # Add QC parameters
    nf,nx,ny = bkg_mean.shape;
    d = 10;
    
    (mean,med,std) = sigma_clipped_stats (bkg_mean[nf/2,nx/2-d:nx/2+d,ny/2-d:ny/2+d]);
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
    fig,ax = plt.subplots();
    ax.imshow (bkg_mean[nf/2,:,:]);
    fig.savefig (output+'_mean.png');

    plt.close("all");
    return hdulist;

def compute_windows (hdrs,bkg,output=None,overwrite=True):
    '''
    Find the location of the fringe on the detector.
    The output file contains a binary (0/1) image.
    '''
    elog = log.trace ('compute_windows');
    
    # Check inputs
    bkg = bkg[0] if type(bkg) == list else bkg;
    
    # Default output
    if output is None:
        output = files.calib_output (hdrs[0])+'_win';

    # Check
    if os.path.exists (output+'.fits') and overwrite is False:
        log.info ('Product already exists');
        return 1;

    # Load files
    hdr,cube = files.load_raw (hdrs, coaddRamp=True);

    # Remove background
    remove_background (cube, bkg);

    # Compute the sum
    log.info ('Compute sum');
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

    # Compute the window map as binary
    pixmap = np.zeros(fmean.shape);
    pixmap[idy_s:idy_e,idx_s:idx_e] = 1;

    # Add QC parameters
    hdr.set ('HIERARCH MIRC QC FRINGE_WIN STARTX',idx_s,'[pix]');
    hdr.set ('HIERARCH MIRC QC FRINGE_WIN NX',idx_e-idx_s,'[pix]');
    hdr.set ('HIERARCH MIRC QC FRINGE_WIN STARTY',idy_s,'[pix]');
    hdr.set ('HIERARCH MIRC QC FRINGE_WIN NY',idy_e-idy_s,'[pix]');
    
    # Create output HDU
    hdu1 = pyfits.PrimaryHDU (fmeancut);
    hdu1.header = hdr;

    # Update header
    hdu1.header['BZERO'] = 0;
    hdu1.header['FILETYPE'] = 'PIXMAP';

    # Set files
    for i,h in enumerate(hdrs):
        hdu1.header['HIERARCH MIRC PRO RAW%i'%i] = h['ORIGNAME'];
    hdu1.header['HIERARCH MIRC PRO BACKGROUND'] = bkg['ORIGNAME'];

    # Write output file
    hdulist = pyfits.HDUList (hdu1);
    files.write (hdulist, output+'.fits');
        
    plt.close("all");
    return fmean;

def compute_preproc (hdrs,bkg,win,output=None,overwrite=True):
    '''
    Compute preproc file
    '''
    elog = log.trace ('compute_preproc');

    # Check inputs
    bkg = bkg[0] if type(bkg) == list else bkg;
    win = win[0] if type(win) == list else win;

    # Default output
    if output is None:
        output = files.reduced_output (hdrs[0])+'_preproc';

    # Check
    if os.path.exists (output+'.fits') and overwrite is False:
        log.info ('Product already exists');
        return 1;
        
    # Load files
    hdr,cube = files.load_raw (hdrs);

    # Remove background
    remove_background (cube, bkg);

    # Check background subtraction in empty region
    empty = np.mean (cube[:,:,30:45,150:250], axis=(0,1));
    (mean,med,std) = sigma_clipped_stats (empty);
    hdr.set ('HIERARCH MIRC QC EMPTY MED',med,'[adu]');
    hdr.set ('HIERARCH MIRC QC EMPTY MEAN',mean,'[adu]');
    hdr.set ('HIERARCH MIRC QC EMPTY STD',std,'[adu]');
    
    # Crop fringe part
    fringe = crop_fringe_window (cube, win);

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
    hdu1.header['HIERARCH MIRC PRO BACKGROUND'] = bkg['ORIGNAME'];
    hdu1.header['HIERARCH MIRC PRO WINDOW'] = win['ORIGNAME'];

    # Write output file
    hdulist = pyfits.HDUList (hdu1);
    files.write (hdulist, output+'.fits');
    
    plt.close("all");
    return fringe;
    
def compute_snr (hdrs,output=None,overwrite=True):

    nr,nf,nx,ny = fringe.shape;
    
    # Compute fft
    log.info ('Compute FFT');
    fringe_ft = fft (fringe, axis=-1);

    # Compute integrated PSD (no coherent integration)
    log.info ('Compute PSD');
    mean_psd = np.mean (np.abs(fringe_ft)**2, (0,1));

    # Figures
    (mean,med,std) = sigma_clipped_stats (mean_psd);
    fig,ax = plt.subplots(3,1);
    ax[0].imshow (mean_psd[:,0:ny/2],vmin=med-5*std,vmax=med+5*std);
    ax[1].plot (mean_psd[:,0:ny/2].T);
    ax[2].plot (mean_psd[:,0:ny/2].T); ax[2].set_ylim (med-3*std, med+3*std);
    fig.savefig (output+'_psd.png');

    # Compute cross-spectra
    log.info ('Compute CSP');
    csd = 0.5 * fringe_ft[:,0:nf-3:4] * np.conj (fringe_ft[:,2:nf-1:4]) + \
          0.5 * fringe_ft[:,1:nf-2:4] * np.conj (fringe_ft[:,3:nf-0:4]);
    mean_psd = np.mean (csd, (0,1)).real;

    # Figures
    (mean,med,std) = sigma_clipped_stats (mean_psd);
    fig,ax = plt.subplots(3,1);
    ax[0].imshow (mean_psd[:,0:ny/2],vmin=med-5*std,vmax=med+5*std);
    ax[1].plot (mean_psd[:,0:ny/2].T);
    ax[2].plot (mean_psd[:,0:ny/2].T); ax[2].set_ylim (med-3*std, med+3*std);
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
