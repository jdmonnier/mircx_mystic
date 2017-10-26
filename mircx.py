import numpy as np
import matplotlib.pyplot as plt

from astropy.stats import sigma_clipped_stats
from astropy.io import fits as pyfits
from astropy.modeling import models, fitting

from scipy.fftpack import fft, ifft
from scipy.signal import medfilt;

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
    
def check_empty_window (cube, hdr, hdrqc):
    ''' Extract empty window from a cube(r,f,xy)'''
    log.info ('Check the empty window');
    
    # Load window
    sx = hdr['HIERARCH MIRC QC EMPTY_WIN STARTX'];
    nx = hdr['HIERARCH MIRC QC EMPTY_WIN NX'];
    sy = hdr['HIERARCH MIRC QC EMPTY_WIN STARTY'];
    ny = hdr['HIERARCH MIRC QC EMPTY_WIN NY'];

    # Crop the fringe window
    empty = cube[:,:,sy:sy+ny,sx:sx+nx]

    # Compute QC
    if hdrqc is not None:
        (mean,med,std) = sigma_clipped_stats (empty);
        hdrqc.set ('HIERARCH MIRC QC EMPTY MED',med,'[adu]');
        hdrqc.set ('HIERARCH MIRC QC EMPTY MEAN',mean,'[adu]');
        hdrqc.set ('HIERARCH MIRC QC EMPTY STD',std,'[adu]');

    return empty;
    
def compute_background (hdrs,output='output_bkg'):
    '''
    Compute BACKGROUND_MEAN file from a sequence of
    BACKGROUND. The output file had the mean and rms over
    all frames, written as ramp.
    '''
    elog = log.trace ('compute_background');

    # Check inputs
    check_hdrs_input (hdrs, required=1);
    
    # Load files
    hdr,cube = files.load_raw (hdrs, coaddRamp=True);

    # Background mean
    log.info ('Compute mean and rms over input files');
    bkg_mean = np.mean (cube, axis=0);
    bkg_std  = np.std (cube, axis=0) / np.sqrt (cube.shape[0]);
    
    # Select which one to plot
    nf,nx,ny = bkg_mean.shape;
    d = 10;
    idf = int(nf/2);
    idx = int(nx/2);
    idy = int(ny/2);
    
    # Add QC parameters
    (mean,med,std) = sigma_clipped_stats (bkg_mean[idf,idx-d:idx+d,idy-d:idy+d]);
    hdr.set ('HIERARCH MIRC QC BKG_MEAN MED',med,'[adu] for frame nf/2');
    hdr.set ('HIERARCH MIRC QC BKG_MEAN STD',std,'[adu] for frame nf/2');

    (smean,smed,sstd) = sigma_clipped_stats (bkg_std[idf,idx-d:idx+d,idy-d:idy+d]);
    hdr.set ('HIERARCH MIRC QC BKG_ERR MED',smed,'[adu] for frame nf/2');
    hdr.set ('HIERARCH MIRC QC BKG_ERR STD',sstd,'[adu] for frame nf/2');
    
    # Create output HDU
    hdu1 = pyfits.PrimaryHDU (bkg_mean);
    hdu1.header = hdr;

    # Update header
    hdu1.header['BZERO'] = 0;
    hdu1.header['BUNIT'] = 'ADU';
    hdu1.header['FILETYPE'] = 'BACKGROUND_MEAN';

    # Create second HDU
    hdu2 = pyfits.ImageHDU (bkg_std);
    
    # Update header
    hdu2.header['BZERO'] = 0;
    hdu2.header['BUNIT'] = 'ADU';
    hdu2.header['EXTNAME'] = 'BACKGROUND_ERR';

    # Write output file
    hdulist = pyfits.HDUList ([hdu1,hdu2]);
    files.write (hdulist, output+'.fits');

    # Figures
    fig,(ax1,ax2,ax3) = plt.subplots (3,1);
    ax1.imshow (bkg_mean[idf,:,:], vmin=med-5*std, vmax=med+5*std);
    ax2.imshow (bkg_mean[idf,:,:], vmin=med-20*std, vmax=med+20*std);
    ax3.imshow (bkg_std[idf,:,:], vmin=smed-20*sstd, vmax=smed+20*sstd);
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

def compute_fringemap (hdrs,bkg,output='output_fringemap'):
    '''
    Find the location of the fringe on the detector.
    '''
    elog = log.trace ('compute_fringemap');
    
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

    # Keep only fringes (supposedly smoothed in x)
    fmap = medfilt (fmean, [1,11]);
    
    # Get spectral limits of profile
    fcut = np.mean (fmap,axis=1);
    fcut /= np.max (fcut);

    idy_s = np.argmax (fcut>0.25);
    idy_e = len(fcut) - np.argmax(fcut[::-1]>0.25);
    
    log.info ('Found limit in spectral direction: %i:%i'%(idy_s,idy_e));

    # Get spatial limits of profile
    fcut = np.mean (fmap[idy_s:idy_e,:],axis=0);
    fcut /= np.max (fcut);
    
    idx_s = np.argmax (fcut>0.25);
    idx_e = len(fcut) - np.argmax(fcut[::-1]>0.25);
    
    log.info ('Found limit in spatial direction: %i:%i'%(idx_s,idx_e));

    # Cut
    fmeancut = fmean[idy_s:idy_e,idx_s:idx_e];

    # Figures
    fig,ax = plt.subplots();
    ax.imshow (fmeancut);
    fig.savefig (output+'_cut.png');

    # Add QC parameters
    hdr.set ('HIERARCH MIRC QC FRINGE_WIN STARTX',idx_s,'[pix]');
    hdr.set ('HIERARCH MIRC QC FRINGE_WIN NX',idx_e-idx_s,'[pix]');
    hdr.set ('HIERARCH MIRC QC FRINGE_WIN STARTY',idy_s,'[pix]');
    hdr.set ('HIERARCH MIRC QC FRINGE_WIN NY',idy_e-idy_s,'[pix]');

    # Add QC parameters
    hdr.set ('HIERARCH MIRC QC EMPTY_WIN STARTX',200,'[pix]');
    hdr.set ('HIERARCH MIRC QC EMPTY_WIN NX',80,'[pix]');
    hdr.set ('HIERARCH MIRC QC EMPTY_WIN STARTY',idy_e+10,'[pix]');
    hdr.set ('HIERARCH MIRC QC EMPTY_WIN NY',15,'[pix]');

    # Check background subtraction in empty region
    check_empty_window (cube, hdr, hdr);
    
    # Create output HDU
    hdu1 = pyfits.PrimaryHDU (fmeancut);
    hdu1.header = hdr;

    # Update header
    hdu1.header['BZERO'] = 0;
    hdu1.header['FILETYPE'] = 'FRINGE_MAP';

    # Set files
    hdu1.header['HIERARCH MIRC PRO BACKGROUND'] = bkg[0]['ORIGNAME'];

    # Write output file
    hdulist = pyfits.HDUList (hdu1);
    files.write (hdulist, output+'.fits');
        
    plt.close("all");
    return fmean;

def compute_beammap (hdrs,bkg,output='output_beammap'):
    '''
    Compute beam file
    '''
    elog = log.trace ('compute_beammap');

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

    # Remove the fringe window (suposedly smoothed in x)
    fmap = fmean - medfilt (fmean, [1,11]);
    fmap = medfilt (fmap, [3,1]);

    # Fit x-position with Gaussian
    fx = np.mean (fmap, axis=0);
    x  = np.arange(len(fx));
    gx_init = models.Gaussian1D (amplitude=np.max(fx), mean=np.argmax(fx), stddev=1.);
    gx = fitting.LevMarLSQFitter()(gx_init, x, fx);
    idx = int(round(gx.mean - 3*gx.stddev));
    nx  = int(round(6*gx.stddev)+1);

    # Get y-position
    fy  = medfilt (np.mean (fmap[:,idx:idx+nx], axis=1), 5);
    fy /= np.max (fy);
    idy = np.argmax (fy>0.25) - 1;
    ny  = (len(fy) - np.argmax(fy[::-1]>0.25)) + 1 - idy;

    # Cut
    fmeancut = fmean[idy:idy+ny,idx:idx+nx];

    # Add QC parameters for window
    name = 'HIERARCH MIRC QC '+hdrs[0]['FILETYPE']+'_WIN';
    hdr.set (name+' STARTX',idx,'[pix]');
    hdr.set (name+' NX',nx,'[pix]');
    hdr.set (name+' STARTY',idy,'[pix]');
    hdr.set (name+' NY',ny,'[pix]');

    # Add QC parameters to for optimal extraction
    hdr.set (name+' SIGMAX',gx.stddev.value,'[pix]');
    hdr.set (name+' MEANX',gx.mean.value,'[pix]');

    # Figures
    fig,ax = plt.subplots(3,1);
    ax[0].imshow (fmap);
    ax[1].plot (fx, label='Data');
    ax[1].plot (x[idx:idx+nx],gx(x[idx:idx+nx]), label='Gaussian');
    ax[1].legend ();
    fig.savefig (output+'_fit.png');

    fig,ax = plt.subplots();
    ax.imshow (fmeancut);
    fig.savefig (output+'_cut.png');

    # Create output HDU
    hdu1 = pyfits.PrimaryHDU (fmeancut);
    hdu1.header = hdr;

    # Update header
    hdu1.header['BZERO'] = 0;
    hdu1.header['FILETYPE'] = hdrs[0]['FILETYPE']+'_MAP';

    # Set files
    hdu1.header['HIERARCH MIRC PRO BACKGROUND'] = bkg[0]['ORIGNAME'];

    # Write output file
    hdulist = pyfits.HDUList (hdu1);
    files.write (hdulist, output+'.fits');
    
    
    plt.close("all");
    return fmean;

def compute_preproc (hdrs,bkg,fmap,output='output_preproc'):
    '''
    Compute preproc file
    '''
    elog = log.trace ('compute_preproc');

    # Check inputs
    check_hdrs_input (hdrs, required=1);
    check_hdrs_input (bkg, required=1);
    check_hdrs_input (fmap, required=1);

    # Load files
    hdr,cube = files.load_raw (hdrs);

    # Remove background
    remove_background (cube, bkg[0]);

    # Check background subtraction in empty region
    check_empty_window (cube, fmap[0], hdr);
    
    # Crop fringe part
    fringe = crop_fringe_window (cube, fmap[0]);

    # Create output HDU
    log.info ('Create file');
    hdu1 = pyfits.PrimaryHDU (fringe);
    hdu1.header = hdr;

    # Update header
    hdu1.header['BZERO'] = 0;
    hdu1.header['BUNIT'] = 'ADU';
    hdu1.header['FILETYPE'] = 'PREPROC';

    # Set files
    hdu1.header['HIERARCH MIRC PRO BACKGROUND'] = bkg[0]['ORIGNAME'];
    hdu1.header['HIERARCH MIRC PRO FRINGE_MAP'] = fmap[0]['ORIGNAME'];

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
    hdu1.header['HIERARCH MIRC PRO BACKGROUND'] = bkg['ORIGNAME'];

    # Write output file
    hdulist = pyfits.HDUList (hdu1);
    files.write (hdulist, output+'.fits');
        
    plt.close("all");
    return fringe;
