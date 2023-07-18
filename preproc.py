import numpy as np;
import os;

import matplotlib.pyplot as plt;
import matplotlib.colors as mcolors;
import matplotlib
matplotlib.use('TkAgg')

from astropy.stats import sigma_clipped_stats;
from astropy.io import fits as pyfits;
from astropy.modeling import models, fitting;
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel

#from skimage.feature import register_translation;
from skimage.registration import phase_cross_correlation;

from scipy import fftpack;
from scipy.signal import medfilt;
from scipy.signal import lombscargle;
from scipy.ndimage.interpolation import shift as subpix_shift;
from scipy.ndimage import gaussian_filter;
from scipy.optimize import least_squares;

from . import log, files, headers, setup, oifits, signal, plot;
from .headers import HM, HMQ, HMP, HMW, rep_nan;

import warnings
warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
warnings.filterwarnings(action='ignore', message='Mean of empty slice')
    
def define_badpixels (bkg, threshold=5.):
    '''
    Define bad pixels from a cube, given an input background
    on which the bad-pixels are detected.
    '''

    # Set log in input header
    hdr = bkg[0];

    # Get bad pixels from rms of polyfit
    # bkg_noise = pyfits.getdata (bkg[0]['ORIGNAME'],0);
    # rms = [];
    # ny = bkg_noise.shape[2];
    # nx = bkg_noise.shape[3];
    # for i in np.arange(ny):
    #     for j in np.arange(nx):
    #         pixel = bkg_noise[0,:,i,j];
    #         frame = np.arange(len(pixel));
    #         p,res,_,_,_ = np.polyfit(frame,pixel,2,full=True);
    #         rms.append(res);
    # rms = np.sqrt(np.array(rms)/len(rms));
    # rms_std = np.std(rms);
    # rms = rms.reshape(ny,nx);
    # thr = threshold;
    # bad_rms = rms > rms_std*thr;
    # log.info ('Found %i bad pixels in RMS'%np.sum (bad_rms));

    # Load background error
    bkg_noise = pyfits.getdata (bkg[0]['ORIGNAME'],0);
    bkg_noise = np.mean (bkg_noise, (0,1));
    delta = bkg_noise - medfilt (bkg_noise, (3,3));
    stat = sigma_clipped_stats (delta);
    thr_mean = threshold;
    bad_mean = np.abs(delta-stat[0])/stat[2] > thr_mean;

    hdr[HMQ+'BADPIX MEAN_THRESHOLD'] = (thr_mean, 'threshold in sigma');
    hdr[HMQ+'BADPIX MEAN_NUMBER'] = (np.sum (bad_mean), 'nb. of badpix');
    log.info ('Found %i bad pixels in MEAN'%np.sum (bad_mean));

    # Load background error
    bkg_noise = pyfits.getdata (bkg[0]['ORIGNAME'],'BACKGROUND_ERR');
    bkg_noise = np.mean (bkg_noise, (0,1));
    delta = bkg_noise - medfilt (bkg_noise, (3,3));
    stat = sigma_clipped_stats (delta);
    thr_err = threshold;
    bad_err = np.abs(delta-stat[0])/stat[2] > thr_err;

    hdr[HMQ+'BADPIX ERR_THRESHOLD'] = (thr_err, 'threshold in sigma');
    hdr[HMQ+'BADPIX ERR_NUMBER'] = (np.sum (bad_err), 'nb. of badpix');
    log.info ('Found %i bad pixels in ERR'%np.sum (bad_err));
    
    # Load background error
    bkg_noise = pyfits.getdata (bkg[0]['ORIGNAME'],'BACKGROUND_NOISE');
    bkg_noise = np.mean (bkg_noise, (0,1));
    delta = bkg_noise - medfilt (bkg_noise, (3,3));
    stat = sigma_clipped_stats (delta);
    thr_noise = threshold;
    bad_noise = np.abs(delta-stat[0])/stat[2] > thr_noise;

    hdr[HMQ+'BADPIX NOISE_THRESHOLD'] = (thr_noise, 'threshold in sigma');
    hdr[HMQ+'BADPIX NOISE_NUMBER'] = (np.sum (bad_noise), 'nb. of badpix');
    log.info ('Found %i bad pixels in NOISE'%np.sum (bad_noise));

    # Ignore the badpixels on the edges
    bad = bad_mean + bad_err + bad_noise;
    bad[0,:] = False; bad[-1,:] = False;
    bad[:,0] = False; bad[:,-1] = False;

    # Return the image of badpixel
    return bad;

def check_empty_window (cube, hdr):
    '''
    Check the level and noise in an empty
    window from a cube(r,f,xy)
    '''
    log.info ('Check the empty window');
    
    # Get dimension
    nr,nf,ny,nx = cube.shape;
    
    # Hardcoded defined
    sx,nx = 200,80;
    sy,ny = int(0.55*ny), int(0.85*ny - 0.55*ny);
    log.info ('Empty window: %i,%i, %i,%i'%(sx,nx,sy,ny));

    # Add QC parameters
    hdr[HMQ+'WIN EMPTY STARTX'] = (sx,'[pix] python-ref');
    hdr[HMQ+'WIN EMPTY STARTY'] = (sy,'[pix] python-ref');
    hdr[HMQ+'WIN EMPTY NX'] = (nx,'[pix]');
    hdr[HMQ+'WIN EMPTY NY'] = (ny,'[pix]');

    # Crop the empty window
    empty = np.mean (cube[:,:,sy:sy+ny,sx:sx+nx], axis=(0,1));

    # Compute QC
    (mean,med,std) = sigma_clipped_stats (empty);
    
    # Set QC
    log.info (HMQ+'EMPTY MED = %.2f [adu]'%med);
    hdr[HMQ+'EMPTY MED'] = (med,'[adu]');
    hdr[HMQ+'EMPTY MEAN'] = (mean,'[adu]');
    hdr[HMQ+'EMPTY STD'] = (std,'[adu]');
    
    return empty;

def compute_background_archive (hdrs, output='output_bkg', filetype='BACKGROUND_MEAN', linear=True): # depricate `linear` after testing
    '''
    BACKGROUND. The output file had the mean and rms over
    all frames, written as ramp.
    '''
    elog = log.trace ('compute_background');
    # Check inputs
    headers.check_input (hdrs, required=1);

    # Load files
    hdr,cube,mjd = files.load_raw (hdrs, coaddRamp='mean',
                                   saturationThreshold=None,
                                   continuityThreshold=None,
                                   linear=linear);
    log.info ('Data size: '+str(cube.shape));

    # Background mean
    log.info ('Compute mean and rms over input files');
    bkg_mean = np.mean (cube, axis=0);
    bkg_err  = np.std (cube, axis=0) / np.sqrt (cube.shape[0]);

    # Load all ramp of first file to measure readout noise
    __,cube,__ = files.load_raw (hdrs[0:1], coaddRamp='none',
                                 saturationThreshold=None,
                                 continuityThreshold=None,
                                 linear=linear);

    # Compute temporal rms
    log.info ('Compute rms over ramp/frame of first file');
    bkg_noise = np.std (cube[:,3:-3,:,:], axis=(0,1));
    
    # Select the region for the QC parameters
    nf,ny,nx = bkg_mean.shape;
    dy,dx = 15,35;
    idf,idy,idx = int(nf/2), int(ny/2), int(nx/2);
    log.info ('Compute QC in box (%i,%i:%i,%i:%i)'%(idf,idy-dy,idy+dy,idx-dx,idx+dx));

    # Add QC parameters
    (mean,med,std) = sigma_clipped_stats (bkg_mean[idf,idy-dy:idy+dy,idx-dx:idx+dx]);
    log.info ('BKG_MEAN MED = %f'%med);
    log.info ('BKG_MEAN STD = %f'%std);
    hdr.set (HMQ+'BKG_MEAN MED',med,'[adu] for frame nf/2');
    hdr.set (HMQ+'BKG_MEAN STD',std,'[adu] for frame nf/2');

    (emean,emed,estd) = sigma_clipped_stats (bkg_err[idf,idy-dy:idy+dy,idx-dx:idx+dx]);
    log.info ('BKG_ERR MED = %f'%emed);
    log.info ('BKG_ERR STD = %f'%estd);
    hdr.set (HMQ+'BKG_ERR MED',emed,'[adu] for frame nf/2');
    hdr.set (HMQ+'BKG_ERR STD',estd,'[adu] for frame nf/2');
    
    (nmean,nmed,nstd) = sigma_clipped_stats (bkg_noise[idy-dy:idy+dy,idx-dx:idx+dx]);
    log.info ('BKG_NOISE MED = %f'%nmed);
    log.info ('BKG_NOISE STD = %f'%nstd);
    hdr.set (HMQ+'BKG_NOISE MED',round(nmed,5),'[adu] for first file');
    hdr.set (HMQ+'BKG_NOISE STD',round(nstd,5),'[adu] for first file');

    # Define quality flag
    hdr[HMQ+'QUALITY'] = (1./(emed+1e-10), 'quality of data');
    
    # Create output HDU
    hdu0 = pyfits.PrimaryHDU (bkg_mean[None,:,:,:]);
    hdu0.header = hdr;

    # Update header
    hdu0.header['FILETYPE'] = filetype;
    hdu0.header['BUNIT'] = 'adu/pixel/frame';
    hdu0.header['SHAPE'] = '(nr,nf,ny,nx)';

    # Create second HDU
    hdu1 = pyfits.ImageHDU (bkg_err[None,:,:,:]);
    hdu1.header['EXTNAME'] = ('BACKGROUND_ERR','uncertainty on background mean');
    hdu1.header['BUNIT'] = 'adu/pixel/frame';
    hdu1.header['SHAPE'] = '(nr,nf,ny,nx)';

    # Create third HDU
    hdu2 = pyfits.ImageHDU (bkg_noise[None,None,:,:]);
    hdu2.header['EXTNAME'] = ('BACKGROUND_NOISE','pixel frame-to-frame noise');
    hdu2.header['BUNIT'] = 'adu/pixel/frame';
    hdu2.header['SHAPE'] = '(nr,nf,ny,nx)';
    
    # Write output file
    hdulist = pyfits.HDUList ([hdu0,hdu1,hdu2]);
    files.write (hdulist, output+'.fits');

    # Figures
    log.info ('Figures');

    # Images of mean
    fig,ax = plt.subplots (2,1);
    fig.suptitle (headers.summary (hdr));
    ax[0].imshow (bkg_mean[idf,:,:], vmin=med-5*std, vmax=med+5*std);
    ax[0].set_ylabel ('Mean (adu) +-5sig');
    ax[1].imshow (bkg_mean[idf,:,:], vmin=med-20*std, vmax=med+20*std);
    ax[1].set_ylabel ('Mean (adu) +-20sig');
    files.write (fig, output+'_mean.png');

    # Images of noise
    fig,ax = plt.subplots (2,1);
    fig.suptitle (headers.summary (hdr));
    ax[0].imshow (bkg_noise, vmin=nmed-5*nstd, vmax=nmed+5*nstd);
    ax[0].set_ylabel ('Noise (adu) +-5sig');
    ax[1].imshow (bkg_noise, vmin=nmed-20*nstd, vmax=nmed+20*nstd);
    ax[1].set_ylabel ('Noise (adu) +-20sig');
    fig.suptitle (headers.summary (hdr));
    files.write (fig, output+'_noise.png');

    # Images of error
    fig,ax = plt.subplots (2,1);
    fig.suptitle (headers.summary (hdr));
    ax[0].imshow (bkg_err[idf,:,:], vmin=emed-5*estd, vmax=emed+5*estd);
    ax[0].set_ylabel ('Err (adu) +-5sig');
    ax[1].imshow (bkg_err[idf,:,:], vmin=emed-20*estd, vmax=emed+20*estd);
    ax[1].set_ylabel ('Err (adu) +-20sig');
    files.write (fig, output+'_err.png');
    
    # Histograms of median
    fig,ax = plt.subplots (2,1);
    fig.suptitle (headers.summary (hdr));
    ax[0].hist (bkg_mean[idf,:,:].flatten(),bins=med+std*np.linspace(-10,20,50));
    ax[0].set_ylabel ("Number of pixels");
    ax[1].hist (bkg_mean[idf,:,:].flatten(),bins=med+std*np.linspace(-10,20,50));
    ax[1].set_ylabel ("Number of pixels");
    ax[1].set_xlabel ("Value at frame nf/2 (adu)");
    ax[1].set_yscale ('log');
    files.write (fig, output+'_histomean.png');

    # Histograms of error
    fig,ax = plt.subplots (2,1);
    fig.suptitle (headers.summary (hdr));
    ax[0].hist (bkg_err.flatten(),bins=emed+estd*np.linspace(-10,20,50));
    ax[0].set_ylabel ("Number of pixels");
    ax[1].hist (bkg_err.flatten(),bins=emed+estd*np.linspace(-10,20,50));
    ax[1].set_ylabel ("Number of pixels");
    ax[1].set_xlabel ("RMS(file)/sqrt(nfile)");
    ax[1].set_yscale ('log');
    files.write (fig, output+'_histoerr.png');

    # Histograms of noise
    fig,ax = plt.subplots (2,1);
    fig.suptitle (headers.summary (hdr));
    ax[0].hist (bkg_noise.flatten(),bins=nmed+nstd*np.linspace(-10,20,50));
    ax[0].set_ylabel ("Number of pixels");
    ax[1].hist (bkg_noise.flatten(),bins=nmed+nstd*np.linspace(-10,20,50));
    ax[1].set_ylabel ("Number of pixels");
    ax[1].set_xlabel ("RMS(frame) for first file");
    ax[1].set_yscale ('log');
    files.write (fig, output+'_histonoise.png');

    # Ramp
    fig,ax = plt.subplots();
    fig.suptitle (headers.summary (hdr));
    ax.plot (np.median (bkg_mean,axis=(1,2)));
    ax.set_xlabel ("Frame");
    ax.set_ylabel ("Median of pixels (adu)");
    files.write (fig, output+'_ramp.png');

    plt.close ("all");
    del elog;
    return hdulist;

def compute_background (hdrs, output='output_bkg', filetype='BACKGROUND_MEAN'): 
    '''
    Compute BACKGROUND_MEAN file from a sequence of
    BACKGROUND. The output file had the mean and rms over
    all frames, written as ramp.

    '''
    elog = log.trace ('compute_background');
    # Check inputs
    # headers.check_input (hdrs, required=1);

    # Load files
    hdr,cube,mjd = files.load_raw_only (hdrs) ; # uses NANs # not differential.
    log.info ('Block Background Data Size (nr,nf,ny,nz): '+str(cube.shape));
    nr,nf,ny,nx = cube.shape;

    badPixelsMap,slopes,noises = bg_outliers_cum(cube) # only use for bg
    # Mark bad pixels in cube. 1=good, nan=bad
    
    cube *= np.reshape(badPixelsMap, (1,1,ny,nx) )
    #hdr,cube,mjd = remove_interference(hdr,cube,mjd)
    hdr,cube,mjd = remove_interference_cum(hdr,cube,mjd)
    breakpoint()
    #Calculate badpixels
    #

    # Someone to detect light in the shutters?
    #   Find median total flux in a frame and look for outliers.
    #   
    #   Recall some pixels enar end of ramp or corners are odd so ignore edges for this.
    #   will fail in nx <5 ny<5 nf<4.. could be less strict if a future problem.
    medians = np.median( np.sum( (cube[:,-2,2:ny-2,2:nx-2]-cube[:,1,2:ny-2:,2:nx-2])/(nf-1), axis=2),axis=1) 
    # Find bad pixels
    # subtract mean ramp and save
    # unwwrap find
    # Find Bad Pixels -
    # Background mean
    log.info ('Compute mean and rms over input files');
    bkg_mean = np.mean (cube, axis=0);
    bkg_err  = np.std (cube, axis=0) / np.sqrt (cube.shape[0]);


    # Compute temporal rms
    log.info ('Compute rms over ramp/frame of first file');
    bkg_noise = np.std (cube[:,3:-3,:,:], axis=(0,1));
    
    # Select the region for the QC parameters
    nf,ny,nx = bkg_mean.shape;
    dy,dx = 15,35;
    idf,idy,idx = int(nf/2), int(ny/2), int(nx/2);
    log.info ('Compute QC in box (%i,%i:%i,%i:%i)'%(idf,idy-dy,idy+dy,idx-dx,idx+dx));

    # Add QC parameters
    (mean,med,std) = sigma_clipped_stats (bkg_mean[idf,idy-dy:idy+dy,idx-dx:idx+dx]);
    log.info ('BKG_MEAN MED = %f'%med);
    log.info ('BKG_MEAN STD = %f'%std);
    hdr.set (HMQ+'BKG_MEAN MED',med,'[adu] for frame nf/2');
    hdr.set (HMQ+'BKG_MEAN STD',std,'[adu] for frame nf/2');

    (emean,emed,estd) = sigma_clipped_stats (bkg_err[idf,idy-dy:idy+dy,idx-dx:idx+dx]);
    log.info ('BKG_ERR MED = %f'%emed);
    log.info ('BKG_ERR STD = %f'%estd);
    hdr.set (HMQ+'BKG_ERR MED',emed,'[adu] for frame nf/2');
    hdr.set (HMQ+'BKG_ERR STD',estd,'[adu] for frame nf/2');
    
    (nmean,nmed,nstd) = sigma_clipped_stats (bkg_noise[idy-dy:idy+dy,idx-dx:idx+dx]);
    log.info ('BKG_NOISE MED = %f'%nmed);
    log.info ('BKG_NOISE STD = %f'%nstd);
    hdr.set (HMQ+'BKG_NOISE MED',round(nmed,5),'[adu] for first file');
    hdr.set (HMQ+'BKG_NOISE STD',round(nstd,5),'[adu] for first file');

    # Define quality flag
    hdr[HMQ+'QUALITY'] = (1./(emed+1e-10), 'quality of data');
    
    # Create output HDU
    hdu0 = pyfits.PrimaryHDU (bkg_mean[None,:,:,:]);
    hdu0.header = hdr;

    # Update header
    hdu0.header['FILETYPE'] = filetype;
    hdu0.header['BUNIT'] = 'adu/pixel/frame';
    hdu0.header['SHAPE'] = '(nr,nf,ny,nx)';

    # Create second HDU
    hdu1 = pyfits.ImageHDU (bkg_err[None,:,:,:]);
    hdu1.header['EXTNAME'] = ('BACKGROUND_ERR','uncertainty on background mean');
    hdu1.header['BUNIT'] = 'adu/pixel/frame';
    hdu1.header['SHAPE'] = '(nr,nf,ny,nx)';

    # Create third HDU
    hdu2 = pyfits.ImageHDU (bkg_noise[None,None,:,:]);
    hdu2.header['EXTNAME'] = ('BACKGROUND_NOISE','pixel frame-to-frame noise');
    hdu2.header['BUNIT'] = 'adu/pixel/frame';
    hdu2.header['SHAPE'] = '(nr,nf,ny,nx)';
    
    # Write output file
    hdulist = pyfits.HDUList ([hdu0,hdu1,hdu2]);
    files.write (hdulist, output+'.fits');

    # Figures
    log.info ('Figures');

    # Images of mean
    fig,ax = plt.subplots (2,1);
    fig.suptitle (headers.summary (hdr));
    ax[0].imshow (bkg_mean[idf,:,:], vmin=med-5*std, vmax=med+5*std);
    ax[0].set_ylabel ('Mean (adu) +-5sig');
    ax[1].imshow (bkg_mean[idf,:,:], vmin=med-20*std, vmax=med+20*std);
    ax[1].set_ylabel ('Mean (adu) +-20sig');
    files.write (fig, output+'_mean.png');

    # Images of noise
    fig,ax = plt.subplots (2,1);
    fig.suptitle (headers.summary (hdr));
    ax[0].imshow (bkg_noise, vmin=nmed-5*nstd, vmax=nmed+5*nstd);
    ax[0].set_ylabel ('Noise (adu) +-5sig');
    ax[1].imshow (bkg_noise, vmin=nmed-20*nstd, vmax=nmed+20*nstd);
    ax[1].set_ylabel ('Noise (adu) +-20sig');
    fig.suptitle (headers.summary (hdr));
    files.write (fig, output+'_noise.png');

    # Images of error
    fig,ax = plt.subplots (2,1);
    fig.suptitle (headers.summary (hdr));
    ax[0].imshow (bkg_err[idf,:,:], vmin=emed-5*estd, vmax=emed+5*estd);
    ax[0].set_ylabel ('Err (adu) +-5sig');
    ax[1].imshow (bkg_err[idf,:,:], vmin=emed-20*estd, vmax=emed+20*estd);
    ax[1].set_ylabel ('Err (adu) +-20sig');
    files.write (fig, output+'_err.png');
    
    # Histograms of median
    fig,ax = plt.subplots (2,1);
    fig.suptitle (headers.summary (hdr));
    ax[0].hist (bkg_mean[idf,:,:].flatten(),bins=med+std*np.linspace(-10,20,50));
    ax[0].set_ylabel ("Number of pixels");
    ax[1].hist (bkg_mean[idf,:,:].flatten(),bins=med+std*np.linspace(-10,20,50));
    ax[1].set_ylabel ("Number of pixels");
    ax[1].set_xlabel ("Value at frame nf/2 (adu)");
    ax[1].set_yscale ('log');
    files.write (fig, output+'_histomean.png');

    # Histograms of error
    fig,ax = plt.subplots (2,1);
    fig.suptitle (headers.summary (hdr));
    ax[0].hist (bkg_err.flatten(),bins=emed+estd*np.linspace(-10,20,50));
    ax[0].set_ylabel ("Number of pixels");
    ax[1].hist (bkg_err.flatten(),bins=emed+estd*np.linspace(-10,20,50));
    ax[1].set_ylabel ("Number of pixels");
    ax[1].set_xlabel ("RMS(file)/sqrt(nfile)");
    ax[1].set_yscale ('log');
    files.write (fig, output+'_histoerr.png');

    # Histograms of noise
    fig,ax = plt.subplots (2,1);
    fig.suptitle (headers.summary (hdr));
    ax[0].hist (bkg_noise.flatten(),bins=nmed+nstd*np.linspace(-10,20,50));
    ax[0].set_ylabel ("Number of pixels");
    ax[1].hist (bkg_noise.flatten(),bins=nmed+nstd*np.linspace(-10,20,50));
    ax[1].set_ylabel ("Number of pixels");
    ax[1].set_xlabel ("RMS(frame) for first file");
    ax[1].set_yscale ('log');
    files.write (fig, output+'_histonoise.png');

    # Ramp
    fig,ax = plt.subplots();
    fig.suptitle (headers.summary (hdr));
    ax.plot (np.median (bkg_mean,axis=(1,2)));
    ax.set_xlabel ("Frame");
    ax.set_ylabel ("Median of pixels (adu)");
    files.write (fig, output+'_ramp.png');

    plt.close ("all");
    del elog;
    return hdulist;

def estimate_windows (cmean, hdr, output='outout_window'):
    '''
    Estimate the position of the fringe and xchan in the 2D image cmean
    It fills many QC parameters in hdr.
    '''
    
    # Get dimensions
    log.info ('Size of cmean: '+str(cmean.shape));
    ny,nx = cmean.shape;

    # Number of spectral channels to extract on plots
    ns = int(setup.nspec (hdr)/2 + 0.5) + 1;
    x  = np.arange (nx);

    # Compute the flux in fringe window
    # (suposedly smoothed in x) 
    fmap = medfilt (cmean, [1,11]);

    # Compute the flux in the photometric window
    # (suposedly sharp in x)
    pmap = medfilt (cmean - fmap, [3,1]);

    # Guess spatial position
    idx = np.argmax (np.mean (pmap, axis=0));

    # Get spectral limit of photometry
    py = np.mean (pmap[:,idx-2:idx+3], axis=1);
    pyc,pyw = signal.getwidth (medfilt (py, 5));

    # Fit spatial of photometry with Gaussian
    px = np.mean (pmap[int(pyc-pyw):int(pyc+pyw),:], axis=0);
    init = models.Gaussian1D (amplitude=np.max(px), mean=np.argmax(px), stddev=1.);
    pfit  = fitting.LevMarLSQFitter()(init, x, px);
    pxc,pxw = pfit.mean.value,pfit.stddev.value;

    log.info ('Max amplitude photo: %f adu/pix/frame'%(pfit.amplitude.value));
    log.info ('Limit photo in spectral direction: %f %f'%(pyc,pyw));
    log.info ('Limit photo in spatial direction: %f %f'%(pxc,pxw));
    
    # Add QC parameters for window
    hdr[HMW+'PHOTO MAX']  = (pfit.amplitude.value,'[adu/pix/frame]');
    hdr[HMW+'PHOTO WIDTHX']  = (pxw,'[pix] spat std');
    hdr[HMW+'PHOTO CENTERX'] = (pxc,'[pix] python-def');
    hdr[HMW+'PHOTO WIDTHY']  = (pyw,'[pix] spec half-size');
    hdr[HMW+'PHOTO CENTERY'] = (pyc,'[pix] python-def');

    # Get spectral limits of fringe
    fy  = np.mean (fmap,axis=1);
    fyc,fyw = signal.getwidth (medfilt (fy, 5));
    
    # Fit spatial of fringe with Gaussian
    fx  = np.mean (fmap[int(fyc-fyw):int(fyc+fyw),:], axis=0);
    init = models.Gaussian1D (amplitude=np.max(fx), mean=np.argmax(fx), stddev=50.);
    fitter = fitting.LevMarLSQFitter();
    ffit = fitter (init, x, fx);
    fxc,fxw = ffit.mean.value,ffit.stddev.value;
        
    log.info ('Max amplitude fringe: %f adu/pix/frame'%(ffit.amplitude.value));
    log.info ('Limit fringe in spectral direction: %f %f'%(fyc,fyw));
    log.info ('Limit fringe in spatial direction: %f %f'%(fxc,fxw));

    # Add QC parameters for window
    hdr[HMW+'FRINGE MAX']  = (ffit.amplitude.value,'[adu/pix/frame]');
    hdr[HMW+'FRINGE WIDTHX']  = (fxw,'[pix] spat std');
    hdr[HMW+'FRINGE CENTERX'] = (fxc,'[pix] python-def');
    hdr[HMW+'FRINGE WIDTHY']  = (fyw,'[pix] spec half-size');
    hdr[HMW+'FRINGE CENTERY'] = (fyc,'[pix] python-def');
    
    # Extract spectrum of photo and fringes
    p_spectra = np.mean (pmap[:,int(pxc-2):int(pxc+3)], axis=1);
    p_spectra /= np.max (p_spectra);

    f_spectra = np.mean (fmap[:,int(fxc-2*fxw):int(fxc+2*fxw)+1], axis=1);
    f_spectra /= np.max (f_spectra);

    # Shift between photo and fringes in spectral direction
    shifty = phase_cross_correlation (p_spectra[:,None],f_spectra[:,None], \
                                   upsample_factor=100)[0][0];

    # Compute shifted spectra
    ps_spectra = subpix_shift (p_spectra, -shifty);

    # Set in header
    hdr[HMW+'PHOTO SHIFTY'] = (shifty,'[pix] shift of PHOTO versus FRINGE');

    # Define quality flag as the SNR
    # quality = ffit.amplitude.value;
    quality = ffit.amplitude.value / np.std (fx - ffit(x));

    # Set quality flag to 0 if bad fit
    if (fxc < 1) or (fxc > nx) or (fxw < 10) or (fxw > nx): quality = 0.0;
    if (pxc < 1) or (pxc > nx) or (pxw < 0.25) or (pxw > 10): quality = 0.0;
    
    # Set quality
    log.info (HMQ+'QUALITY = %f'%quality);
    hdr[HMQ+'QUALITY'] = (quality, 'quality of data');

    # Figures
    log.info ('Figures');
    
    # Figures of photo
    fig,ax = plt.subplots(3,1);
    fig.suptitle (headers.summary (hdr));
    ax[0].imshow (pmap);
    ax[1].plot (px, label='Data');
    ax[1].plot (x,pfit(x), label='Gaussian');
    ax[1].set_ylabel ('adu/pix/fr');
    ax[1].legend ();
    ax[2].imshow (pmap[int(pyc-ns):int(pyc+ns+1),int(pxc-2):int(pxc+3)]);
    files.write (fig, output+'_pfit.png');

    # Figures of fringe
    fig,ax = plt.subplots(3,1);
    fig.suptitle (headers.summary (hdr));
    ax[0].imshow (fmap);
    ax[1].plot (fx, label='Medfilt Data');
    ax[1].plot (x,ffit(x), label='Gaussian');
    # ax[1].plot (cmean[int(fyc-ns):int(fyc+ns+1)+1,:].mean(axis=0), label='Raw Data');
    # ax[1].set_ylim (bottom=0, top=1.2*ffit(x).max());
    ax[1].set_ylabel ('adu/pix/fr');
    ax[1].legend ();
    ax[2].imshow (fmap[int(fyc-ns):int(fyc+ns+1)+1,int(fxc-2*fxw):int(fxc+2*fxw)]);
    files.write (fig, output+'_ffit.png');

    # Shifted spectra
    fig,ax = plt.subplots(2,1);
    fig.suptitle (headers.summary (hdr));
    ax[0].imshow (cmean);
    ax[1].plot (f_spectra / np.sum (f_spectra), label='fringe');
    ax[1].plot (p_spectra / np.sum (p_spectra), label='photo');
    ax[1].plot (ps_spectra / np.sum (ps_spectra), label='shifted photo');
    ax[1].legend ();
    files.write (fig, output+'_cut.png');

    return pmap, fmap;
    
def compute_beam_map (hdrs,bkg,flat,threshold,output='output_beam_map',filetype='BEAM_MAP', linear=True): # depricate `linear` after testing
    '''
    Compute BEAM_MAP product.
    '''
    elog = log.trace ('compute_beam_map');

    # Check inputs
    headers.check_input (hdrs, required=1);
    headers.check_input (bkg, required=1, maximum=1);
    headers.check_input (flat, required=1, maximum=1);
    
    # Load background
    log.info ('Load %s'%bkg[0]['ORIGNAME']);
    bkg_cube = pyfits.getdata (bkg[0]['ORIGNAME'],0);
    
    # Compute bad pixels position from background
    bad_img = define_badpixels (bkg,threshold);

    # Load flat
    log.info ('Load %s'%flat[0]['ORIGNAME']);
    flat_img = pyfits.getdata (flat[0]['ORIGNAME'],'FLAT');

    # Crop the FLAT image. For now, this is not working.
    # Either the image are not defined with the same orientation,
    # or the flat is not valid anymores
    idy, idx = setup.crop_ids (hdrs[0]);
    flat_img = flat_img[idy,:][:,idx];

    # Load files
    hdr,cube,mjd = files.load_raw (hdrs, coaddRamp='sum', background=bkg_cube,
                                   badpix=bad_img, flat=None, output=output,
                                   linear=linear);

    # Check background subtraction in empty region
    check_empty_window (cube, hdr);
    
    # Get dimensions
    log.info ('Data size: '+str(cube.shape));
    nr,nf,ny,nx = cube.shape;

    # Compute the sum
    log.info ('Compute sum over ramps and frames');
    csum = np.sum (cube, axis=(0,1));

    # Estimate windows position
    pmap, fmap = estimate_windows (csum, hdr, output=output);

    # File
    log.info ('Create file');
    
    # First HDU
    hdu0 = pyfits.PrimaryHDU (csum[None,None,:,:]);
    hdu0.header = hdr;
    hdu0.header['FILETYPE'] = filetype;
    hdu0.header['BUNIT'] = ('adu/pixel','sum over ramp and frame');
    hdu0.header['SHAPE'] = '(nr,nf,ny,nx)';    

    # Set files
    hdu0.header[HMP+'BACKGROUND_MEAN'] = os.path.basename (bkg[0]['ORIGNAME'])[-40:];
    
    # Write output file
    hdulist = pyfits.HDUList ([hdu0]);
    files.write (hdulist, output+'.fits');
    
    plt.close("all");
    del elog;
    return hdulist;
    
def compute_beam_profile (hdrs,output='output_beam_profile',filetype='BEAM_PROFILE'):
    '''
    Compute BEAM_PROFILE product, by simply summing the BEAM_MAP.
    The output product contains
    keywords defining the fringe window and the photometric
    windows, as well as the spectral shift between them.
    '''
    elog = log.trace ('compute_beam_profile');

    # Check inputs
    headers.check_input (hdrs, required=1);

    # Load header
    hdr = pyfits.getheader (hdrs[0]['ORIGNAME']);

    # For sum over all files 
    csum = 0.0;
    
    # Load data as images
    for h in hdrs:
        f = h['ORIGNAME'];
        log.info ('Load file %s'%f);
        csum = csum + pyfits.getdata (f).astype(float).sum (axis=(0,1));

    # Estimate windows position
    pmap, fmap = estimate_windows (csum, hdr, output=output);

    # File
    log.info ('Create file');
    
    # First HDU
    hdu0 = pyfits.PrimaryHDU (csum[None,None,:,:]);
    hdu0.header = hdr;
    hdu0.header['FILETYPE'] = filetype;
    hdu0.header['BUNIT'] = ('adu/pixel','sum over ramp and frame');
    hdu0.header['SHAPE'] = '(nr,nf,ny,nx)';

    # Write output file
    hdulist = pyfits.HDUList ([hdu0]);
    files.write (hdulist, output+'.fits');
    
    plt.close("all");
    del elog;
    return hdulist;

def compute_preproc (hdrs,bkg,flat,bmaps,threshold,output='output_preproc',filetype='PREPROC', linear=True): # depricate `linear` after testing
    '''
    Compute preproc file. The first HDU contains the
    fringe window. The second HDU contains the 6 photometries
    already extracted and re-aligned spectrally
    '''
    
    elog = log.trace ('compute_preproc');

    # Check inputs
    headers.check_input (hdrs,  required=1);
    headers.check_input (bkg,   required=1, maximum=1);
    headers.check_input (flat, required=1, maximum=1);
    headers.check_input (bmaps, required=1, maximum=6);

    # Load background
    log.info ('Load %s'%bkg[0]['ORIGNAME']);
    bkg_cube = pyfits.getdata (bkg[0]['ORIGNAME'],0);
    
    # Compute bad pixels position
    bad_img = define_badpixels (bkg,threshold);

    # Load flat
    log.info ('Load %s'%flat[0]['ORIGNAME']);
    flat_img = pyfits.getdata (flat[0]['ORIGNAME'],'FLAT');

    # Crop the FLAT image. For now, this is not working.
    # Either the image are not defined with the same orientation,
    # or the flat is not valid anymores
    idy, idx = setup.crop_ids (hdrs[0]);
    flat_img = flat_img[idy,:][:,idx];
        
    # Load files
    hdr,cube,mjd = files.load_raw (hdrs, background=bkg_cube,
                                   badpix=bad_img, flat=None, output=output,
                                   linear=linear);

    # Get dimensions
    log.info ('Data size: '+str(cube.shape));
    
    # Check background subtraction in empty region
    check_empty_window (cube, hdr);

    # Extract the fringe as the middle of all provided map
    fxc0 = np.mean ([b['MIRC QC WIN FRINGE CENTERX'] for b in bmaps if b!=[]]);
    fyc0 = np.mean ([b['MIRC QC WIN FRINGE CENTERY'] for b in bmaps if b!=[]]);
                   
    # Define the closest integer
    fxc = int(round(fxc0));
    fyc = int(round(fyc0));
    log.info ('FRINGE CENTERX/Y = %i,%i'%(fxc,fyc));

    # Expected size on spatial and spectral direction are hardcoded
    fxw = int(setup.fringe_widthx (hdr) / 2);
    pxw = int(setup.photo_widthx (hdr) / 2 + 1.5);
    ns  = int(setup.nspec (hdr)/2 + 1.5);

    # Check that we can deal with this
    # spectrum size, otherwise crop more
    overflow1 = ns - fyc;
    overflow2 = (fyc+ns+1) - cube.shape[2];
    if overflow1 > 0 or overflow2 > 0:
        overflow = np.maximum (overflow1, overflow2);
        log.warning ('Hard window too short, reduce spectrum from %i to %i'%(ns,ns-overflow));
        ns -= overflow;
    
    # Keep track of crop value
    hdr[HMW+'FRINGE STARTX'] = (fxc-fxw, '[pix] python-def');
    hdr[HMW+'FRINGE STARTY'] = (fyc-ns, '[pix] python-def');

    hdr[HMW+'FRINGE NX'] = (2*fxw+1, '[pix]');
    hdr[HMW+'FRINGE NY'] = (2*ns+1, '[pix]');

    # Photometry
    hdr[HMW+'PHOTO NX'] = (2*pxw+1, '[pix]');
    hdr[HMW+'PHOTO NY'] = (2*ns+1, '[pix]');

    # Extract fringe
    fringe = cube[:,:,fyc-ns:fyc+ns+1,fxc-fxw:fxc+fxw+1];

    # Robust measure of total flux in fringe
    value = np.sum (medfilt (np.mean (fringe, axis=(0,1)), (1,11)));
    hdr[HMW+'FRINGE MEAN'] = (value,'[adu/frame] total flux');
    log.info ('FRINGE MEAN = %.2f [adu/frame]'%value);

    # Same for photometries
    nr,nf,ny,nx = fringe.shape;

    photos = np.zeros ((6,nr,nf,ny,2*pxw+1));
    for bmap in bmaps:
        if bmap == []: continue;
        log.info ('Use %s: %s'%(bmap['FILETYPE'],bmap['ORIGNAME']));
        beam = int(bmap['FILETYPE'][4:5]) - 1;
        
        # Get the position of the photo spectra
        pxc = int(round(bmap['MIRC QC WIN PHOTO CENTERX']));
        pyc = int(round(bmap['MIRC QC WIN PHOTO CENTERY']));
        log.info ('PHOTO%i CENTERX/Y = %i,%i'%(beam,pxc,pyc));

        # Set the required crop in header
        hdr[HMW+'PHOTO%i STARTX'%(beam)] = (pxc-pxw, '[pix] python-def');
        hdr[HMW+'PHOTO%i STARTY'%(beam)] = (pyc-ns, '[pix] python-def');
        photos[beam,:,:,:,:] = cube[:,:,pyc-ns:pyc+ns+1,pxc-pxw:pxc+pxw+1];
        
        # Robust measure of max flux in photometry
        value = np.sum (medfilt (np.mean (photos[beam,:,:,:,:], axis=(0,1)), (3,1)));
        hdr[HMW+'PHOTO%i MEAN'%(beam)] = (value,'[adu/frame], total flux');
        log.info ('PHOTO%i MEAN = %.2f [adu/frame]'%(beam,value));

    # Figures
    log.info ('Figures');

    # Fringe and photo mean
    fig,ax = plt.subplots(2,1);
    fig.suptitle (headers.summary (hdr));
    ax[0].imshow (np.mean (fringe,axis=(0,1)));
    ax[1].imshow (np.swapaxes (np.mean (photos,axis=(1,2)), 0,1).reshape((ny,-1)));
    for b in np.arange(1,6):
        plt.axvline (x=b*(2*pxw+1) - 0.5, color='w', linestyle='--');
    files.write (fig, output+'_mean.png');

    # Spectra
    fig,ax = plt.subplots();
    fig.suptitle (headers.summary (hdr));
    ax.plot (np.mean (fringe, axis=(0,1,3)), '--', label='fringes');
    ax.plot (np.mean (photos, axis=(1,2,4)).T);
    ax.set_ylabel ('adu/pix/fr');
    ax.legend ();
    files.write (fig, output+'_spectra.png');

    # Time continuity
    fig,ax = plt.subplots();
    fig.suptitle (headers.summary (hdr));
    time_ms = (mjd - mjd[0,0]) * 24*3600*1e3;
    ax.plot (np.diff(time_ms.flatten()));
    ax.set_xlabel ('frame number');
    ax.set_ylabel ('delta time in ms');
    files.write (fig, output+'_timecont.png');
    
    # File
    log.info ('Create file');
    
    # First HDU
    hdu0 = pyfits.PrimaryHDU (fringe.astype('float32'));
    hdu0.header = hdr;
    hdu0.header['BUNIT'] = 'adu/pixel/frame';
    hdu0.header['FILETYPE'] = filetype;
    hdu0.header['SHAPE'] = '(nr,nf,ny,nx)';
    
    # Set files
    hdu0.header[HMP+'BACKGROUND_MEAN'] = os.path.basename (bkg[0]['ORIGNAME'])[-40:];

    # Set the input calibration file
    for bm in bmaps:
        hdu0.header[HMP+bm['FILETYPE']] = os.path.basename (bm['ORIGNAME'])[-50:];

    # Second HDU with photometries
    hdu1 = pyfits.ImageHDU (photos.astype('float32'));
    hdu1.header['BUNIT'] = 'adu/pixel/frame';
    hdu1.header['EXTNAME'] = 'PHOTOMETRY_PREPROC';
    hdu1.header['SHAPE'] = '(nb,nr,nf,ny,nx)';

    # Third HDU with MJD
    hdu2 = pyfits.ImageHDU (mjd);
    hdu2.header['BUNIT'] = 'day';
    hdu2.header['EXTNAME'] = 'MJD';
    hdu2.header['SHAPE'] = '(nr,nf)';
    
    # Write file
    hdulist = pyfits.HDUList ([hdu0,hdu1,hdu2]);
    files.write (hdulist, output+'.fits');
    
    plt.close("all");
    del elog;
    return hdulist;

def remove_interference(hdr,cube,mjd):

    # Removes sinusoidal-like interferograms from the data. Assumes RAW data -- no differences.
    # assume the block of data are CONTINUOUS in time. not gaps in the block.. The block creation
    # should require continuous file blocks in time not just in file #.. Using RESTART0  
    #  
    # some common camera problems.
    #  last frame of a blck is sometimes bad -- very high value.
    #  top row is usually bad (all zeros) . sometimes more than 1 row if using polaization mode (this dpends on # of subwindows)
    #  camera could get reset during block so there is a discontinuity in the sine wave.. will need to look for that in order to
    #  avoid lomb scargle which is just terribly slow and data is almost always equally spaced...
    #  problem here is making a routine that will always work for all data... eventually even nbin != 1

    nbin=hdr['NBIN']
    nr,nf,ny,nx = cube.shape

    # find the empty rows (near top of the array).. depends on number of subarray windows
    test = np.median(cube[:,-2,:,:],axis=(0,2)) - np.median( cube[:,0,:,:],axis=(0,2) )
    maxrows=np.max(np.argwhere(test > .1) ) # assume its alwasy the top rows that are bad.
    data = np.diff (cube[:,:-1,0:maxrows+1,:],axis=1)
    # Find the columns with lowest fluxes (but no zero..)
    
    data -= np.mean(data,axis=1,keepdims=True) # median average flux from each pixel per ramp.
                                                 # this could more like a smooth high pass filter
    col_values = np.mean(data,axis=(0,1,2))
    insort=np.argsort(col_values)
    numfaint=16
    subdata=data[:,:,:,insort[0:numfaint]]

    data = np.mean(data,axis=3)
    subdata= np.mean(subdata,axis=3)

    # average across row. # change to only be lowest pixels.
    nr1,nf1,ny1 = data.shape

    fullarray = np.zeros((nr,nf,ny))
    fullarray[:,0:nf-2,0:maxrows+1]=data
    #data=subdata
    
    # how to re accumulate if you want.
    #data2 = np.concatenate((data1[:,0,:][:,None,:],data1),axis=1).cumsum(axis=1)
    
    #Note due double precision of MJD, the best we can do is ~1/1000 of a frame numerical precisions!
    rowtimes =np.linspace(0,hdr['EXPOSURE']/1000./ny*(maxrows)/nbin,maxrows+1) #seconds
    fullrowtimes =np.linspace(0,hdr['EXPOSURE']/1000./ny*(ny-1),ny) #seconds

    mjd0=np.min(mjd)    
    #time=(mjd-mjd0)[:,:,None]*24*3600.+rowtimes3[None,None,:]
    time=(mjd[:,0:-2]+hdr['EXPOSURE']/1000/2./24/3600.  - mjd0)[:,:,None]*24.*3600. +rowtimes[None,None,:] #seconds
    fulltime=(mjd + hdr['EXPOSURE']/1000/2./24/3600.  - mjd0)[:,:,None]*24.*3600. +fullrowtimes[None,None,:] #seconds

    # find EXACT FREQUENCY 
    intime=np.argsort(fulltime.ravel())

    zeropad=np.zeros((len(intime)*15))
    signal=np.concatenate( (fullarray.ravel()[intime],zeropad))
    signal=np.where(np.isnan(signal),0,signal)
    ft= np.fft.rfft(signal)
    hz=np.fft.rfftfreq(len(signal), d=hdr['EXPOSURE']/ny/1000.)
    hzmax=hz[np.argmax(np.abs(ft))]
    log.debug('Interference Removal: Peak of FFT at %f Hz. '%(hzmax))
    if hzmax < 80:
        log.warning('Peak of FFT at %f Hz. Recalculating peak to isolate interference'%(hzmax))
        insubset = np.argwhere( (hz > 80) )
        hzmax=hz[insubset[np.argmax(np.abs(ft[insubset]))]]
        log.warning('New Peak of FFT at %f Hz. '%(hzmax))
    
    #calc period using fft method
    period=1./hzmax; 
    #tbin1,dbin1,data3=avg_fold(time,data,period,num=ny*40) # working well!
    tbin,dbin,data2=avg_fold(time,data,period,num=ny*4) # working well!
    tbin3,dbin3,data3=avg_fold(time,subdata,period,num=ny*4) # working well!


    fullphases = (fulltime % period) / period

    fullarray2 = np.zeros((nr,nf,ny))
    fullarray2[:,0:nf-2,0:maxrows+1]=data2-np.mean(data2)
    signal2=np.concatenate( (fullarray2.ravel()[intime],zeropad))
    signal2=np.where(np.isnan(signal2),0,signal2)
    ft2= np.fft.rfft(signal2)
    ft3= np.fft.rfft(signal-signal2)

    breakpoint() ;  
    time_data1 = np.reshape(time_data[:,:,None],(nr1,nf1,ny1))
    ft= np.fft.rfft(data.ravel())
    times=time_data.ravel()
    #times=times-np.floor(times[0])
    


    data=np.append(data, np.mean(data,axis=1,keep=True),1)
    data[:,-1,:,:]=data[:,-2,:,:]

    #mjd = 0.5 * (mjd[:,0:-1] + mjd[:,1:])[:,0:-1];


    if hdr[0]['NBIN'] == 1:
        #subtract average flux from each pixel (or smoothed version?)
        #bias = 
        data_sum = np.sum (data, axis=(3)); # sum rows since nloops ruins this axis anyway for timing.

    else: # nbin >1
        log.info ('NBIN>1: No proper interference removal YET. Using old method of reference columns. ');
        breakpoint()
        ids = np.append (np.arange(15), data.shape[-1] - np.arange(1,15));
        bias = np.median (data[:,:,:,ids],axis=3);
        bias = gaussian_filter (bias,[0,0,1]);
        data = data - bias[:,:,:,None];
    return hdr,data,mjd;

def remove_interference_cum(hdr,cube,mjd):

    # Removes sinusoidal-like interferograms from the data. Assumes RAW data -- no differences.
    # assume the block of data are CONTINUOUS in time. not gaps in the block.. The block creation
    # should require continuous file blocks in time not just in file # or the fft won't work 
    #  
    # some common camera problems.
    #  last frame of a blck is sometimes bad -- very high value.
    #  top row is usually bad (all zeros) . sometimes more than 1 row if using polaization mode (this dpends on # of subwindows)
    #  camera could get reset during block so there is a discontinuity in the sine wave.. will need to look for that in order to
    #  avoid lomb scargle which is just terribly slow and data is almost always equally spaced...
    #  problem here is making a routine that will always work for all data... eventually even nbin != 1

    nbin=hdr['NBIN']
    tint=hdr['EXPOSURE']/1000. #seconds
    nloops=hdr['NLOOPS']
    nr,nf,ny,nx = cube.shape

    data = np.diff (cube,axis=1,prepend=np.nan) # keep dimensions same.
    #the first and last reads can show offsets
    #data[;,0:1,:,:]=np.nan
    #data[:,-1,:,:]=np.nan

    # Find the columns with lowest fluxes (but no zero..)
    data -= np.nanmean(data,axis=1,keepdims=True,dtype=np.float64) # median average flux from each pixel per ramp.
                                                 # this could more like a smooth high pass filter
    alldata = data.ravel()


    col_values = np.nanmean(data,axis=(0,1,2))

    insort=np.argsort(col_values)
    numfaint=16
    subdata=data[:,:,:,insort[0:numfaint]]

    data = np.nanmean(data,axis=3)
    #data = data[:,:,:,200]

    #data -=np.nanmean(data,axis=1,keepdims=True)
    #data2 = np.nanmean(data,axis=3)

    subdata= np.nanmean(subdata,axis=3) # unused for now. could use if problems with peaks...

    # average across row. # change to only be lowest pixels.
    #nr1,nf1,ny1 = data.shape
    
    #data=subdata
    
    # how to re accumulate if you want.
    #data2 = np.concatenate((data1[:,0,:][:,None,:],data1),axis=1).cumsum(axis=1)
    
    #Note due double precision of MJD, the best we can do is ~1/1000 of a frame numerical precisions!
    rowtimes =np.linspace(0,tint/ny*(ny-1),ny) #seconds

    mjd0=np.min(mjd)    
    #time=(mjd-mjd0)[:,:,None]*24*3600.+rowtimes3[None,None,:]
    time=(mjd - tint/2./24/3600.  - mjd0)[:,:,None]*24.*3600. +rowtimes[None,None,:] #seconds
    # first entry is null. start at 1:


    
    # find EXACT FREQUENCY 
    #intime=np.argsort(time.ravel())
    zeropad=np.zeros(len(time.ravel())*15)
    data -=np.nanmean(data,dtype=np.float64)
    data_test = 25.*np.cos(time.ravel()*2*np.pi*93.990813)

    signal=np.concatenate( (data.ravel(),zeropad) )
    signal_test=np.concatenate( (data_test.ravel(),zeropad) )
    
    signal_test = np.where(np.isnan(signal),0,signal_test)
    signal = np.where(np.isnan(signal),0,signal)
        
    ft= np.fft.rfft(signal)
    ft_test= np.fft.rfft(signal_test)

    #tt=time.ravel()
    hz=np.fft.rfftfreq(len(signal), d=tint/ny )
    
    insubset=np.argmax(np.abs(ft))
    hzmax=hz[insubset]

    log.debug('Interference Removal: Peak of FFT at %f Hz. '%(hzmax))
    
    if hzmax < 80:
        log.warning('Peak of FFT at %f Hz. Recalculating peak to isolate interference'%(hzmax))
        insubset = np.argwhere( (hz > 80) )
        hzmax=hz[insubset[np.argmax(np.abs(ft[insubset]))]]
        log.warning('New Peak of FFT at %f Hz. '%(hzmax))
    
    nearmax=insubset+np.array([-2,-1,0,1,2])
    xtemp=hz[nearmax]
    ytemp = np.abs(ft[nearmax])
    ytemp=ytemp/ytemp.max()
    poly3=np.polyfit(xtemp,ytemp,2)
    hzmax = -poly3[1]/(2*poly3[0])
    log.debug('Interference Removal: Interpolated Peak of FFT at %f Hz. '%(hzmax))
    
    rp0=np.interp(hzmax,hz,np.real(ft))/(len(signal)/2)
    ip0=np.interp(hzmax,hz,np.imag(ft))/(len(signal)/2)
    rp2=np.interp(hzmax*2,hz,np.real(ft))/(len(signal)/2)
    ip2=np.interp(hzmax*2,hz,np.imag(ft))/(len(signal)/2)


    
    # in order to match this up with a signal in raw data, lets propogate a perfect sinewave through the same analysis.
    sinetime=(mjd - tint/24/3600.  - mjd0)[:,:,None]*24.*3600. +rowtimes[None,None,:] #seconds
    sinecube = 1.0*np.cos(sinetime*2*np.pi*hzmax)+1.0*np.cos(sinetime*2*np.pi*hzmax*2)
    sinedata = np.diff (sinecube,axis=1,prepend=np.nan) # keep dimensions same.
    sinedata -=np.nanmean(sinedata,dtype=np.float64)
    sinesignal=np.concatenate( (sinedata.ravel(),zeropad) )
    sinesignal = np.where(np.isnan(np.concatenate( (data.ravel(),zeropad) )),0,sinesignal) # use same nans
    sineft= np.fft.rfft(sinesignal)
    
    sinerp0=np.interp(hzmax,hz,np.real(sineft))/(len(sinesignal)/2)
    sineip0=np.interp(hzmax,hz,np.imag(sineft))/(len(sinesignal)/2)
    sinerp2=np.interp(hzmax*2,hz,np.real(sineft))/(len(sinesignal)/2)
    sineip2=np.interp(hzmax*2,hz,np.imag(sineft))/(len(sinesignal)/2)

    cval = complex(rp0,ip0)/complex(sinerp0,sineip0)
    cval2 = complex(rp2,ip2)/complex(sinerp2,sineip2)

    sinecube2 = np.abs(cval)*np.cos(sinetime*2*np.pi*hzmax+np.angle(cval)) + np.abs(cval2)*np.cos(sinetime*2*np.pi*hzmax*2+np.angle(cval2))
    sinedata2 = np.diff (sinecube2,axis=1,prepend=np.nan) # keep dimensions same.
    sinedata2 -=np.nanmean(sinedata2,dtype=np.float64)
    diffdata=data-sinedata2

    diffsignal=np.concatenate( (diffdata.ravel(),zeropad) )
    diffsignal = np.where(np.isnan(np.concatenate( (data.ravel(),zeropad) )),0,diffsignal) # use same nans
    diffft= np.fft.rfft(diffsignal)
    
    tbin,dbin,data2=avg_fold(time,data,period,num=period/(tint/ny)*2) #period/tint*ny) # working well!
    tbin2,dbin2,data3=avg_fold(time,diffdata,period,num=period/(tint/ny)*2) #period/tint*ny) # working well!

    cubesum=np.nanmean(cube,axis=(3),dtype=np.float64)
    cubesum -= gaussian_filter(cubesum,(0,.02/tint,0),mode='nearest')
    tcube,dcube,data3=avg_fold(sinetime,cubesum,period,num=period/(tint/ny)*2) #period/tint*ny) # working well!


    cubesum
    np.nanmean(cube,axis=(0),keepdims=True,dtype=np.float64).astype('float32')



    #calc period using fft method
    period=1./hzmax; 
    #tbin1,dbin1,data3=avg_fold(time,data,period,num=ny*40) # working well!
    tbin,dbin,data2=avg_fold(time,data,period,num=period/(tint/ny)*2) #period/tint*ny) # working well!
    dbin -= np.mean(dbin)
    phases=(time % period)/period
    data_fullsin=np.interp(phases,tbin,dbin)
    breakpoint()
    ntotal = nr*nf*nbin*ny*nloops*nx
    # this seems intractable -- JDM :(
    nphases=100
    phase_model = np.linspace(0,1,nphases+1,endpoint=True)
    counters = np.arange(ntotal,dtype='int')
    counters6d = np.reshape(counters,(nr,nf,nbin,ny,nloops,nx))
    phis6d=(((counters6d*(tint/nbin/ny/nloops/nx)) % period) / period)
    phis6d_index = (phis6d // (1./nphases)).astype('int') 
    phis2d_index =  np.reshape(np.transpose(phis6d_index,(2,4,0,1,3,5)),(nloops*nbin,nr*nf*ny*nx))
    
    #phis6d.itemsize*phis6d.size/1e9
    cube0 = (cube-np.nanmedian(cube,axis=(0),keepdims=True)).astype('float32')
    cube6d = np.repeat(np.repeat(np.reshape(cube0,(nr,nr,1,ny,1,nx)),nloops,axis=4),nbin,axis=2)
    cube2d=  np.reshape(np.transpose(cube6d,(2,4,0,1,3,5)),(nloops*nbin,nr*nf*ny*nx))

    cube1d = cube0.ravel()
    # phis2d_index goes with cube1d

    waveform = np.zeros(nphases+1)
    for i in range(nphases):
        waveform[i]=np.nanmean(np.extract(phis6d_index == i, cube6d))


    fitmatrix=np.zeros( (nphases,nphases) )
    yvector=np.zeros( (nphases) )
    #remove Nans to speed up things.
    goodin=np.squeeze(np.argwhere(np.isnan(cube1d)==False))
    goodphi2d=phis2d_index[:,goodin]
    goodcube2d=cube2d[:,goodin]

    #goodphi2d=np.extract(cube2d != np.nan, phis2d_index)
    #goodcube2d=np.extract(cube2d != np.nan, cube2d)
    print('Try making matrix. try using pn library and check timing!')
    
    breakpoint()
    for k in range(nphases): # top part too slow.
        yvector[k]=np.sum(np.extract(goodphi2d == k,goodcube2d) )/(nloops*nbin)
        Akj = np.sum(np.where(goodphi2d ==k, 1./nloops/nbin,0),axis=0)
        good_j = np.squeeze(np.argwhere(Akj != 0))
        goodphi2d_sub=goodphi2d[:,good_j] #not working
        Akj_sub = Akj[good_j]
        good_i=np.unique(goodphi2d_sub)
        # speed up next section 
        for i in good_i:
            Aij_sub = np.sum(np.where(goodphi2d_sub ==i, 1./nloops/nbin,0),axis=0)
            fitmatrix[k,i]=np.sum(Aij_sub*Akj_sub)
            print('k: %i i: %i  yvector %f Matrix %f'%(k,i, yvector[k],fitmatrix[k,i]))
    invmat=np.linalg.inv(fitmatrix)
    waveform2 = np.matmul(invmat,yvector)
    plt.plot(waveform )
    plt.plot(waveform2,'.')
    plt.show()
    # check how well this removes artfifacts by feeding cube back into the avgfold analysis and FT.
    breakpoint()
    print(i,waveform[i])
    index1=np.where( (phis6d)>phase_model[0]) & (phis6d < phase_model[1])
    # if I had  phase,waveform
    #waveform4d=np.nanmean(np.interp(phis6d,phase_model,waveform),axis=(2,4)
    

    # try another approach
    num_cycles = np.int(np.floor(len(alldata)*(tint/ny/nx)/period)-1)
    nbar= np.linspace(0,num_cycles,num_cycles,endpoint=False)
    num_phis=np.int(np.floor(period/(tint/nx/ny/nloops)))

    phi0=np.linspace(0,1,num_phis,endpoint=True)
    result=np.zeros(num_phis)
    offset=np.min(time)/period
    for j in range(num_phis):
        R=period*(phi0[j]+nbar-offset)/(tint/ny) 
        R1 = (np.floor(R)*nx).astype(int)
        R2 = ((R % (1./nloops) /(1./nloops))*nx).astype(int)
        result[j]=np.nanmean(alldata[R1+R2])
        print(j,num_phis,result[j])
    dbin-=dbin.mean()
    result-=result.mean()
    #plt.plot(phi0,result)
    #plt.plot(tbin,dbin)
    #plt.show()
    #breakpoint()

    #tbin3,dbin3,data3=avg_fold(time,subdata,period,num=ny*) # working well!
    #data2 -=np.nanmean(data2)
    #signal2=np.concatenate( ( (data2).ravel(),zeropad))
    #signal2 = np.where(np.isnan(signal2),0,signal2)
    #ft2= np.fft.rfft(signal2)
    #ft3=np.fft.rfft(signal-signal2)
    #plt.plot(hz,np.abs(ft2))
    #plt.plot(hz,np.abs(ft))
    #plt.plot(hz,np.abs(ft3))
    #plt.xlim(1220,1224)
    #plt.xlim(93.9,94.1)

    #plt.xlim(85,100)
    #plt.show()
    #breakpoint()
    #Working at 200:1 level for first peak.

    data = np.diff (cube,axis=1,prepend=np.nan) # keep dimensions same.
    data -= np.nanmean(data,axis=(0),keepdims=True)
    
    # Find the columns with lowest fluxes (but no zero..)
    data_sinusoid = np.repeat(data2[:,:,:,np.newaxis],nx,axis=3)
    data_fix=data-data_sinusoid

    data_alt = np.repeat(np.nanmedian(data[:,:,:,0:15],axis=3,keepdims=True),nx,axis=3)
    data_fix2=data-data_alt
    data_alt2 = np.repeat(np.nanmean(data[:,:,:,0:15],axis=3,keepdims=True),nx,axis=3)
    data_fix3=data-data_alt2
    #estimating based on first 15 ros working better than sinusoid model...
    #more to look into.
    # note last read in ramp is slightly off from normal. maybe should mark bad or fix.
    plt.plot(data[1,:,:,:].ravel(),'o')
    plt.plot(data_sinusoid[1,:,:,:].ravel(),'.')
    plt.show()

    data_fullsin2 = np.repeat(data_fullsin[:,:,:,np.newaxis],nx,axis=3)
    cube2=data_fullsin2.cumsum(axis=1)
    cube2 = np.concatenate((cube[:,0,:,:][:,None,:],data_fullsin2[:,1:,:,:]),axis=1).cumsum(axis=1)

    breakpoint()

    fullphases = (fulltime % period) / period
    fullarray2=np.interp(fullphases,tbin,dbin)

    breakpoint() ;  
    time_data1 = np.reshape(time_data[:,:,None],(nr1,nf1,ny1))
    ft= np.fft.rfft(data.ravel())
    times=time_data.ravel()
    #times=times-np.floor(times[0])
    



    data=np.append(data, np.mean(data,axis=1,keep=True),1)
    data[:,-1,:,:]=data[:,-2,:,:]

    #mjd = 0.5 * (mjd[:,0:-1] + mjd[:,1:])[:,0:-1];


    if hdr[0]['NBIN'] == 1:
        #subtract average flux from each pixel (or smoothed version?)
        #bias = 
        data_sum = np.sum (data, axis=(3)); # sum rows since nloops ruins this axis anyway for timing.

    else: # nbin >1
        log.info ('NBIN>1: No proper interference removal YET. Using old method of reference columns. ');
        breakpoint()
        ids = np.append (np.arange(15), data.shape[-1] - np.arange(1,15));
        bias = np.median (data[:,:,:,ids],axis=3);
        bias = gaussian_filter (bias,[0,0,1]);
        data = data - bias[:,:,:,None];
    return hdr,data,mjd;

def avg_fold(time0,data0,period,num=10,t0=0.0):
    nr,nf,ny = data0.shape
    num=int(num)
    data=np.extract( np.isnan(data0.ravel()) == False,data0.ravel() )
    time=np.extract( np.isnan(data0.ravel()) == False,time0.ravel() )
    
    phases = ((time-t0) % period) / period
    index=np.argsort(phases)
    tbin=np.linspace(0,1, num,endpoint=False)
    binwidth=tbin[1]-tbin[0]
    tbin += binwidth/2.
    dbin=np.zeros(num)

    for i in range(num): 
        dbin[i]=np.median(np.extract((phases >= tbin[i]-binwidth/2) & (phases < tbin[i]+binwidth/2),data))
        #dbin[i]=np.median(data[index[np.argwhere((phases[index] >= tbin[i]-binwidth/2) & (phases[index] < tbin[i]+binwidth/2))]])

    tbin1 = np.concatenate( ([tbin[-1]-1],tbin,[tbin[0]+1]))
    dbin1 = np.concatenate( ([dbin[-1]],dbin,[dbin[0]] ))
    
    data2=np.interp(((time0.ravel()-t0) % period) /period,tbin1,dbin1)
    data2=np.reshape(data2,(nr,nf,ny))
    data2=np.where(np.isnan(data0),np.nan,data2)
    return tbin1,dbin1,data2


def bg_outliers_cum(cube):
    # only use to find bad pixel sfor backgrounds, not files with data in it!
    # Nans are used when pixels saturated
    nr,nf,ny,nx = cube.shape
    diffdata = np.diff (cube,axis=1)
    
    # First remove pixels with large dark current
    #ma_data = np.masked_invalid(data) # resistant to some bad shutters by medians across time.
    median_counts = np.nanmedian(np.nanmean(diffdata,axis=1),axis=0)
    slope,slope_rms = outlier_stats(median_counts)
    nz_counts = np.nanmedian(np.nanstd(diffdata,axis=1),axis=0)
    nz,nz_rms = outlier_stats(nz_counts)
    # will need to check the defaults for MYSTIC
    
    badPixelMap = np.where( (median_counts > (np.max( (slope+5*slope_rms,2*slope)))), np.nan, 1.0)
    badPixelMap *= np.where( (median_counts < (np.min( (slope-5*slope_rms,.5*slope)))) , np.nan, 1.0)
    badPixelMap *= np.where( (nz_counts > (np.max( (nz+5*nz_rms,1.5*nz)))) , np.nan, 1.0)
    badPixelMap *= np.where( (nz_counts < (np.min( (nz-5*nz_rms,.5*nz)))) , np.nan, 1.0)

    return badPixelMap,median_counts,nz_counts

def outlier_stats(x):
    # Measure the percentile intervals and then estimate Standard Deviation of the distribution, 
    # both from median to the 90th percentile and from the 10th to 90th percentile
    # returns median, robust sigma
    p90 = np.nanpercentile(x, 90)
    p10 = np.nanpercentile(x, 10)
    p50 = np.nanmedian(x)
    # p50 to p90 is 1.2815 sigma
    #rSig = (p90-p50)/1.2815
    #print("Robust Sigma=", rSig)

    rSig = (p90-p10)/(2*1.2815)
    #print("Robust Sigma=", rSig)
    return p50,rSig 

def smooth(x,width=5, kernel='boxcar'):
    if kernel == 'boxcar': the_kernel = Box1DKernel(width)
    if kernel == 'gaussian': the_kernel = Gaussian1DKernel(width)

    smoothed_data_box = convolve(x, the_kernel)
    return smooth_data_box
