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
from scipy.ndimage import gaussian_filter;
from scipy.optimize import least_squares;

from . import log, files, headers, setup, oifits, signal, plot;
from .headers import HM, HMQ, HMP, HMW, rep_nan;


def define_badpixels (bkg):
    '''
    Define bad pixels from a cube, given an input background
    on which the bad-pixels are detected.
    '''

    # Set log in input header
    hdr = bkg[0];

    # Load background error
    bkg_noise = pyfits.getdata (bkg[0]['ORIGNAME'],0);
    bkg_noise = np.mean (bkg_noise, (0,1));
    delta = bkg_noise - medfilt (bkg_noise, (3,3));
    stat = sigma_clipped_stats (delta);
    thr_mean = 40.0;
    bad_mean = np.abs(delta-stat[0])/stat[2] > thr_mean;

    hdr[HMQ+'BADPIX MEAN_THRESHOLD'] = (thr_mean, 'threshold in sigma');
    hdr[HMQ+'BADPIX MEAN_NUMBER'] = (np.sum (bad_mean), 'nb. of badpix');
    log.info ('Found %i bad pixels in MEAN'%np.sum (bad_mean));

    # Load background error
    bkg_noise = pyfits.getdata (bkg[0]['ORIGNAME'],'BACKGROUND_ERR');
    bkg_noise = np.mean (bkg_noise, (0,1));
    delta = bkg_noise - medfilt (bkg_noise, (3,3));
    stat = sigma_clipped_stats (delta);
    thr_err = 20.0;
    bad_err = np.abs(delta-stat[0])/stat[2] > thr_err;

    hdr[HMQ+'BADPIX ERR_THRESHOLD'] = (thr_err, 'threshold in sigma');
    hdr[HMQ+'BADPIX ERR_NUMBER'] = (np.sum (bad_err), 'nb. of badpix');
    log.info ('Found %i bad pixels in ERR'%np.sum (bad_err));
    
    # Load background error
    bkg_noise = pyfits.getdata (bkg[0]['ORIGNAME'],'BACKGROUND_NOISE');
    bkg_noise = np.mean (bkg_noise, (0,1));
    delta = bkg_noise - medfilt (bkg_noise, (3,3));
    stat = sigma_clipped_stats (delta);
    thr_noise = 15.0;
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

def compute_background (hdrs,output='output_bkg'):
    '''
    Compute BACKGROUND_MEAN file from a sequence of
    BACKGROUND. The output file had the mean and rms over
    all frames, written as ramp.
    '''
    elog = log.trace ('compute_background');

    # Check inputs
    headers.check_input (hdrs, required=1);

    # Load files
    hdr,cube = files.load_raw (hdrs, coaddRamp=True, saturationThreshold=False, continuityThreshold=False);
    log.info ('Data size: '+str(cube.shape));

    # Background mean
    log.info ('Compute mean and rms over input files');
    bkg_mean = np.mean (cube, axis=0);
    bkg_err  = np.std (cube, axis=0) / np.sqrt (cube.shape[0]);

    # Load all ramp of first file to measure readout noise
    __,cube = files.load_raw (hdrs[0:1], coaddRamp=False, saturationThreshold=False, continuityThreshold=False);

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
    hdu0.header['FILETYPE'] = 'BACKGROUND_MEAN';
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
    return hdulist;

def compute_beammap (hdrs,bkg,output='output_beammap'):
    '''
    Compute BEAM_MAP product. The output product contains
    keywords defining the fringe window and the photometric
    windows, as well as the spectral shift between them.
    '''
    elog = log.trace ('compute_beammap');

    # Check inputs
    headers.check_input (hdrs, required=1);
    headers.check_input (bkg, required=1, maximum=1);
    
    # Load background
    log.info ('Load background %s'%bkg[0]['ORIGNAME']);
    bkg_cube = pyfits.getdata (bkg[0]['ORIGNAME'],0);
    
    # Compute bad pixels position
    badpix = define_badpixels (bkg);
    
    # Load files
    hdr,cube = files.load_raw (hdrs, coaddRamp=True, background=bkg_cube,
                               badpix=badpix, output=output);
    log.info ('Data size: '+str(cube.shape));

    # Check background subtraction in empty region
    check_empty_window (cube, hdr);

    # Get dimensions
    nr,nf,ny,nx = cube.shape;

    # Number of spectral channels to extract on plots
    ns = int(setup.nspec (hdr)/2 + 0.5) + 1;
    x  = np.arange (nx);

    # Compute the sum
    log.info ('Compute mean over ramps and frames');
    cmean = np.mean (cube, axis=(0,1));

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
    hdr[HMW+'PHOTO WIDTHX']  = (pxw,'[pix] spatial std');
    hdr[HMW+'PHOTO CENTERX'] = (pxc,'[pix] python-def');
    hdr[HMW+'PHOTO WIDTHY']  = (pyw,'[pix] spectral half-size');
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
    hdr[HMW+'FRINGE WIDTHX']  = (fxw,'[pix] spatial std');
    hdr[HMW+'FRINGE CENTERX'] = (fxc,'[pix] python-def');
    hdr[HMW+'FRINGE WIDTHY']  = (fyw,'[pix] spectral half-size');
    hdr[HMW+'FRINGE CENTERY'] = (fyc,'[pix] python-def');
    
    # Extract spectrum of photo and fringes
    p_spectra = np.mean (pmap[:,int(pxc-2):int(pxc+3)], axis=1);
    p_spectra /= np.max (p_spectra);

    f_spectra = np.mean (fmap[:,int(fxc-2*fxw):int(fxc+2*fxw)+1], axis=1);
    f_spectra /= np.max (f_spectra);

    # Shift between photo and fringes in spectral direction
    shifty = register_translation (p_spectra[:,None],f_spectra[:,None], \
                                   upsample_factor=100)[0][0];

    # Compute shifted spectra
    ps_spectra = subpix_shift (p_spectra, -shifty);

    # Set in header
    hdr[HMW+'PHOTO SHIFTY'] = (shifty,'[pix] shift of PHOTO versus FRINGE');

    # Define quality flag
    quality = ffit.amplitude.value;
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
    ax[1].plot (fx, label='Data');
    ax[1].plot (x,ffit(x), label='Gaussian');
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

    # File
    log.info ('Create file');
    
    # First HDU
    hdu0 = pyfits.PrimaryHDU (cmean[None,None,:,:]);
    hdu0.header = hdr;
    hdu0.header['FILETYPE'] = hdrs[0]['FILETYPE']+'_MAP';
    hdu0.header['BUNIT'] = ('adu/pixel/frame','mean over ramp and frame');
    hdu0.header['SHAPE'] = '(nr,nf,ny,nx)';

    # Set files
    hdu0.header[HMP+'BACKGROUND_MEAN'] = os.path.basename (bkg[0]['ORIGNAME'])[-40:];

    # Second HDU
    hdu1 = pyfits.ImageHDU (fmap[None,None,:,:]);
    hdu1.header['EXTNAME'] = ('FRINGE_MAP','imprint of fringe flux');
    hdu1.header['BUNIT'] = ('adu/pixel/frame','mean over ramp and frame');
    hdu1.header['SHAPE'] = '(nr,nf,ny,nx)';

    # Third HDU
    hdu2 = pyfits.ImageHDU (pmap[None,None,:,:]);
    hdu2.header['EXTNAME'] = ('PHOTOMETRY_MAP','imprint of photometry flux');
    hdu2.header['BUNIT'] = ('adu/pixel/frame','mean over ramp and frame');
    hdu2.header['SHAPE'] = '(nr,nf,ny,nx)';
    
    # Write output file
    hdulist = pyfits.HDUList ([hdu0,hdu1,hdu2]);
    files.write (hdulist, output+'.fits');
    
    plt.close("all");
    return hdulist;

def compute_preproc (hdrs,bkg,bmaps,output='output_preproc'):
    '''
    Compute preproc file. The first HDU contains the
    fringe window. The second HDU contains the 6 photometries
    already extracted and re-aligned spectrally
    '''
    
    elog = log.trace ('compute_preproc');

    # Check inputs
    headers.check_input (hdrs,  required=1);
    headers.check_input (bkg,   required=1, maximum=1);
    headers.check_input (bmaps, required=1, maximum=6);

    # Load background
    log.info ('Load background %s'%bkg[0]['ORIGNAME']);
    bkg_cube = pyfits.getdata (bkg[0]['ORIGNAME'],0);
    
    # Compute bad pixels position
    badpix = define_badpixels (bkg);
    
    # Load files
    hdr,cube = files.load_raw (hdrs, background=bkg_cube, badpix=badpix, output=output);
    log.info ('Data size: '+str(cube.shape));

    # Check background subtraction in empty region
    check_empty_window (cube, hdr);

    # Extract the fringe as the middle of all provided map
    fxc0 = np.mean ([b['MIRC QC WIN FRINGE CENTERX'] for b in bmaps]);
    fyc0 = np.mean ([b['MIRC QC WIN FRINGE CENTERY'] for b in bmaps]);
                   
    # Define the closest integer
    fxc = int(round(fxc0));
    fyc = int(round(fyc0));
    log.info ('FRINGE CENTERX/Y = %i,%i'%(fxc,fyc));

    # Expected size on spatial and spectral direction are hardcoded 
    fxw = int(setup.fringe_widthx (hdr) / 2);
    pxw = int(setup.photo_widthx (hdr) / 2 + 1.5);
    ns  = int(setup.nspec (hdr)/2 + 2.5);
    
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
    
    # File
    log.info ('Create file');
    
    # First HDU
    hdu0 = pyfits.PrimaryHDU (fringe.astype('float32'));
    hdu0.header = hdr;
    hdu0.header['BUNIT'] = 'adu/pixel/frame';
    hdu0.header['FILETYPE'] += '_PREPROC';
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
    hdu1.header['SHAPE'] = '(nt,nr,nf,ny,nx)';
    
    # Write file
    hdulist = pyfits.HDUList ([hdu0,hdu1]);
    files.write (hdulist, output+'.fits');
    
    plt.close("all");
    return hdulist;

