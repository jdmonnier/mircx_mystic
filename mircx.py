import numpy as np;

import matplotlib.pyplot as plt;
from matplotlib.colors import LogNorm;

from astropy.stats import sigma_clipped_stats;
from astropy.io import fits as pyfits;
from astropy.modeling import models, fitting;

from skimage.feature import register_translation;

from scipy.fftpack import fft, ifft;
from scipy.signal import medfilt;
from scipy.ndimage.interpolation import shift as subpix_shift;
from scipy.ndimage import gaussian_filter;

from . import log, files, headers, setup;
from .headers import HM, HMQ, HMP, HMW;

def gaussian_filter_cpx (input,sigma,**kwargs):
    ''' Gaussian filter of a complex array '''
    return gaussian_filter (input.real,sigma,**kwargs) + \
           gaussian_filter (input.imag,sigma,**kwargs) * 1.j;
    
def getwidth (curve, threshold=None):
    '''
    Compute the width of curve around its maximum,
    given a threshold. Return the tuple (center,fhwm)
    '''
    
    if threshold is None:
        threshold = 0.5*np.max (curve);

    # Find rising point
    f = np.argmax (curve > threshold) - 1;
    first = f + (threshold - curve[f]) / (curve[f+1] - curve[f]);
    
    # Find lowering point
    l = len(curve) - np.argmax (curve[::-1] > threshold) - 1;
    last = l + (threshold - curve[l]) / (curve[l+1] - curve[l]);
    
    return 0.5*(last+first), 0.5*(last-first)
    
def check_empty_window (cube, hdr):
    ''' Extract empty window from a cube(r,f,xy)'''
    log.info ('Check the empty window');
    
    # Hardcoded defined
    sx,nx = (200,80);
    sy,ny = (45,55)

    # Add QC parameters
    hdr[HMQ+'WIN EMPTY STARTX'] = (sx,'[pix] python-ref');
    hdr[HMQ+'WIN EMPTY STARTY'] = (sy,'[pix] python-ref');
    hdr[HMQ+'WIN EMPTY NX'] = (nx,'[pix]');
    hdr[HMQ+'WIN EMPTY NY'] = (ny,'[pix]');

    # Crop the empty window
    empty = np.mean (cube[:,:,sy:sy+ny,sx:sx+nx], axis=(0,1));

    # Compute QC
    (mean,med,std) = sigma_clipped_stats (empty);
    hdr[HMQ+'EMPTY MED'] = (med,'[adu]');
    hdr[HMQ+'EMPTY MEAN'] = (mean,'[adu]');
    hdr[HMQ+'EMPTY STD'] = (std,'[adu]');

    return empty;

def extract_maps (hdr, bmaps):
    '''
    Load the maps from a sery of BEAM_MAP
    They are cropped to mach the windows in hdr
    '''

    # Setup
    npx = hdr[HMW+'PHOTO NX'];
    npy = hdr[HMW+'PHOTO NY'];
    nfx = hdr[HMW+'FRINGE NX'];
    nfy = hdr[HMW+'FRINGE NY'];
    
    fringe_map = np.zeros ((6,1,1,nfy,nfx));
    photo_map  = np.zeros ((6,1,1,npy,npx));
    shifty     = np.zeros (6);
    
    # Loop for the necessary map
    for bmap in bmaps:
        if bmap == []: continue;
            
        log.info ('Load BEAM_MAP file %s'%bmap['ORIGNAME']);
        mean_map = pyfits.getdata (bmap['ORIGNAME']);
        beam = int(bmap['FILETYPE'][4:5]) - 1;

        fsx = hdr[HMW+'FRINGE STARTX'];
        fsy = hdr[HMW+'FRINGE STARTY'];
        fringe_map[beam,:,:,:,:] = mean_map[:,:,fsy:fsy+nfy,fsx:fsx+nfx];
        
        psx = hdr[HMW+'PHOTO%i STARTX'%(beam)];
        psy = hdr[HMW+'PHOTO%i STARTY'%(beam)];
        photo_map[beam,:,:,:,:] = mean_map[:,:,psy:psy+npy,psx:psx+npx];

        shifty[beam] = bmap[HMW+'PHOTO SHIFTY'] - psy + fsy;

    return fringe_map, photo_map, shifty;

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
    hdr,cube = files.load_raw (hdrs, coaddRamp=True);

    # Background mean
    log.info ('Compute mean and rms over input files');
    bkg_mean = np.mean (cube, axis=0);
    bkg_std  = np.std (cube, axis=0) / np.sqrt (cube.shape[0]);

    # Load all ramp of first file to get temporal variance
    __,cube = files.load_raw (hdrs[0:1], coaddRamp=False);

    # Compute temporal rms
    log.info ('Compute rms over ramp/frame of first file');
    bkg_noise = np.std (cube[:,3:-3,:,:], axis=(0,1));
    
    # Select which one to plot
    nf,ny,nx = bkg_mean.shape;
    dy,dx = 35,15;
    idf = int(nf/2);
    idx = int(nx/2);
    idy = int(ny/2);
    
    # Add QC parameters
    (mean,med,std) = sigma_clipped_stats (bkg_mean[idf,idy-dy:idy+dy,idx-dx:idx+dx]);
    hdr.set (HMQ+'BKG_MEAN MED',med,'[adu] for frame nf/2');
    hdr.set (HMQ+'BKG_MEAN STD',std,'[adu] for frame nf/2');

    (smean,smed,sstd) = sigma_clipped_stats (bkg_std[idf,idy-dy:idy+dy,idx-dx:idx+dx]);
    hdr.set (HMQ+'BKG_ERR MED',smed,'[adu] for frame nf/2');
    hdr.set (HMQ+'BKG_ERR STD',sstd,'[adu] for frame nf/2');
    
    (smean,nmed,nstd) = sigma_clipped_stats (bkg_noise[idy-dy:idy+dy,idx-dx:idx+dx]);
    hdr.set (HMQ+'BKG_NOISE MED',round(nmed,5),'[adu] for first file');
    hdr.set (HMQ+'BKG_NOISE STD',round(nstd,5),'[adu] for first file');

    # Define quality flag
    hdr[HMQ+'QUALITY'] = (1./(smed+1e-10), 'quality of data');
    
    # Create output HDU
    hdu0 = pyfits.PrimaryHDU (bkg_mean[None,:,:,:]);
    hdu0.header = hdr;

    # Update header
    hdu0.header['BUNIT'] = 'ADU';
    hdu0.header['FILETYPE'] = 'BACKGROUND_MEAN';

    # Create second HDU
    hdu1 = pyfits.ImageHDU (bkg_std[None,:,:,:]);
    hdu1.header['BUNIT'] = 'ADU';
    hdu1.header['EXTNAME'] = 'BACKGROUND_ERR';

    # Create third HDU
    hdu2 = pyfits.ImageHDU (bkg_noise[None,None,:,:]);
    hdu2.header['BUNIT'] = 'ADU';
    hdu2.header['EXTNAME'] = 'BACKGROUND_NOISE';
    
    # Write output file
    hdulist = pyfits.HDUList ([hdu0,hdu1,hdu2]);
    files.write (hdulist, output+'.fits');

    # Figures
    log.info ('Figures');

    # Images
    fig,ax = plt.subplots (3,1);
    ax[0].imshow (bkg_mean[idf,:,:], vmin=med-5*std, vmax=med+5*std, interpolation='none');
    ax[0].set_ylabel ('Mean');
    ax[1].imshow (bkg_mean[idf,:,:], vmin=med-20*std, vmax=med+20*std, interpolation='none');
    ax[1].set_ylabel ('Mean');
    ax[2].imshow (bkg_std[idf,:,:], vmin=smed-20*sstd, vmax=smed+20*sstd, interpolation='none');
    ax[2].set_ylabel ('Mean_err');
    fig.savefig (output+'_mean.png');

    # Histograms of median
    fig,ax = plt.subplots (2,1);
    ax[0].hist (bkg_mean[idf,:,:].flatten(),bins=med+std*np.linspace(-10,10,50));
    ax[0].set_ylabel ("Number of pixels");
    ax[0].grid ();
    ax[1].hist (bkg_mean[idf,:,:].flatten(),bins=med+std*np.linspace(-10,10,50));
    ax[1].set_ylabel ("Number of pixels");
    ax[1].set_xlabel ("Value at frame nf/2 (adu)");
    ax[1].set_yscale ('log');
    ax[1].grid ();
    fig.savefig (output+'_histo.png');

    # Histograms of noise
    fig,ax = plt.subplots (2,1);
    ax[0].hist (bkg_noise.flatten(),bins=nmed+nstd*np.linspace(-1.1,10,50));
    ax[0].set_ylabel ("Number of pixels");
    ax[0].grid ();
    ax[1].hist (bkg_noise.flatten(),bins=nmed+nstd*np.linspace(-1.1,10,50));
    ax[1].set_ylabel ("Number of pixels");
    ax[1].set_xlabel ("Value for first file");
    ax[1].set_yscale ('log');
    ax[1].grid ();
    fig.savefig (output+'_histonoise.png');

    # Ramp
    fig,ax = plt.subplots();
    ax.plot (np.median (bkg_mean,axis=(1,2)));
    ax.set_xlabel ("Frame");
    ax.set_ylabel ("Median of pixels (adu)");
    fig.savefig (output+'_ramp.png');

    plt.close("all");
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
    
    # Load files
    hdr,cube = files.load_raw (hdrs, coaddRamp=True);

    # Get dimensions
    nr,nf,ny,nx = cube.shape;
    x  = np.arange (nx);

    # Number of spectral channels to extract on plots
    ns = int(setup.get_nspec (hdr)/2 + 0.5) + 1;

    # Remove background
    log.info ('Load background %s'%bkg[0]['ORIGNAME']);
    cube -= pyfits.getdata (bkg[0]['ORIGNAME'],0);

    # Check background subtraction in empty region
    check_empty_window (cube, hdr);

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
    pyc,pyw = getwidth (medfilt (py, 5));

    # Fit spatial of photometry with Gaussian
    px = np.mean (pmap[int(pyc-pyw):int(pyc+pyw),:], axis=0);
    init = models.Gaussian1D (amplitude=np.max(px), mean=np.argmax(px), stddev=1.);
    pfit  = fitting.LevMarLSQFitter()(init, x, px);
    pxc,pxw = pfit.mean.value,pfit.stddev.value;

    log.info ('Found limit photo in spectral direction: %f %f'%(pyc,pyw));
    log.info ('Found limit photo in spatial direction: %f %f'%(pxc,pxw));
    
    # Add QC parameters for window
    hdr[HMW+'PHOTO WIDTHX']  = (pxw,'[pix]');
    hdr[HMW+'PHOTO CENTERX'] = (pxc,'[pix] python-def');
    hdr[HMW+'PHOTO WIDTHY']  = (pyw,'[pix]');
    hdr[HMW+'PHOTO CENTERY'] = (pyc,'[pix] python-def');

    # Get spectral limits of fringe
    fy  = np.mean (fmap,axis=1);
    fyc,fyw = getwidth (medfilt (fy, 5));
    
    # Fit spatial of fringe with Gaussian
    fx  = np.mean (fmap[int(fyc-fyw):int(fyc+fyw),:], axis=0);
    init = models.Gaussian1D (amplitude=np.max(fx), mean=np.argmax(fx), stddev=50.);
    ffit  = fitting.LevMarLSQFitter()(init, x, fx);
    fxc,fxw = ffit.mean.value,ffit.stddev.value;
        
    log.info ('Found limit fringe in spectral direction: %f %f'%(fyc,fyw));
    log.info ('Found limit fringe in spatial direction: %f %f'%(fxc,fxw));

    # Add QC parameters for window
    hdr[HMW+'FRINGE WIDTHX']  = (fxw,'[pix]');
    hdr[HMW+'FRINGE CENTERX'] = (fxc,'[pix] python-def');
    hdr[HMW+'FRINGE WIDTHY']  = (fyw,'[pix]');
    hdr[HMW+'FRINGE CENTERY'] = (fyc,'[pix] python-def');
    
    # Extract spectrum of photo and fringes
    p_spectra = np.mean (pmap[:,int(pxc-2):int(pxc+3)], axis=1);
    p_spectra /= np.max (p_spectra);

    f_spectra = np.mean (fmap[:,int(fxc-2*fxw):int(fxc+2*fxw)+1], axis=1);
    f_spectra /= np.max (f_spectra);

    # Shift between photo and fringes in spectral direction
    shifty = register_translation (p_spectra[:,None],f_spectra[:,None],upsample_factor=100)[0][0];

    # Set in header
    hdr[HMW+'PHOTO SHIFTY'] = (shifty,'[pix] shift of PHOTO versus FRINGE');

    # Define quality flag
    hdr[HMQ+'QUALITY'] = (np.max (fmap), 'quality of data');

    # Figures
    log.info ('Figures');
    
    # Figures of photo
    fig,ax = plt.subplots(3,1);
    ax[0].imshow (pmap, interpolation='none');
    ax[1].plot (px, label='Data');
    ax[1].plot (x,pfit(x), label='Gaussian');
    ax[1].set_ylabel ('adu/pix/fr');
    ax[1].legend ();
    ax[2].imshow (pmap[int(pyc-ns):int(pyc+ns+1),int(pxc-2):int(pxc+3)], interpolation='none');
    fig.savefig (output+'_pfit.png');

    # Figures of fringe
    fig,ax = plt.subplots(3,1);
    ax[0].imshow (fmap, interpolation='none');
    ax[1].plot (fx, label='Data');
    ax[1].plot (x,ffit(x), label='Gaussian');
    ax[1].set_ylabel ('adu/pix/fr');
    ax[1].legend ();
    ax[2].imshow (fmap[int(fyc-ns):int(fyc+ns+1)+1,int(fxc-2*fxw):int(fxc+2*fxw)], interpolation='none');
    fig.savefig (output+'_ffit.png');

    # Shifted spectra
    fig,ax = plt.subplots(2,1);
    ax[0].imshow (cmean, interpolation='none');
    ax[1].plot (f_spectra, label='fringe');
    ax[1].plot (p_spectra, label='photo');
    ax[1].plot (subpix_shift (p_spectra, -shifty), label='shifted photo');
    ax[1].legend ();
    fig.savefig (output+'_cut.png');

    # File
    log.info ('Create file');
    
    # First HDU
    hdu0 = pyfits.PrimaryHDU (cmean[None,None,:,:]);
    hdu0.header = hdr;
    hdu0.header['FILETYPE'] = hdrs[0]['FILETYPE']+'_MAP';

    # Set files
    hdu0.header[HMP+'BACKGROUND_MEAN'] = bkg[0]['ORIGNAME'];

    # Second HDU
    hdu1 = pyfits.ImageHDU (fmap);
    hdu1.header['EXTNAME'] = 'FRINGE_MAP';

    # Third HDU
    hdu2 = pyfits.ImageHDU (pmap);
    hdu2.header['EXTNAME'] = 'PHOTOMETRY_MAP';
    
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

    # Load files
    hdr,cube = files.load_raw (hdrs);

    # Remove background
    log.info ('Load background %s'%bkg[0]['ORIGNAME']);
    cube -= pyfits.getdata (bkg[0]['ORIGNAME'],0);

    # Check background subtraction in empty region
    check_empty_window (cube, hdr);

    # We use the fringe window of the first BEAM_MAP
    # (FIME: could be improved)

    # Extract the fringe as the middle of all provided map
    fxc0 = np.mean ([b['MIRC QC WIN FRINGE CENTERX'] for b in bmaps]);
    fyc0 = np.mean ([b['MIRC QC WIN FRINGE CENTERY'] for b in bmaps]);
                   
    # Define the closest integer
    fxc = int(round(fxc0));
    fyc = int(round(fyc0));

    # Expected size on spatial and spectral direction are hardcoded 
    fxw = int(setup.get_fringe_widthx (hdr) / 2);
    pxw = int(setup.get_photo_widthx (hdr) / 2 + 1.5);
    ns  = int(setup.get_nspec (hdr)/2 + 3.5);
    
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

    # Same for photometries
    nr,nf,ny,nx = fringe.shape;
    photos = np.zeros ((6,nr,nf,ny,2*pxw+1));
    for bmap in bmaps:
        if bmap == []: continue;
        
        # Get the position of the photo spectra
        pxc = int(round(bmap['MIRC QC WIN PHOTO CENTERX']));
        pyc = int(round(bmap['MIRC QC WIN PHOTO CENTERY']));

        # Set the required crop in header
        beam = int(bmap['FILETYPE'][4:5]) - 1;
        hdr[HMW+'PHOTO%i STARTX'%(beam)] = (pxc-pxw, '[pix] python-def');
        hdr[HMW+'PHOTO%i STARTY'%(beam)] = (pyc-ns, '[pix] python-def');
        photos[beam,:,:,:,:] = cube[:,:,pyc-ns:pyc+ns+1,pxc-pxw:pxc+pxw+1];
        
        
    # Figures
    log.info ('Figures');

    # Fringe and photo mean
    fig,ax = plt.subplots(2,1);
    ax[0].imshow (np.mean (fringe,axis=(0,1)), interpolation='none');
    ax[1].imshow (np.swapaxes (np.mean (photos,axis=(1,2)), 0,1).reshape((ny,-1)), interpolation='none');
    fig.savefig (output+'_mean.png');

    # Spectra
    fig,ax = plt.subplots();
    ax.plot (np.mean (fringe, axis=(0,1,3)), '--', label='fringes');
    ax.plot (np.mean (photos, axis=(1,2,4)).T);
    ax.set_ylabel ('adu/pix/fr');
    ax.legend ();
    fig.savefig (output+'_spectra.png');
    
    # File
    log.info ('Create file');
    
    # First HDU
    hdu0 = pyfits.PrimaryHDU (fringe);
    hdu0.header = hdr;
    hdu0.header['BUNIT'] = 'ADU';
    hdu0.header['FILETYPE'] += '_PREPROC';
    
    # Set files
    hdu0.header[HMP+'BACKGROUND_MEAN'] = bkg[0]['ORIGNAME'];
    hdu0.header[HMP+'FRINGE_MAP'] = bmaps[0]['ORIGNAME'];
    for bmap in bmaps:
        hdu0.header[HMP+bmap['FILETYPE']] = bmap['ORIGNAME'];

    # Second HDU with photometries
    hdu1 = pyfits.ImageHDU (photos);
    hdu1.header['BUNIT'] = 'ADU';
    hdu1.header['EXTNAME'] = 'PHOTOMETRY_PREPROC';
    
    # Write file
    hdulist = pyfits.HDUList ([hdu0,hdu1]);
    files.write (hdulist, output+'.fits');
    
    plt.close("all");
    return hdulist;

def compute_rts (hdrs, bmaps, output='output_rts'):
    '''
    Compute the RTS
    '''
    elog = log.trace ('compute_rts');

    # Check inputs
    headers.check_input (hdrs,  required=1, maximum=1);
    headers.check_input (bmaps, required=1, maximum=6);
    f = hdrs[0]['ORIGNAME'];

    # Load DATA
    log.info ('Load PREPROC file %s'%f);
    hdr = pyfits.getheader (f);
    fringe = pyfits.getdata (f);
    photo  = pyfits.getdata (f, 'PHOTOMETRY_PREPROC');
    nr,nf,ny,nx = fringe.shape

    # Get fringe and photo maps
    fringe_map, photo_map, shifty = extract_maps (hdr, bmaps);

    # Compute the expected position of lbd0    
    fcy = np.mean ([h[HMW+'FRINGE CENTERY'] for h in bmaps]) - hdr[HMW+'FRINGE STARTY'];
    
    # Build wavelength
    lbd0,dlbd = setup.get_lbd0 (hdr);
    lbd = (np.arange (ny) - fcy) * dlbd + lbd0;

    # Define profile for optimal extraction of photometry
    # The same profile is used for all spectral channels
    # profile is normalised to be flux-conservative
    profile  = np.mean (photo_map, axis=3, keepdims=True);
    profile *= np.sum (profile,axis=-1, keepdims=True) / np.sum (profile**2,axis=-1, keepdims=True);

    # Optimal extraction of photometry with profile
    log.info ('Extract photometry with profile');
    photo = np.sum (photo * profile, axis=-1);

    # Construct kappa-matrix
    # Use the provided kappa-matrix if any
    # so that the photo signal is the
    # normalisation to the fringe
    log.info ('Use kappa-matrix');
    kappa = np.sum (medfilt (fringe_map,[1,1,1,1,11]), axis=-1) / \
            np.sum (photo_map * profile, axis=-1);

    # kappa is defined so that photo is the
    # total number of adu in the fringe
    photok = photo * kappa;

    # Smooth photometry
    log.info ('Smooth photometry by 2 frames');
    photok = gaussian_filter (photok,(0,0,2,0));

    # Spectral shift of photometry to align with fringes
    for b in range(6):
        log.info ('Shift photometry %i by %.3f pixels'%(b,+shifty[b]));
        photok[b,:,:,:] = subpix_shift (photok[b,:,:,:], [0,0,+shifty[b]]);

    # Temporal / Spectral averaging of photometry
    # to be discussed
    log.info ('Temporal / Spectral averaging of photometry');
    spectra  = np.mean (photok, axis=(1,2), keepdims=True);
    spectra /= np.sum (spectra,axis=3, keepdims=True);
    injection = np.sum (photok, axis=3, keepdims=True);
    photok = spectra*injection;
    
    

    # Compute flux in fringes
    log.info ('Compute dc in fringes');
    fringe_map  = medfilt (fringe_map,[1,1,1,1,11]);
    fringe_map /= np.sum (fringe_map, axis=-1,keepdims=True);
    
    cont = np.zeros ((nr,nf,ny,nx));
    for b in range(6):
        cont += photok[b,:,:,:,None] * fringe_map[b,:,:,:,:];
        
    # QC about the fringe dc
    log.info ('Compute QC about dc');
    photodc_mean  = np.mean (cont,axis=(2,3));
    fringedc_mean = np.mean (fringe,axis=(2,3));
    hdr[HMQ+'DC MEAN'] = np.sum (fringedc_mean) / np.sum (photodc_mean);
    
    poly_dc = np.polyfit (photodc_mean.flatten(), fringedc_mean.flatten(), 2);
    hdr[HMQ+'DC ORDER0'] = (poly_dc[0],'[adu] fit DC(photo)');
    hdr[HMQ+'DC ORDER1'] = (poly_dc[1],'[adu/adu] fit DC(photo)');
    hdr[HMQ+'DC ORDER2'] = (poly_dc[2],'[adu/adu2] fit DC(photo)');
        
    # Subtract continuum
    log.info ('Subtract dc');
    fringe_hf = fringe - cont;

    # Model (x,f)
    log.info ('Model of data');
    nfq = int(nx/2);
    x = 1. * np.arange(nx) / nx;
    f = 1. * np.arange(1,nfq+1);

    # Scale to ensure the frequencies fall
    # into integer pixels (max freq in 40)
    freqs = setup.get_base_freq (hdr);
    scale0 = 40. / np.abs (freqs * nx).max();
    ifreqs = np.round(freqs * scale0 * nx).astype(int);

    # Compute DFT. The amplitude of the complex number
    # correspond to the total adu in the fringe enveloppe
    model = np.zeros ((nx,nfq*2+1));
    cf = 0.j + np.zeros ((nr*nf,ny,nfq+1));
    for y in np.arange(ny):
        log.info ('Fit channel %i'%y);
        amp = np.ones (nx) / nx;
        model[:,0] = amp;
        scale = lbd0 / lbd[y] / scale0;
        model[:,1:nfq+1] = amp[:,None] * 2 * np.cos (2.*np.pi * x[:,None] * f[None,:] * scale);
        model[:, nfq+1:] = amp[:,None] * 2 * np.sin (2.*np.pi * x[:,None] * f[None,:] * scale);
        cfc = np.tensordot (model,fringe_hf[:,:,y,:],axes=([0],[2])).reshape((nx,nr*nf)).T;
        cf[:,y,0]  = cfc[:,0];
        cf[:,y,1:] = cfc[:,1:nfq+1] - 1.j * cfc[:,nfq+1:];
    cf.shape = (nr,nf,ny,nfq+1);

    # DFT at fringe frequencies
    base_dft  = cf[:,:,:,np.abs(ifreqs)];
    
    # DFT at bias frequencies
    ibias = np.abs (ifreqs).max() + 4 + np.arange (5);
    bias_dft  = cf[:,:,:,ibias];

    # Compute unbiased PSD for plots (without coherent average
    # thus the bias is larger than in the base data).
    cf_upsd  = np.abs(cf[:,:,:,0:nx/2])**2;
    cf_upsd -= np.mean (cf_upsd[:,:,:,ibias],axis=-1,keepdims=True);

    # Figures
    log.info ('Figures');
    
    # Check dc
    fig,ax = plt.subplots ();
    ax.hist2d (photodc_mean.flatten(), fringedc_mean.flatten(),
               bins=40, norm=LogNorm());
    plt.plot (photodc_mean.flatten(),np.poly1d(poly_dc)(photodc_mean.flatten()),'--');
    plt.plot (photodc_mean.flatten(),photodc_mean.flatten(),'-');
    ax.set_xlabel('fringe dc'); ax.set_ylabel('sum of photo');
    ax.grid();
    fig.savefig (output+'_dccorr.png');

    # Integrated spectra
    fig,ax = plt.subplots (2,1);
    ax[0].plot (lbd*1e6,np.mean (fringe,axis=(0,1,3)),'--', label='fringes and photo');
    ax[0].plot (lbd*1e6,np.mean (photo,axis=(1,2)).T);
    ax[0].legend(); ax[0].grid();
    ax[0].set_ylabel ('adu/pix/frame');
    
    val = np.mean (fringe,axis=(0,1,3));
    val /= np.max (medfilt (val,3), keepdims=True);
    ax[1].plot (lbd*1e6,val,'--', label='fringes and photo * kappa * map');
    val = np.mean (photok, axis=(1,2));
    val /= np.max (medfilt (val,(1,3)), axis=1, keepdims=True);
    ax[1].plot (lbd*1e6,val.T);
    ax[1].legend(); ax[1].grid();
    ax[1].set_ylabel ('normalized');
    ax[1].set_xlabel ('lbd (um)');
    fig.savefig (output+'_spectra.png');
    
    # Power densities
    fig,ax = plt.subplots (2,1);
    ax[0].imshow ( np.mean (cf_upsd, axis=(0,1)));
    for f in ifreqs: ax[0].axvline (np.abs(f), color='k', linestyle='--');
    ax[1].plot ( np.mean (cf_upsd, axis=(0,1))[ny/2,:]);
    ax[1].set_xlim (0,cf_upsd.shape[-1]);
    ax[1].grid();
    fig.savefig (output+'_psd.png');

    # File
    log.info ('Create file');

    # First HDU
    hdu0 = pyfits.PrimaryHDU ([]);
    hdu0.header = hdr;
    hdu0.header['FILETYPE'] = 'RTS';
    hdu0.header[HMP+'PREPROC'] = hdrs[0]['ORIGNAME'];

    # Set DFT of fringes, bias, photometry and lbd
    hdu1 = pyfits.ImageHDU (base_dft.real);
    hdu1.header['EXTNAME'] = 'BASE_DFT_REAL';
    hdu1.header['BUNIT'] = ('adu','adu in the fringe envelope');
    
    hdu2 = pyfits.ImageHDU (base_dft.imag);
    hdu2.header['EXTNAME'] = 'BASE_DFT_IMAG';
    hdu1.header['BUNIT'] = ('adu','adu in the fringe envelope');
    
    hdu3 = pyfits.ImageHDU (bias_dft.real);
    hdu3.header['EXTNAME'] = 'BIAS_DFT_REAL';
    hdu1.header['BUNIT'] = ('adu','adu in the fringe envelope');
    
    hdu4 = pyfits.ImageHDU (bias_dft.imag);
    hdu4.header['EXTNAME'] = 'BIAS_DFT_IMAG';
    hdu1.header['BUNIT'] = ('adu','adu in the fringe envelope');
    
    hdu5 = pyfits.ImageHDU (np.transpose (photo,axes=(1,2,3,0)));
    hdu5.header['EXTNAME'] = 'PHOTOMETRY';
    hdu1.header['BUNIT'] = ('adu','adu in the fringe envelope');

    hdu6 = pyfits.ImageHDU (lbd);
    hdu6.header['EXTNAME'] = 'WAVELENGTH';
    hdu6.header['BUNIT'] = 'm';
        
    # Write file
    hdulist = pyfits.HDUList ([hdu0,hdu1,hdu2,hdu3,hdu4,hdu5,hdu6]);
    files.write (hdulist, output+'.fits');
    
    plt.close("all");
    return hdulist;

def compute_vis (hdrs, output='output_vis', ncoher=3.0):
    '''
    Compute the VIS
    '''
    elog = log.trace ('compute_vis');

    # Check inputs
    headers.check_input (hdrs,  required=1, maximum=1);
    f = hdrs[0]['ORIGNAME'];

    # Get data
    log.info ('Load RTS file %s'%f);
    hdr = pyfits.getheader (f);
    base_dft  = pyfits.getdata (f, 'BASE_DFT_IMAG') * 1.j;
    base_dft += pyfits.getdata (f, 'BASE_DFT_REAL');
    bias_dft  = pyfits.getdata (f, 'BIAS_DFT_IMAG') * 1.j;
    bias_dft += pyfits.getdata (f, 'BIAS_DFT_REAL');
    photo     = pyfits.getdata (f, 'PHOTOMETRY');
    lbd       = pyfits.getdata (f, 'WAVELENGTH');

    # Dimensions
    nr,nf,ny,nb = base_dft.shape;

    # Compute lbd0 and dlbd
    lbd0 = np.mean (lbd);
    dlbd = np.mean (np.diff (lbd));

    # Do coherent integration
    log.info ('Coherent integration over %.1f frames'%ncoher);
    hdr[HMQ+'NFRAME_COHER'] = (ncoher,'nb. of frames integrated coherently');
    base_dft = gaussian_filter_cpx (base_dft,(0,ncoher,0,0),mode='constant',truncate=2.0);
    bias_dft = gaussian_filter_cpx (bias_dft,(0,ncoher,0,0),mode='constant',truncate=2.0);
    photo = gaussian_filter (photo,(0,ncoher,0,0),mode='constant',truncate=2.0);
            
    # Compute group-delay in [m] and broad-band power
    log.info ('Compute GD');
    base_gd  = np.angle (np.sum (base_dft[:,:,1:,:] * np.conj (base_dft[:,:,:-1,:]), axis=2, keepdims=True));
    base_gd /= (1./(lbd0) - 1./(lbd0+dlbd)) * (2*np.pi);

    phasor = np.exp (2.j*np.pi * base_gd / lbd[None,None,:,None]);
    base_powerbb = np.abs (np.sum (base_dft * phasor, axis=2, keepdims=True))**2;

    # Compute group-delay and broad-band power for bias
    bias_gd  = np.angle (np.sum (bias_dft[:,:,1:,:] * np.conj (bias_dft[:,:,:-1,:]), axis=2,keepdims=True));
    bias_gd /= (1./(lbd0) - 1./(lbd0+dlbd)) * (2*np.pi);
        
    phasor = np.exp (2.j*np.pi * bias_gd / lbd[None,None,:,None]);
    bias_powerbb = np.mean (np.abs (np.sum (bias_dft * phasor, axis=2,keepdims=True))**2,axis=-1,keepdims=True);
    
    # Broad-band SNR
    base_snrbb = base_powerbb / bias_powerbb + 1;

    # Compute power per spectral channels
    bias_power = np.mean (np.abs (bias_dft)**2,axis=-1,keepdims=True);
    base_power = np.abs (base_dft)**2; 

    # Compute norm power
    log.info ('Compute norm power');
    base = setup.get_base_beam ();
    norm_power = photo[:,:,:,base[:,0]] * photo[:,:,:,base[:,1]];
    
    # QC for power
    log.info ('Compute QC for power');
    for b,name in enumerate (setup.get_base_name ()):
        val = np.mean (norm_power[:,:,ny/2,b], axis=(0,1));
        hdr[HMQ+'NORM'+name+' MEAN'] = (val,'Norm Power at lbd0');
        val = np.mean (base_power[:,:,ny/2,b], axis=(0,1));
        hdr[HMQ+'POWER'+name+' MEAN'] = (val,'Fringe Power at lbd0');
        val = np.std (base_power[:,:,ny/2,b], axis=(0,1));
        hdr[HMQ+'POWER'+name+' STD'] = (val,'Fringe Power at lbd0');
        val = np.mean (base_snrbb[:,:,:,b]);
        hdr[HMQ+'SNR'+name+' MEAN'] = (val,'Broad-band SNR');
        val = np.std (base_snrbb[:,:,:,b]);
        hdr[HMQ+'SNR'+name+' STD'] = (val,'Broad-band SNR');
    val = np.mean (bias_power[:,:,ny/2,:], axis=(0,1,-1));
    hdr[HMQ+'BIAS MEAN'] = (val,'Bias Power at lbd0');

    # Compute mean SNR over ramp
    # TODO: probably should another time-constant
    base_snr = np.mean (base_snrbb, axis=1,keepdims=True);

    # TODO: Bootstrap over baseline

    # Compute flag from averaged SNR over the ramp
    base_flag = 1.0 * (base_snr > 5.0);
    base_flag[base_flag == 0.0] = np.nan;

    # Compute visibility
    log.info ('Compute VIS');
    # base_power *= base_flag;
    # norm_power *= base_flag;
    # bias_power *= base_flag;
    vis = np.nanmean (base_power - bias_power, axis=(0,1)) / np.nanmean (norm_power, axis=(0,1));

    # QC for VIS
    for b,name in enumerate (setup.get_base_name ()):
        val = headers.rep_nan (vis[ny/2,b]);
        hdr[HMQ+'VISS'+name+' MEAN'] = (val,'visibility at lbd0');

    # Figures
    log.info ('Figures');

    # Correlation
    fig,axes = plt.subplots (5,3);
    for b,ax in enumerate(axes.flatten()):
        ax.plot ( np.mean (norm_power[:,:,ny/2,b],1), np.mean (base_power[:,:,ny/2,b],1), 'o');
        ax.grid();
    fig.savefig (output+'_norm_power.png');
    
    # SNR, GD and FLAGs
    fig,ax = plt.subplots (3,1);
    ax[0].imshow (np.log10 (np.mean (base_snrbb,axis=(1,2))).T);
    ax[0].grid(); ax[0].set_ylabel ('log10 (SNR_bb)');
    ax[1].imshow (np.mean (base_gd,axis=(1,2)).T * 1e6);
    ax[1].grid(); ax[1].set_ylabel ('gdelay (um)');
    ax[1].set_xlabel ('ramp');
    ax[2].imshow (np.mean (base_flag,axis=(1,2)).T);
    ax[2].grid(); ax[1].set_ylabel ('gdelay (um)');
    ax[2].set_xlabel ('flag');
    fig.savefig (output+'_snr_gd.png');

    # File
    log.info ('Create file');

    # First HDU
    hdu0 = pyfits.PrimaryHDU ([]);
    hdu0.header = hdr;
    hdu0.header['FILETYPE'] = 'VIS';
    hdu0.header[HMP+'RTS'] = hdrs[0]['ORIGNAME'];
    
    # Write file
    hdulist = pyfits.HDUList ([hdu0]);
    files.write (hdulist, output+'.fits');
            
    plt.close("all");
    return hdulist;
    
