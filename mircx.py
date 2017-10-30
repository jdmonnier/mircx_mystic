import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from astropy.stats import sigma_clipped_stats
from astropy.io import fits as pyfits
from astropy.modeling import models, fitting

from skimage.feature import register_translation

from scipy.fftpack import fft, ifft
from scipy.signal import medfilt;
from scipy.ndimage.interpolation import shift as subpix_shift
from scipy.ndimage import gaussian_filter;

from . import log, files, headers, setup

def gaussian_filter_cpx (input,sigma,**kwargs):
    return gaussian_filter (input.real,sigma,**kwargs) + \
           gaussian_filter (input.imag,sigma,**kwargs) * 1.j;

def check_hdrs_input (hdrs, required=1):
    ''' Check the input when provided as hdrs'''

    # Ensure a list
    if type (hdrs) is not list:
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
    bkg_data  = pyfits.getdata (hdr['ORIGNAME'],0);
    
    # Remove background
    log.info ('Remove background');
    cube -= bkg_data;

def crop_window (cube, hdr, cx, dx, cy, dy):
    ''' Extract fringe window from a cube(r,f,xy)'''
    log.debug ('Extract region');

    sx = hdr['* STARTX'][0];
    nx = hdr['* NX'][0];
    sy = hdr['* STARTY'][0];
    ny = hdr['* NY'][0];
    
    # Return the crop
    return cube[:,:,sy:sy+ny,sx:sx+nx];
    
def check_empty_window (cube, hdr):
    ''' Extract empty window from a cube(r,f,xy)'''
    log.info ('Check the empty window');
    
    # Hardcoded defined
    sx,nx = (200,80);
    sy,ny = (45,55)

    # Add QC parameters
    hdr.set ('HIERARCH MIRC QC WIN EMPTY STARTX',sx,'[pix]');
    hdr.set ('HIERARCH MIRC QC WIN EMPTY NX',nx,'[pix]');
    hdr.set ('HIERARCH MIRC QC WIN EMPTY STARTY',sx,'[pix]');
    hdr.set ('HIERARCH MIRC QC WIN EMPTY NY',nx,'[pix]');

    # Crop the empty window
    empty = np.mean (cube[:,:,sy:sy+ny,sx:sx+nx], axis=(0,1));

    # Compute QC
    (mean,med,std) = sigma_clipped_stats (empty);
    hdr.set ('HIERARCH MIRC QC EMPTY MED',med,'[adu]');
    hdr.set ('HIERARCH MIRC QC EMPTY MEAN',mean,'[adu]');
    hdr.set ('HIERARCH MIRC QC EMPTY STD',std,'[adu]');

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
    hdu1 = pyfits.PrimaryHDU (bkg_mean[None,:,:,:]);
    hdu1.header = hdr;

    # Update header
    headers.set_revision (hdu1.header);
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
    ax1.imshow (bkg_mean[idf,:,:], vmin=med-5*std, vmax=med+5*std, interpolation='none');
    ax2.imshow (bkg_mean[idf,:,:], vmin=med-20*std, vmax=med+20*std, interpolation='none');
    ax3.imshow (bkg_std[idf,:,:], vmin=smed-20*sstd, vmax=smed+20*sstd, interpolation='none');
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

def getwidth (curve, threshold=None):
    
    if threshold is None:
        threshold = 0.5*np.max (curve);
        
    f = np.argmax (curve > threshold) - 1;
    first = f + (threshold - curve[f]) / (curve[f+1] - curve[f]);
    
    l = len(curve) - np.argmax (curve[::-1] > threshold) - 1;
    last = l + (threshold - curve[l]) / (curve[l+1] - curve[l]);
    
    return 0.5*(last+first), 0.5*(last-first)
    
def compute_beammap (hdrs,bkg,output='output_beammap'):
    '''
    Compute BEAM_MAP product. The output product contains
    keywords defining the fringe window and the photometric
    windows, as well as the spectral shift between them.
    '''
    elog = log.trace ('compute_beammap');

    # Check inputs
    check_hdrs_input (hdrs, required=1);
    check_hdrs_input (bkg, required=1);
    
    # Load files
    hdr,cube = files.load_raw (hdrs, coaddRamp=True);

    # Get dimensions
    nr,nf,ny,nx = cube.shape;
    x  = np.arange (nx);

    # Number of spectral channels to extract on plots
    ns = int(setup.get_nspec (hdr)/2 + 0.5) + 1;

    # Remove background
    remove_background (cube, bkg[0]);

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
    name = 'HIERARCH MIRC QC WIN PHOTO ';
    hdr.set (name+'WIDTHX',pxw,'[pix]');
    hdr.set (name+'CENTERX',pxc,'[pix] python-def');
    hdr.set (name+'WIDTHY',pyw,'[pix]');
    hdr.set (name+'CENTERY',pyc,'[pix] python-def');


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
    name = 'HIERARCH MIRC QC WIN FRINGE ';
    hdr.set (name+'WIDTHX',fxw,'[pix]');
    hdr.set (name+'CENTERX',fxc,'[pix] python-def');
    hdr.set (name+'WIDTHY',fyw,'[pix]');
    hdr.set (name+'CENTERY',fyc,'[pix] python-def');
    

    # Extract spectrum of photo and fringes
    p_spectra = np.mean (pmap[:,int(pxc-2):int(pxc+3)], axis=1);
    p_spectra /= np.max (p_spectra);

    f_spectra = np.mean (fmap[:,int(fxc-2*fxw):int(fxc+2*fxw)+1], axis=1);
    f_spectra /= np.max (f_spectra);

    # Shift between photo and fringes in spectral direction
    shifty = register_translation (p_spectra[:,None],f_spectra[:,None],upsample_factor=100)[0][0];
    
    name = 'HIERARCH MIRC QC WIN PHOTO SHIFTY';
    hdr.set (name,shifty,'[pix] shift of PHOTO versus FRINGE');
    
    
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

    # Figures
    fig,ax = plt.subplots(2,1);
    ax[0].imshow (cmean, interpolation='none');
    ax[1].plot (f_spectra, label='fringe');
    ax[1].plot (p_spectra, label='photo');
    ax[1].plot (subpix_shift (p_spectra, -shifty), label='shifted photo');
    ax[1].legend ();
    fig.savefig (output+'_cut.png');

    # Create output HDU
    hdu1 = pyfits.PrimaryHDU (cmean[None,None,:,:]);
    hdu1.header = hdr;

    # Update header
    hdu1.header['BZERO'] = 0;
    hdu1.header['FILETYPE'] = hdrs[0]['FILETYPE']+'_MAP';

    # Set files
    headers.set_revision (hdu1.header);
    hdu1.header['HIERARCH MIRC PRO BACKGROUND_MEAN'] = bkg[0]['ORIGNAME'];

    # Write output file
    hdulist = pyfits.HDUList (hdu1);
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
    check_hdrs_input (hdrs,  required=1);
    check_hdrs_input (bkg,   required=1);
    check_hdrs_input (bmaps, required=1);

    # Load files
    hdr,cube = files.load_raw (hdrs);

    # Remove background
    remove_background (cube, bkg[0]);

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
    
    # Crop fringe windows
    log.info ('Extract fringe region');
    fringe = cube[:,:,fyc-ns:fyc+ns+1,fxc-fxw:fxc+fxw+1];

    # Keep track of crop value
    hdr['HIERARCH MIRC QC WIN FRINGE STARTX'] = (fxc-fxw, '[pix] python-def');
    hdr['HIERARCH MIRC QC WIN FRINGE STARTY'] = (fyc-ns, '[pix] python-def');

    # Keep track of these values
    hdr['HIERARCH MIRC QC WIN FRINGE CENTERX'] = (fxc0-(fxc-fxw),'[pix]');
    hdr['HIERARCH MIRC QC WIN FRINGE CENTERY'] = (fyc0-(fyc-ns),'[pix]');

    # Init photometries and map to zero
    nr,nf,ny,nx = fringe.shape;
    photos      = np.zeros ((6,nr,nf,ny,2*pxw+1));
    photo_maps  = np.zeros ((6,1,1,ny,2*pxw+1));
    fringe_maps = np.zeros ((6,1,1,ny,2*fxw+1));

    # Loop on provided BEAM_MAP
    for beam in range(6):

        # Loop for the necessary map
        bmap = [ b for b in bmaps if b['FILETYPE'] == 'BEAM%i_MAP'%(beam+1) ];
        if len (bmap) != 1:
            log.warning ('Cannot extract photometry %i'%beam);
            continue;
        else:
            log.debug ('Extract photometry %i'%beam);
            bmap = bmap[0];

        # Get the position of the photo spectra
        pxc = int(round(bmap['MIRC QC WIN PHOTO CENTERX']));
        pyc = int(round(bmap['MIRC QC WIN PHOTO CENTERY']));
            
        # Extract photometric data
        log.info ('Extract photo region');
        photos[beam,:,:,:,:] = cube[:,:,pyc-ns:pyc+ns+1,pxc-pxw:pxc+pxw+1]

        # Extract photo_map and fringe_map data
        bcube = pyfits.getdata (bmap['ORIGNAME'], 0);
        photo_maps[beam,:,:,:,:]  = bcube[:,:,pyc-ns:pyc+ns+1,pxc-pxw:pxc+pxw+1];
        fringe_maps[beam,:,:,:,:] = bcube[:,:,fyc-ns:fyc+ns+1,fxc-fxw:fxc+fxw+1];

        # Keep track of crop value
        name = 'HIERARCH MIRC QC WIN PHOTO%i'%beam;
        hdr[name+' STARTX'] = (pxc-pxw, '[pix] python-def');
        hdr[name+' STARTY'] = (pyc-ns, '[pix] python-def');

        # Keep track of shift
        shifty = bmap['MIRC QC WIN PHOTO SHIFTY'] - pyc + fyc;
        hdr[name+' SHIFTY'] = (shifty, '[pix]');
        

    # Figure
    fig,ax = plt.subplots(2,1);
    ax[0].imshow (np.mean (fringe,axis=(0,1)), interpolation='none');
    ax[1].imshow (np.swapaxes (np.mean (photos,axis=(1,2)), 0,1).reshape((ny,-1)), interpolation='none');
    fig.savefig (output+'_mean.png');

    # Figures
    fig,ax = plt.subplots();
    ax.plot (np.mean (fringe, axis=(0,1,3)), '--', label='fringes');
    ax.plot (np.mean (photos, axis=(1,2,4)).T);
    ax.set_ylabel ('adu/pix/fr');
    ax.legend ();
    fig.savefig (output+'_spectra.png');
    
    # Create output HDU
    log.info ('Create file');
    hdu1 = pyfits.PrimaryHDU (fringe);
    hdu1.header = hdr;
    
    # Update header
    hdu1.header['BZERO'] = 0;
    hdu1.header['BUNIT'] = 'ADU';
    hdu1.header['FILETYPE'] += '_PREPROC';
    
    # Set files
    headers.set_revision (hdu1.header);
    hdu1.header['HIERARCH MIRC PRO BACKGROUND_MEAN'] = bkg[0]['ORIGNAME'];
    hdu1.header['HIERARCH MIRC PRO FRINGE_MAP'] = bmaps[0]['ORIGNAME'];
    for bmap in bmaps:
        hdu1.header['HIERARCH MIRC PRO '+bmap['FILETYPE']] = bmap['ORIGNAME'];

    # Second HDU with photometries
    hdu2 = pyfits.ImageHDU (photos);
    hdu2.header['BUNIT'] = 'ADU';
    hdu2.header['EXTNAME'] = 'PHOTOMETRY_PREPROC';

    # Second HDU with fringe map
    hdu3 = pyfits.ImageHDU (fringe_maps);
    hdu3.header['BUNIT'] = 'ADU';
    hdu3.header['EXTNAME'] = 'FRINGE_MAP';

    # Second HDU with photometries map
    hdu4 = pyfits.ImageHDU (photo_maps);
    hdu4.header['BUNIT'] = 'ADU';
    hdu4.header['EXTNAME'] = 'PHOTOMETRY_MAP';
    
    # Write file
    hdulist = pyfits.HDUList ([hdu1,hdu2,hdu3,hdu4]);
    files.write (hdulist, output+'.fits');
    
    plt.close("all");
    return hdulist;

def compute_snr (hdrs, output='output_snr'):
    elog = log.trace ('compute_snr');

    # Check inputs
    check_hdrs_input (hdrs,  required=1);
    f = hdrs[0]['ORIGNAME'];

    # Load data
    log.info ('Load PREPROC file %s'%f);
    hdr = pyfits.getheader (f);
    fringe = pyfits.getdata (f);
    photo  = pyfits.getdata (f, 'PHOTOMETRY_PREPROC');
    photo_map  = pyfits.getdata (f, 'PHOTOMETRY_MAP');
    fringe_map = pyfits.getdata (f, 'FRINGE_MAP');

    nr,nf,ny,nx = fringe.shape

    # Build wavelength
    lbd0,dlbd = setup.get_lbd0 (hdr);
    lbd = (np.arange (ny) - hdr['HIERARCH MIRC QC WIN FRINGE CENTERY']) * dlbd + lbd0;

    # Optimal extraction of  photometry
    # (same profile for all spectral channels)
    log.info ('Extract photometry');
    profile = np.mean (photo_map, axis=3, keepdims=True);
    profile = profile * np.sum (profile,axis=-1, keepdims=True) / np.sum (profile**2,axis=-1, keepdims=True);
    photo = np.sum (photo * profile, axis=-1);

    # Spectral shift of photometry to align with fringes
    for b in range(6):
        shifty = hdr['HIERARCH MIRC QC WIN PHOTO%i SHIFTY'%b];
        log.info ('Shift photometry %i by -%.3f pixels'%(b,shifty));
        photo[b,:,:,:] = subpix_shift (photo[b,:,:,:], [0,0,-shifty]);
    
    # Temporal / Spectral averaging of photometry
    # to be discussed
    log.info ('Temporal / Spectral averaging of photometry');
    spectra  = np.mean (photo, axis=(1,2), keepdims=True);
    spectra /= np.sum (spectra,axis=3, keepdims=True);
    injection = np.sum (photo, axis=3, keepdims=True);
    photo = spectra*injection;

    # Smooth photometry
    log.info ('Smooth photometry');
    photo = gaussian_filter (photo,(0,0,2,0));

    # Construct kappa-matrix
    log.info ('Construct kappa-matrix');
    kappa  = medfilt (fringe_map,[1,1,1,1,11]);
    kappa  = kappa / np.sum (photo_map * profile, axis=-1,keepdims=True);

    # Compute flux in fringes
    log.info ('Compute dc in fringes');
    cont = np.zeros ((nr,nf,ny,nx));
    for b in range(6):
        cont += photo[b,:,:,:,None] * kappa[b,:,:,:,:];

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

    # Compute DFT
    model = np.zeros ((nx,nfq*2+1));
    cf = 0.j + np.zeros ((nr*nf,ny,nfq+1));
    for y in np.arange(ny):
        log.info ('Fit channel %i'%y);
        amp = np.ones (nx);
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
    
    # Add additional frequencies to ifreqs
    # to get the bias power
    ibias = np.abs (ifreqs).max() + 4 + np.arange (5);
    bias_dft  = cf[:,:,:,ibias];

    # Do coherence integration
    ncoher = 3.;
    log.info ('Coherent integration over %.1f frames'%ncoher);
    base_dft = gaussian_filter_cpx (base_dft,(0,ncoher,0,0),mode='constant');
    bias_dft = gaussian_filter_cpx (bias_dft,(0,ncoher,0,0),mode='constant');
        
    # Compute power and unbias it
    bias_power = np.mean (np.abs (bias_dft)**2,axis=-1,keepdims=True);
    base_power = np.abs (base_dft)**2 - bias_power;

    # Compute group-delay in [m]
    gdelay = np.angle (np.sum (base_dft[:,:,1:,:] * np.conj (base_dft[:,:,:-1,:]), axis=2));
    gdelay /= (1./(lbd0) - 1./(lbd0+dlbd)) * (2*np.pi);

    # Compute unbiased PSD for plots
    cf_upsd  = np.abs(cf[:,:,:,0:nx/2])**2;
    cf_upsd -= np.mean (cf_upsd[:,:,:,ibias],axis=3,keepdims=True);

    log.info ('Figures');
    
    # Check photometry
    fig,ax = plt.subplots ();
    ax.hist2d (np.mean (fringe,axis=(2,3)).flatten(),
                np.mean (cont,axis=(2,3)).flatten(),
                bins=40, norm=LogNorm());
    ax.set_xlabel('fringe dc'); ax.set_ylabel('sum of photo');
    fig.savefig (output+'_contcorr.png');

    # Integrated spectra
    fig,ax = plt.subplots ();
    ax.plot (lbd*1e6,np.mean (fringe,axis=(0,1,3)),'--', label='fringes');
    ax.plot (lbd*1e6,np.mean (photo,axis=(1,2)).T, label='photo');
    ax.legend(); ax.grid();
    ax.set_xlabel ('lbd (um)'); ax.set_ylabel ('adu/pix/frame');
    fig.savefig (output+'_spectra.png');
    
    # Power densities
    fig,ax = plt.subplots (3,1);
    ax[0].imshow ( np.mean (cf_upsd, axis=(0,1)));
    for f in ifreqs: ax[0].axvline (np.abs(f), color='k', linestyle='--');
    ax[1].plot ( np.mean (cf_upsd, axis=(0,1))[ny/2,:]);
    ax[1].grid();
    ax[2].imshow (np.sum (base_power,axis=(0,1)));
    fig.savefig (output+'_psd.png');

    # Power SNR
    fig,ax = plt.subplots (2,1);
    ax[0].plot (np.log10 (np.mean (base_power/bias_power + 1,axis=1)[:,ny/2,:]));
    ax[0].grid(); ax[0].set_ylabel ('log10 (SNR)');
    ax[1].plot (np.mean (gdelay,axis=1) * 1e6);
    ax[1].grid(); ax[1].set_ylabel ('gdelay (um)');
    fig.savefig (output+'_snr_gd.png');
        
    plt.close('all');
    pass;
