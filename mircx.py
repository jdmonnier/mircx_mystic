import numpy as np
import matplotlib.pyplot as plt

from astropy.stats import sigma_clipped_stats
from astropy.io import fits as pyfits
from astropy.modeling import models, fitting

from skimage.feature import register_translation

from scipy.fftpack import fft, ifft
from scipy.signal import medfilt;
from scipy.ndimage.interpolation import shift as subpix_shift

from . import log, files, headers, setup

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
    bkg_data  = pyfits.getdata (hdr['ORIGNAME'],0);
    
    # Remove background
    log.info ('Remove background');
    cube -= bkg_data[None,:,:,:];

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
    hdu1 = pyfits.PrimaryHDU (bkg_mean);
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
        threshold = 0.25*np.max (curve);
        
    first = np.argmax (curve > threshold);
    last = len(curve) - np.argmax (curve[::-1] > threshold) - 1;
    
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
    hdu1 = pyfits.PrimaryHDU (cmean);
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
    fxc = int(round(np.mean ([b['MIRC QC WIN FRINGE CENTERX'] for b in bmaps])));
    fyc = int(round(np.mean ([b['MIRC QC WIN FRINGE CENTERY'] for b in bmaps])));

    # Expected size on spatial and spectral direction are hardcoded 
    fxw = int(setup.get_fringe_widthx (hdr) / 2);
    pxw = int(setup.get_photo_widthx (hdr) / 2 + 1.5);
    ns = int(setup.get_nspec (hdr)/2 + 3.5);
    
    # Crop fringe windows
    log.info ('Extract fringe region');
    fringe = cube[:,:,fyc-ns:fyc+ns+1,fxc-fxw:fxc+fxw+1];

    # Keep track
    hdr['HIERARCH MIRC QC WIN FRINGE STARTX'] = (fxc-fxw, '[pix] python-def');
    hdr['HIERARCH MIRC QC WIN FRINGE STARTY'] = (fyc-ns, '[pix] python-def');

    # Init photometries and map to zero
    nr,nf,ny,nx = fringe.shape;
    photos      = np.zeros ((6,nr,nf,ny,2*pxw+1));
    photo_maps  = np.zeros ((6,ny,2*pxw+1));
    fringe_maps = np.zeros ((6,ny,2*fxw+1));

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

        # Get the position of the spectra
        pxc = int(round(bmap['MIRC QC WIN PHOTO CENTERX']));
        pyc = int(round(bmap['MIRC QC WIN PHOTO CENTERY']));
            
        # Extract data
        log.info ('Extract photo region');
        photos[beam,:,:,:,:] = cube[:,:,pyc-ns:pyc+ns+1,pxc-pxw:pxc+pxw+1]

        bcube = pyfits.getdata (bmap['ORIGNAME'], 0);
        photo_maps[beam,:,:]  = bcube[pyc-ns:pyc+ns+1,pxc-pxw:pxc+pxw+1];
        fringe_maps[beam,:,:] = bcube[fyc-ns:fyc+ns+1,fxc-fxw:fxc+fxw+1];

        # Keep track of shift and crop value, to recenter spectra        
        shifty = bmap['MIRC QC WIN PHOTO SHIFTY'] - pyc + fyc;

        name = 'HIERARCH MIRC QC WIN PHOTO%i'%beam;
        hdr[name+' SHIFTY'] = (shifty, '[pix]');
        hdr[name+' STARTX'] = (pxc-pxw, '[pix] python-def');
        hdr[name+' STARTY'] = (pyc-ns, '[pix] python-def');


    # Figure
    fig,ax = plt.subplots(2,1);
    ax[0].imshow (np.mean (fringe,axis=(0,1)), interpolation='none');
    ax[1].imshow (np.swapaxes (np.mean (photos,axis=(1,2)), 0,1).reshape((ny,-1)), interpolation='none');
    fig.savefig (output+'_mean.png');

    # Figures
    fig,ax = plt.subplots();
    ax.plot (np.mean (fringe, axis=(0,1,3)), '--', label='fringes');
    ax.plot (np.mean (photos, axis=(1,2,4)).T);
    ax[1].set_ylabel ('adu/pix/fr');
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

def compute_snr (hdrs):
    pass;
