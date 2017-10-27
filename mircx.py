import numpy as np
import matplotlib.pyplot as plt

from astropy.stats import sigma_clipped_stats
from astropy.io import fits as pyfits
from astropy.modeling import models, fitting

from skimage.feature import register_translation

from scipy.fftpack import fft, ifft
from scipy.signal import medfilt;
from scipy.ndimage.interpolation import shift as subpix_shift

from . import log, files, headers

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

def crop_window (cube, hdr):
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

    # Remove background
    remove_background (cube, bkg[0]);

    # Check background subtraction in empty region
    check_empty_window (cube, hdr);

    # Compute the sum
    log.info ('Compute mean over ramps and frames');
    cmean = np.mean (cube, axis=(0,1));
    cmean_cut = cmean * 0.0;

    # Compute the flux in fringe window
    # (suposedly smoothed in x)
    fmap = medfilt (cmean, [1,11]);

    # Compute the flux in the photometric window
    # (suposedly sharp in x)
    pmap = medfilt (cmean - fmap, [3,1]);

    # Fit spatial of photometry with Gaussian
    px = np.mean (pmap, axis=0);
    x  = np.arange(len(px));
    gx_init = models.Gaussian1D (amplitude=np.max(px), mean=np.argmax(px), stddev=1.);
    gx = fitting.LevMarLSQFitter()(gx_init, x, px);
    idx = int(round(gx.mean - 3*gx.stddev));
    nx  = int(round(6*gx.stddev)+1);

    log.info ('Found limit photo in spatial direction: %i %i'%(idx,nx));
    
    # Get spectral limit of photometry
    p0y = np.mean (pmap[:,idx:idx+nx], axis=1);
    py  = medfilt (p0y, 5);
    py /= np.max (py);
    idy = np.argmax (py>0.25) - 2;
    ny  = (len(py) - np.argmax(py[::-1]>0.25)) + 3 - idy;

    log.info ('Found limit photo in spectral direction: %i %i'%(idy,ny));
    
    # Cut x-chan
    pcut = cmean[idy:idy+ny,idx:idx+nx];
    cmean_cut[idy:idy+ny,idx:idx+nx] = pcut;

    # Add QC parameters for window
    name = 'HIERARCH MIRC QC WIN PHOTO ';
    hdr.set (name+'STARTX',idx,'[pix] python-def');
    hdr.set (name+'NX',nx,'[pix]');
    hdr.set (name+'STARTY',idy,'[pix]');
    hdr.set (name+'NY',ny,'[pix]');

    # Add QC parameters to for optimal extraction
    hdr.set (name+'WIDTHX',gx.stddev.value,'[pix]');
    hdr.set (name+'CENTERX',gx.mean.value,'[pix] python-def');
    hdr.set (name+'WIDTHY',0.5*ny,'[pix]');
    hdr.set (name+'CENTERY',ny+0.5*ny,'[pix] python-def');

    # Figures
    fig,ax = plt.subplots(3,1);
    ax[0].imshow (pmap, interpolation='none');
    ax[1].plot (px, label='Data');
    ax[1].plot (x[idx:idx+nx],gx(x[idx:idx+nx]), label='Gaussian');
    ax[1].legend ();
    ax[2].imshow (pcut, interpolation='none');
    fig.savefig (output+'_pfit.png');

    # Get spectral limits of fringe
    f0y = np.mean (fmap,axis=1);
    fy  = medfilt (f0y, 5);
    fy /= np.max (fy);
    idy = np.argmax (fy>0.25) - 1;
    ny  = (len(fy) - np.argmax(fy[::-1]>0.25)) + 1 - idy;
    
    log.info ('Found limit fringe in spectral direction: %i %i'%(idy,ny));

    # Fit spatial of fringe with Gaussian
    fx = np.mean (fmap[idy:idy+ny,:], axis=0);
    x  = np.arange(len(fx));
    gx_init = models.Gaussian1D (amplitude=np.max(fx), mean=np.argmax(fx), stddev=50.);
    gx = fitting.LevMarLSQFitter()(gx_init, x, fx);
    idx = int(round(gx.mean - 3*gx.stddev));
    nx  = int(round(6*gx.stddev)+1);
        
    log.info ('Found limit fringe in spatial direction: %i %i'%(idx,nx));

    # Cut fringe
    fcut = cmean[idy:idy+ny,idx:idx+nx];
    cmean_cut[idy:idy+ny,idx:idx+nx] = fcut;

    # Add QC parameters for window
    name = 'HIERARCH MIRC QC WIN FRINGE ';
    hdr.set (name+'STARTX',idx,'[pix] python-def');
    hdr.set (name+'NX',nx,'[pix]');
    hdr.set (name+'STARTY',idy,'[pix]');
    hdr.set (name+'NY',ny,'[pix]');

    # Add QC parameters to for optimal extraction
    hdr.set (name+'WIDTHX',gx.stddev.value,'[pix]');
    hdr.set (name+'CENTERX',gx.mean.value,'[pix] python-def');
    hdr.set (name+'WIDTHY',0.5*ny,'[pix]');
    hdr.set (name+'CENTERY',ny+0.5*ny,'[pix] python-def');

    # Figures
    fig,ax = plt.subplots(3,1);
    ax[0].imshow (fmap, interpolation='none');
    ax[1].plot (fx, label='Data');
    ax[1].plot (x[idx:idx+nx],gx(x[idx:idx+nx]), label='Gaussian');
    ax[1].legend ();
    ax[2].imshow (fcut, interpolation='none');
    fig.savefig (output+'_ffit.png');

    # Shift between photo and fringes in spectral direction
    p0y /= np.max (p0y);
    f0y /= np.max (f0y);
    shifty = register_translation(p0y[:,None],f0y[:,None],upsample_factor=100)[0][0];
    
    name = 'HIERARCH MIRC QC WIN PHOTO SHIFTY';
    hdr.set (name,shifty,'[pix] shift of PHOTO versus FRINGE');

    # Figures
    fig,ax = plt.subplots(3,1);
    ax[0].imshow (cmean, interpolation='none');
    ax[1].imshow (cmean_cut, interpolation='none');
    ax[2].plot (f0y, label='fringe');
    ax[2].plot (p0y, label='photo');
    ax[2].plot (subpix_shift (p0y, -shifty), label='shifted photo');
    ax[2].legend ();
    fig.savefig (output+'_cut.png');

    # Create output HDU
    hdu1 = pyfits.PrimaryHDU (cmean_cut);
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
    hdrwin0 = bmaps[0]["* QC WIN FRINGE *"];
    
    # Crop fringe windows
    log.info ('Crop fringe region');
    fringe = crop_window (cube, hdrwin0);

    # Init photometries and map to zero
    nr,nf,ny,nx = fringe.shape;
    photos      = np.zeros ((6,nr,nf,ny,1));
    photo_maps  = np.zeros ((6,ny,1));
    fringe_maps = np.zeros ((6,ny,nx));

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

        # Shift value
        shifty = bmap['MIRC QC WIN PHOTO SHIFTY'] - \
                 bmap['MIRC QC WIN PHOTO STARTY'] + \
                 bmap['MIRC QC WIN FRINGE STARTY'];
        n = np.minimum (bmap['MIRC QC WIN PHOTO NY'],ny);

        # Get data
        bcube = pyfits.getdata (bmap['ORIGNAME'], 0);
        hdrwin = bmap["* QC WIN PHOTO *"];
        
        # Crop photometry and collapse
        # (FIXME, TODO: can be improved)
        photo = crop_window (cube, hdrwin);
        photo = np.sum (photo, axis=3)[:,:,:,None];

        # Shift spectra and set in cube
        photo = subpix_shift (photo, [0,0,-shifty,0]);
        photos[beam,:,:,0:n,:] = photo[:,:,0:n,:];

        # Same on map photometry
        photo_map = crop_window (bcube[None,None,:,:], hdrwin);
        photo_map = np.sum (photo_map, axis=3)[:,:,:,None];
        
        photo_map = subpix_shift (photo_map, [0,0,-shifty,0]);
        photo_maps[beam,0:n,:] = photo_map[0,0,0:n,:];

        # Same on map fringe
        fringe_map = crop_window (bcube[None,None,:,:], hdrwin0)[0,0,:,:];
        fringe_maps[beam,:,:] = fringe_map; 
        
    # Figures
    fig,ax = plt.subplots();
    ax.plot (np.mean (fringe, axis=(0,1,3)), '--', label='fringes');
    ax.plot (np.mean (photos, axis=(1,2,4)).T);
    ax.legend ();
    fig.savefig (output+'_spectra.png');
    
    # Create output HDU
    log.info ('Create file');
    hdu1 = pyfits.PrimaryHDU (fringe);
    hdu1.header = hdr;
    
    # Update header
    hdu1.header += hdrwin;
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
