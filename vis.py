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
from scipy.ndimage import gaussian_filter, uniform_filter, median_filter;
from scipy.optimize import least_squares, curve_fit;
from scipy.ndimage.morphology import binary_closing, binary_opening;
from scipy.ndimage.morphology import binary_dilation, binary_erosion;

from . import log, files, headers, setup, oifits, signal, plot, qc;
from .headers import HM, HMQ, HMP, HMW, rep_nan;

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

    return fringe_map, photo_map;

def compute_speccal (hdrs, output='output_speccal', filetype='SPEC_CAL',
                     ncoher=3, nfreq=4096, fitorder=2):
    '''
    Compute the SPEC_CAL from list of PREPROC
    '''
    elog = log.trace ('compute_speccal');

    # Check inputs
    headers.check_input (hdrs,  required=1);

    # Loop on files to compute their PSD
    for ih,h in enumerate(hdrs):
        f = h['ORIGNAME'];
        
        # Load file
        log.info ('Load PREPROC file %i over %i (%s)'%(ih+1,len(hdrs),f));
        hdr = pyfits.getheader (f);
        fringe = pyfits.getdata (f).astype(float);

        # Verbose on data size
        nr,nf,ny,nx = fringe.shape;
        log.info ('Data size: '+str(fringe.shape));

        # Define output
        if ih == 0:
            correl = np.zeros ((ny,nx*2-1));
            spectrum = np.zeros (ny);

        # Accumulate spectrum
        log.info ('Accumulate spectrum');
        tmp = medfilt (np.mean (fringe, axis=(0,1)), (1,11));
        spectrum += np.mean (tmp, axis=-1);
        
        # Remove the mean DC-shape
        log.info ('Compute the mean DC-shape');
        fringe_map = np.mean (fringe, axis=(0,1), keepdims=True);
        fringe_map /= np.sum (fringe_map);

        log.info ('Compute the mean DC-flux');
        fringe_dc = np.sum (fringe, axis=(2,3), keepdims=True);

        log.info ('Remove the DC');
        fringe -= fringe_map * fringe_dc;
        
        # Coherence integration
        log.info ('Coherent integration');
        fringe = gaussian_filter (fringe,(0,ncoher,0,0),mode='constant',truncate=2.0);

        # We accumulate the full-window auto-correlation
        # instead of the FFT**2 because this allows to oversampled
        # the integrated PSD after the incoherent integration.
        log.info ('Accumulate auto-correlation');
        data = fringe.reshape (nr*nf,ny,nx);
        for y in range(ny):
            for s in range(nr*nf):
                tmp = np.correlate (data[s,y,:],data[s,y,:],mode='full');
                correl[y,:] += tmp;

    # Compute valid channels based on spectrum
    spec = median_filter (spectrum, 3, mode='nearest');
    is_valid = spec > (0.25 * np.max (spec));

    # Morphology to avoid isolated rejected point
    # tmp = np.insert (np.insert (is_valid, 0, is_valid[0]), -1, is_valid[-1]);
    # is_valid = binary_closing (tmp, structure=[1,1,1])[1:-1];

    # Get center of spectrum
    fyc,fyw = signal.getwidth (spectrum);
    log.info ('Expect center of spectrum (lbd0) on %f'%fyc);

    # Build expected wavelength table
    lbd0,dlbd = setup.lbd0 (hdr);
    lbd = (np.arange (ny) - fyc) * dlbd + lbd0;

    # Model for the pic position at lbd0
    freq0 = np.abs (setup.base_freq (hdr));
    delta0 = np.min (freq0) / 6;
    
    # Frequencies in pix-1
    freq = 1.0 * np.arange (nfreq) / nfreq;

    # Used dataset is restricted to interesting range
    idmin = np.argmax (freq > 0.75*freq0.min());
    idmax = np.argmax (freq > 1.25*freq0.max());

    # Compute zero-padded PSD
    log.info ('PSD with huge zero-padding %i'%nfreq);
    psd = np.abs (fftpack.fft (correl, n=nfreq, axis=-1, overwrite_x=False));

    # Remove bias and normalise to the maximum in the interesting range
    psd -= np.median (psd[:,idmax:], axis=-1, keepdims=True);
    norm = np.max (psd[:,idmin:idmax], axis=1, keepdims=True);
    psd /= norm;

    # Correlate each wavelength channel with a template
    log.info ('Correlated PSD with model');
    res = [];
    for y in range (ny):
        s0 = lbd[y] / lbd0;
        args = (freq[idmin:idmax],freq0,delta0,psd[y,idmin:idmax]);
        res.append (least_squares (signal.psd_projection, s0, args=args, bounds=(0.8*s0,1.2*s0)));
        log.info ('Best merit 1-c=%.4f found at s/s0=%.4f'%(res[-1].fun[0],res[-1].x[0]/s0));

    # Get wavelengths
    yfit = 1.0 * np.arange (ny);
    lbdfit = np.array([r.x[0]*lbd0 for r in res]);

    log.info ('Compute QC');
    
    # Compute quality of projection
    projection = (1. - res[int(ny/2)].fun[0]) * norm[int(ny/2),0];
    log.info ('Projection quality = %g'%projection);

    # Typical difference with prediction
    delta = np.median (np.abs (lbd-lbdfit));
    log.info ('Median delta = %.3f um'%(delta*1e6));

    # Set quality to zero if clearly wrong fit
    if delta > 0.075e-6:
        log.warning ('Spectral calibration is probably faulty, set QUALITY to 0');
        projection = 0.0;

    # Set QC
    hdr[HMQ+'QUALITY'] = (projection, 'quality of data');
    hdr[HMQ+'DELTA MEDIAN'] = (delta, '[m] median difference');

    # Compute position on detector of lbd0
    lbd0 = 1.6e-6;
    s = np.argsort (lbdfit);
    try:     y0 = hdr[HMW+'FRINGE STARTY'] + np.interp (lbd0, lbdfit[s], yfit[s]);
    except:  y0 = -99.0
    hdr[HMQ+'YLBD0'] = (y0, 'ypos of %.3fum in cropped window'%(lbd0*1e6));
    log.info (HMQ+'YLBD0 = %e'%y0);

    # Compute a better version of the wavelength
    # by fitting a quadratic law, optional
    lbdlaw = lbdfit.copy ();

    if (fitorder > 0 and is_valid.sum() > 5):
        log.info ('Fit measure with order %i polynomial'%fitorder);
        hdr[HMQ+'LBDFIT_ORDER'] = (fitorder, 'order to fit the lbd solution (0 is no fit)');
        
        # Run a quadratic fit on valid values, except the
        # edges of the spectra.
        is_fit = binary_erosion (is_valid, structure=[1,1,1]);
        poly = np.polyfit (yfit[is_fit], lbdfit[is_fit], deg=fitorder);

        # Replace the fitted values by the polynomial
        lbdlaw[is_fit] = np.poly1d (poly)(yfit[is_fit]);
    else:
        log.info ('Keep raw measure (no fit of lbd solution)');

    log.info ('Figures');

    # Polynomial fit
    fig,ax = plt.subplots (2,sharex=True);
    fig.suptitle ('Polynomial fit\n(dark blue=poly, cyan=valid channels, light cyan = all channels)');
    ax[0].plot (yfit[is_fit],lbdfit[is_fit] * 1e6,'-', c='blue', alpha=1);
    ax[0].plot (yfit[is_valid],lbdlaw[is_valid] * 1e6,'o-', c='cyan', alpha=0.25);
    ax[0].plot (yfit,lbdlaw * 1e6,'o-', c='cyan', alpha=0.15);
    ax[1].plot (yfit[is_fit],(lbdlaw[is_fit]-lbdfit[is_fit]) * 1e9,'o-', c='cyan');
    ax[0].set_ylabel ('Polynomial fit [um]');
    ax[1].set_ylabel ('Residual [nm]');
    ax[1].set_xlabel ('Detector line (python-def)');
    files.write (fig,output+'_polynomial.png');
    
    # Spectrum
    fig,ax = plt.subplots ();
    fig.suptitle ('Mean Spectrum of all observations');
    ax.plot (spectrum,'o-', alpha=0.3);
    ax.plot (spectrum / is_valid,'o-');
    ax.set_ylabel ('Mean spectrum');
    ax.set_xlabel ('Detector line (python-def)');
    files.write (fig,output+'_spectrum.png');
    
    # Figures of PSD with model
    fig,axes = plt.subplots (ny,sharex=True);
    fig.suptitle ('Observed PSD (orange) and scaled template (blue)');
    for y in range (ny):
        ax = axes.flatten()[y];
        ax.plot (freq,signal.psd_projection (res[y].x[0], freq, freq0, delta0, None), c='blue');
        ax.plot (freq,psd[y,:], c='orange');
        ax.set_xlim (0,1.3*np.max(freq0));
        ax.set_ylim (0,1.1);
    files.write (fig,output+'_psdmodel.png');

    # Effective wavelength
    fig,ax = plt.subplots ();
    fig.suptitle ('Guess calib. (orange) and Fitted calib, (blue)');
    ax.plot (yfit,lbdlaw * 1e6,'o-', c='blue', alpha=0.5);
    ax.plot (yfit[is_fit],lbdfit[is_fit] * 1e6,'o-', c='blue', alpha=0.5);
    ax.plot (yfit,lbd * 1e6,'o-', c='orange', alpha=0.25);
    ax.set_ylabel ('lbd (um)');
    ax.set_xlabel ('Detector line (python-def)');
    ax.set_ylim (1.45,1.8);
    files.write (fig,output+'_lbd.png');

    # PSD
    fig,ax = plt.subplots (2,1);
    fig.suptitle (headers.summary (hdr));
    ax[0].imshow (correl,aspect='auto');
    ax[1].plot (psd[:,0:int(nfreq/2)].T);
    files.write (fig,output+'_psd.png');

    # File
    log.info ('Create file');

    # First HDU
    hdu0 = pyfits.PrimaryHDU (lbdlaw);
    hdu0.header = hdr;
    hdu0.header['FILETYPE'] = filetype;
    hdu0.header['BUNIT'] = 'm';

    # Save input files
    for h in hdrs:
        npp = len (hdr['*MIRC PRO PREPROC*']);
        hdr['HIERARCH MIRC PRO PREPROC%i'%(npp+1,)] = h['ORIGNAME'][-30:];

    # Other HDU
    hdu1 = pyfits.ImageHDU (lbdfit);
    hdu1.header['EXTNAME'] = ('LBDFIT','RAW wavelength');
    hdu1.header['BUNIT'] = 'm';
    
    hdu2 = pyfits.ImageHDU (spectrum);
    hdu2.header['EXTNAME'] = ('SPECTRUM','mean spectrum');
    hdu2.header['BUNIT'] = 'adu';
    
    hdu3 = pyfits.ImageHDU (is_valid.astype(int));
    hdu3.header['EXTNAME'] = ('IS_VALID','valid channels');
    hdu3.header['BUNIT'] = 'bool';

    # Write file
    hdulist = pyfits.HDUList ([hdu0,hdu1,hdu2,hdu3]);
    files.write (hdulist, output+'.fits');
    
    plt.close ("all");
    return hdulist;
    
def compute_rts (hdrs, profiles, kappas, speccal,
                 output='output_rts', filetype='RTS',
                 psmooth=0,
                 save_all_freqs=False):
    '''
    Compute the RTS
    '''
    elog = log.trace ('compute_rts');

    # Check inputs
    save_all_freqs = headers.clean_option (save_all_freqs);
    headers.check_input (hdrs,  required=1);
    headers.check_input (profiles, required=1, maximum=6);
    headers.check_input (kappas, required=1, maximum=6);
    headers.check_input (speccal, required=1, maximum=1);

    # Load the wavelength table
    f = speccal[0]['ORIGNAME'];
    log.info ('Load SPEC_CAL file %s'%f);
    lbd = pyfits.getdata (f);

    # Get valid spectral channels
    is_valid = (pyfits.getdata (f,'IS_VALID') == 1);
    lbd = lbd[is_valid];
    
    # Load DATA
    f = hdrs[0]['ORIGNAME'];
    log.info ('Load PREPROC file %s'%f);
    hdr = pyfits.getheader (f);
    fringe = pyfits.getdata (f).astype(float);
    photo  = pyfits.getdata (f, 'PHOTOMETRY_PREPROC').astype(float);
    mjd    = pyfits.getdata (f, 'MJD');

    # Load other files if any
    for h in hdrs[1:]:
        f = h['ORIGNAME'];
        log.info ('Load PREPROC file %s'%f);
        fringe = np.append (fringe, pyfits.getdata (f).astype(float), axis=0);
        photo  = np.append (photo, pyfits.getdata (f, 'PHOTOMETRY_PREPROC').astype(float), axis=1);
        mjd    = np.append (mjd, pyfits.getdata (f, 'MJD'), axis=0);

    # Dimensions
    nr,nf,ny,nx = fringe.shape
    log.info ('fringe.shape = %s'%str(fringe.shape));
    log.info ('mean(fringe) = %f adu/pix/frame'%np.mean(fringe,axis=(0,1,2,3)));

    # Saturation checks
    fsat  = 1.0 * np.sum (np.mean (np.sum (fringe,axis=1),axis=0)>40000) / (ny*nx);
    log.info (HMQ+'FRAC_SAT = %.3f'%rep_nan (fsat));
    hdr[HMQ+'FRAC_SAT'] = (rep_nan (fsat), 'fraction of saturated pixel');

    # Get fringe and photo maps
    log.info ('Read data for photometric and fringe profiles');
    fringe_map, photo_map = extract_maps (hdr, profiles);
    
    # Define profile for optimal extraction of photometry
    # The same profile is used for all spectral channels
    log.info ('Compute profile');
    profile = np.mean (photo_map, axis=3, keepdims=True);

    # Remove edge of the profile
    profile /= np.sum (profile,axis=-1, keepdims=True) + 1e-20;
    flag = profile > 0.25;
    flag[:,:,:,:,1:]  += (profile[:,:,:,:,:-1] > 0.25);
    flag[:,:,:,:,:-1] += (profile[:,:,:,:,1:] > 0.25);
    profile[~flag] = 0.0;

    # Profile is normalised to be flux-conservative
    # Maybe not good to have profile when photon-counting
    profile *= np.sum (profile,axis=-1, keepdims=True) / \
               (np.sum (profile**2,axis=-1, keepdims=True)+1e-20);
    
    # Plot the profile and compare to data
    fig,axes = plt.subplots (3,2);
    fig.suptitle (headers.summary (hdr));
    for b in range(6):
        ax = axes.flatten()[b];
        val = np.mean (profile[b,:,:,:,:],axis=(0,1,2));
        ax.plot (val / (np.mean (val)+1e-20), label='profile');
        val = np.mean (photo[b,:,:,:,:],axis=(0,1,2));
        ax.plot (val / (np.mean (val)+1e-20), label='xchan');
    axes[0,0].legend();
    files.write (fig,output+'_profile.png');

    # Optimal extraction of photometry with profile
    log.info ('Extract photometry with profile');
    photo = np.sum (photo * profile, axis=-1);

    # Shift between photo and fringes in spectral direction
    log.info ('Compute spectral offsets in beam_map');
    shifty = np.zeros (6);
    upper = np.sum (medfilt (fringe_map,[1,1,1,1,11]), axis=(1,2,4));
    lower = np.sum (medfilt (photo_map,[1,1,1,1,1]) * profile, axis=(1,2,4));
    for b in range (6):
        shifty[b] = register_translation (lower[b,:,None],upper[b,:,None],
                                              upsample_factor=100)[0][0];

    # Re-align photometry (all with the same)
    log.info ('Register photometry to fringe');
    for b in range(6):
        photo[b,:,:,:] = subpix_shift (photo[b,:,:,:], [0,0,-shifty[b]]);

    # Keep only valid channels
    log.info ('Keep only valid channels');
    photo  = photo[:,:,:,is_valid];
    fringe = fringe[:,:,is_valid,:];
    fringe_map = fringe_map[:,:,:,is_valid,:];
    photo_map  = photo_map[:,:,:,is_valid,:];

    # Plot photometry versus time
    log.info ('Plot photometry');
    fig,axes = plt.subplots (3,2,sharex=True);
    fig.suptitle ('Xchan flux (adu) \n' + headers.summary (hdr));
    plot.compact (axes);
    for b in range (6):
        data = np.mean (photo[b,:,:,:], axis=(1,2));
        ax = axes.flatten()[b];
        ax.plot (data);
        ax.set_ylim (np.minimum (np.min (data), 0.0));
    ax.set_xlabel ('Ramp #');
    files.write (fig,output+'_photo.png');

    # Plot ramp of flux in fringe
    log.info ('Plot fringe ramp');
    fig,ax = plt.subplots ();
    fig.suptitle (headers.summary (hdr));
    ax.plot (np.mean (fringe, axis=(0,3)));
    ax.set_ylabel ('Mean fringe flux (adu)');
    ax.set_xlabel ('Frame in ramp');
    files.write (fig,output+'_fringeramp.png');

    # Get data for kappa_matrix
    log.info ('Read data for kappa matrix');
    fringe_kappa, photo_kappa = extract_maps (hdr, kappas);
    
    # Build kappa from input data.
    # kappa(nb,nr,nf,ny)
    log.info ('Build kappa-matrix with profile, filtering and registration, and keep valid');
    upper = np.sum (medfilt (fringe_kappa,[1,1,1,1,11]), axis=-1);
    lower = np.sum (medfilt (photo_kappa,[1,1,1,1,1]) * profile, axis=-1);
    for b in range(6):
        lower[b,:,:,:] = subpix_shift (lower[b,:,:,:], [0,0,-shifty[b]]);

    upper = upper[:,:,:,is_valid];
    lower = lower[:,:,:,is_valid];

    kappa = upper / (lower + 1e-20);

    # Set invalid kappas to zero
    kappa[kappa > 1e3] = 0.0;
    kappa[kappa < 0.] = 0.0;
        
    # Kappa-matrix as spectrum
    log.info ('Plot kappa');
    fig,axes = plt.subplots (3,2);
    fig.suptitle (headers.summary (hdr));
    for b in range (6):
        ax = axes.flatten()[b];
        val = np.mean (upper, axis=(1,2));
        val /= np.max (medfilt (val,(1,3)), axis=1, keepdims=True) + 1e-20;
        ax.plot (lbd*1e6,val[b,:],'--', label='upper');
        val = np.mean (lower, axis=(1,2));
        val /= np.max (medfilt (val,(1,3)), axis=1, keepdims=True) + 1e-20;
        ax.plot (lbd*1e6,val[b,:], label='lower');
        val = np.mean (kappa, axis=(1,2));
        val /= np.max (medfilt (val,(1,3)), axis=1, keepdims=True) + 1e-20;
        ax.plot (lbd*1e6,val[b,:], label='kappa');
        ax.set_ylim ((0.1,1.5));
        ax.set_ylabel ('normalized');
    axes[0,0].legend();
    files.write (fig,output+'_kappa.png');

    # Kappa-matrix
    fig,ax = plt.subplots (1);
    fig.suptitle (headers.summary (hdr));
    ax.imshow (np.mean (kappa,axis=(1,2)));
    files.write (fig,output+'_kappaimg.png');

    # kappa is defined so that photok is the
    # total number of adu in the fringe
    log.info ('Compute photok');
    photok = photo * kappa;

    # QC about the fringe dc
    log.info ('Compute fringedc / photok');
    photok_sum = np.sum (photok,axis=(0,3));
    fringe_sum = np.sum (fringe,axis=(2,3));
    dc_ratio = np.sum (fringe_sum) / np.sum (photok_sum);
    hdr[HMQ+'DC MEAN'] = (rep_nan (dc_ratio), 'fringe/photo');

    # Scale the photometry to the fringe DC. FIXME: this is done
    # for all wavelength together, not per-wavelength.
    log.info ('Scale the photometries by %.4f'%dc_ratio);
    photok *= dc_ratio;

    # We save this estimation of the photometry
    # for the further visibility normalisation
    log.info ('Save photometry for normalisation');
    photok0 = photok.copy();

    # Smooth photometry
    if psmooth > 0:
        log.info ('Smooth photometry by sigma=%i frames'%psmooth);
        photok = gaussian_filter (photok,(0,0,psmooth,0),mode='constant');

    # Warning because of saturation
    log.info ('Deal with saturation in the filtering');
    isok  = 1.0 * (np.sum (fringe,axis=(2,3)) != 0);
    trans = gaussian_filter (isok,(0,psmooth),mode='constant');
    photok *= isok[None,:,:,None] / np.maximum (trans[None,:,:,None],1e-10);

    # Temporal / Spectral averaging of photometry
    # to be discussed (note that this is only for the
    # continuum removal, not for normalisation)
    log.info ('Temporal / Spectral averaging of photometry');
    spectra  = np.mean (photok, axis=(1,2), keepdims=True);
    spectra /= np.sum (spectra, axis=3, keepdims=True) + 1e-20;
    injection = np.sum (photok, axis=3, keepdims=True);
    photok = spectra*injection;
     
    # Compute flux in fringes. fringe_map is normalised
    log.info ('Compute dc in fringes');
    # fringe_map  = medfilt (fringe_map, [1,1,1,1,11]);
    # fringe_map  = median_filter (fringe_map, size=[1,1,1,1,11],mode='nearest');
    fringe_map /= np.sum (fringe_map, axis=-1, keepdims=True) + 1e-20;
    cont = np.einsum ('Brfy,Brfyx->rfyx', photok, fringe_map);

    # Check dc
    log.info ('Figure of DC in fringes');
    fig,ax = plt.subplots ();
    fig.suptitle (headers.summary (hdr));
    cont_mean = np.mean (cont,axis=(2,3));
    fringe_mean = np.mean (fringe,axis=(2,3));
    ax.hist2d (cont_mean.flatten(), fringe_mean.flatten(),
               bins=40, norm=mcolors.LogNorm());
    xvalues = np.array ([np.min (cont_mean), np.max (cont_mean)]);
    ax.plot (xvalues,xvalues,'g-',label='y = x');
    ax.set_ylabel('fringe dc');
    ax.set_xlabel('sum of photo * kappa * map');
    ax.legend (loc=2);
    files.write (fig,output+'_dccorr.png');

    # Save integrated spectra and profile before
    # subtracting the continuum
    cont_img   = np.mean (cont, axis=(0,1));
    fringe_img = np.mean (fringe, axis=(0,1));
    fringe_spectra = np.mean (fringe_img, axis=1);
    fringe_sum = fringe.sum (axis=3);

    # Remove the DC predicted from xchan
    log.info ('Subtract dc with profiles predicted from xchan');
    fringe -= cont;
    del cont;

    # Remove the residual DC with a mean profile
    log.info ('Subtract residual dc with mean profile');
    fringe_meanmap = fringe_map.mean (axis=0);
    fringe_meanmap /= np.sum (fringe_meanmap, axis=-1, keepdims=True) + 1e-20;
    dcres = fringe.sum (axis=-1, keepdims=True);
    fringe -= dcres * fringe_meanmap;
    del dcres, fringe_meanmap;

    # Check residual
    log.info ('Figure of DC residual');
    fig,axes = plt.subplots (2, 1, sharex=True);
    fig.suptitle (headers.summary (hdr));
    axes[0].plot (fringe_img[int(ny/2),:], label='fringe');
    axes[0].plot (cont_img[int(ny/2),:], label='cont');
    axes[0].legend();
    axes[1].plot (np.mean (fringe[int(ny/2),:],axis=(0,1)), label='res');
    axes[1].set_xlabel('x (spatial direction)');
    axes[1].legend();
    files.write (fig,output+'_dcres.png');

    # Model (x,f)
    log.info ('Model of data');
    nfq = int(nx/2);
    f = 1. * np.arange(1,nfq+1);
    x = 1. * np.arange(nx) / nx;
    x -= np.mean (x);

    # fres is the spatial frequency at the
    # reference wavelength lbd0
    lbd0,dlbd = setup.lbd0 (hdr);
    freqs = setup.base_freq (hdr);
    
    # Scale to ensure the frequencies fall
    # into integer pixels (max freq is 40 or 72)
    ifreq_max = setup.ifreq_max (hdr);
    scale0 = 1.0 * ifreq_max / np.abs (freqs * nx).max();

    # Compute the expected scaling
    log.info ("ifreqs as float");
    log.info (freqs * scale0 * nx);

    # Compute the expected scaling
    log.info ("ifreqs as integer");
    ifreqs = np.round (freqs * scale0 * nx).astype(int);
    log.info (ifreqs);

    # Dimensions
    nb = len(ifreqs);
    nr,nf,ny,nx = fringe.shape

    # Compute DFT. The amplitude of the complex number corresponds
    # to the sum of the amplitude sum(A) of the oscillation A.cos(x)
    # in the fringe enveloppe.
    model = np.zeros ((nx,nfq*2+1));
    cf = 0.j + np.zeros ((nr*nf,ny,nfq+1));
    for y in np.arange(ny):
        log.info ('Project channel %i (centered)'%y);
        amp = np.ones (nx);
        model[:,0] = amp;
        scale = lbd0 / lbd[y] / scale0;
        model[:,1:nfq+1] = amp[:,None] * 2. * np.cos (2.*np.pi * x[:,None] * f[None,:] * scale);
        model[:, nfq+1:] = amp[:,None] * 2. * np.sin (2.*np.pi * x[:,None] * f[None,:] * scale);
        cfc = np.tensordot (model,fringe[:,:,y,:],axes=([0],[2])).reshape((nx,nr*nf)).T;
        cf[:,y,0]  = cfc[:,0];
        cf[:,y,1:] = cfc[:,1:nfq+1] - 1.j * cfc[:,nfq+1:];
    cf.shape = (nr,nf,ny,nfq+1);

    # Free input fringes images
    if (save_all_freqs == False):
        log.info ('Free fringe');
        del fringe;        

    # DFT at fringe frequencies
    log.info ('Extract fringe frequency');
    base_dft  = cf[:,:,:,np.abs(ifreqs)];

    # Take complex conjugated for negative frequencies
    idx = ifreqs < 0.0;
    base_dft[:,:,:,idx] = np.conj(base_dft[:,:,:,idx]);

    # DFT at bias frequencies
    ibias = np.abs (ifreqs).max() + 4 + np.arange (15);
    bias_dft  = cf[:,:,:,ibias];

    # Compute unbiased PSD for plots (without coherent average
    # thus the bias is larger than in the base data).
    cf_upsd  = np.abs(cf[:,:,:,0:int(nx/2)])**2;
    cf_upsd -= np.mean (cf_upsd[:,:,:,ibias],axis=-1,keepdims=True);

    # Free DFT images
    if (save_all_freqs == False):
        log.info ('Free cf');
        del cf;

    log.info ('Compute crude vis2 with various coherent');
        
    # Compute crude normalisation for vis2
    bbeam = setup.base_beam ();
    norm = np.mean (photok0[:,:,int(ny/2),:], axis=(0,1));
    norm = 4. * norm[bbeam[:,0]] * norm[bbeam[:,1]];
    
    # Compute the coherent flux for various integration
    # for plots, to track back vibrations
    nc = np.array([2, 5, 10, 15, 25, 50]);
    vis2 = np.zeros ((nb, len(nc)));
    for i,n in enumerate(nc):
        # Coherent integration, we process only the central channel
        base_s  = signal.uniform_filter_cpx (base_dft[:,:,int(ny/2),:],(0,n,0),mode='constant');
        bias_s  = signal.uniform_filter_cpx (bias_dft[:,:,int(ny/2),:],(0,n,0),mode='constant');
        # Unbiased visibility, based on cross-spectrum with 1-shift
        b2    = np.mean (np.mean (np.real (bias_s[:,1:,:] * np.conj(bias_s[:,0:-1,:])), axis=(0,1,2)));
        power = np.mean (np.real (base_s[:,1:,:] * np.conj(base_s[:,0:-1,:])), axis=(0,1)) - b2;
        vis2[:,i] = power / norm;
        
    log.info ('Compute QC DECOHER_TAU0');
    
    # Time and model
    fps = hdr['HIERARCH MIRC FRAME_RATE'];
    time  = 1.0 * nc / fps * 1e3;
    timem = np.linspace (1e-6, time.max(), 1000);
    vis2m = np.zeros ((nb,len(timem)));
    vis2h  = np.zeros (nb)

    # QC parameters
    for b,name in enumerate (setup.base_name ()):
        # Time where we lose half the coherence
        vis2h[b] = np.interp (0.5 * vis2[b,0], vis2[b,::-1], time[::-1]);
        hdr[HMQ+'DECOHER'+name+'_HALF'] = (vis2h[b], '[ms] time for half V2');
        # Tau0 from model assuming 5/3
        try:
            popt, pcov = curve_fit (signal.decoherence, time, vis2[b,:], p0=[vis2[b,0], 0.01]);
            vis2m[b,:] = signal.decoherence (timem, popt[0], popt[1]);
            hdr[HMQ+'DECOHER'+name+'_TAU0'] = (popt[1], '[ms] coherence time with 5/3');
        except:
            log.warning ("Fail to fit on baseline %i, continue anyway"%b);
        
    # Figures
    log.info ('Figures');

    # Plot the decoherence
    fig,axes = plt.subplots (5,3, sharex=True);
    fig.suptitle (headers.summary (hdr));
    plot.base_name (axes);
    plot.compact (axes);
    for i,ax in enumerate (axes.flatten()):
        ax.plot (time, vis2[i,:],'o-');
        ax.plot (timem,vis2m[i,:],'-',alpha=0.5);
        plot.scale (ax, vis2h[i], h=0.2, fmt="%.1f ms");
        ax.set_ylim (0);
    axes.flatten()[13].set_xlabel ('Coherent integration [ms]    (FPS=%f)'%fps);
    files.write (fig,output+'_vis2coher.png');
    
    # Integrated spectra
    fig,ax = plt.subplots (2,1);
    fig.suptitle (headers.summary (hdr));
    val = fringe_spectra;
    val /= np.max (medfilt (val,3), keepdims=True) + 1e-20;
    ax[0].plot (lbd*1e6,val,':', lw=5, label='fringes and photo');
    val = np.mean (photo, axis=(1,2));
    val /= np.max (medfilt (val,(1,3)), axis=1, keepdims=True) + 1e-20;
    ax[0].plot (lbd*1e6,val.T,alpha=0.5);
    ax[0].legend(loc=4);
    ax[0].set_ylabel ('normalized');
    
    val = fringe_spectra;
    val /= np.max (medfilt (val,3), keepdims=True) + 1e-20;
    ax[1].plot (lbd*1e6,val,':', lw=5, label='fringes and photo * kappa');
    val = np.mean (photok0,axis=(0,1,2));
    val /= np.max (medfilt (val,3), keepdims=True) + 1e-20;
    ax[1].plot (lbd*1e6,val,'--', lw=5);
    val = np.mean (photok0, axis=(1,2));
    val /= np.max (medfilt (val,(1,3)), axis=1, keepdims=True) + 1e-20;
    ax[1].plot (lbd*1e6,val.T,alpha=0.5);
    ax[1].legend (loc=4);
    ax[1].set_ylabel ('normalized');
    ax[1].set_xlabel ('lbd (um)');
    files.write (fig,output+'_spectra.png');
    
    # Power densities
    fig,ax = plt.subplots (2,1);
    fig.suptitle (headers.summary (hdr));
    ax[0].imshow (np.mean (cf_upsd, axis=(0,1)),aspect='auto');
    for f in ifreqs: ax[0].axvline (np.abs(f), color='k', linestyle='--', alpha=0.5);
    ax[0].axvline (np.abs(ibias[0]), color='r', linestyle='--', alpha=0.3);
    ax[0].axvline (np.abs(ibias[-1]), color='r', linestyle='--', alpha=0.3);
    ax[1].plot (np.mean (cf_upsd, axis=(0,1))[int(ny/2),:]);
    ax[1].set_xlim (0,cf_upsd.shape[-1]);
    files.write (fig,output+'_psd.png');

    # File
    log.info ('Create file');

    # First HDU
    hdu = pyfits.PrimaryHDU ([]);
    hdu.header = hdr;
    hdu.header['FILETYPE'] = filetype;
    hdu.header[HMP+'PREPROC'] = os.path.basename (hdrs[0]['ORIGNAME'])[-30:];

    # Set the input calibration file
    for pro in profiles:
        name = pro['FILETYPE'].split('_')[0]+'_PROFILE';
        hdu.header[HMP+name] = os.path.basename (pro['ORIGNAME'])[-30:];

    # Set the input calibration file
    for kap in kappas:
        name = kap['FILETYPE'].split('_')[0]+'_KAPPA';
        hdu.header[HMP+name] = os.path.basename (kap['ORIGNAME'])[-30:];

    # Start a list
    hdus = [hdu];

    # Set DFT of fringes, bias, photometry and lbd
    hdu = pyfits.ImageHDU (base_dft.real.astype('float32'));
    hdu.header['EXTNAME'] = ('BASE_DFT_REAL','total flux in the fringe envelope');
    hdu.header['BUNIT'] = 'adu';
    hdu.header['SHAPE'] = '(nr,nf,ny,nb)';
    hdus.append (hdu);
    
    hdu = pyfits.ImageHDU (base_dft.imag.astype('float32'));
    hdu.header['EXTNAME'] = ('BASE_DFT_IMAG','total flux in the fringe envelope');
    hdu.header['BUNIT'] = 'adu'
    hdu.header['SHAPE'] = '(nr,nf,ny,nb)';
    hdus.append (hdu);
    
    hdu = pyfits.ImageHDU (bias_dft.real.astype('float32'));
    hdu.header['EXTNAME'] = ('BIAS_DFT_REAL','total flux in the fringe envelope');
    hdu.header['BUNIT'] = 'adu';
    hdu.header['SHAPE'] = '(nr,nf,ny,nbias)';
    hdus.append (hdu);
    
    hdu = pyfits.ImageHDU (bias_dft.imag.astype('float32'));
    hdu.header['EXTNAME'] = ('BIAS_DFT_IMAG','total flux in the fringe envelope');
    hdu.header['BUNIT'] = 'adu';
    hdu.header['SHAPE'] = '(nr,nf,ny,nbias)';
    hdus.append (hdu);
    
    hdu = pyfits.ImageHDU (np.transpose (photok0,axes=(1,2,3,0)).astype('float32'));
    hdu.header['EXTNAME'] = ('PHOTOMETRY','total flux in the fringe envelope');
    hdu.header['BUNIT'] = 'adu'
    hdu.header['SHAPE'] = '(nr,nf,ny,nt)';
    hdus.append (hdu);
    
    hdu = pyfits.ImageHDU (lbd);
    hdu.header['EXTNAME'] = ('WAVELENGTH','effective wavelength');
    hdu.header['BUNIT'] = 'm';
    hdu.header['SHAPE'] = '(ny)';
    hdus.append (hdu);

    hdu = pyfits.ImageHDU (np.transpose (kappa,axes=(1,2,3,0)));
    hdu.header['EXTNAME'] = ('KAPPA','ratio total_fringe/total_photo');
    hdu.header['SHAPE'] = '(nr,nf,ny,nt)';
    hdus.append (hdu);

    hdu = pyfits.ImageHDU (mjd);
    hdu.header['EXTNAME'] = ('MJD','time of each frame');
    hdu.header['BUNIT'] = 'day';
    hdu.header['SHAPE'] = '(nr,nf)';
    hdus.append (hdu);
    
    hdu = pyfits.ImageHDU (ifreqs);
    hdu.header['EXTNAME'] = ('IFREQ','spatial frequencies of fringes');
    hdu.header['BUNIT'] = 'pix';
    hdu.header['SHAPE'] = '(nf)';
    hdus.append (hdu);
    
    hdu = pyfits.ImageHDU (ibias);
    hdu.header['EXTNAME'] = ('IBIAS','spatial frequencies for bias');
    hdu.header['BUNIT'] = 'pix';
    hdu.header['SHAPE'] = '(nf)';
    hdus.append (hdu);

    if (save_all_freqs):
        log.info ("Save all frequencies for John's test");
        hdu = pyfits.ImageHDU (cf.real.astype('float32'));
        hdu.header['EXTNAME'] = ('ALL_DFT_REAL','total flux in the fringe envelope');
        hdu.header['BUNIT'] = 'adu';
        hdu.header['SHAPE'] = '(nr,nf,ny,nb)';
        hdus.append (hdu);
    
        hdu = pyfits.ImageHDU (cf.imag.astype('float32'));
        hdu.header['EXTNAME'] = ('ALL_DFT_IMAG','total flux in the fringe envelope');
        hdu.header['BUNIT'] = 'adu'
        hdu.header['SHAPE'] = '(nr,nf,ny,nb)';
        hdus.append (hdu);

        hdu = pyfits.ImageHDU (fringe_sum.astype('float32'));
        hdu.header['EXTNAME'] = ('FRINGE_SUM','total flux in the fringe');
        hdu.header['BUNIT'] = 'adu'
        hdu.header['SHAPE'] = '(nr,nf,ny)';
        hdus.append (hdu);

    # Write file
    hdulist = pyfits.HDUList (hdus);
    files.write (hdulist, output+'.fits');
                
    plt.close("all");
    return hdulist;

def compute_vis (hdrs, coeff, output='output_oifits', filetype='OIFITS',
                 ncoher=3, nincoher=5,
                 snr_threshold=3.0, flux_threshold=20.0,
                 avgphot=True, ncs=2, nbs=2, gdAttenuation=True,
                 vis_reference='self'):
    '''
    Compute the OIFITS from the RTS
    '''
    elog = log.trace ('compute_vis');

    # Check inputs
    headers.check_input (hdrs, required=1);
    headers.check_input (coeff, required=0, maximum=1);

    # Get data
    f = hdrs[0]['ORIGNAME'];
    log.info ('Load RTS file %s'%f);
    hdr = pyfits.getheader (f);
    base_dft  = pyfits.getdata (f, 'BASE_DFT_IMAG').astype(float) * 1.j;
    base_dft += pyfits.getdata (f, 'BASE_DFT_REAL').astype(float);
    bias_dft  = pyfits.getdata (f, 'BIAS_DFT_IMAG').astype(float) * 1.j;
    bias_dft += pyfits.getdata (f, 'BIAS_DFT_REAL').astype(float);
    photo     = pyfits.getdata (f, 'PHOTOMETRY').astype(float);
    mjd       = pyfits.getdata (f, 'MJD');
    lbd       = pyfits.getdata (f, 'WAVELENGTH').astype(float);

    # Load other files if any
    for h in hdrs[1:]:
        f = h['ORIGNAME'];
        log.info ('Load RTS file %s'%f);
        base_dft = np.append (base_dft, \
                   pyfits.getdata (f, 'BASE_DFT_IMAG').astype(float) * 1.j + \
                   pyfits.getdata (f, 'BASE_DFT_REAL').astype(float), axis=0);
        bias_dft = np.append (bias_dft, \
                   pyfits.getdata (f, 'BIAS_DFT_IMAG').astype(float) * 1.j + \
                   pyfits.getdata (f, 'BIAS_DFT_REAL').astype(float), axis=0);
        photo    = np.append (photo, \
                   pyfits.getdata (f, 'PHOTOMETRY').astype(float), axis=0);
        mjd      = np.append (mjd, pyfits.getdata (f, 'MJD'), axis=0);
                   
    # Dimensions
    nr,nf,ny,nb = base_dft.shape;
    log.info ('Data size: '+str(base_dft.shape));
        
    # Check parameters consistency
    if ncs + ncoher + 2 > nf:
        raise ValueError ('ncs+ncoher+2 should be less than nf (nf=%i)'%nf);

    # Compute lbd0 and dlbd    
    if hdr['CONF_NA'] == 'H_PRISM20' :
        lbd0 = np.mean (lbd);
        dlbd = np.mean (np.diff (lbd));
        
    elif (hdr['CONF_NA'] == 'H_PRISM40') or  (hdr['CONF_NA'] == 'H_PRISM50'):
        lbd0 = np.mean (lbd[2:-2]);
        dlbd = np.mean (np.diff (lbd[2:-2]));
    
    else :
        lbd0 = np.mean (lbd[4:-4]);
        dlbd = np.mean (np.diff (lbd[4:-4]));
    
    # Verbose spectral resolution
    log.info ('lbd0=%.3e, dlbd=%.3e um (R=%.1f)'%(lbd0*1e6,dlbd*1e6,np.abs(lbd0/dlbd)));

    # Coherence length
    coherence_length = lbd0**2 / np.abs (dlbd);

    # Spectral channel for QC (not exactly center of band
    # because this one is not working for internal light)
    y0 = int(ny/2) - 2;

    # Check if nan in photometry
    nnan = np.sum (np.isnan (photo));
    if nnan > 0: log.warning ('%i NaNs in photometry'%nnan);
        
    # Check if nan in fringe
    nnan = np.sum (np.isnan (base_dft));
    if nnan > 0: log.warning ('%i NaNs in fringe'%nnan);

    log.info ('Mean photometries: %e'%np.mean (photo));

    # Do spectro-temporal averaging of photometry
    if avgphot is True:
        log.info ('Do spectro-temporal averaging of photometry');
        hdr[HMP+'AVGPHOT'] = (True,'spectro-temporal averaging of photometry');
        
        for b in range (6):
            # Compute the matrix with mean, slope
            spectrum = np.mean (photo[:,:,:,b], axis=(0,1));
            M = np.array ([spectrum, spectrum * (lbd - lbd0)*1e6]);
            # Invert system 
            ms = np.einsum ('rfy,ys->rfs', photo[:,:,:,b], np.linalg.pinv (M));
            photo[:,:,:,b] = np.einsum ('rfs,sy->rfy',ms,M);
            
        log.info ('Mean photometries: %e'%np.mean (photo));
    else:
        log.info ('No spectro-temporal averaging of photometry');
        hdr[HMP+'AVGPHOT'] = (False,'spectro-temporal averaging of photometry');
        
    # Do coherent integration
    log.info ('Coherent integration over %i frames'%ncoher);
    base_dft = signal.uniform_filter_cpx (base_dft,(0,ncoher,0,0),mode='constant');
    bias_dft = signal.uniform_filter_cpx (bias_dft,(0,ncoher,0,0),mode='constant');

    # Smooth photometry over the same amount (FIXME: be be discussed)
    log.info ('Smoothing of photometry over %i frames'%ncoher);
    photo = uniform_filter (photo,(0,ncoher,0,0),mode='constant');

    #  Remove edges
    log.info ('Remove edge of coherence integration for each ramp');
    edge = int(ncoher/2);
    base_dft = base_dft[:,edge:nf-edge,:,:];
    bias_dft = bias_dft[:,edge:nf-edge,:,:];
    photo    = photo[:,edge:nf-edge,:,:];

    # New size
    nr,nf,ny,nb = base_dft.shape;

    # Add QC
    qc.flux (hdr, y0, photo);

    nscan = 64;
    log.info ('Compute 2d FFT (nscan=%i)'%nscan);

    # Compute FFT over the lbd direction, thus OPD-scan
    base_scan  = np.fft.fftshift (np.fft.fft (base_dft, n=nscan, axis=2), axes=2);
    bias_scan  = np.fft.fftshift (np.fft.fft (bias_dft, n=nscan, axis=2), axes=2);

    # Compute power in the scan, average the scan over the ramp.
    # Therefore the coherent integration is the ramp, hardcoded.
    if ncs > 0:
        log.info ('Compute OPD-scan Power with offset of %i frames'%ncs);
        base_scan = np.real (base_scan[:,ncs:,:,:] * np.conj(base_scan[:,0:-ncs,:,:]));
        bias_scan = np.real (bias_scan[:,ncs:,:,:] * np.conj(bias_scan[:,0:-ncs,:,:]));
        base_scan = np.mean (base_scan, axis=1, keepdims=True);
        bias_scan = np.mean (bias_scan, axis=1, keepdims=True);
    else:
        log.info ('Compute OPD-scan Power without offset');
        base_scan = np.mean (np.abs(base_scan)**2,axis=1, keepdims=True);
        bias_scan = np.mean (np.abs(bias_scan)**2,axis=1, keepdims=True);

    # Incoherent integration over several ramp
    if nincoher > 0:
        log.info ('Incoherent integration over %i ramps'%nincoher);
        base_scan = signal.uniform_filter (base_scan,(nincoher,0,0,0),mode='constant');
        bias_scan = signal.uniform_filter (bias_scan,(nincoher,0,0,0),mode='constant');
    else:
        log.info ('Incoherent integration over 1 ramp');

    # Observed noise, whose statistic is independent of averaging
    base_scan -= np.median (base_scan, axis=2, keepdims=True);
    bias_scan -= np.median (bias_scan, axis=2, keepdims=True);
    base_powerbb_np = base_scan[:,:,int(nscan/2),:][:,:,None,:];
    base_powerbb    = np.max (base_scan, axis=2, keepdims=True);
    bias_powerbb    = np.mean (np.max (bias_scan, axis=2, keepdims=True), axis=-1, keepdims=True);

    # Scale for gd in [um]
    log.info ('Compute GD');
    scale_gd = 1. / (lbd0**-1 - (lbd0+dlbd)**-1) / nscan;
    base_gd  = (np.argmax (base_scan, axis=2)[:,:,None,:] - int(nscan/2)) * scale_gd;
    gd_range = scale_gd * nscan / 2;
    
    # Broad-band SNR
    log.info ('Compute SNR');
    base_snr = base_powerbb / bias_powerbb;
    base_snr[~np.isfinite (base_snr)] = 0.0;

    # Smooth SNR along the ramp (actually done before)
    # base_snr = np.mean (base_snr,axis=1,keepdims=True);
    # base_gd  = np.mean (base_gd,axis=1,keepdims=True);

    # Copy before bootstrap
    base_snr0 = base_snr.copy ();
    base_gd0 = base_gd.copy ();

    # Smooth SNR
    log.info ('Stabilize SNR over over few ramps');
    base_snr = gaussian_filter (base_snr,(2,0,0,0),mode='constant',truncate=2.0);

    # Bootstrap over baseline. Maybe the GD should be
    # boostraped and averaged as a phasor
    base_snr, base_gd = signal.bootstrap_triangles (base_snr, base_gd);

    # Add the QC about raw SNR
    qc.snr (hdr, y0, base_snr0, base_snr);
    
    # Reduce norm power far from white-fringe
    if gdAttenuation == True or gdAttenuation == 'TRUE':
        log.info ('Apply coherence envelope of %.1f um'%(coherence_length*1e6));
        attenuation = np.exp (-(np.pi * base_gd / coherence_length)**2);
    else:
        log.info ('Dont apply coherence envelope');
        attenuation = base_gd * 0.0 + 1.0;

    # Compute selection flag from SNR
    log.info ('SNR selection > %.2f'%snr_threshold);
    hdr[HMQ+'SNR_THRESHOLD'] = (snr_threshold, 'to accept fringe');
    base_flag  = 1. * (base_snr > snr_threshold);

    # Compute selection flag from GD
    log.info ('GD selection: enveloppe > 0.2');
    base_flag *= (attenuation**2 > 0.2);

    # Mean flux
    bbeam = setup.base_beam ();
    photo_mean = np.nanmean (photo, axis=(1,2), keepdims=True);
    
    # Compute selection flag from SNR
    log.info ('Flux selection > %.2f'%flux_threshold);
    hdr[HMQ+'FLUX_THRESHOLD'] = (flux_threshold, 'to accept fringe');
    base_flag  *= (photo_mean[:,:,:,bbeam[:,0]] > flux_threshold);
    base_flag  *= (photo_mean[:,:,:,bbeam[:,1]] > flux_threshold);

    # TODO: Add selection on mean flux in ramp, gd...

    # Morphological operation
    log.info ('Closing/opening of selection');
    structure = np.ones ((2,1,1,1));

    base_flag0 = base_flag.copy ();
    base_flag = 1.0 * binary_closing (base_flag, structure=structure);
    base_flag = 1.0 * binary_opening (base_flag, structure=structure);

    # Replace 0 by nan to perform nanmean and nanstd
    base_flag1 = base_flag.copy ();
    base_flag[base_flag == 0.0] = np.nan;

    # Compute the time stamp of each ramp
    mjd_ramp = mjd.mean (axis=1);

    # Save the options in file HEADER
    hdr[HMP+'NCOHER'] = (ncoher,'[frame] coherent integration');
    hdr[HMP+'NINCOHER'] = (nincoher,'[ramp] incoherent integration');
    hdr[HMP+'NCS'] = (ncs,'[frame] cross-spectrum shift');
    hdr[HMP+'NBS'] = (nbs,'[frame] bi-spectrum shift');
    
    # Create the file
    hdulist = oifits.create (hdr, lbd, y0=y0);

    # Compute OI_FLUX
    log.info ('Compute Flux by simple mean, without selection');
    
    p_flux = np.nanmean (photo, axis=1);
    
    oifits.add_flux (hdulist, mjd_ramp, p_flux, output=output,y0=y0);

    # Compute OI_VIS2
    if ncs > 0:
        log.info ('Compute Cross Spectrum with offset of %i frames'%ncs);
        bias_power = np.real (bias_dft[:,ncs:,:,:] * np.conj(bias_dft[:,0:-ncs,:,:]));
        base_power = np.real (base_dft[:,ncs:,:,:] * np.conj(base_dft[:,0:-ncs,:,:]));
    else:
        log.info ('Compute Cross Spectrum without offset');
        bias_power = np.abs (bias_dft)**2;
        base_power = np.abs (base_dft)**2;

    # Average over the frames in ramp
    base_power = np.nanmean (base_power*base_flag, axis=1);
    bias_power = np.nanmean (bias_power, axis=1);

    # Average over the frames in ramp
    photo_power = photo[:,:,:,setup.base_beam ()];
    photo_power = 4 * photo_power[:,:,:,:,0] * photo_power[:,:,:,:,1] * attenuation**2;
    photo_power = np.nanmean (photo_power*base_flag, axis=1);

    oifits.add_vis2 (hdulist, mjd_ramp, base_power, bias_power, photo_power, output=output, y0=y0);
    
    c_cpx  = base_dft.copy ();
    
    # Compute OI_VIS

    if vis_reference == 'self':
        log.info ('Compute VIS by self-tracking');
        c_cpx *= np.exp (2.j*np.pi * base_gd / lbd[None,None,:,None]);
        phi = np.mean (c_cpx, axis=2, keepdims=True);
        phi = signal.uniform_filter_cpx (phi, (0,ncoher,0,0), mode='constant');
        c_cpx *= np.exp (-1.j * np.angle (phi));
        c_cpx  = np.nanmean (c_cpx * base_flag, axis=1);

    elif vis_reference == 'spec-diff':
        log.info ('Compute VIS by taking spectral-differential');
        c_cpx = c_cpx[:,:,1:,:] * np.conj(c_cpx[:,:,:-1,:]);
        c_cpx = np.insert(c_cpx,np.size(c_cpx,2),np.nan,axis=2);
        c_cpx = np.nanmean (c_cpx * base_flag, axis=1);

    else:
        raise ValueError("vis_reference is unknown");
        
    c_norm = photo[:,:,:,setup.base_beam ()];
    c_norm = 4 * c_norm[:,:,:,:,0] * c_norm[:,:,:,:,1] * attenuation**2;
    c_norm = np.sqrt (np.maximum (c_norm, 0));
    c_norm = np.nanmean (c_norm*base_flag, axis=1);
    
    oifits.add_vis (hdulist, mjd_ramp, c_cpx, c_norm, output=output, y0=y0);

    # Compute OI_T3
    if nbs > 0:
        log.info ('Compute Bispectrum with offset of %i frames'%nbs);
        t_cpx = (base_dft*base_flag)[:,:,:,setup.triplet_base()];
        t_cpx = t_cpx[:,2*nbs:,:,:,0] * t_cpx[:,nbs:-nbs,:,:,1] * np.conj (t_cpx[:,:-2*nbs,:,:,2]);
    else:
        log.info ('Compute Bispectrum without offset');
        t_cpx = (base_dft*base_flag)[:,:,:,setup.triplet_base()];
        t_cpx = t_cpx[:,:,:,:,0] * t_cpx[:,:,:,:,1] * np.conj (t_cpx[:,:,:,:,2]);

    # Load BBIAS_COEFF
    if coeff == []:
        log.info ('No BBIAS_COEFF file');
    else:
        f = coeff[0]['ORIGNAME'];
        log.info ('Load BBIAS_COEFF file %s'%f);
        bbias_coeff0 = pyfits.getdata (f, 'C0');
        bbias_coeff1 = pyfits.getdata (f, 'C1');
        bbias_coeff2 = pyfits.getdata (f, 'C2');

        # Debias with C0
        log.info ('Debias with C0');
        t_cpx -= bbias_coeff0[None,None,:,None]/(ncoher*ncoher*ncoher);

        # Debias with C1
        log.info ('Debias with C1');
        Ntotal = photo.sum (axis=-1,keepdims=True);
        t_cpx -= bbias_coeff1[None,None,:,None] * Ntotal[:,:np.size(t_cpx,1),:,:]/(ncoher*ncoher);

        # Debias with C2
        log.info ('Debias with C2');
        xps = np.real (base_dft[:,ncs:,:,:] * np.conj(base_dft[:,0:-ncs,:,:]));
        xps0 = np.real (bias_dft[:,ncs:,:,:] * np.conj(bias_dft[:,0:-ncs,:,:]));
        xps -= np.mean (xps0, axis=-1, keepdims=True);
        Ptotal = xps[:,:,:,setup.triplet_base()].sum (axis=-1);
        t_cpx = t_cpx[:,:-1,:,:];
        t_cpx -= bbias_coeff2[None,None,:,None] * Ptotal[:,:,:,:]/ncoher;
    
    # Normalisation, FIXME: take care of the shift
    t_norm = photo[:,:,:,setup.triplet_beam()];
    t_norm = t_norm[:,:,:,:,0] * t_norm[:,:,:,:,1] * t_norm[:,:,:,:,2];

    t_att  = attenuation[:,:,:,setup.triplet_base()];
    t_att  = t_att[:,:,:,:,0] * t_att[:,:,:,:,1] * t_att[:,:,:,:,2];

    t_cpx = np.nanmean (t_cpx, axis=1);
    t_norm = np.nanmean (t_norm * t_att, axis=1);

    oifits.add_t3 (hdulist, mjd_ramp, t_cpx, t_norm, output=output, y0=y0);

    # Figures
    log.info ('Figures');

    # Plot the 'opd-scan'
    fig,axes = plt.subplots (5,3, sharex=True);
    fig.suptitle (headers.summary (hdr));
    plot.base_name (axes);
    plot.compact (axes);
    for i,ax in enumerate (axes.flatten()): ax.imshow (base_scan[:,0,:,i].T,aspect='auto');
    files.write (fig,output+'_base_trend.png');

    # Plot the trend
    fig,axes = plt.subplots (5,3, sharex=True);
    fig.suptitle (headers.summary (hdr));
    plot.base_name (axes);
    plot.compact (axes);
    for i,ax in enumerate (axes.flatten()): ax.imshow (bias_scan[:,0,:,i].T,aspect='auto');
    files.write (fig,output+'_bias_trend.png');

    # SNR
    fig,axes = plt.subplots (5,3, sharex=True);
    fig.suptitle ('SNR versus ramp \n' + headers.summary (hdr));
    plot.base_name (axes);
    plot.compact (axes);
    d0 = np.mean (base_snr0,axis=(1,2));
    d1 = np.mean (base_snr,axis=(1,2));
    for b in range (15):
        ax = axes.flatten()[b];
        ax.axhline (snr_threshold,color='r', alpha=0.2);
        ax.plot (d1[:,b]);
        ax.plot (d0[:,b],'--', alpha=0.5);
        ax.set_yscale ('log');
    ax.set_xlabel ('Ramp #');
    files.write (fig,output+'_snr.png');

    # GD
    fig,axes = plt.subplots (5,3, sharex=True);
    fig.suptitle (headers.summary (hdr));
    plot.base_name (axes);
    plot.compact (axes);
    d0 = np.mean (base_gd0,axis=(1,2)) * 1e6;
    d1 = np.mean (base_gd,axis=(1,2)) * 1e6;
    lim = 1.05 * gd_range * 1e6;
    for b in range (15):
        ax = axes.flatten()[b];
        # lim = 1.05 * np.max (np.abs (d0[:,b]));
        ax.plot (d1[:,b]);
        ax.plot (d0[:,b],'--', alpha=0.5);
        ax.set_ylim (-lim,+lim);
    ax.set_xlabel ('Ramp #');
    files.write (fig,output+'_gd.png');

    # Photo
    fig,axes = plt.subplots (5,3, sharex=True);
    fig.suptitle ('Flux in fringe \n' + headers.summary (hdr));
    plot.compact (axes);
    for b in range (15):
        ax = axes.flatten()[b];
        data = np.nanmean (photo_mean[:,:,:,bbeam[b,:]], axis=(1,2));
        ax.plot (data[:,0]);
        ax.plot (data[:,1]);
        ax.axhline (flux_threshold,color='r', alpha=0.2);
        ax.set_ylim (1.0);
        ax.set_yscale ('log');
    ax.set_xlabel ('Ramp #');
    files.write (fig,output+'_flux.png');
    
    # Plot the fringe selection
    fig,axes = plt.subplots (5,3, sharex=True);
    fig.suptitle (headers.summary (hdr));
    plot.base_name (axes);
    plot.compact (axes);
    for b in range (15):
        axes.flatten()[b].plot (base_flag0[:,0,0,b], 'o', alpha=0.3, markersize=4);
        axes.flatten()[b].plot (base_flag1[:,0,0,b], 'o', alpha=0.3, markersize=2);
        axes.flatten()[b].set_ylim (-.2,1.2);
    files.write (fig,output+'_selection.png');

    # SNR versus GD
    fig,axes = plt.subplots (5,3, sharex=True);
    fig.suptitle (headers.summary (hdr));
    plot.base_name (axes);
    plot.compact (axes);
    d0 = np.mean (base_gd,axis=(1,2)) * 1e6;
    d1 = np.mean (base_snr0,axis=(1,2));
    lim = 1.05 * gd_range * 1e6;
    for b in range (15):
        axes.flatten()[b].axhline (snr_threshold,color='r', alpha=0.2);
        axes.flatten()[b].plot (d0[:,b], d1[:,b],'+');
        axes.flatten()[b].set_xlim (-lim,+lim);
    files.write (fig,output+'_snrgd.png');
    
    # POWER versus GD
    fig,axes = plt.subplots (5,3, sharex=True);
    fig.suptitle (headers.summary (hdr));
    plot.base_name (axes);
    plot.compact (axes);
    d0 = np.mean (base_gd,axis=(1,2)) * 1e6;
    d1 = np.mean (base_powerbb,axis=(1,2));
    d2 = np.mean (base_powerbb_np,axis=(1,2));
    lim = 1.05 * gd_range * 1e6;
    for b in range (15):
        axes.flatten()[b].plot (d0[:,b], d1[:,b],'+');
        axes.flatten()[b].plot (d0[:,b], d2[:,b],'+r',alpha=0.5);
        axes.flatten()[b].set_xlim (-lim,+lim);
    files.write (fig,output+'_powergd.png');
    
    # File
    log.info ('Create file');

    # First HDU
    hdulist[0].header['FILETYPE'] = filetype;
    hdulist[0].header[HMP+'RTS'] = os.path.basename (hdrs[0]['ORIGNAME'])[-30:];
    
    # Write file
    files.write (hdulist, output+'.fits');
            
    plt.close("all");
    return hdulist;
    
