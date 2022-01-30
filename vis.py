import numpy as np;
import os;
import pdb;

import matplotlib.pyplot as plt;
import matplotlib.colors as mcolors;

from astropy.stats import sigma_clipped_stats;
from astropy.io import fits as pyfits;
from astropy.modeling import models, fitting;

from skimage.feature import register_translation;

from scipy import fftpack;
from scipy.signal import medfilt;
from scipy.ndimage.interpolation import shift as subpix_shift;
from scipy.ndimage import gaussian_filter, uniform_filter, median_filter,shift;
from scipy.optimize import least_squares, curve_fit, brute;
from scipy.ndimage.morphology import binary_closing, binary_opening;
from scipy.ndimage.morphology import binary_dilation, binary_erosion;
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from itertools import permutations # could make self but easier if in anaconda

from . import files, headers, mircx_mystic_log, setup, oifits, signal, plot, qc;
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
            
        mircx_mystic_log.info ('Load BEAM_MAP file %s'%bmap['ORIGNAME']);
        mean_map = pyfits.getdata (bmap['ORIGNAME']);
        beam = int(bmap['FILETYPE'][4:5]) - 1;

        # Check that this xchan was extracted in hdr
        if HMW+'PHOTO%i STARTX'%(beam) not in hdr:
            mircx_mystic_log.info ('Beam %i not preproc in this file, skip'%(beam+1));
            continue;

        # Crop fringe window
        fsx = hdr[HMW+'FRINGE STARTX'];
        fsy = hdr[HMW+'FRINGE STARTY'];
        fringe_map[beam,:,:,:,:] = mean_map[:,:,fsy:fsy+nfy,fsx:fsx+nfx];

        # Crop xchan window
        psx = hdr[HMW+'PHOTO%i STARTX'%(beam)];
        psy = hdr[HMW+'PHOTO%i STARTY'%(beam)];
        photo_map[beam,:,:,:,:] = mean_map[:,:,psy:psy+npy,psx:psx+npx];

    return fringe_map, photo_map;

def compute_speccal (hdrs, output='output_speccal', filetype='SPEC_CAL',
                     ncoher=3, nfreq=4096, fitorder=2):
    '''
    Compute the SPEC_CAL from list of PREPROC
    '''
    elog = mircx_mystic_log.trace ('compute_speccal');

    # Check inputs
    headers.check_input (hdrs,  required=1);

    # Loop on files to compute their PSD
    for ih,h in enumerate(hdrs):
        f = h['ORIGNAME'];
        
        # Load file
        mircx_mystic_log.info ('Load PREPROC file %i over %i (%s)'%(ih+1,len(hdrs),f));
        hdr = pyfits.getheader (f);
        fringe = pyfits.getdata (f).astype(float);

        # Verbose on data size
        nr,nf,ny,nx = fringe.shape;
        mircx_mystic_log.info ('Data size: '+str(fringe.shape));

        # Define output
        if ih == 0:
            correl = np.zeros ((ny,nx*2-1));
            spectrum = np.zeros (ny);

        # Accumulate spectrum
        mircx_mystic_log.info ('Accumulate spectrum');
        tmp = medfilt (np.mean (fringe, axis=(0,1)), (1,11));
        spectrum += np.mean (tmp, axis=-1);
        
        # Remove the mean DC-shape
        mircx_mystic_log.info ('Compute the mean DC-shape');
        fringe_map = np.mean (fringe, axis=(0,1), keepdims=True);
        fringe_map /= np.sum (fringe_map);

        mircx_mystic_log.info ('Compute the mean DC-flux');
        fringe_dc = np.sum (fringe, axis=(2,3), keepdims=True);

        mircx_mystic_log.info ('Remove the DC');
        fringe -= fringe_map * fringe_dc;
        
        # Coherence integration
        mircx_mystic_log.info ('Coherent integration');
        fringe = uniform_filter (fringe,(0,ncoher,0,0),mode='constant');

        # We accumulate the full-window auto-correlation
        # instead of the FFT**2 because this allows to oversampled
        # the integrated PSD after the incoherent integration.
        mircx_mystic_log.info ('Accumulate auto-correlation');
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
    mircx_mystic_log.info ('Expect center of spectrum (lbd0) on %f'%fyc);

    # Build expected wavelength table
    lbdref,lbd0,dlbd = setup.lbd0 (hdr);
    lbd = (np.arange (ny) - fyc) * dlbd + lbd0;

    # Model for the pic position at lbd0
    freq0 = np.abs (setup.base_freq (hdr)) * lbdref / lbd0;
    delta0 = np.min (freq0) / 6;
    
    # Frequencies in pix-1
    freq = 1.0 * np.arange (nfreq) / nfreq;

    # Used dataset is restricted to interesting range
    idmin = np.argmax (freq > 0.75*freq0.min());
    idmax = np.argmax (freq > 1.25*freq0.max());

    # Compute zero-padded PSD
    mircx_mystic_log.info ('PSD with huge zero-padding %i'%nfreq);
    psd = np.abs (fftpack.fft (correl, n=nfreq, axis=-1, overwrite_x=False));

    # Remove bias and normalise to the maximum in the interesting range
    psd -= np.median (psd[:,idmax:], axis=-1, keepdims=True);
    norm = np.max (psd[:,idmin:idmax], axis=1, keepdims=True);
    psd /= norm;

    # Correlate each wavelength channel with a template
    mircx_mystic_log.info ('Fit PSD with model');
    res = [];
    for y in range (ny):
        args = (freq[idmin:idmax],freq0,delta0,psd[y,idmin:idmax]);
        s0 = lbd[y] / lbd0;
        # Fit at expected position
        res.append (least_squares (signal.psd_projection, s0, args=args, bounds=(0.8*s0,1.2*s0)));
        # Explore around
        for s00 in np.linspace (0.8*s0,1.2*s0, 20):
            rr = least_squares (signal.psd_projection, s00, args=args, bounds=(0.8*s00,1.2*s00));
            if (rr.fun[0] < res[-1].fun[0]): res[-1] = rr;
        mircx_mystic_log.info ('Best merit 1-c=%.4f found at s/s0=%.4f'%(res[-1].fun[0],res[-1].x[0]/s0));

    # Get wavelengths
    yfit = 1.0 * np.arange (ny);
    lbdfit = np.array([r.x[0]*lbd0 for r in res]);
    
    # Compute a better version of the wavelength
    # by fitting a quadratic law, optional
    lbdlaw = lbdfit.copy ();

    if (fitorder > 0 and is_valid.sum() > 5):
        mircx_mystic_log.info ('Fit measure with order %i polynomial'%fitorder);
        hdr[HMQ+'LBDFIT_ORDER'] = (fitorder, 'order to fit the lbd solution (0 is no fit)');
        
        # Run a quadratic fit on valid values, except the
        # edges of the spectra.
        is_fit = binary_erosion (is_valid, structure=[1,1,1]);
        poly = np.polyfit (yfit[is_fit], lbdfit[is_fit], deg=fitorder);

        # Replace the fitted values by the polynomial
        lbdlaw[is_fit] = np.poly1d (poly)(yfit[is_fit]);
    else:
        mircx_mystic_log.info ('Keep raw measure (no fit of lbd solution)');

    mircx_mystic_log.info ('Compute QC');
    
    # Compute quality of projection
    projection = (1. - res[int(ny/2)].fun[0]) * norm[int(ny/2),0];
    mircx_mystic_log.info ('Projection quality = %g'%projection);

    # Typical difference with prediction
    delta = np.median (np.abs (lbd-lbdfit));
    mircx_mystic_log.info ('Median delta = %.3f um'%(delta*1e6));

    # Residual of fit
    rms_res = np.std (lbdlaw[is_fit]-lbdfit[is_fit]);
    med_res = np.median (np.abs(lbdlaw[is_fit]-lbdfit[is_fit]));

    # Set quality to zero if clearly wrong fit
    if rms_res > 10e-9 or med_res > 10e-9:
        mircx_mystic_log.warning ('Spectral calibration is probably faulty, set QUALITY to 0');
        projection = 0.0;

    # Set QC
    hdr[HMQ+'QUALITY'] = (rep_nan (projection), 'quality of data');
    hdr[HMQ+'DELTA MEDIAN'] = (rep_nan (delta), '[m] median difference');
    hdr[HMQ+'RESIDUAL STD']    = (rep_nan (rms_res), '[m] std residual');
    hdr[HMQ+'RESIDUAL MEDIAN'] = (rep_nan (med_res), '[m] median residual');

    # Compute position on detector of lbdref
    s = np.argsort (lbdfit);
    try:     y0 = hdr[HMW+'FRINGE STARTY'] + np.interp (lbdref, lbdfit[s], yfit[s]);
    except:  y0 = -99.0
    hdr[HMQ+'LBDREF'] = (rep_nan (lbdref), '[m] lbdref');
    hdr[HMQ+'YLBDREF'] = (rep_nan (y0), 'ypos of %.3fum in cropped window'%(lbdref*1e6));
    mircx_mystic_log.info (HMQ+'YLBDREF = %e  (%.3fum)'%(y0,lbdref*1e6));

    mircx_mystic_log.info ('Figures');

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
    ax.set_ylim (lbd.min() * 1e6 - 0.05,lbd.max() * 1e6 + 0.05);
    files.write (fig,output+'_lbd.png');

    # PSD
    fig,ax = plt.subplots (2,1);
    fig.suptitle (headers.summary (hdr));
    ax[0].imshow (correl,aspect='auto');
    ax[1].plot (psd[:,0:int(nfreq/2)].T);
    files.write (fig,output+'_psd.png');

    # File
    mircx_mystic_log.info ('Create file');

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
    elog = mircx_mystic_log.trace ('compute_rts');

    # Check inputs
    save_all_freqs = headers.clean_option (save_all_freqs);
    headers.check_input (hdrs,  required=1);
    headers.check_input (profiles, required=1, maximum=6);
    headers.check_input (kappas, required=1, maximum=6);
    headers.check_input (speccal, required=1, maximum=1);

    # Load the wavelength table
    f = speccal[0]['ORIGNAME'];
    mircx_mystic_log.info ('Load SPEC_CAL file %s'%f);
    lbd = pyfits.getdata (f);

    # Get valid spectral channels
    is_valid = (pyfits.getdata (f,'IS_VALID') == 1);
    lbd = lbd[is_valid];
    
    # Load DATA
    f = hdrs[0]['ORIGNAME'];
    mircx_mystic_log.info ('Load PREPROC file %s'%f);
    hdr = pyfits.getheader (f);
    fringe = pyfits.getdata (f).astype(float);
    photo  = pyfits.getdata (f, 'PHOTOMETRY_PREPROC').astype(float);
    mjd    = pyfits.getdata (f, 'MJD');

    # Load other files if any
    for h in hdrs[1:]:
        f = h['ORIGNAME'];
        mircx_mystic_log.info ('Load PREPROC file %s'%f);
        fringe = np.append (fringe, pyfits.getdata (f).astype(float), axis=0);
        photo  = np.append (photo, pyfits.getdata (f, 'PHOTOMETRY_PREPROC').astype(float), axis=1);
        mjd    = np.append (mjd, pyfits.getdata (f, 'MJD'), axis=0);

    # Dimensions
    nr,nf,ny,nx = fringe.shape
    mircx_mystic_log.info ('fringe.shape = %s'%str(fringe.shape));
    mircx_mystic_log.info ('mean(fringe) = %f adu/pix/frame'%np.mean(fringe,axis=(0,1,2,3)));

    # Saturation checks
    fsat  = 1.0 * np.sum (np.mean (np.sum (fringe,axis=1),axis=0)>40000) / (ny*nx);
    mircx_mystic_log.info (HMQ+'FRAC_SAT = %.3f'%rep_nan (fsat));
    hdr[HMQ+'FRAC_SAT'] = (rep_nan (fsat), 'fraction of saturated pixel');

    # Get fringe and photo maps
    mircx_mystic_log.info ('Read data for photometric and fringe profiles');
    fringe_map, photo_map = extract_maps (hdr, profiles);
    
    # Define profile for optimal extraction of photometry
    # The same profile is used for all spectral channels
    # JDM2020: what is spectrum is curved?
    mircx_mystic_log.info ('Compute profile');
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
    mircx_mystic_log.info ('Extract photometry with profile');
    photo = np.sum (photo * profile, axis=-1);

    # Shift between photo and fringes in spectral direction
    mircx_mystic_log.info ('Compute spectral offsets in beam_map');
    shifty = np.zeros (6);
    upper = np.sum (medfilt (fringe_map,[1,1,1,1,11]), axis=(1,2,4));
    lower = np.sum (medfilt (photo_map,[1,1,1,1,1]) * profile, axis=(1,2,4));
    for b in range (6):
        shifty[b] = register_translation (lower[b,:,None],upper[b,:,None],
                                              upsample_factor=100)[0][0];

    # Re-align photometry
    mircx_mystic_log.info ('Register photometry to fringe');
    for b in range(6):
        photo[b,:,:,:] = subpix_shift (photo[b,:,:,:], [0,0,-shifty[b]]);

    # Keep only valid channels
    mircx_mystic_log.info ('Keep only valid channels');
    photo  = photo[:,:,:,is_valid];
    fringe = fringe[:,:,is_valid,:];
    fringe_map = fringe_map[:,:,:,is_valid,:];
    photo_map  = photo_map[:,:,:,is_valid,:];

    # Plot photometry versus time
    mircx_mystic_log.info ('Plot photometry');
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
    mircx_mystic_log.info ('Plot fringe ramp');
    fig,ax = plt.subplots ();
    fig.suptitle (headers.summary (hdr));
    ax.plot (np.mean (fringe, axis=(0,3)));
    ax.set_ylabel ('Mean fringe flux (adu)');
    ax.set_xlabel ('Frame in ramp');
    files.write (fig,output+'_fringeramp.png');

    # Get data for kappa_matrix
    mircx_mystic_log.info ('Read data for kappa matrix');
    fringe_kappa, photo_kappa = extract_maps (hdr, kappas);
    
    # Build kappa from input data.
    # kappa(nb,nr,nf,ny)
    mircx_mystic_log.info ('Build kappa-matrix with profile, filtering and registration, and keep valid');
    upper = np.sum (medfilt (fringe_kappa,[1,1,1,1,11]), axis=-1);
    lower = np.sum (medfilt (photo_kappa,[1,1,1,1,1]) * profile, axis=-1);
    for b in range(6):
        lower[b,:,:,:] = subpix_shift (lower[b,:,:,:], [0,0,-shifty[b]]);

    upper = upper[:,:,:,is_valid];
    lower = lower[:,:,:,is_valid];

    kappa = upper / (lower + 1e-20);

    # Set invalid kappas to zero
    skappa = setup.kappa (hdr);
    kappa[kappa > skappa*10] = 0.0;
    kappa[kappa < skappa/10] = 0.0;

    # Kappa-matrix as spectrum
    mircx_mystic_log.info ('Plot kappa');
    spec_upper = np.mean (upper, axis=(1,2));
    spec_lower = np.mean (lower, axis=(1,2));
    spec_kappa = np.mean (kappa, axis=(1,2));
    
    # Scaling kappa spectrum, the xchan flux and the kappa
    # are scaled with a factor to be closer to 1.0
    norm = np.max (medfilt (spec_upper,(1,3)), axis=1, keepdims=True) + 1e-20;
    spec_upper = spec_upper / norm;
    spec_lower = spec_lower / norm * skappa;
    spec_kappa = spec_kappa / skappa;
        
    fig,axes = plt.subplots (3,2);
    fig.suptitle (headers.summary (hdr));
    for b in range (6):
        ax = axes.flatten()[b];
        ax.plot (lbd*1e6,spec_upper[b,:],'--', label='fringe');
        ax.plot (lbd*1e6,spec_lower[b,:], label='xchan x %.1f'%skappa);
        ax.plot (lbd*1e6,spec_kappa[b,:], label='kappa / %.1f'%skappa);
        ax.set_ylim ((0.,2));
        ax.set_ylabel ('normalized flux');
    axes[0,0].legend();
    files.write (fig,output+'_kappa.png');

    # Kappa-matrix as image
    fig,ax = plt.subplots (1);
    fig.suptitle (headers.summary (hdr));
    ax.imshow (np.mean (kappa,axis=(1,2)));
    files.write (fig,output+'_kappaimg.png');

    # kappa is defined so that photok is the
    # total number of adu in the fringe
    mircx_mystic_log.info ('Compute photok');
    photok = photo * kappa;

    # QC about the fringe dc
    mircx_mystic_log.info ('Compute fringedc / photok');
    photok_sum = np.sum (photok,axis=(0,3));
    fringe_sum = np.sum (fringe,axis=(2,3));
    dc_ratio = np.sum (fringe_sum) / np.sum (photok_sum);
    hdr[HMQ+'DC MEAN'] = (rep_nan (dc_ratio), 'fringe/photo');

    # Scale the photometry to the fringe DC. FIXME: this is done
    # for all wavelength together, not per-wavelength.
    mircx_mystic_log.info ('Scale the photometries by %.4f'%dc_ratio);
    photok *= dc_ratio;

    # We save this estimation of the photometry
    # for the further visibility normalisation
    mircx_mystic_log.info ('Save photometry for normalisation');
    photok0 = photok.copy();

    # Smooth photometry
    if psmooth > 0:
        mircx_mystic_log.info ('Smooth photometry by sigma=%i frames'%psmooth);
        photok = uniform_filter (photok,(0,0,psmooth,0),mode='nearest');

    # Warning because of saturation
    mircx_mystic_log.info ('Deal with saturation in the filtering');
    isok  = 1.0 * (np.sum (fringe,axis=(2,3)) != 0);
    trans = uniform_filter (isok,(0,psmooth),mode='nearest');
    photok *= isok[None,:,:,None] / np.maximum (trans[None,:,:,None],1e-10);

    # Temporal / Spectral averaging of photometry
    # to be discussed (note that this is only for the
    # continuum removal, not for normalisation)
    mircx_mystic_log.info ('Temporal / Spectral averaging of photometry');
    spectra  = np.mean (photok, axis=(1,2), keepdims=True);
    spectra /= np.sum (spectra, axis=3, keepdims=True) + 1e-20;
    injection = np.sum (photok, axis=3, keepdims=True);
    photok = spectra*injection;
     
    # Compute flux in fringes. fringe_map is normalised
    mircx_mystic_log.info ('Compute dc in fringes');
    # fringe_map  = medfilt (fringe_map, [1,1,1,1,11]);
    # fringe_map  = median_filter (fringe_map, size=[1,1,1,1,11],mode='nearest');
    fringe_map /= np.sum (fringe_map, axis=-1, keepdims=True) + 1e-20;
    cont = np.einsum ('Brfy,Brfyx->rfyx', photok, fringe_map);

    # Check dc
    mircx_mystic_log.info ('Figure of DC in fringes');
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
    mircx_mystic_log.info ('Subtract dc with profiles predicted from xchan');
    fringe -= cont;
    del cont;

    # Remove the residual DC with a mean profile
    mircx_mystic_log.info ('Subtract residual dc with mean profile');
    fringe_meanmap = fringe_map.mean (axis=0);
    fringe_meanmap /= np.sum (fringe_meanmap, axis=-1, keepdims=True) + 1e-20;
    dcres = fringe.sum (axis=-1, keepdims=True);
    fringe -= dcres * fringe_meanmap;
    del dcres, fringe_meanmap;

    # Check residual
    mircx_mystic_log.info ('Figure of DC residual');
    fig,axes = plt.subplots (2, 1, sharex=True);
    fig.suptitle (headers.summary (hdr));
    axes[0].plot (fringe_img[int(ny/2),:], label='fringe');
    axes[0].plot (cont_img[int(ny/2),:], label='cont');
    axes[0].legend();
    axes[1].plot (np.mean (fringe[:,:,int(ny/2),:],axis=(0,1)), label='res');
    axes[1].set_xlabel('x (spatial direction)');
    axes[1].legend();
    files.write (fig,output+'_dcres.png');

    # Model (x,f)
    mircx_mystic_log.info ('Model of data');
    nfq = int(nx/2);
    f = 1. * np.arange(1,nfq+1);
    x = 1. * np.arange(nx) / nx;
    x -= np.mean (x);

    # fres is the spatial frequency at the
    # reference wavelength lbd0
    lbdref,lbd0,dlbd = setup.lbd0 (hdr);
    freqs = setup.base_freq (hdr) * lbdref / lbd0;
    
    # Scale to ensure the frequencies fall
    # into integer pixels (max freq is 40 or 72)
    ifreq_max = setup.ifreq_max (hdr);
    scale0 = 1.0 * ifreq_max / np.abs (freqs * nx).max();

    # Compute the expected scaling
    mircx_mystic_log.info ("ifreqs as float");
    mircx_mystic_log.info (freqs * scale0 * nx);

    # Compute the expected scaling
    mircx_mystic_log.info ("ifreqs as integer");
    ifreqs = np.round (freqs * scale0 * nx).astype(int);
    mircx_mystic_log.info (ifreqs);

    # Dimensions
    nb = len(ifreqs);
    nr,nf,ny,nx = fringe.shape

    # Compute DFT. The amplitude of the complex number corresponds
    # to the sum of the amplitude sum(A) of the oscillation A.cos(x)
    # in the fringe enveloppe.
    model = np.zeros ((nx,nfq*2+1));
    cf = 0.j + np.zeros ((nr*nf,ny,nfq+1));
    for y in np.arange(ny):
        mircx_mystic_log.info ('Project channel %i (centered)'%y);
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
        mircx_mystic_log.info ('Free fringe');
        del fringe;        

    # DFT at fringe frequencies
    mircx_mystic_log.info ('Extract fringe frequency');
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
        mircx_mystic_log.info ('Free cf');
        del cf;

    mircx_mystic_log.info ('Compute crude vis2 with various coherent');
        
    # Compute crude normalisation for vis2
    bbeam = setup.base_beam ();
    norm = np.mean (photok0[:,:,:,int(ny/2)], axis=(1,2));
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
        
    mircx_mystic_log.info ('Compute QC DECOHER_TAU0');
    
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
        hdr[HMQ+'DECOHER'+name+'_HALF'] = (rep_nan (vis2h[b]), '[ms] time for half V2');
        # Tau0 from model assuming 5/3
        try:
            popt, pcov = curve_fit (signal.decoherence, time, vis2[b,:], p0=[vis2[b,0], 0.01]);
            vis2m[b,:] = signal.decoherence (timem, popt[0], popt[1]);
            hdr[HMQ+'DECOHER'+name+'_TAU0'] = (rep_nan (popt[1]), '[ms] coherence time with 5/3');
        except:
            mircx_mystic_log.warning ("Fail to fit on baseline %i, continue anyway"%b);
        
    # Figures
    mircx_mystic_log.info ('Figures');

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
    mircx_mystic_log.info ('Create file');

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


    nscan =  np.int(2**(np.ceil(np.log2(ny*2+1)))); #64; round to next highest factor of 2?
    base_scan  = np.fft.fftshift (np.fft.fft (base_dft, n=nscan, axis=2), axes=2); 
    base_scan_avg = np.mean(np.abs(base_scan)**2,axis=(0,1))
    #log.info('JDM!!!!',base_scan.shape)
    hdu = pyfits.ImageHDU (base_scan_avg.astype('float32'));
    hdu.header['EXTNAME'] = ('GROUP_DELAY_AVG','Average Group Delay in File');
    hdu.header['BUNIT'] = 'powspec adu'
    hdu.header['SHAPE'] = '(ny_zpad,nb)';
    hdus.append (hdu);

    temp_photo = np.transpose (photok0,axes=(1,2,3,0))
    temp_photo = np.mean(temp_photo,axis=(0,1)  )
    hdu = pyfits.ImageHDU (temp_photo.astype('float32'));
    hdu.header['EXTNAME'] = ('PHOTOMETRY_AVG','Mean Photometry in File');
    hdu.header['BUNIT'] = 'adu'
    hdu.header['SHAPE'] = '(ny,nt)';
    hdus.append (hdu);

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
        mircx_mystic_log.info ("Save all frequencies for John's test");
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
                 ncoher=3, gdt_tincoh=0.5,
                 snr_threshold=3.0, flux_threshold=20.0,
                 gd_threshold=.5,
                 avgphot=True, ncs=2, nbs=2, gd_attenuation=False,
                 vis_reference='self'):
    '''
    Compute the OIFITS from the RTS
    '''
    elog = mircx_mystic_log.trace ('compute_vis');

    # Check inputs
    headers.check_input (hdrs, required=1);
    headers.check_input (coeff, required=0, maximum=1);

    # Get data
    f = hdrs[0]['ORIGNAME'];
    mircx_mystic_log.info ('Load RTS file %s'%f);
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
        mircx_mystic_log.info ('Load RTS file %s'%f);
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
    mircx_mystic_log.info ('Data size: '+str(base_dft.shape));
        
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
    mircx_mystic_log.info ('lbd0=%.3e, dlbd=%.3e um (R=%.1f)'%(lbd0*1e6,dlbd*1e6,np.abs(lbd0/dlbd)));

    # Coherence length
    coherence_length = lbd0**2 / np.abs (dlbd);

    # Spectral channel for QC (not exactly center of band
    # because this one is not working for internal light)
    y0 = int(ny/2) - 2;

    # Check if nan in photometry
    nnan = np.sum (np.isnan (photo));
    if nnan > 0: mircx_mystic_log.warning ('%i NaNs in photometry'%nnan);
        
    # Check if nan in fringe
    nnan = np.sum (np.isnan (base_dft));
    if nnan > 0: mircx_mystic_log.warning ('%i NaNs in fringe'%nnan);

    mircx_mystic_log.info ('Mean photometries: %e'%np.mean (photo));
    photo_original=photo.copy()
    # Do spectro-temporal averaging of photometry
    # JDM2020 The principal components should be basd on shutters not mean of data...!!
    if avgphot is True:
        mircx_mystic_log.info ('Do spectro-temporal averaging of photometry');
        hdr[HMP+'AVGPHOT'] = (True,'spectro-temporal averaging of photometry');
        
        for b in range (6):
            # Compute the matrix with mean, slope
            # JDM. we should use the shutter spectrum probably (higher snr)
            # this method does cause some problems with edge channels...
            spectrum = np.mean (photo[:,:,:,b], axis=(0,1));
            M = np.array ([spectrum, spectrum * (lbd - lbd0)*1e6]);
            # Invert system 
            ms = np.einsum ('rfy,ys->rfs', photo[:,:,:,b], np.linalg.pinv (M));
            photo[:,:,:,b] = np.einsum ('rfs,sy->rfy',ms,M);
            
        mircx_mystic_log.info ('Mean photometries: %e'%np.mean (photo));
    else:
        mircx_mystic_log.info ('No spectro-temporal averaging of photometry');
        hdr[HMP+'AVGPHOT'] = (False,'spectro-temporal averaging of photometry');
        
    # Do coherent integration
    mircx_mystic_log.info ('Coherent integration over %i frames'%ncoher);
    base_dft = signal.uniform_filter_cpx (base_dft,(0,ncoher,0,0),mode='constant');
    bias_dft = signal.uniform_filter_cpx (bias_dft,(0,ncoher,0,0),mode='constant');

    # Smooth photometry over the same amount 
    mircx_mystic_log.info ('Smoothing of photometry over %i frames'%ncoher);
    photo = uniform_filter (photo,(0,ncoher,0,0),mode='constant');

    #  Remove edges
    mircx_mystic_log.info ('Remove edge of coherence integration for each ramp');
    edge = int(ncoher/2);
    base_dft = base_dft[:,edge:nf-edge,:,:];
    bias_dft = bias_dft[:,edge:nf-edge,:,:];
    photo    = photo[:,edge:nf-edge,:,:];

    # New size
    nr,nf,ny,nb = base_dft.shape;

    # Add QC
    qc.flux (hdr, y0, photo);

    nscan =  np.int(2**(np.ceil(np.log2(ny*2+1)))); #64; round to next highest factor of 2?

    mircx_mystic_log.info ('Compute 2d FFT (nscan=%i)'%nscan);

    ## iterate 2 times to improve group delay
    base_dft_original = base_dft.copy ()
    bias_dft_original = bias_dft.copy ()

    #simulate_temp=(.6+.4*np.exp(-2.j*np.pi*(33.e-6/lbd)))*np.exp(-2.j*np.pi*(15.e-6/lbd))
    #base_dft_original *= 0. 
    #base_dft_original =(base_dft_original+1)*simulate_temp[None,None,:,None]
 
    # JDM will design to work best for short files chunks like 3-10 seconds.
    # we need to add an aveargin step to mircx_calibrate!

    # before we start,  we should find a 'good' part of the scans to create a template for the
    # opd analysis.  lets do a traditional gdt, smooth the snr. we can in theory iterate on this, but
    # likely not needed. I believe we don't need to have this be based on the same time chunk for each
    # baseline but can be based on best snr for a given baseline even if not at same time as others best snr.
    # this will lead to funny discreenacies between <vis2> and <visamp> but .. this can be detected later maybe.
    
    #gdt_tincoh = 0.5 # seconds  put as keyword!
    nincoh=np.maximum(np.minimum(np.ceil(gdt_tincoh*1000/hdr['EXPOSURE']/nf),nr//3),1)
    scale_gd = 1. / (lbd0**-1 - (lbd0+dlbd)**-1) / nscan;

    #GDT I. Identify Reference Ramp Region with best overall GDT SNR
    #Ia. Use long smoothing length here to be robust to low SNR conditions.

    #opds_centroid, base_snr, base_scan = fft_gdt(base_dft_original, bias_dft_original, nscan, \
    #    ncs=ncs, nincoh=np.minimum(20,nr//5), method='centroid');
    #gdt_phasor = np.exp (2.j*np.pi * opds_centroid*scale_gd / lbd[None,None,:,None]);
    
    #default 5 seconds of ramps for smoothing length or 1/3 of scan (e.,g for short 3 sec scans)

    opds_peak,     snr_peak,     base_scan_peak     = fft_gdt(base_dft_original, bias_dft_original, nscan, \
        ncs=ncs, nincoh=np.minimum(np.ceil(3000/hdr['EXPOSURE']/nf),nr//3),method='peak');
    #used to 5000
    #used ot nr//3

    #Ib. Using all fringes, find best ramp region with strongest peak fringes.
    # givre preference to center region...
    ramp_wts = snr_peak*np.exp(-.5*(opds_peak/(nscan/4))**2)
    ref_ramp = np.argmax(np.mean(np.log10( np.maximum(ramp_wts,np.abs(snr_threshold))),axis=3),axis=0)[0]
    # check taht this work .. weight toward center.

    #Ic. Binary DETECTOR
    #  Fit 2-gaussian model to the reference scan and look for statistically signifincant secondary peaks.
    #  Behavior of GDT will depend on if we find a 'near equal' binary or not.
    ref_opds_raw = opds_peak[ref_ramp,:,:,:] 
    ref_snr_raw = snr_peak[ref_ramp,:,:,:]

    # w/o tracking, we can double peaks in this important mean scan used for fitting gaussian, leading to
    # false peaks. I will do a centroid guiding (which is robust to binaries in order to have a better shape 
    # while using the peaks in the reference to account for the shifts.
    #opds_centroid,     snr_centroid,     base_scan_centroid    = fft_gdt(base_dft_original, bias_dft_original, nscan, \
    #    ncs=ncs, nincoh=nincoh,method='peak');
    #JDM prevous method use dto centroid. but ran into problems for low snr data. trying to see if peak will work
    # though might run into some bistable jumps for close binaries....  will use a shorter window. for accumulating
    # the OPD scan.
    #opds_centroid = opds_centroid-opds_centroid[ref_ramp,:,:,:]+opds_peak[ref_ramp,:,:,:]
    #gdt_phasor = np.exp (2.j*np.pi * opds_centroid*scale_gd/ lbd[None,None,:,None]);
    #opds_peak,     snr_peak,     base_scan_peak    = fft_gdt(base_dft_original*gdt_phasor, bias_dft_original, nscan, \
    #    ncs=ncs, nincoh=np.minimum(np.ceil(1000/hdr['EXPOSURE']/nf),nr//5),method='peak')
    #used to be 5000.    
    # #used ot nr//3

    #ref_opds_raw = opds_peak[ref_ramp,:,:,:] 
    #ref_snr_raw = snr_peak[ref_ramp,:,:,:]
    #ref_scan = np.squeeze(base_scan_peak[ref_ramp,0,:,:])
    #binary_flag, seps, opds,snrs = binary_detector(ref_scan, ref_snr_raw, ref_opds_raw,\
    #    scale=scale_gd*1e6,title=headers.summary(hdr)+"\n OPD SCANS",file=output+'_OPD_AVG.png') # might find peaks around bright singles, will require tuning


    ###################
    # New method to find separations:
    ac_flags,ac_seps,ac_snrs,ac_ratios=binary_detector_amps(base_dft,bias_dft,photo,nscan,ncs=ncs,scale=scale_gd*1e6,title=headers.summary(hdr)+"\n OPD AC SCANS",file=output+'_OPD_AC.png')
    ac_snrs_ratio = np.squeeze(ac_snrs[:,1]/ac_snrs[:,0])

    binary_flag = ac_flags
    snrs=ac_snrs # may have to copy other variables for 'exhaustive' method to work now....

    #return
    #Try something simple. if at least two baselines have flux ratio > 0.25 then treat as binary. (could make higher)
    # could be less restrictive I think with new imaging method
    #if np.any( (snrs[:,1]/snrs[:,0])*binary_flag > .25) and np.sum(binary_flag) >=2 :
    #strong_binary_condition = np.where((snrs[:,1]/snrs[:,0])*binary_flag > .25, True, False) 
    strong_binary_condition = np.where((snrs[:,1]/snrs[:,0])*binary_flag > .1, True, False) 
    if np.sum(strong_binary_condition) >= 2 and np.sum(binary_flag) >= 3:
        binary=True
    else:
        binary=False


    if binary:
        
        #identify components.  Use trick that center of two components can be a reference.   
        #boot_snr = np.amin(snrs,axis=1)*binary_flag
        #boot_opd = np.mean(opds,axis=1)*binary_flag # possivbel problem for aliased fringes....       
        #snr_jdm, opd_jdm, topd_jdm = signal.bootstrap_triangles_jdm (boot_snr[None,None,None,:], boot_opd[None,None,None,:]);

        #gd_method = 'gravity'
        #gd_method = 'exhaustive'
        #gd_method = 'torder' 

        # note only imaging and torder method works right with AC method.  other methods use 'binary flag'
        sep_method = 'imaging'

        if sep_method == 'imaging':

            im_res = lbd0/330. # could go fine.
            im_fov = lbd0/330. * lbd0/dlbd *2 
            dimx= (np.ceil(im_fov//im_res)).astype(int)
            im_x = np.linspace(+im_fov/2,-im_fov/2,dimx)
            im_y = -1*im_x

            im_xv, im_yv = np.meshgrid(im_x, im_y)
            uv_all=setup.base_uv(hdr)
            if 'STS_E' in headers.summary(hdr):
                sts_x0=np.array([6.61180   ,   30.4028  ,   0.00000   ,  18.7844  , 24.9115  ,  12.6828  ])
                sts_y0=np.array([13.3628   ,   3.37246  ,   5.29695   , -15.2014  , 15.2014  , -13.3395  ])
                sts_sep=50.00 # mas.
                mean_index = 1.4428026 #for 1.50, 1.65, 1.75 mu:  1.4446177 1.4428026  1.4415308
                sts_x = 2*mean_index*sts_x0*1e-6/(sts_sep*np.pi/1000./3600./180)
                sts_y = 2*mean_index*sts_y0*1e-6/(sts_sep*np.pi/1000./3600./180)
                for i in range(15):
                    tel_in=setup.base_beam()[i]
                    uv_all[0,i]=sts_x[tel_in[1]]-sts_x[tel_in[0]]
                    uv_all[1,i]=sts_y[tel_in[1]]-sts_y[tel_in[0]]


                
            im=np.zeros((dimx,dimx))
            for i in np.argwhere(binary_flag == True):
                # if STS/Etalon use artifical uv coverage to create synthetic binary at 1/4 FOV.
                proj_delay = im_xv*uv_all[0,i] + im_yv*uv_all[1,i]
                im += ac_snrs_ratio[i]*np.exp(-.5*( (proj_delay-ac_seps[i]*scale_gd)/(2.*scale_gd))**2)
                im += ac_snrs_ratio[i]*np.exp(-.5*( (proj_delay+ac_seps[i]*scale_gd)/(2.*scale_gd))**2)

                # also add alias.. should I?
                if ac_seps[i] <0:
                    alias_delay = ac_seps[i]+nscan
                else:
                    alias_delay = ac_seps[i]-nscan
                im += ac_snrs_ratio[i]*np.exp(-.5*( (proj_delay-alias_delay*scale_gd)/(2.*scale_gd))**2)
                im += ac_snrs_ratio[i]*np.exp(-.5*( (proj_delay+alias_delay*scale_gd)/(2.*scale_gd))**2)

            # using best imaging spot, use this to determine signs and aliasing of opds.
            # then reconstruct image.
            i_y,i_x = np.unravel_index(im.argmax(), im.shape)
            peak_x = im_x[i_x]
            peak_y = im_y[i_y]

            #make image!
            fig,axes = plt.subplots (1,1, sharex=True);
            fig.suptitle (headers.summary (hdr));
            factor = 1000.*3600.*180./np.pi
            extent = +im_fov/2*factor,-im_fov/2*factor,-im_fov/2*factor,+im_fov/2*factor
            plt.imshow(im,extent=extent);
            plt.xlim(+im_fov/2*factor,-im_fov/2*factor)
            plt.ylim(-im_fov/2*factor,+im_fov/2*factor)
            plt.xlabel('Milli-Arcseconds')
            plt.ylabel('Milli-Arcseoncds')
            plt.plot(peak_x*factor,peak_y*factor,'rx')
            plt.plot(-peak_x*factor,-peak_y*factor,'rx')
            axes.text(0.01, 0.95, 'Binary Found at %.2f, %.2f milliarcseconds'%(peak_x*factor,peak_y*factor),\
                verticalalignment='top', horizontalalignment='left',\
                transform=axes.transAxes, color='yellow', fontsize=10)
            files.write (fig,output+'_binary_image.png');
            plt.close("all");

         

            # now convert BACK to delays.
            newopd_m = peak_x*uv_all[0,:] + peak_y*uv_all[1,:]
            newopd = newopd_m/scale_gd
            newsnr=np.zeros(nb)+100.
            snr_torder, opd_torder, topd_torder = signal.bootstrap_triangles_jdm (newsnr[None,None,None,:],newopd[None,None,None,:])
            snr_torder=np.squeeze(snr_torder)
            opd_torder=np.squeeze(opd_torder)

            tsep_vector = np.squeeze(topd_torder)
            sep_vector  = np.squeeze(opd_torder)


        if sep_method == 'torder':
            # features
            # >exhaustive
            # >ignores delays > nscan/2
            # >robust to some 'bad' delays
            #iterate over all possible tel ORDERS.
            #there is one where all opds are positive and match delays.
            mircx_mystic_log.info('Starting t-order method:')
            tperm = list(permutations([0,1,2,3,4,5]))
            tnum=len(tperm)
            tperm = tperm[0:tnum//2]
            tnum=len(tperm)
            metric=np.zeros(tnum)
            # make quick image !?

            #helper arrays
            tels = setup.beam_tel(hdr)
            base_arr = np.zeros([6,6],dtype='int')
            counter=0
            for i in range(5):
                for j in range(i+1,6):
                    base_arr[i,j]=counter
                    base_arr[j,i]=counter
                    counter =counter +1


            for i_p,p in enumerate(tperm):
                newopd=np.zeros(nb)
                newsnr=np.zeros(nb)
                counter=0
                for i in range(5):
                    for j in range(i+1,6):
                        i1=np.minimum(p[i],p[j])
                        i2=np.maximum(p[i],p[j])
                        index=np.argwhere(np.bitwise_and(setup.base_beam()[:,0] == i1, setup.base_beam()[:,1] == i2))
                        newopd[counter] = ac_seps[index]
                        newsnr[counter] = ac_snrs_ratio[index]
                        counter = counter + 1
                print(p)
                # for metric. Check self-consinstencies of all triangles with wts, ignoring aliased delays
                # allow for 
                # mark longest baseline bad. 
                # if one of edge channels is S1, then also mark S2.
                newsnr[4]=0. # mark 1-6 baseline as bad
                if tels[p[0]] == 'S1':
                    newsnr[8]=0.
                if tels[p[5]] == 'S1':
                    newsnr[3] =0.
                # this should catch the two most most likely baselines to be aliased.
                # if more baselines are aliased, then all bets are off and a fast efficient
                # algorithm is still elusive. maybe best to actually make an "image"

                #cp1a=newopd[setup.triplet_base()[:,0]]
                cp1= newopd[setup.triplet_base()[:,0]]+ newopd[setup.triplet_base()[:,1]]
                cp2 = newopd[setup.triplet_base()[:,2]]
                cp_wt = np.amin(newsnr[setup.triplet_base()],axis=1)
                # to avoid aliasing... 
                # remove longest baseline.
                #index=np.argwhere(np.bitwise_and(np.abs(cp2) < (nscan*1/2) , cp_wt > 0))
                index=np.argwhere(cp_wt >0 )
                metric0=np.sqrt(np.average ( (cp1[index]-cp2[index])**2,weights=cp_wt[index]))
                print(cp1.astype(int))
                print(cp2.astype(int))
                print( ( cp1-cp2).astype(int))
                print(metric0)
                metric[i_p]=metric0

                print('Check')

            inbest = metric.argmin()
            pbest = tperm[inbest]
            # recover
            counter = 0
            for i in range(5):
                for j in range(i+1,6):
                    i1=pbest.index(i)
                    i2=pbest.index(j)
                    newopd[counter] = np.where(i1 < i2, ac_seps[counter], -1*ac_seps[counter])
                    newsnr[counter] = ac_snrs_ratio[counter]
                    counter =counter+1
            newsnr *= 100
            newsnr[base_arr[pbest[0],pbest[5]]]=0.
            if tels[pbest[0]] == 'S1':
                newsnr[base_arr[pbest[1],pbest[5]]]=0.
            if tels[pbest[5]] == 'S1':
                newsnr[base_arr[pbest[0],pbest[4]]]=0.


            snr_torder, opd_torder, topd_torder = signal.bootstrap_triangles_jdm (newsnr[None,None,None,:],newopd[None,None,None,:])
            snr_torder=np.squeeze(snr_torder)
            opd_torder=np.squeeze(opd_torder)

            tsep_vector = np.squeeze(B_topd_boot-A_topd_boot)
            sep_vector  = np.squeeze(B_opd_boot-A_opd_boot)

        if sep_method == 'exhaustive':
            nbase=12 # only search using the top nbase baselines.

            #start with strongest SNR peak. This will be my REFERENCE
            #then also drop the lowest couple of baselines to avoid wrapround problems and speed up.
            # drop baselines near edge.
            search_opds = opds.copy()
            search_snrs = snrs.copy()
            sort_method = 'sort_opd' # 'sort_snr'
            if sort_method=='sort_snr':
                mircx_mystic_log.info("Using sort_snr Method")
                b_ref = np.argmax(snrs[:,0]) # REF.
                search_snrs[b_ref,1]=-1
                cutoff = np.sort(search_snrs[:,1])[15-nbase]
                search_snrs[:,0]= np.where(search_snrs[:,1] <= cutoff, 0, search_snrs[:,0])
                search_snrs[:,1]= np.where(search_snrs[:,1] <= cutoff, 0, search_snrs[:,1])
                search_snrs[b_ref,0]=snrs[b_ref,0]
                goodin=np.squeeze(np.argwhere(search_snrs[:,1]>0))
                fixedin=np.squeeze(np.argwhere(search_snrs[:,1]==0))
                search_opds[fixedin,1]=search_opds[fixedin,0] # to help when missing tels.

            if sort_method=='sort_opd':
                mircx_mystic_log.info("Using sort_opd Method")
                mircx_mystic_log.info(search_snrs.T.astype(int))
                mircx_mystic_log.info(search_opds.T.astype(int))
                #remove baselines with largest opds. though remove zero snrs first. easy way is to assign large opds if snr =0
                search_opds = np.where(search_snrs ==0, 500., search_opds)
                temp= np.max(np.abs(search_opds),axis=1)
                if nbase != 15: 
                    cutoff = np.sort(temp)[nbase+1]
                    search_snrs[:,0]= np.where(temp >= cutoff, 0, search_snrs[:,0])
                    search_snrs[:,1]= np.where(temp >= cutoff, 0, search_snrs[:,1])
                b_ref = np.argmax(search_snrs[:,0]) # find strongest remaining fringe1
                search_snrs[b_ref,1]=0 #remove fringe2.
                mircx_mystic_log.info(search_snrs.T.astype(int))
                mircx_mystic_log.info(search_opds.T.astype(int))
                goodin=np.squeeze(np.argwhere(search_snrs[:,1]>0))
                fixedin=np.squeeze(np.argwhere(search_snrs[:,1]==0))
                search_opds[fixedin,1]=search_opds[fixedin,0] # to help when missing tels.

            merit_in=np.squeeze(np.argwhere(search_snrs[:,0]>0))
            
            num_combinations = 2**(nbase-1) # since top baseline is fixed.
            counter = np.arange(num_combinations)
            test_gd =  np.zeros( (num_combinations,nb))[:,None,None,:]
            test_snr = np.zeros( (num_combinations,nb))[:,None,None,:]

            if nbase != 15:
                for i_test in fixedin:
                    #log.info ('Filling in fixed OPDS for baseline %i '%i_test);
                    test_gd[:,0,0,i_test] = (search_opds.T)[counter % 1, i_test]
                    test_snr[:,0,0,i_test] = (search_snrs.T)[counter % 1 , i_test]
            for i_perm,i_test in enumerate(goodin):
                #log.info ('Filling in both OPDS for baseline %i '%i_test);
                test_gd[:,0,0,i_test] = (search_opds.T)[(counter // 2**i_perm) % 2, i_test]
                test_snr[:,0,0,i_test] = (search_snrs.T)[(counter // 2**i_perm) % 2 , i_test]
            #log.info ('Starting mega bootstrap for  %i  combinations'%num_combinations);
            test_snr_jdm, test_gd_jdm, test_results_jdm = signal.bootstrap_triangles_jdm (test_snr, test_gd);
            diff=np.squeeze((test_gd-test_gd_jdm)**2)
            diff_subset=diff[:,merit_in] # include only baselines we used in fit.
            merit=np.sqrt(np.mean(diff_subset,axis=1))
    

            #use Best one.    
            bestorder=np.argsort(merit)
            bestone=bestorder[0]
            mircx_mystic_log.info ('rms error  %.1f um'%(merit[bestone]*scale_gd*1e6));
            np.base_repr(bestorder[0],base=2)
            tsep_vector = -1*(np.squeeze(test_results_jdm[bestone,:])-np.squeeze(topd_jdm))*2 
            sep_vector  = -1*(np.squeeze(test_gd_jdm[bestone,:])-np.squeeze(opd_jdm) )*2

            # use all the opds.snrs, useful when we 'reflect' the opds..
            #A_val_opd =np.squeeze(test_gd[bestone,:])
            #A_val_snr =np.squeeze(test_snr[bestone,:])
            #B_val_opd = np.where(A_val_opd == opds[:,0], opds[:,1], opds[:,0] )
            #B_val_snr = np.where(A_val_opd == opds[:,0], snrs[:,1], snrs[:,0] )  

            # use the pruned set!
            A_val_opd =np.squeeze( test_gd[bestone,0,0,:])
            A_val_snr =np.squeeze(test_snr[bestone,0,0,:])
            B_val_opd = np.where(A_val_opd == opds[:,0], search_opds[:,1], search_opds[:,0] )
            B_val_snr = np.where(A_val_opd == opds[:,0], search_snrs[:,1], search_snrs[:,0] )  
            B_val_snr[b_ref] = snrs[b_ref,1]
            Atest, A_opd_boot, A_topd_boot = signal.bootstrap_triangles_jdm (A_val_snr[None,None,None,:],A_val_opd[None,None,None,:])
            Btest, B_opd_boot, B_topd_boot = signal.bootstrap_triangles_jdm (B_val_snr[None,None,None,:],B_val_opd[None,None,None,:])
            
            diffA=np.squeeze((A_val_opd-A_opd_boot)**2)[merit_in]
            diffB=np.squeeze((B_val_opd-B_opd_boot)**2)[merit_in]

            meritA=np.sqrt(np.mean(diffA))
            meritB=np.sqrt(np.mean(diffB))
            
            '''log.info ('A rms error  %.1f um'%(meritA*scale_gd*1e6));
            log.info ('B rms error  %.1f um'%(meritB*scale_gd*1e6));
            log.info(snrs.T.astype(int))
            log.info(opds.T.astype(int))
            log.info(merit_in.T.astype(int))

            log.info(search_snrs.T.astype(int))
            log.info(search_opds.T.astype(int))
            log.info(A_val_snr.T.astype(int))
            log.info(B_val_snr.T.astype(int))
            log.info(A_val_opd.T.astype(int))
            log.info(B_val_opd.T.astype(int)) '''

            
            tsep_vector = np.squeeze(B_topd_boot-A_topd_boot)
            sep_vector  = np.squeeze(B_opd_boot-A_opd_boot)


            # add wraparound snrs peaks?
            #Ag_topds, Ag_opds =gd_gravity_solver((A_val_opd)[None,None,None,:], (A_val_opd)[None,None,None,:], start_topd=A_topd_boot, softlength=2.,niter=50)
            #Bg_topds, Bg_opds =gd_gravity_solver((B_val_opd)[None,None,None,:], (B_val_opd)[None,None,None,:], start_topd=B_topd_boot, softlength=2.,niter=50)

            #tsep_vector = np.squeeze(Bg_topds-Ag_topds)[1:]
            #sep_vector  = np.squeeze(Bg_opds-Ag_opds)




            #A0_topds, A0_opds =gd_gravity_solver((opds.T)[None,None,:,:], (snrs.T)[None,None,:,:], start_topd=topd_jdm, softlength=2.,niter=50)

            
            # once I have the best fit, I can plug into gravity solver to deal with wraparound problems.
            #us bitwise inverse of bestorder to separate solve for B. should work I think.
             
            #there are failure modes!


        if sep_method == 'gravity':
            A_sep = np.full(nb,0.0)
            A_wts  =np.full(nb,0.0)
            in_strong = (snrs[:,0]*binary_flag).argmax()
            A_sep[in_strong]=-1*seps[in_strong]/2
            A_wts[in_strong]=snrs[in_strong,0]

            # note. this algorithms is not the best. but might work with tweaking.
            AB_topds, AB_opds =gd_gravity_solver((opds.T)[None,None,:,:], (snrs.T)[None,None,:,:], start_topd=topd_jdm, softlength=nscan/4.,niter=200)
            # We might want to add cross-check here! since this is critical to get a good solution!
            tsep_vector = (np.squeeze(AB_topds)[1:]-np.squeeze(topd_jdm))*2 
            sep_vector  = (np.squeeze(AB_opds)-np.squeeze(opd_jdm) )*2


        #gd_method='double-delta'
        gd_method='gravity'
        if gd_method == 'gravity':
            opds_peak,     snr_peak,     base_scan_peak     = fft_gdt(base_dft_original, bias_dft_original, nscan, \
                ncs=ncs, nincoh=nincoh,method='peak');

            bestsnr_index = np.argmax(base_scan_peak,axis=2)[:,:,None,:]
            bestsnr_snr = np.max(base_scan_peak,axis=2,keepdims=True)

            wlen = np.maximum(nscan/10, 2.*nscan/ny) # last part for low-res modes. first part for high-res modes.
            #gaussian fitting too slow in general but helpful in our BEST separation determinatino. now we use a faster
            #method to find top 3 peaks 
            norm_scan_mask1 = np.where( (np.abs( np.arange(nscan)[None,None,:,None]-bestsnr_index) < wlen) \
            |   (np.abs( np.arange(nscan)[None,None,:,None]-bestsnr_index) > (nscan-wlen)), 0., base_scan_peak)
            bestsnr1_index = np.argmax(norm_scan_mask1,axis=2)[:,:,None,:]
            bestsnr1_snr = np.max(norm_scan_mask1,axis=2,keepdims=True)

            #don't calculate the 3rd to save time.
            #norm_scan_mask2 = np.where( (np.abs( np.arange(nscan)[None,None,:,None]-bestsnr1_index) < wlen) \
            #|   (np.abs( np.arange(nscan)[None,None,:,None]-bestsnr1_index) > (nscan-wlen)), 0., norm_scan_mask1)
            #bestsnr2_index = np.argmax(norm_scan_mask2,axis=2)[:,:,None,:]
            #bestsnr2_snr = np.max(norm_scan_mask2,axis=2,keepdims=True)

            #try to determine 'sign' of the sep_vector: Lets define component A as the higher snr one.
            median_sep = np.median(bestsnr1_index-bestsnr_index,axis=0)
            alias_flag = np.where(np.abs(sep_vector) > nscan/2, False, True)
            temp_index = np.argwhere(alias_flag*binary_flag) # tried getting masked arrays workimng but failed... JDM
            sep_sign = np.median(np.sign(median_sep*sep_vector).ravel()[temp_index]) # not foolproof.
            
            A_index = np.where(np.sign(bestsnr1_index-bestsnr_index) == np.sign(sep_sign*sep_vector), bestsnr_index,bestsnr1_index)
            B_index = np.where(np.sign(bestsnr1_index-bestsnr_index) != np.sign(sep_sign*sep_vector), bestsnr_index,bestsnr1_index)
            A_snr = np.where(np.sign(bestsnr1_index-bestsnr_index) == np.sign(sep_sign*sep_vector), bestsnr_snr,bestsnr1_snr)
            B_snr = np.where(np.sign(bestsnr1_index-bestsnr_index) != np.sign(sep_sign*sep_vector), bestsnr_snr,bestsnr1_snr)
            #still problem with aliased baseline having wrong sign.
            # at this end of this section, A is the higher snr one, B is lower snr one and the separation is defined by sep_sign
            # +sign = matches sep_vector from last step, otherwise its swapped.  will need to account for this for starting conditions
            # in nex step.

            # keep top 3
            #bestsnr_snrs = np.concatenate( (bestsnr_snr,bestsnr1_snr,bestsnr2_snr),axis=2)
            #bestsnr_indices = np.concatenate( (bestsnr_index,bestsnr1_index,bestsnr2_index),axis=2)-(nscan//2)

            #keep top 2
            #bestsnr_snrs = np.concatenate( (bestsnr_snr,bestsnr1_snr),axis=2)
            #bestsnr_indices = np.concatenate( (bestsnr_index,bestsnr1_index),axis=2)-(nscan//2)
            bestsnr_snrs = np.concatenate( (A_snr,B_snr),axis=2)
            bestsnr_indices = np.concatenate( (A_index,B_index),axis=2)-(nscan//2)
            bestsnr_indices = np.where(bestsnr_snrs < np.abs(snr_threshold),0, bestsnr_indices) # set low snr fringes to position 0

            # now we do a double gravity solver ... then plot results!
            #binary_flag2d = np.where(bestsnr1_snr > np.abs(snr_threshold), True, False)
            #boot_snr = np.amin(bestsnr_snrs[:,:,0:2,:],axis=2,keepdims=True)*binary_flag2d
            #boot_opd = np.mean(bestsnr_indices[:,:,0:2,:],axis=2,keepdims=True)*binary_flag2d # possivbel problem for aliased fringes....       
            #rather tha solve for MEAN. maybe Better to 
            #snr_jdm, opd_jdm, topd_jdm = signal.bootstrap_triangles_jdm (boot_snr, boot_opd);
            #topd_jdm  = topd_jdm-.5*tsep_vector[None,None,None,:] # move from COM to peak.
            # start at zero...
            # flag separations that are likely aliased.
            #alias_flag = np.where(np.abs(sep_vector) > nscan/2, False, True)
            snrA_jdm, opdA_jdm, topdA_jdm = signal.bootstrap_triangles_jdm (A_snr*alias_flag[None,None,None,:], (A_index-nscan//2)*alias_flag[None,None,None,:]);
            snrB_jdm, opdB_jdm, topdB_jdm = signal.bootstrap_triangles_jdm (B_snr*alias_flag[None,None,None,:], (B_index-nscan//2)*alias_flag[None,None,None,:])
            topd_jdm = topdA_jdm # use higher snr component.
            if sep_sign == -1:
                topd_jdm = topdA_jdm-tsep_vector[None,None,None,:] #switch starting point to other B to maintain tsep_vector.
            #Note. Here. A is the brightest (likely near zero) and B is fainter. the sepvector can switch sign depending.
            # this was painful to work out. 

            #topd_jdm = np.zeros((nr,1,1,5)) 
            #if this doesn't work, could try the above mean method but filtering out long baselines.
            #topd_jdm1 = topd_jdm+tsep_vector[None,None,None,:]
            #topd_jdm2 = topd_jdm-tsep_vector[None,None,None,:]
            
            AB_topds, AB_opds, AB_pot =gd_gravity2_solver(bestsnr_indices, bestsnr_snrs, tsep_vector, start_topd=topd_jdm, softlength=nscan/4.,niter=200,nscan=nscan)
            #AB_topds1, AB_opds1, AB_pot1 =gd_gravity2_solver(bestsnr_indices, bestsnr_snrs, tsep_vector, start_topd=topd_jdm1, softlength=nscan/4.,niter=200)
            #AB_topds2, AB_opds2, AB_pot2  =gd_gravity2_solver(bestsnr_indices, bestsnr_snrs, tsep_vector, start_topd=topd_jdm2, softlength=nscan/4.,niter=200)
            #might add this check for multiple starting positions!

            #opds_peak = scale_gd*1e6*opds_peak
            #opds_centroid = scale_gd*1e6*opds_centroid
            A_opds = np.squeeze(AB_opds) 
            B_opds = A_opds + sep_vector[None,:]
            imshow_gdt(base_scan_peak ,opds1=A_opds*scale_gd*1e6,opds2=B_opds*scale_gd*1e6, scale=scale_gd*1e6,title=headers.summary(hdr)+"\nOPD",file=output+'_base_trend.png')
        
            tracking_snr0  = bestsnr_snr # use the brigthest peak.
            tracking_opds0 = (A_opds-A_opds[ref_ramp,:])[:,None,None,:]
            tracking_snr, tracking_opds, tracking_topds = signal.bootstrap_triangles_jdm (tracking_snr0, tracking_opds0);
            base_gd = 0.5*(A_opds[:,None,None,:]+B_opds[:,None,None,:])
                # used fcenter or attentuation correciotn which we sholdn't do for binaries!
            #in0=tracking_opds[]
            #tracking_snrs = base_scan_peak[]
            #tracking_snrs = base_scan_peak[]

        if gd_method == 'double-delta':
                mircx_mystic_log.info("Starting with double-delta method.")

                # maybe waste of time, but getting reference positions using new separations:
                opds_peak,     snr_peak,     base_scan_peak     = fft_gdt2(sep_vector,base_dft_original, bias_dft_original, nscan, \
                            ncs=ncs, nincoh=np.minimum(np.ceil(3000/hdr['EXPOSURE']/nf),nr//3),method='peak');
                # keep same ref ramp location but get new reference.

                #Ic. Binary DETECTOR
                #  Fit 2-gaussian model to the reference scan and look for statistically signifincant secondary peaks.
                #  Behavior of GDT will depend on if we find a 'near equal' binary or not.
                ref_opds_raw = opds_peak[ref_ramp,:,:,:] 
                ref_snr_raw = snr_peak[ref_ramp,:,:,:]

                opds_peak,     snr_peak,     base_scan_peak     = fft_gdt2(sep_vector,base_dft_original, bias_dft_original, nscan, \
                        ncs=ncs, nincoh=nincoh,method='peak');
                tracking_snr0 = snr_peak.copy()
                tracking_opds0 = opds_peak #- ref_opds_raw # opds_peak[ref_ramp,:,:,:]
                tracking_snr, tracking_opds, tracking_topds = signal.bootstrap_triangles_jdm (tracking_snr0, tracking_opds0)
                #imshow_gdt(base_scan_peak ,opds1=opds_peak*scale_gd*1e6, opds2=tracking_opds*scale_gd*1e6,scale=scale_gd*1e6,title=headers.summary(hdr)+"\nOPD",file=output+'_base_trend.png')
                A_opds = tracking_opds
                B_opds = tracking_opds + sep_vector[None,None,None,:]
                # for imaging
                opds_peak1,     snr_peak1,     base_scan_peak1     = fft_gdt(base_dft_original, bias_dft_original, nscan, \
                    ncs=ncs, nincoh=nincoh,method='peak');
                imshow_gdt(base_scan_peak1 ,opds1=A_opds*scale_gd*1e6,opds2=B_opds*scale_gd*1e6, scale=scale_gd*1e6,title=headers.summary(hdr)+"\nOPD",file=output+'_base_trend.png')
                base_gd = tracking_opds.copy() # for later plotting

                mircx_mystic_log.info("Finished with double-delta method.")

    else:
        #SINGLE
        opds_peak,     snr_peak,     base_scan_peak     = fft_gdt(base_dft_original, bias_dft_original, nscan, \
            ncs=ncs, nincoh=nincoh,method='peak');

        tracking_snr0 = snr_peak.copy()
        tracking_opds0 = opds_peak -opds_peak[ref_ramp,:,:,:]
        tracking_snr, tracking_opds, tracking_topds = signal.bootstrap_triangles_jdm (tracking_snr0, tracking_opds0)
        imshow_gdt(base_scan_peak ,opds1=opds_peak*scale_gd*1e6, opds2=tracking_opds*scale_gd*1e6,scale=scale_gd*1e6,title=headers.summary(hdr)+"\nOPD",file=output+'_base_trend.png')

        base_gd = tracking_opds.copy() # for later plotting

    tracking_opds = tracking_opds*scale_gd
    tracking_opds0 = tracking_opds0*scale_gd
    base_gd = base_gd*scale_gd

    #if snr_threshold <0: # signal that this is FOREGROUND. do no fringe tracking but don't flag as bad.
    #    tracking_opds = tracking_opds*0.0
    #    tracking_opds0 = tracking_opds0*0.0
    #    base_gd = base_gd*0.0
    gdt_phasor = np.exp (2.j*np.pi * tracking_opds/ lbd[None,None,:,None]);

    temp_snr, temp_opd, oivis_scan = fft_gdt(base_dft_original*gdt_phasor, bias_dft_original, nscan, ncs=ncs, nincoh=nincoh,method='peak');
    # only use for this is for PLOTTING. w/ base_flag
        
    # Add the QC about raw SNR

    qc.snr (hdr, y0, tracking_snr0, tracking_snr);
    
    # Reduce norm power far from white-fringe
    #JDM this needs checked if there is a correlation.
    if gd_attenuation == True or gd_attenuation == 'TRUE':
        mircx_mystic_log.info ('Apply coherence envelope of %.1f um'%(coherence_length*1e6));
        attenuation = np.sinc(base_gd/coherence_length) 
    else:
        mircx_mystic_log.info ('Dont apply coherence envelope');
        attenuation = base_gd * 0.0 + 1.0;

    # Compute selection flag from SNR
    mircx_mystic_log.info ('SNR selection > %.2f'%snr_threshold);
    hdr[HMQ+'SNR_THRESHOLD'] = (snr_threshold, 'to accept fringe');
    snr_smooth_filter=np.maximum(np.minimum(nincoh*4, nr//4),3).astype(int) # if median snr is above bar then keep. smooth to avoid biased over selection near threshold
    median_snr = median_filter(tracking_snr,size=(snr_smooth_filter,1,1,1),mode='nearest',origin=0)
    if snr_threshold > 0:
        base_flag  = 1. * (median_snr > snr_threshold);
    else:
        mircx_mystic_log.info('For this foreground data we accept all fringes but we do not track on them')
        base_flag = tracking_snr *0.0 + 1.0

    # Compute selection flag from GD
    mircx_mystic_log.info ('GDT Motion selection MU > %.2f'%(gd_threshold*.5*coherence_length*1e6));
    hdr[HMQ+'GD_THRESHOLD'] = (gd_threshold*.5*coherence_length*1e6, 'microns to accept fringe');
    gd_smoothl=np.maximum(np.minimum(nincoh*4,nr//4).astype(int),3) # smooth before filtering. 
    # SLIGHTLY CHANGE interpretation. we look for deviations away from the reference position, not 0.
    median_gd = median_filter(tracking_opds,size=(gd_smoothl,1,1,1),mode='nearest')
    base_flag *= (np.abs(median_gd) < gd_threshold*.5*coherence_length);
    # we define dlbd based on pixel so the trackingw indowis 0.5 coherence length.
    #JDM slight worry this will cause bias if target has barely detectable fringes and so will only keep
    #a biased sample.


    # for fringe power and delay trhesholds, we will extend bad flags cine our smoothing lengths are slong
    base_flag=uniform_filter(base_flag,(np.minimum(nincoh*4+1,nr//4),0,0,0))  # extend bad flags a bit to get all bad data since
                                                        # we are smoothing a lot
                                                        # might be overkill for flux drop outs.... (?)

    # Mean flux
    bbeam = setup.base_beam ();
    photo_mean = np.nanmean (photo, axis=(1,2), keepdims=True);
    


    # Compute selection flag from SNR
    mircx_mystic_log.info ('Flux selection > %.2f'%flux_threshold);
    hdr[HMQ+'FLUX_THRESHOLD'] = (flux_threshold, 'to accept fringe');
    base_flag  *= (photo_mean[:,:,:,bbeam[:,0]] > flux_threshold);
    base_flag  *= (photo_mean[:,:,:,bbeam[:,1]] > flux_threshold);

    # TODO: Add selection on mean flux in ramp, gd...

    # Morphological operation
    #log.info ('Closing/opening of selection');
    #structure = np.ones ((2,1,1,1));

    #base_flag0 = base_flag.copy ();
    #base_flag = 1.0 * binary_closing (base_flag, structure=structure);
    #base_flag = 1.0 * binary_opening (base_flag, structure=structure);


    # add option to flag bad if gdt doesn't close within X microns.
    # might be useful. would remove baselines that don't close.
    # tricky to calculate probably -- easier if we only require all baselines to close that have high snr.



    base_flag0 = base_flag.copy ();
    # reset flags for testing JDM
    #base_flag  = 1. * (median_snr > -1e10);

    # Replace 0 by nan to perform nanmean and nanstd
 

    base_flag[base_flag < 1] = np.nan;

    #base_flag[base_flag == 0.0] = np.nan;


    # Compute the time stamp of each ramp
    mjd_ramp = mjd.mean (axis=1);

    # Save the options in file HEADER
    hdr[HMP+'NCOHER'] = (ncoher,'[frame] coherent integration');
    hdr[HMP+'NINCOHER'] = (nincoh,'[ramp] incoherent integration');
    hdr[HMP+'NCS'] = (ncs,'[frame] cross-spectrum shift');
    hdr[HMP+'NBS'] = (nbs,'[frame] bi-spectrum shift');
    
    # Create the file
    hdulist = oifits.create (hdr, lbd, y0=y0);

    # Compute OI_FLUX
    mircx_mystic_log.info ('Compute Flux by simple mean, without selection');
    
    p_flux = np.nanmean (photo, axis=1);
    
    # I can't really flag on flux since flags are done per baseline.
    oifits.add_flux (hdulist, mjd_ramp, p_flux, output=output,y0=y0);



    # Compute OI_VIS2
    if ncs > 0:
        mircx_mystic_log.info ('Compute Cross Spectrum with offset of %i frames'%ncs);
        bias_power = np.real (bias_dft_original[:,ncs:,:,:] * np.conj(bias_dft_original[:,0:-ncs,:,:]));
        base_power = np.real (base_dft_original[:,ncs:,:,:] * np.conj(base_dft_original[:,0:-ncs,:,:]));
    else:
        mircx_mystic_log.info ('Compute Cross Spectrum without offset');
        bias_power = np.abs (bias_dft_original)**2;
        base_power = np.abs (base_dft_original)**2;

    # Average over the frames in ramp
    base_power = np.nanmean (base_power*base_flag, axis=1);
    bias_power = np.nanmean (bias_power, axis=1);

    # Average over the frames in ramp
    photo_power = photo[:,:,:,setup.base_beam ()];
    #totflux = np.nansum(photo,axis=(1,3))
    #bp=np.nanmean(bias_power,axis=2)
    photo_power = 4 * photo_power[:,:,:,:,0] * photo_power[:,:,:,:,1] * attenuation**2;
    photo_power = np.nanmean (photo_power*base_flag, axis=1);
    #note there is a time shift between photo and dft that should get fixed.
     
    nchunk=10 # makes sure each chunk is longer than nincoh (for oi-vis, not needed for oi-vis2,t3phi)

    #nr_smooth= nr//nchunk
    # JDM: what about if most of the frame is bad due to bad tracking or something? or missing fringes?
    # we need to make a list cognizant of this before chunking. 
    # 
    #index_bin=(np.arange(nr_smooth)[None,:])+(np.arange(nchunk)*nr_smooth)[:,None]
    #mjd_ramp0=np.nanmean(mjd_ramp[index_bin],axis=1)
    #base_power0=np.nanmean(base_power[index_bin,:,:],axis=1)
    #bias_power0=np.nanmean(bias_power[index_bin,:,:],axis=1)
    #photo_power0=np.nanmean(photo_power[index_bin,:,:],axis=1)


    #oifits.add_vis2 (hdulist, mjd_ramp, base_power, bias_power, photo_power, output=output, y0=y0);
    oifits.add_vis2 (hdulist, mjd_ramp, base_power, bias_power, photo_power, output=output, y0=y0,nchunk=nchunk);

    # Compute OI_VIS
    if vis_reference == 'self':

        mircx_mystic_log.info ('Compute VIS by self-tracking');
        hdulist[0].header[HMP+'VIS_REF'] = ('SELF', 'vis reference');

        #probalby should weight the next amp by gd_weights. 
        c_cpx  =  base_dft_original * np.exp (2.j*np.pi * (tracking_opds) / lbd[None,None,:,None]);
        temp_opds_peak,     temp_snr_peak,     temp_base_scan    = fft_gdt(c_cpx, bias_dft_original, nscan, \
        ncs=ncs, nincoh=nincoh,method='peak');
        
        #phase_ref_method = "time-shift" # "classic"
        phase_ref_method = "bias-free" # "classic" "time-shift"
        if phase_ref_method == 'bias-free':
            mircx_mystic_log.info('using bias-free method of oi-vis')
            #JDM may need to deal with baseflag earlier.
            #JDM figured out (hopefully) how to bias correct the cvis without time-shifting.
            c_cpx_ref = np.mean (c_cpx, axis=2, keepdims=True);
            #c_cpx0 = c_cpx*np.conj(c_cpx_ref) - np.mean(bias_dft_original*np.conj(bias_dft_original),axis=2,keepdim=True)/ny
            #c_cpx0 = c_cpx*np.conj(c_cpx_ref) - bias_dft_original*np.conj(bias_dft_original)/ny #bias corrected cvis like Vis2
            c_cpx0 = c_cpx[:,ncs:,:,:] * np.conj(c_cpx_ref[:,0:-ncs,:,:]) - np.real(bias_dft_original[:,ncs:,:,:]*np.conj(bias_dft_original[:,0:-ncs,:,:])/ny)
       # bias_power = np.real (bias_dft_original[:,ncs:,:,:] * np.conj(bias_dft_original[:,0:-ncs,:,:]));
       # base_power = np.real (base_dft_original[:,ncs:,:,:] * np.conj(base_dft_original[:,0:-ncs,:,:]));
            #c_cpx0_norm2 = c_cpx_ref*conj(c_cpx_ref) - bias_ref*conj(bias_ref) #not needed until end.
            #c_cpx0 = c_cpx0/np.sqrt(np.mean(c_cpx0_norm2,axis=(0,1),keepdims=True)) #not needed until end.

            #iterate with dphase in case of crazy wild dphases.
            c_cpx1= np.nanmean(c_cpx0*base_flag,axis=(0,1),keepdims=1)
            dphase_phasor = np.exp (-1.j * np.angle (c_cpx1));
            c_cpx_ref = np.mean (c_cpx*dphase_phasor, axis=2, keepdims=True);
            #c_cpx0 = c_cpx*np.conj(c_cpx_ref)- bias_dft_original*np.conj(bias_dft_original)/ny
            c_cpx0 = c_cpx[:,ncs:,:,:] * np.conj(c_cpx_ref[:,0:-ncs,:,:]) - np.real(bias_dft_original[:,ncs:,:,:]*np.conj(bias_dft_original[:,0:-ncs,:,:])/ny)

            #c_cpx0 = c_cpx*np.conj(c_cpx_ref)- bias_dft_original*np.conj(bias_dft_original)/ny

            # do it again with the EVEN BETTER dphase estimate! 2 more times to converge!
            c_cpx1= np.nanmean(c_cpx0*base_flag,axis=(0,1),keepdims=1)
            dphase_phasor = np.exp (-1.j * np.angle (c_cpx1)); # need to put the ref_ramp slopes back on.
            c_cpx_ref = np.mean (c_cpx*dphase_phasor, axis=2, keepdims=True);
            #c_cpx0 = c_cpx*np.conj(c_cpx_ref)- bias_dft_original*np.conj(bias_dft_original)/ny
            c_cpx0 = c_cpx[:,ncs:,:,:] * np.conj(c_cpx_ref[:,0:-ncs,:,:]) - np.real(bias_dft_original[:,ncs:,:,:]*np.conj(bias_dft_original[:,0:-ncs,:,:])/ny)

            #c_cpx0 = c_cpx*np.conj(c_cpx_ref) - bias_dft_original*np.conj(bias_dft_original)/ny 

            # do it again with the EVEN BETTER dphase estimate! 2 more times to converge!
            #c_cpx1= np.nanmean(c_cpx0*base_flag,axis=(0,1),keepdims=1) # final answer except normalization.

            bias_ref = np.mean(bias_dft_original,axis=2,keepdims=True);
            #c_cpx0_norm2 = c_cpx_ref*np.conj(c_cpx_ref) - bias_ref*np.conj(bias_ref)
            c_cpx0_norm2 = np.real( c_cpx_ref[:,ncs:,:,:]*np.conj(c_cpx_ref[:,:-ncs,:,:])- bias_ref[:,ncs:,:,:]*np.conj(bias_ref[:,0:-ncs,:,:]))
            if snr_threshold > 0: #don't divide for foreground/background.
                c_cpx = c_cpx0/np.sqrt(np.maximum(np.nanmean(c_cpx0_norm2*base_flag,axis=(0,1),keepdims=True),0))
            #realy this normalization shiuld be carried into OI-vis like for oi_vis2 so bootstrap more accurate.
            # but this should be ok for the average.


        elif phase_ref_method == 'time-shift':
            mircx_mystic_log.info('using time-shift method of oi-vis')
            # scipy shift does not support complex nuumbres.
            c_cpx_prime_real =  shift(np.real(c_cpx), (0,+ncoher,0,0), mode='constant', cval=0.0) + \
                                shift(np.real(c_cpx), (0,-ncoher,0,0), mode='constant', cval=0.0) 
            c_cpx_prime_imag  = shift(np.imag(c_cpx), (0,+ncoher,0,0), mode='constant', cval=0.0) + \
                                shift(np.imag(c_cpx), (0,-ncoher,0,0), mode='constant', cval=0.0) 
            c_cpx_prime = c_cpx_prime_real + 1.j*c_cpx_prime_imag

                           #if near end of ramp will zero out and only use one data set for reference.
            phi = np.mean (c_cpx_prime, axis=2, keepdims=True);
            c_cpx0 = c_cpx*np.exp (-1.j * np.angle (phi));
            c_cpx1= np.nanmean(c_cpx0*base_flag,axis=(0,1),keepdims=1) #dphase estimate.
            dphase_phasor = np.exp (-1.j * np.angle (c_cpx1)); # need to put the ref_ramp slopes back on.
            
            phi = np.mean (c_cpx_prime*dphase_phasor, axis=2, keepdims=True);
            c_cpx0 = c_cpx*np.exp (-1.j * np.angle (phi));            
            c_cpx1= np.nanmean(c_cpx0*base_flag,axis=(0,1),keepdims=1) #dphase estimate.
            dphase_phasor = np.exp (-1.j * np.angle (c_cpx1)); # need to put the ref_ramp slopes back on.

            phi = np.mean (c_cpx_prime*dphase_phasor, axis=2, keepdims=True);
            c_cpx0 = c_cpx*np.exp (-1.j * np.angle (phi));
            c_cpx1= np.nanmean(c_cpx0*base_flag,axis=(0,1),keepdims=1)
            dphase_phasor = np.exp (-1.j * np.angle (c_cpx1)); # need to put the ref_ramp slopes back on.

            phi = np.mean (c_cpx_prime*dphase_phasor, axis=2, keepdims=True);
            c_cpx = c_cpx*np.exp (-1.j * np.angle (phi));


        elif phase_ref_method == 'classic':
            phi = np.mean (c_cpx, axis=2, keepdims=True);
            c_cpx0 = c_cpx*np.exp (-1.j * np.angle (phi));
            #maybe should iterate here to first remove differential phase signature so the 'average' phase is not jumping about
            #due to the rapid fluctuations.
            c_cpx1= np.nanmean(c_cpx0*base_flag,axis=(0,1),keepdims=1)
            dphase_phasor = np.exp (-1.j * np.angle (c_cpx1)); # need to put the ref_ramp slopes back on.
            phi = np.mean (c_cpx*dphase_phasor, axis=2, keepdims=True);
            c_cpx0 = c_cpx*np.exp (-1.j * np.angle (phi));
            # do it again with the EVEN BETTER dphase estimate! 2 more times to converge!
            c_cpx1= np.nanmean(c_cpx0*base_flag,axis=(0,1),keepdims=1)
            dphase_phasor = np.exp (-1.j * np.angle (c_cpx1)); # need to put the ref_ramp slopes back on.
            phi = np.mean (c_cpx*dphase_phasor, axis=2, keepdims=True);
            c_cpx0 = c_cpx*np.exp (-1.j * np.angle (phi));
            c_cpx1= np.nanmean(c_cpx0*base_flag,axis=(0,1),keepdims=1)
            dphase_phasor = np.exp (-1.j * np.angle (c_cpx1)); # need to put the ref_ramp slopes back on.
            phi = np.mean (c_cpx*dphase_phasor, axis=2, keepdims=True);
            c_cpx = c_cpx*np.exp (-1.j * np.angle (phi));
        else: 
            raise ValueError("phase_ref_reference is unknown");


        #show results.
        mircx_mystic_log.info ('software fringe tracked: Compute OPD-scan Power with offset of %i frames'%ncs);
        #
        #base_scan0 = np.real (base_scan0[:,ncs:,:,:] * np.conj(base_scan0[0:,0:-ncs,:,:]))
        #base_flag[5:8,0,0,3]=np.nan
        #base_flag[12:15,0,0,8]=np.nan

        imshow_gdt(temp_base_scan,opds1=None,opds2=None, scale=scale_gd*1e6,title=output+"tracking",file=output+"_base_trend_tracking.png",flag=np.squeeze(base_flag));

        temp_opds_peak, temp_snr_peak, temp_base_scan_peaks    = fft_gdt(base_dft_original*gdt_phasor, bias_dft_original, nscan, \
            ncs=ncs, nincoh=nr,method='peak')
        #prob. should use base_flag....
        temp_base_scan = np.squeeze(temp_base_scan_peaks[ref_ramp,:,:,:])
        temp_ref_opds_raw = temp_opds_peak[ref_ramp,:,:,:] 
        temp_ref_snr_raw = temp_snr_peak[ref_ramp,:,:,:]
        temp_binary_flag, temp_seps, temp_opds,temp_snrs = binary_detector(temp_base_scan, temp_ref_snr_raw, temp_ref_opds_raw,\
             scale=scale_gd*1e6,title=headers.summary(hdr)+"\n OPD SCANS",file=output+'_OPD_AVG.png') # might find peaks around bright singles, will require tuning

        #temp_opds_peak2,     temp_snr_peak2,     temp_base_scan2    = fft_gdt(c_cpx, bias_dft_original, nscan, \
        #ncs=ncs, nincoh=nincoh,method='peak');
        #imshow_gdt(temp_base_scan2,opds1=None,opds2=None, scale=scale_gd*1e6,title=output+"tracking",file=output+"_base_trend_tracking_dphase_correction.png",flag=np.squeeze(base_flag));
        
        c_cpx = np.nanmean (c_cpx * base_flag, axis=1);

    elif vis_reference == 'spec-diff':
        mircx_mystic_log.info ('Compute VIS by taking spectral-differential');
        hdulist[0].header[HMP+'VIS_REF'] = ('SPEC-DIFF', 'vis reference');
        c_cpx  = base_dft_original * np.exp (2.j*np.pi * base_gd / lbd[None,None,:,None]);

        c_cpx = c_cpx[:,:,1:,:] * np.conj(c_cpx[:,:,:-1,:]);
        c_cpx = np.insert(c_cpx,np.size(c_cpx,2),np.nan,axis=2);
        c_cpx = np.nanmean (c_cpx * base_flag, axis=1);


    elif vis_reference == 'absolute':
        mircx_mystic_log.info ('Compute VIS without subtracting mean/gd');
        hdulist[0].header[HMP+'VIS_REF'] = ('ABSOLUTE', 'vis reference');
        c_cpx  = np.nanmean (c_cpx * base_flag, axis=1);

    else:
        raise ValueError("vis_reference is unknown");



        #JDM can we maintain closure phases? yes! but best to do it on the average quantities and might be fragile.
        # A) From closure phases with visphi and subgtract from action t3phi, lambda by lambda.
        # B) the difference in phase should be flat with lambda!
        # C) find anverage triangle phase difference.
        # D) solve for phases_base that will match the differences and apply them.
        # E) will this work at low SNR? the hope is but maybe not. lathough the harm is just adding in offsets.
        # F) could instead find self-consistent of cphases first. Then do differences. better for low snr t3phi?
        # are there any biases in the visphi like t3phi?
        # can I get access to the averaged quantities after calls to add_oivis. no. they have different flags since t3phi requies
        # all 3 baselines valid while oi-vis doesn't... though I am thinking of forcing this? but won't have to if I get this fix working
        #
        # I see that visphi doesn't calibrate well now since I don't track to zero opd for cals... Hmmmmm... 
        # I want to maintain the mean OPDs for later aclibration but I also wanted to calibrate the dphase a bit using the cals.
        # Hmmm..


        
    c_norm = photo[:,:,:,setup.base_beam ()];

    #c_norm = 4 * c_norm[:,:,:,:,0] * c_norm[:,:,:,:,1] * attenuation**2;
    #c_norm = np.sqrt (np.maximum (c_norm, 0));
    # average first over ramp, then multiple... 
    c_norm = 4 * np.nanmean(c_norm[:,:,:,:,0],axis=1,keepdims=1) * np.nanmean(c_norm[:,:,:,:,1],axis=1,keepdims=1) * attenuation**2;
    c_norm = np.sqrt (np.maximum (c_norm, 0));

    c_norm = np.nanmean (c_norm*base_flag, axis=1);
    
    #JDM due to time correlations, we should bin in time first before sending
    #oifits.add_vis.
    #nchunk=10
    #nr_smooth= nr//nchunk
    #mjd_ramp0=np.nanmean(mjd_ramp[index_bin],axis=1)
    #c_cpx0=np.nanmean(c_cpx[index_bin,:,:],axis=1)
    #c_norm0=np.nanmean(c_norm[index_bin,:,:],axis=1)

    #oifits.add_vis (hdulist, mjd_ramp, c_cpx, c_norm, output=output, y0=y0);
    oifits.add_vis (hdulist, mjd_ramp, c_cpx, c_norm, output=output, y0=y0,nchunk=nchunk);

    # Compute OI_T3
    if nbs > 0:
        mircx_mystic_log.info ('Compute Bispectrum with offset of %i frames'%nbs);
        t_cpx = (base_dft_original*base_flag)[:,:,:,setup.triplet_base()];
        t_cpx = t_cpx[:,2*nbs:,:,:,0] * t_cpx[:,nbs:-nbs,:,:,1] * np.conj (t_cpx[:,:-2*nbs,:,:,2]);
    else:
        mircx_mystic_log.info ('Compute Bispectrum without offset');
        t_cpx = (base_dft_original*base_flag)[:,:,:,setup.triplet_base()];
        t_cpx = t_cpx[:,:,:,:,0] * t_cpx[:,:,:,:,1] * np.conj (t_cpx[:,:,:,:,2]);

    # Load BBIAS_COEFF
    if coeff == []:
        mircx_mystic_log.info ('No BBIAS_COEFF file');
    else:
        f = coeff[0]['ORIGNAME'];
        mircx_mystic_log.info ('Load BBIAS_COEFF file %s'%f);
        bbias_coeff0 = pyfits.getdata (f, 'C0');
        bbias_coeff1 = pyfits.getdata (f, 'C1');
        bbias_coeff2 = pyfits.getdata (f, 'C2');

        # Get rid of bad channels in bbias
        mean0 = np.nanmean (bbias_coeff0);
        std0 = np.nanstd (bbias_coeff0);
        mean1 = np.nanmean (bbias_coeff1);
        std1 = np.nanstd (bbias_coeff1);
        mean2 = np.nanmean (bbias_coeff2);
        std2 = np.nanstd (bbias_coeff2);
        idx1 = abs(bbias_coeff0-mean0)>3*std0;
        idx2 = abs(bbias_coeff1-mean1)>3*std1;
        idx3 = abs(bbias_coeff2-mean2)>3*std2;
        idx = idx1+idx2+idx3;

        idx = np.where(idx==True);
        bbias_coeff0[idx] = np.nan;
        bbias_coeff1[idx] = np.nan;
        bbias_coeff2[idx] = np.nan;

        # Debias with C0
        mircx_mystic_log.info ('Debias with C0');
        t_cpx -= bbias_coeff0[None,None,:,None]/(ncoher*ncoher*ncoher);

        # Debias with C1
        mircx_mystic_log.info ('Debias with C1');
        Ntotal = photo.sum (axis=-1,keepdims=True);
        t_cpx -= bbias_coeff1[None,None,:,None] * Ntotal[:,:np.size(t_cpx,1),:,:]/(ncoher*ncoher);

        # Debias with C2
        mircx_mystic_log.info ('Debias with C2');
        xps = np.real (base_dft_original[:,ncs:,:,:] * np.conj(base_dft_original[:,0:-ncs,:,:]));
        xps0 = np.real (bias_dft_original[:,ncs:,:,:] * np.conj(bias_dft_original[:,0:-ncs,:,:]));
        xps -= np.mean (xps0, axis=-1, keepdims=True);
        Ptotal = xps[:,:,:,setup.triplet_base()].sum (axis=-1);
        t_cpx = t_cpx[:,:-1,:,:];
        t_cpx -= bbias_coeff2[None,None,:,None] * Ptotal[:,:,:,:]/ncoher;
    
    # Normalisation, FIXME: take care of the shift

    t_norm = photo[:,:,:,setup.triplet_beam()];
    # JDM: this original code multiplied before averaging over ramps. probably should do that to avoid low flux issues.
    #t_norm = t_norm[:,:,:,:,0] * t_norm[:,:,:,:,1] * t_norm[:,:,:,:,2];
    t_norm = 8.*np.nanmean(t_norm[:,:,:,:,0],axis=1,keepdims=True)*np.nanmean(t_norm[:,:,:,:,1],axis=1,keepdims=True)*np.nanmean(t_norm[:,:,:,:,2],axis=1,keepdims=True)

    t_att  = attenuation[:,:,:,setup.triplet_base()];
    t_att  = t_att[:,:,:,:,0] * t_att[:,:,:,:,1] * t_att[:,:,:,:,2];

    t_cpx = np.nanmean (t_cpx, axis=1);
    t_norm = np.nanmean (t_norm * t_att, axis=1);

    #index_bin=(np.arange(nr_smooth)[None,:])+(np.arange(nchunk)*nr_smooth)[:,None]
    #mjd_ramp0=np.nanmean(mjd_ramp[index_bin],axis=1)
    #t_cpx0=np.nanmean(t_cpx[index_bin,:,:],axis=1)
    #t_norm0=np.nanmean(t_norm[index_bin,:,:],axis=1)

    #oifits.add_t3 (hdulist, mjd_ramp, t_cpx, t_norm, output=output, y0=y0);
    oifits.add_t3 (hdulist, mjd_ramp, t_cpx, t_norm, output=output, y0=y0,nchunk=nchunk);

    # Figures
    mircx_mystic_log.info ('Figures');

    # Plot the 'opd-scan'

    #fig,axes = plt.subplots (5,3, sharex=True);
    #fig.suptitle (headers.summary (hdr));
    #plot.base_name (axes);
    ###plot.compact (axes);
    #for i,ax in enumerate (axes.flatten()): ax.imshow (base_scan[:,0,:,i].T,aspect='auto');
    #files.write (fig,output+'_base_trend.png');



    # Plot the trend
    base_scan  = np.fft.fftshift (np.fft.fft (base_dft_original, n=nscan, axis=2), axes=2); 
    bias_scan  = np.fft.fftshift (np.fft.fft (bias_dft_original, n=nscan, axis=2), axes=2);
        # Compute power in the scan, average the scan over the rampp
        # Therefore the coherent integration is the ramp, hardcoded.
    if ncs > 0:
        #log.info ('Compute OPD-scan Power with offset of %i frames'%ncs);
        base_scan = np.real (base_scan[:,ncs:,:,:] * np.conj(base_scan[:,0:-ncs,:,:]));
        bias_scan = np.real (bias_scan[:,ncs:,:,:] * np.conj(bias_scan[:,0:-ncs,:,:]));
        base_scan = np.mean (base_scan, axis=1, keepdims=True);
        bias_scan = np.mean (bias_scan, axis=1, keepdims=True);
    else:
        #log.info ('Compute OPD-scan Power without offset');
        base_scan = np.mean (np.abs(base_scan)**2,axis=1, keepdims=True);
        bias_scan = np.mean (np.abs(bias_scan)**2,axis=1, keepdims=True);

    # Incoherent integration over several ramp. default 5?
    base_scan = signal.uniform_filter (base_scan,(nincoh,0,0,0),mode='constant');
    bias_scan = signal.uniform_filter (bias_scan,(nincoh,0,0,0),mode='constant');

    # Observed noise, whose statistic is independent of averaging
    base_scan -= np.median (base_scan, axis=2, keepdims=True); 
    bias_scan -= np.median (bias_scan, axis=2, keepdims=True);
    base_powerbb_np = base_scan[:,:,int(nscan/2),:][:,:,None,:];
    base_powerbb    = np.max (base_scan, axis=2, keepdims=True);
    bias_powerbb    = np.mean (np.max (bias_scan, axis=2, keepdims=True), axis=-1, keepdims=True);

    #base_gd  = np.argmax (base_scan, axis=2)[:,:,None,:] - nscan//2 # index space  
    #Use our previous extraction.
    base_snr = base_powerbb / bias_powerbb;
    base_snr[~np.isfinite (base_snr)] = 0.0;
    #base_gd=base_gd*scale_gd

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
    d0 = np.mean (tracking_snr0,axis=(1,2));
    d1 = np.mean (tracking_snr,axis=(1,2));
    for b in range (15):
        ax = axes.flatten()[b];
        ax.axhline (snr_threshold,color='r', alpha=0.2);
        ax.plot (d1[:,b]);
        ax.plot (d0[:,b],'--', alpha=0.5);
        ax.plot(median_snr[:,0,0,b],'+')
        ax.set_yscale ('log');
        ax.set_xlabel ('Ramp #');
        #if b==3:
        #    log.info("showing median baseline %i"%b)
        #    log.info(snr_smooth_filter)
        #    log.info(tracking_snr[:,0,0,b])
        #    log.info(median_snr[:,0,0,b])
        #median_snr = median_filter(tracking_snr,size=(snr_smooth_filter,1,1,1),mode='nearest',origin=0)

    files.write (fig,output+'_snr.png');

    # GD
    fig,axes = plt.subplots (5,3, sharex=True);
    fig.suptitle (headers.summary (hdr));
    plot.base_name (axes);
    plot.compact (axes);
    d0 = np.mean (tracking_opds,axis=(1,2)) * 1e6;
    gd_range = scale_gd * nscan / 2;

    lim = 1.05 * gd_range * 1e6;
    for b in range (15):
        ax = axes.flatten()[b];
        # lim = 1.05 * np.max (np.abs (d0[:,b]));
        ax.plot (d0[:,b]);
        #ax.plot (d0[:,b],'--', alpha=0.5);
        #ax.plot (d2,'--')
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
        #ax.set_ylim (1.0);
        #ax.set_yscale ('log');
    ax.set_xlabel ('Ramp #');
    files.write (fig,output+'_flux.png');
    
    # Plot the fringe selection
    fig,axes = plt.subplots (5,3, sharex=True);
    fig.suptitle (headers.summary (hdr));
    plot.base_name (axes);
    plot.compact (axes);
    for b in range (15):
        axes.flatten()[b].plot (base_flag0[:,0,0,b], 'o', alpha=0.3, markersize=4);
        #axes.flatten()[b].plot (base_flag1[:,0,0,b], 'o', alpha=0.3, markersize=2);
        axes.flatten()[b].set_ylim (-.2,1.2);
    files.write (fig,output+'_selection.png');

    # SNR versus GD
    fig,axes = plt.subplots (5,3, sharex=True);
    fig.suptitle (headers.summary (hdr));
    plot.base_name (axes);
    plot.compact (axes);
    d0 = np.mean (base_gd,axis=(1,2))*1e6
    d1 = np.mean (tracking_snr0,axis=(1,2));
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
    mircx_mystic_log.info ('Create file');

    # First HDU
    hdulist[0].header['FILETYPE'] = filetype;
    hdulist[0].header[HMP+'RTS'] = os.path.basename (hdrs[0]['ORIGNAME'])[-30:];
    
    # Write file
    files.write (hdulist, output+'.fits');
            
    plt.close("all");
    return hdulist;

def gd_gravity_solver(opds, snrs, start_topd=None, softlength=2.,niter=200):
    # JDM finds that worsk much bettre with long softlengths, liek nscan/4
    #starting point!
    if np.any(start_topd == None):
        bestsnr_snr_jdm, bestsnr_gd_jdm, start_topd = signal.bootstrap_triangles_jdm (snrs[:,:,[0],:],opds[:,:,[0],:])
    
    nr0,nf0,np0,nt0=start_topd.shape
    new_tel_history=np.zeros((nr0,nf0,np0,nt0+1,niter))
    new_tel= np.concatenate( (np.zeros((nr0,nf0,np0,1)),start_topd),axis=3)
    ii=0
    for i_test in range(niter):
        tel_gravity, tel_potential, tel_opds = signal.get_gd_gravity(new_tel, snrs,opds,softlength=softlength)
        mircx_mystic_log.info("GRAVITY : step %i topds %.1f  %.1f %.1f %.1f %.1f %.1f  pots %.1f %.1f %.1f %.1f %.1f %.1f  "\
            %(i_test,new_tel[ii,0,0,0],new_tel[ii,0,0,1],new_tel[ii,0,0,2],new_tel[ii,0,0,3],new_tel[ii,0,0,4],new_tel[ii,0,0,5],\
                tel_potential[ii,0,0,0],tel_potential[ii,0,0,1],tel_potential[ii,0,0,2],tel_potential[ii,0,0,3],tel_potential[ii,0,0,4],tel_potential[ii,0,0,5]))
        diff_tel = tel_gravity - tel_gravity[:,:,:,0][:,:,:,None]
        diff_tel = diff_tel/np.amax(np.abs(diff_tel),axis=3,keepdims=True)
        new_tel = new_tel+diff_tel
        new_tel_history[:,:,:,:,i_test]=new_tel
    
    new_tels_avg = signal.uniform_filter(new_tel_history,(0,0,0,0,np.minimum(10,niter)),mode='reflect')[:,:,:,:,-1]
    tel_gravity, tel_potential, tel_opds = signal.get_gd_gravity(new_tels_avg, snrs,opds,softlength=softlength)

    return new_tels_avg, tel_opds


def gd_gravity2_solver(opds, snrs, tsep_vector, start_topd=None, softlength=2.,niter=200,nscan=None):
    # JDM finds that worsk much bettre with long softlengths, liek nscan/4
    #starting point!
    tsep6_vector=np.concatenate(([0.0], tsep_vector))
    if np.any(start_topd == None):
        bestsnr_snr_jdm, bestsnr_gd_jdm, start_topd = signal.bootstrap_triangles_jdm (snrs[:,:,[0],:],opds[:,:,[0],:])
    
    nr0,nf0,np0,nt0=start_topd.shape
    new_tel_history=np.zeros((nr0,nf0,np0,nt0+1,niter))
    new_tel= np.concatenate( (np.zeros((nr0,nf0,np0,1)),start_topd),axis=3)
    ii=0
    for i_test in range(niter):
        tel_gravity1, tel_potential1, tel_opds1 = signal.get_gd_gravity(new_tel, snrs,opds,softlength=softlength,nscan=nscan)
        tel_gravity2, tel_potential2, tel_opds2 = signal.get_gd_gravity(new_tel+tsep6_vector[None,None,None,:], snrs,opds,softlength=softlength,nscan=nscan)
        tel_gravity = tel_gravity1+tel_gravity2
        tel_potential = tel_potential1+tel_potential2
        #log.info("GRAVITY2: step %i topds %.1f  %.1f %.1f %.1f %.1f %.1f  pots %.1f %.1f %.1f %.1f %.1f %.1f  "\
        #    %(i_test,new_tel[ii,0,0,0],new_tel[ii,0,0,1],new_tel[ii,0,0,2],new_tel[ii,0,0,3],new_tel[ii,0,0,4],new_tel[ii,0,0,5],\
        #        tel_potential[ii,0,0,0],tel_potential[ii,0,0,1],tel_potential[ii,0,0,2],tel_potential[ii,0,0,3],tel_potential[ii,0,0,4],tel_potential[ii,0,0,5]))
        diff_tel = tel_gravity - tel_gravity[:,:,:,0][:,:,:,None]
        diff_tel = diff_tel/np.amax(np.abs(diff_tel),axis=3,keepdims=True)
        new_tel = new_tel+diff_tel
        new_tel_history[:,:,:,:,i_test]=new_tel
    
    new_tels_avg = signal.uniform_filter(new_tel_history,(0,0,0,0,np.minimum(10,niter)),mode='reflect')[:,:,:,:,-1]
    tel_gravity0, tel_potential0, tel_opds = signal.get_gd_gravity(new_tels_avg, snrs,opds,softlength=softlength,nscan=nscan)

    return new_tels_avg, tel_opds, tel_potential

def binary_detector(ref_scan, ref_snr_raw, ref_opds_raw,scale=1.0,title="Results of Binary Fits",file='default_binary_fits.png'):
    nscan,nb = ref_scan.shape
    gauss2_scan = ref_scan.copy() # not needed surely.
    gauss2_params=np.zeros( (nb,7))
    x=np.arange(nscan)-nscan//2
    ierrs = np.zeros(nb)
    for i_b in range(nb):
        # this is kludge-fest. probably won't be robust.. :(
        perr=.2
        aerr=2.
        y=np.squeeze(ref_scan[:,i_b]) 
        yerr=y*perr+aerr
        model_gauss = models.Gaussian1D() #+models.Polynomial1D(degree=0)
        model_gauss.amplitude=ref_snr_raw[0,0,0,i_b]
        model_gauss.mean=ref_opds_raw[0,0,0,i_b]
        model_gauss.mean.bounds=[-nscan/2,+nscan/2]
        model_gauss.stddev=2.0 
        model_gauss.stddev.min=1.
        model_gauss.mean.fixed =True
        fitter_gauss = fitting.LevMarLSQFitter()
        best_fit_gauss = fitter_gauss(model_gauss, x, y)
        diff= y-best_fit_gauss(x)
        # avoid choosing near the original gaussian. punish within 3sigma.
        diff=diff*(1.-np.exp(-.5*((x-best_fit_gauss.mean.value)/(best_fit_gauss.stddev*3))**2))
        peak2=np.argmax(diff)
        model_gauss2 = models.Gaussian1D()+models.Gaussian1D()+models.Polynomial1D(degree=0) #+models.Polynomial1D(degree=0)
        model_gauss2.amplitude_0=best_fit_gauss.amplitude
        model_gauss2.mean_0=best_fit_gauss.mean
        model_gauss2.mean_0.bounds=[-nscan/2,+nscan/2]
        model_gauss2.stddev_0= best_fit_gauss.stddev
        model_gauss2.stddev_0.min=1.
        model_gauss2.amplitude_1=np.max(diff)
        model_gauss2.mean_1=x[peak2]
        model_gauss2.mean_1.bounds=[-nscan/2,+nscan/2]
        model_gauss2.stddev_1= best_fit_gauss.stddev
        model_gauss2.stddev_1.min=1.
        model_gauss2.mean_0.fixed =True
        model_gauss2.mean_1.fixed =True
        best_fit_gauss2 = fitter_gauss(model_gauss2, x, y )
        model_gauss2.parameters =best_fit_gauss2.parameters
        model_gauss2.mean_0.fixed =False
        model_gauss2.mean_1.fixed =False
        model_gauss2.mean_0.bounds=[-nscan/2,+nscan/2]
        model_gauss2.mean_1.bounds=[-nscan/2,+nscan/2]
        model_gauss2.stddev_0.min=1.
        model_gauss2.stddev_1.min=1.
        best_fit_gauss2 = fitter_gauss(model_gauss2, x, y )
        ierr=fitter_gauss.fit_info['ierr']
        ierrs[i_b]=ierr
        if ierr >4:
            mircx_mystic_log.info("Detected Gauss2 Fit Failure baseline %i ierr: %i"%(i_b,ierr))
        gauss2_params[i_b,:]=best_fit_gauss2.parameters
        gauss2_scan[:,i_b]=best_fit_gauss2(x)
    #Binary detector!
    seps = (gauss2_params[:,4]-gauss2_params[:,1])
    nsig = np.abs((gauss2_params[:,1]-gauss2_params[:,4])/np.maximum(gauss2_params[:,2],gauss2_params[:,5]))
    wt2 = np.minimum(gauss2_params[:,3],gauss2_params[:,0])
    wt1 = np.maximum(gauss2_params[:,3],gauss2_params[:,0])
    peakratio = wt2/wt1
    # not a binary if a) too close b) two big raito c) second peak too weak.
    # based on looking at cals we have to ignore companions close to main peak with a formula. 
    peakratio_limit =   .025+.975*np.exp(-.5*(.5*nsig)**2) #.05+.95*np.exp(-.5*(.65*nsig)**2)
    binary_detector_flags = (peakratio > peakratio_limit)  & (wt2 > 5) & (nsig >3.0) & (ierrs <= 4) \
        & (np.abs(gauss2_params[:,1]) < nscan/2) & (np.abs(gauss2_params[:,4]) < nscan/2) # might find peaks around bright singles.
    mircx_mystic_log.info("b,peakratio,wt1,wt2,nsig,ierr  %i %f %f %f %f %i"%(11,peakratio[11],wt1[11],wt2[11],nsig[11],ierrs[11]))
   #peakratio_limit =   .025+.975*np.exp(-.5*(.625*nsig)**2) #.05+.95*np.exp(-.5*(.65*nsig)**2)
    #binary_detector_flags = (peakratio > peakratio_limit)  & (wt2 > 3) & (nsig >2.0) # might find peaks around bright singles.

    # wtmin = np.minimum( (peakratio > peakratio_limit), wt2/5.) # if <1 will flag 0. 
    seps=seps*binary_detector_flags
    snr1=gauss2_params[:,0]
    snr2=gauss2_params[:,3]*binary_detector_flags
    opd1=gauss2_params[:,1]
    opd2=gauss2_params[:,4]*binary_detector_flags
    snrs = np.concatenate( (snr1[:,None],snr2[:,None]),axis=1)
    opds = np.concatenate( (opd1[:,None],opd2[:,None]),axis=1)

    fig,axes = plt.subplots (5,3, sharex=True);
    fig.suptitle(title);
    fig.subplots_adjust (wspace=0.3, hspace=0.1);
    plot.base_name (axes);
    plot.compact (axes);

    x = (np.arange (nscan)-nscan//2)*scale;
    for b,ax in enumerate(axes.flatten()):
        ax.plot(x,ref_scan[:,b],'r-')
        ax.plot(x,gauss2_scan[:,b],'g-',alpha=.5)
        ax.axvline(opd1[b]*scale,color='b')
        if binary_detector_flags[b]:
            ax.axvline(opd2[b]*scale,color='b',linestyle='--')
            ax.text(0.01, 0.95, 'Binary\nDetected',\
                verticalalignment='top', horizontalalignment='left',\
                transform=ax.transAxes, color='red', fontsize=5)
            ax.plot(gauss2_params[b,1]*scale,gauss2_params[b,0],'ro')
            ax.plot(gauss2_params[b,4]*scale,gauss2_params[b,3],'rx')
            ax.set_xlim(-nscan/2*scale,+nscan/2*scale)
        else:
            ax.text(0.01, 0.95, 'No Binary\nDetected',\
                verticalalignment='top', horizontalalignment='left',\
                transform=ax.transAxes, color='green', fontsize=5)
        
    plt.setp(axes[-1, :], xlabel='OPD $\mu$m')
    plt.setp(axes[:, 0], ylabel='SNR')  
    files.write (fig,file,dpi=256)

    return binary_detector_flags, seps, opds,snrs
    # if binaries are wihin about 1/5 of the cohenence length, then we think it doesn't matter so much since we think
    # the brightest peak should be found more consistently....  mainly needed for wide binaries. Could add another check here
    # to only worry abougt WIDE binaries. 

        #plt.plot(x,y,'o')
        #plt.plot(x,best_fit_gauss(x))
        #plt.plot(x,best_fit_gauss2(x))
        #lt.show()

def fft_gdt(base_dft_original, bias_dft_original, nscan, ncs=1, nincoh= 1, method='peak'):
    '''workhorse routine to calc group delays from complex visibility data
        method: 'peak' uses peak of the oversampled scan fft.  best if we are sure only 1 peak.
                'centroid' = uses values abobve 10% of peak to do a weighted average. 
                           useful for BINARIES since peak are bistable if near unity flux ratio.
        ncs: number of frames to shift for cross spectrum
        nincoh: smoothing in ramp-space 
    '''
    nr,nf,ny,nb = base_dft_original.shape
    base_scan  = np.fft.fftshift (np.fft.fft (base_dft_original, n=nscan, axis=2), axes=2); 
    bias_scan  = np.fft.fftshift (np.fft.fft (bias_dft_original, n=nscan, axis=2), axes=2);
        # Compute power in the scan, average the scan over the rampp
        # Therefore the coherent integration is the ramp, hardcoded.
    if ncs > 0:
        #log.info ('Compute OPD-scan Power with offset of %i frames'%ncs);
        base_scan = np.real (base_scan[:,ncs:,:,:] * np.conj(base_scan[:,0:-ncs,:,:]));
        bias_scan = np.real (bias_scan[:,ncs:,:,:] * np.conj(bias_scan[:,0:-ncs,:,:]));
        base_scan = np.mean (base_scan, axis=1, keepdims=True);
        bias_scan = np.mean (bias_scan, axis=1, keepdims=True);
    else:
        #log.info ('Compute OPD-scan Power without offset');
        base_scan = np.mean (np.abs(base_scan)**2,axis=1, keepdims=True);
        bias_scan = np.mean (np.abs(bias_scan)**2,axis=1, keepdims=True);

    # Incoherent integration over several ramp. default 5?
    base_scan = signal.uniform_filter (base_scan,(nincoh,0,0,0),mode='constant');
    bias_scan = signal.uniform_filter (bias_scan,(nincoh,0,0,0),mode='constant');

    # Observed noise, whose statistic is independent of averaging
    base_scan -= np.median (base_scan, axis=2, keepdims=True); 
    bias_scan -= np.median (bias_scan, axis=2, keepdims=True);

    if method=='centroid':
        wts0=(np.arange(nscan)[None,None,:,None])-nscan//2
    # to avoid biases from imperfect background subtraction., lets set all values less than 0.1x peak as zero.after all looking for bright objects
    # that might mess up our gdt later that is based on brightest pixel.
        base_scan_threshold = np.where(base_scan < (0.1*np.amax(base_scan,axis=2,keepdims=True)), 0, base_scan)
        bias_scan_threshold = np.where(bias_scan < (0.1*np.amax(bias_scan,axis=2,keepdims=True)), 0, bias_scan)

        base_gd = np.nanmean(base_scan_threshold*wts0,axis=2,keepdims=True)/np.nanmean(base_scan,axis=2,keepdims=True)
        bias_powerbb    = np.mean (np.mean(bias_scan, axis=2, keepdims=True), axis=-1, keepdims=True);
        base_snr = np.nanmean(base_scan,axis=2,keepdims=True)/bias_powerbb
        base_snr[~np.isfinite (base_snr)] = 0.0;

    elif method=='peak':
        base_powerbb    = np.max (base_scan, axis=2, keepdims=True);
        bias_powerbb    = np.mean (np.max (bias_scan, axis=2, keepdims=True), axis=-1, keepdims=True);

        base_gd  = np.argmax (base_scan, axis=2)[:,:,None,:] - nscan//2 # index space  
        base_snr = base_powerbb / bias_powerbb;
        base_snr[~np.isfinite (base_snr)] = 0.0;
        base_scan /= bias_powerbb #normalize base_scan Duh.

    return base_gd, base_snr, base_scan

def fft_gdt2(sep_vector,base_dft_original, bias_dft_original, nscan, ncs=1, nincoh= 1, method='peak'):
    '''workhorse routine to calc group delays from complex visibility data
        method: 'peak' uses peak of the oversampled scan fft.  best if we are sure only 1 peak.
                'centroid' = uses values abobve 10% of peak to do a weighted average. 
                           useful for BINARIES since peak are bistable if near unity flux ratio.
        ncs: number of frames to shift for cross spectrum
        nincoh: smoothing in ramp-space 
    '''
    nr,nf,ny,nb = base_dft_original.shape
    base_scan  = np.fft.fftshift (np.fft.fft (base_dft_original, n=nscan, axis=2), axes=2); 
    bias_scan  = np.fft.fftshift (np.fft.fft (bias_dft_original, n=nscan, axis=2), axes=2);
        # Compute power in the scan, average the scan over the rampp
        # Therefore the coherent integration is the ramp, hardcoded.
    if ncs > 0:
        #log.info ('Compute OPD-scan Power with offset of %i frames'%ncs);
        base_scan = np.real (base_scan[:,ncs:,:,:] * np.conj(base_scan[:,0:-ncs,:,:]));
        bias_scan = np.real (bias_scan[:,ncs:,:,:] * np.conj(bias_scan[:,0:-ncs,:,:]));
        base_scan = np.mean (base_scan, axis=1, keepdims=True);
        bias_scan = np.mean (bias_scan, axis=1, keepdims=True);
    else:
        #log.info ('Compute OPD-scan Power without offset');
        base_scan = np.mean (np.abs(base_scan)**2,axis=1, keepdims=True);
        bias_scan = np.mean (np.abs(bias_scan)**2,axis=1, keepdims=True);

    # Incoherent integration over several ramp. default 5?
    base_scan = signal.uniform_filter (base_scan,(nincoh,0,0,0),mode='constant');
    bias_scan = signal.uniform_filter (bias_scan,(nincoh,0,0,0),mode='constant');

    # now do the double delta function
    for i_b in range(nb): # I"m sure theres a clever way to vectorize this...
        base_scan[:,:,:,i_b] = base_scan[:,:,:,i_b] + shift(base_scan[:,:,:,i_b],(0,0,-np.round(sep_vector[i_b]).astype(int)),mode='wrap')
        bias_scan[:,:,:,i_b] = bias_scan[:,:,:,i_b] + shift(bias_scan[:,:,:,i_b],(0,0,-np.round(sep_vector[i_b]).astype(int)),mode='wrap')


    # Observed noise, whose statistic is independent of averaging
    base_scan -= np.median (base_scan, axis=2, keepdims=True); 
    bias_scan -= np.median (bias_scan, axis=2, keepdims=True);

    if method=='centroid':
        wts0=(np.arange(nscan)[None,None,:,None])-nscan//2
    # to avoid biases from imperfect background subtraction., lets set all values less than 0.1x peak as zero.after all looking for bright objects
    # that might mess up our gdt later that is based on brightest pixel.
        base_scan_threshold = np.where(base_scan < (0.1*np.amax(base_scan,axis=2,keepdims=True)), 0, base_scan)
        bias_scan_threshold = np.where(bias_scan < (0.1*np.amax(bias_scan,axis=2,keepdims=True)), 0, bias_scan)

        base_gd = np.nanmean(base_scan_threshold*wts0,axis=2,keepdims=True)/np.nanmean(base_scan,axis=2,keepdims=True)
        bias_powerbb    = np.mean (np.mean(bias_scan, axis=2, keepdims=True), axis=-1, keepdims=True);
        base_snr = np.nanmean(base_scan,axis=2,keepdims=True)/bias_powerbb
        base_snr[~np.isfinite (base_snr)] = 0.0;

    elif method=='peak':
        base_powerbb    = np.max (base_scan, axis=2, keepdims=True);
        bias_powerbb    = np.mean (np.max (bias_scan, axis=2, keepdims=True), axis=-1, keepdims=True);

        base_gd  = np.argmax (base_scan, axis=2)[:,:,None,:] - nscan//2 # index space  
        base_snr = base_powerbb / bias_powerbb;
        base_snr[~np.isfinite (base_snr)] = 0.0;
        base_scan /= bias_powerbb #normalize base_scan Duh.

    return base_gd, base_snr, base_scan
    
def imshow_gdt(base_scan,opds1=None,opds2=None, scale=1.0,title="Cool GDT Plots",file="default_imshow_gdt.png",flag=None):
    bs=np.squeeze(base_scan)
    nr,nf,nscan,nb = base_scan.shape
    fig,axes = plt.subplots (5,3, sharex=True);
    fig.suptitle (title);
    plot.base_name (axes);
    plot.compact (axes);
    for i,ax in enumerate (axes.flatten()): 
        ax.imshow (np.sqrt(np.abs(bs[:,:,i].T)),aspect='auto',cmap=plt.cm.Reds,extent=[0,nr,-scale*nscan/2, +scale*nscan/2]);
        if np.any(opds1 != None):
            ax.plot(np.arange(nr),np.squeeze(opds1)[:,i].ravel()+3*scale,'b-',alpha=1,linewidth=.2)
            ax.plot(np.arange(nr),np.squeeze(opds1)[:,i].ravel()-3*scale,'b-',alpha=1,linewidth=.2)

        #ax.set_xlabel('Ramp')
        #ax.set_ylabel('OPD microns')
        #ax.set_xlabel('Ramp', fontsize=4)
        #ax.set_ylabel('$\mu$', size = 4)
        if np.any(opds2 != None):
                #ax.plot(np.arange(nr),np.squeeze(opds2)[:,i].ravel(),'g,',alpha=.2)
                ax.plot(np.arange(nr),np.squeeze(opds2)[:,i].ravel()+3*scale,'g-',alpha=1,linewidth=.3)
                ax.plot(np.arange(nr),np.squeeze(opds2)[:,i].ravel()-3*scale,'g-',alpha=1,linewidth=.3)
        # set axis common for clarity labels
        if np.any(flag != None):
                test=np.isnan(flag[:,i])
                xvect=np.argwhere(test == True)
                for xl0 in xvect:
                    ax.axvline (xl0, color='b', linestyle='--', alpha=0.3);
        plt.xlim(0,nr)
        plt.ylim(-scale*nscan/2, +scale*nscan/2)
    plt.setp(axes[-1, :], xlabel='Ramp')
    plt.setp(axes[:, 0], ylabel='OPD $\mu$m')  

        #ax.plot(np.arange(nr),np.linspace(-.5*coherence_length/2*1e6,+.5*coherence_length/2*1e6,num=nr),'o')
    files.write (fig,file ,dpi=256);


def binary_detector_amps(base_dft,bias_dft,photo,nscan,ncs=1,scale=1.0,title="Results of Autocorrelation Fits",file='default_binary_ac_fits.png'):
    nr,nf,ny,nb = base_dft.shape
    scale=np.abs(scale)

    photo_power = photo[:,:,:,setup.base_beam ()];
    photo_power = 4 * photo_power[:,:,:,:,0] * photo_power[:,:,:,:,1] 
    photo_power = np.nanmean (photo_power, axis=(0,1));
    photo_power /= np.mean(photo_power,axis=0,keepdims=1)

    fringe_power = np.mean(np.real(base_dft[:,ncs:,:,:]*np.conj(base_dft[:,0:-ncs,:,:])),axis=(0,1))
    bias_power =   np.mean(np.real(bias_dft[:,ncs:,:,:]*np.conj(bias_dft[:,0:-ncs,:,:])),axis=(0,1))
    mean_bp = np.mean(bias_power,axis=1,keepdims=1)
    fringe_power = fringe_power-mean_bp
    bias_power=bias_power-mean_bp
    
    envelopes = photo_power; (photo_power*np.mean(fringe_power,axis=0,keepdims=1))
    # now... envelopes hsould never go too low, but there could be problems with kappa matrix, or 
    # missing flux so. 
    if np.any(envelopes < .05):
        mircx_mystic_log.warning("At least on envelope has <5% of mean value. possible bad kappa matrix.")  
        envelopes = np.maximum(envelopes,.05)
    fringe_power = fringe_power / envelopes
    bias_power =   bias_power /  envelopes
    han=np.hanning(ny+2)[1:ny+1]
    fringe_power *= han[:,None] # np.hanning(ny+)[:,None]
    bias_power *=han[:,None] # np.hanning(ny)[:,None]
    #fringe_power = np.zeros((ny,nb))+han[:,None]
    fringe_peaks = np.abs(np.fft.rfft(fringe_power,axis=0,n=nscan))
    bias_peaks   = np.abs(np.fft.rfft(bias_power,axis=0,n=nscan))
    
    nz = np.mean(np.amax(bias_peaks,axis=0))
    fringe_peaks /= nz
    bias_peaks /=nz

    # fit binary
    ref_scan = fringe_peaks
    nscan2,nb = ref_scan.shape
    gauss2_scan = ref_scan.copy() # not needed surely.
    gauss2_params=np.zeros( (nb,7))
    gauss1_scan = ref_scan.copy()
    x=np.arange(nscan2)
    ierrs = np.zeros(nb)
    for i_b in range(nb):
        # this is kludge-fest. probably won't be robust.. :(
        perr=.2
        aerr=2.
        y=np.squeeze(ref_scan[:,i_b]) 
        yerr=y*perr+aerr
        model_gauss = models.Gaussian1D() #+models.Polynomial1D(degree=0)
        model_gauss.amplitude=y[0]
        model_gauss.mean=0.0
        model_gauss.mean.bounds=[0,nscan2]
        model_gauss.stddev=3.0 
        model_gauss.stddev.bounds=[2.5,5.0] # if outside this range, then wierd affect of binary.
        model_gauss.mean.fixed =True
        model_gauss.amplitude.fixed=True
        fitter_gauss = fitting.LevMarLSQFitter()
        best_fit_gauss = fitter_gauss(model_gauss, x[0:4], y[0:4])
        gauss1_scan[:,i_b]=best_fit_gauss(x)
        diff= y-best_fit_gauss(x)
        # avoid choosing near the original gaussian. punish within sigma.
        diff=diff*(1.-np.exp(-.5*((x-best_fit_gauss.mean.value)/(best_fit_gauss.stddev*2))**2))
        peak2=np.argmax(diff)
        model_gauss2 = models.Gaussian1D()+models.Gaussian1D()+models.Polynomial1D(degree=0) #+models.Polynomial1D(degree=0)
        model_gauss2.amplitude_0=y[0]
        model_gauss2.amplitude_0.fixed=True
        model_gauss2.mean_0=0.0
        model_gauss2.mean_0.bounds=[0,+nscan2]
        model_gauss2.stddev_0= 2.6
        model_gauss2.stddev_0.bounds=[2.5,5.0]
        model_gauss2.amplitude_1=np.max(diff)
        model_gauss2.amplitude_1.min=0.0
        model_gauss2.mean_1=np.maximum(x[peak2],2)
        model_gauss2.mean_1.bounds=[2,+nscan2]
        model_gauss2.stddev_1= 2.6 #start small.
        model_gauss2.stddev_1.bounds=[2.5,6.0]
        model_gauss2.mean_0.fixed =True
        model_gauss2.mean_1.fixed =True
        model_gauss2.c0_2.min=y.min()
        best_fit_gauss2 = fitter_gauss(model_gauss2, x, y )
        model_gauss2.parameters =best_fit_gauss2.parameters
        model_gauss2.amplitude_0.fixed=False
        model_gauss2.amplitude_0.min=0.0
        model_gauss2.amplitude_1.min=0.0
        model_gauss2.c0_2.min=y.min()
        model_gauss2.mean_0.fixed =True
        model_gauss2.mean_1.fixed =False
        model_gauss2.mean_0.bounds=[0,+nscan2]
        model_gauss2.mean_1.bounds=[2,+nscan2]
        model_gauss2.stddev_0.bounds=[2.5,5]
        model_gauss2.stddev_1.bounds=[2.5,6.]
        best_fit_gauss2 = fitter_gauss(model_gauss2, x, y )
        ierr=fitter_gauss.fit_info['ierr']
        ierrs[i_b]=ierr
        if ierr >4:
            mircx_mystic_log.info("Detected Gauss2 Fit Failure baseline %i ierr: %i"%(i_b,ierr))
        gauss2_params[i_b,:]=best_fit_gauss2.parameters
        gauss2_scan[:,i_b]=best_fit_gauss2(x)


        
    #Binary detector!
    seps = (gauss2_params[:,4]-gauss2_params[:,1])
    nsig = np.abs((gauss2_params[:,1]-gauss2_params[:,4])/np.maximum(gauss2_params[:,2],gauss2_params[:,5]))
    wt2 = np.minimum(gauss2_params[:,3],gauss2_params[:,0])
    wt1 = np.maximum(gauss2_params[:,3],gauss2_params[:,0])
    peakratio = wt2/wt1
    # not a binary if a) too close b) two big raito c) second peak too weak.
    # based on looking at cals we have to ignore companions close to main peak with a formula. 
    peakratio_limit =   .025+.975*np.exp(-.5*(.75*nsig)**2) #.05+.95*np.exp(-.5*(.65*nsig)**2)
    binary_detector_flags = (peakratio > peakratio_limit)  & (wt2 > 3) & (nsig >2.0) & (ierrs <= 4) \
        & (np.abs(gauss2_params[:,1]) < nscan/2) & (np.abs(gauss2_params[:,4]) < nscan/2) # might find peaks around bright singles.
    mircx_mystic_log.info("b,peakratio,wt1,wt2,nsig,ierr  %i %f %f %f %f %i"%(11,peakratio[11],wt1[11],wt2[11],nsig[11],ierrs[11]))
   #peakratio_limit =   .025+.975*np.exp(-.5*(.625*nsig)**2) #.05+.95*np.exp(-.5*(.65*nsig)**2)
    #binary_detector_flags = (peakratio > peakratio_limit)  & (wt2 > 3) & (nsig >2.0) # might find peaks around bright singles.

    # wtmin = np.minimum( (peakratio > peakratio_limit), wt2/5.) # if <1 will flag 0. 
    seps=seps*binary_detector_flags
    snr1=gauss2_params[:,0]
    snr2=gauss2_params[:,3]*binary_detector_flags
    opd1=gauss2_params[:,1]
    opd2=gauss2_params[:,4]*binary_detector_flags
    snrs = np.concatenate( (snr1[:,None],snr2[:,None]),axis=1)
    opds = np.concatenate( (opd1[:,None],opd2[:,None]),axis=1)

    fig,axes = plt.subplots (5,3, sharex=True);
    fig.suptitle(title);
    fig.subplots_adjust (wspace=0.3, hspace=0.1);
    plot.base_name (axes);
    plot.compact (axes);

    x = np.arange (nscan2)*scale;
    for b,ax in enumerate(axes.flatten()):
        ax.plot(x,ref_scan[:,b],'r-')
        ax.plot(x,gauss2_scan[:,b],'g-',alpha=.5)
        ax.axvline(opd1[b]*scale,color='b')
        if binary_detector_flags[b]:
            ax.axvline(opd2[b]*scale,color='b',linestyle='--')
            ax.text(0.2, 0.95, 'Binary\nDetected',\
                verticalalignment='top', horizontalalignment='left',\
                transform=ax.transAxes, color='red', fontsize=5)
            ax.plot(gauss2_params[b,1]*scale,gauss2_params[b,0],'ro')
            ax.plot(gauss2_params[b,4]*scale,gauss2_params[b,3],'rx')
            ax.set_xlim(0,+nscan2*scale)
        else:
            ax.text(0.2, 0.95, 'No Binary\nDetected',\
                verticalalignment='top', horizontalalignment='left',\
                transform=ax.transAxes, color='green', fontsize=5)
        
    plt.setp(axes[-1, :], xlabel='OPD $\mu$m')
    plt.setp(axes[:, 0], ylabel='SNR')  
    files.write (fig,file,dpi=256)

    # we can infver the ratios of peaks to get the underlying ratios.
    ratios = 2.*(snrs[:,1]/snrs[:,0])/(1.+snrs[:,1]/snrs[:,0])**2
    
    return binary_detector_flags, seps, snrs,ratios









