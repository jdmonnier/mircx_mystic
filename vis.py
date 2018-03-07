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
from scipy.ndimage import gaussian_filter, uniform_filter;
from scipy.optimize import least_squares;
from scipy.ndimage.morphology import binary_closing, binary_opening;
from scipy.ndimage.morphology import binary_dilation;

from . import log, files, headers, setup, oifits, signal, plot;
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

def compute_speccal (hdrs, output='output_speccal', ncoher=3.0, nfreq=4096):
    '''
    Compute the SPEC_CAL
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
    yfit = hdr[HMW+'FRINGE STARTY'] + np.arange (ny);
    lbdfit = np.array([r.x[0]*lbd0 for r in res]);

    log.info ('Compute QC');
    
    # Compute quality factor
    projection = (1. - res[int(ny/2)].fun[0]) * norm[int(ny/2),0];
    hdr[HMQ+'QUALITY'] = (projection, 'quality of data');
    log.info (HMQ+'QUALITY = %e'%projection);

    log.info ('Figures');
    
    # Figures of PSD with model
    fig,axes = plt.subplots (ny,sharex=True);
    fig.suptitle ('Observed PSD (orange) and scaled template (blue)');
    for y in range (ny):
        ax = axes.flatten()[y];
        ax.plot (freq,signal.psd_projection (res[y].x[0], freq, freq0, delta0, None));
        ax.plot (freq,psd[y,:]);
        ax.set_xlim (0,1.3*np.max(freq0));
        ax.set_ylim (0,1.1);
    files.write (fig,output+'_psdmodel.png');

    # Effective wavelength
    fig,ax = plt.subplots ();
    fig.suptitle ('Guess calib. (orange) and Fitted calib, (blue)');
    ax.plot (yfit,lbdfit * 1e6,'o-');
    ax.plot (yfit,lbd * 1e6,'o-');
    ax.set_ylabel ('lbd (um)');
    ax.set_xlabel ('Detector line (python-def)');
    ax.set_ylim (1.45,1.8);
    files.write (fig,output+'_lbd.png');

    # PSD
    fig,ax = plt.subplots (2,1);
    fig.suptitle (headers.summary (hdr));
    ax[0].imshow (correl);
    ax[1].plot (psd[:,0:int(nfreq/2)].T);
    files.write (fig,output+'_psd.png');

    # File
    log.info ('Create file');

    # First HDU
    hdu0 = pyfits.PrimaryHDU (lbdfit);
    hdu0.header = hdr;
    hdu0.header['FILETYPE'] = 'SPEC_CAL';
    hdu0.header['BUNIT'] = 'm';

    # Save input files
    for h in hdrs:
        npp = len (hdr['*MIRC PRO PREPROC*']);
        hdr['HIERARCH MIRC PRO PREPROC%i'%(npp+1,)] = h['ORIGNAME'][-50:];

    # Write file
    hdulist = pyfits.HDUList ([hdu0]);
    files.write (hdulist, output+'.fits');
    
    plt.close ("all");
    return hdulist;
    
def compute_rts (hdrs, profiles, kappas, speccal, output='output_rts', psmooth=2):
    '''
    Compute the RTS
    '''
    elog = log.trace ('compute_rts');

    # Check inputs
    headers.check_input (hdrs,  required=1, maximum=1);
    headers.check_input (profiles, required=1, maximum=6);
    headers.check_input (kappas, required=1, maximum=6);
    headers.check_input (speccal, required=1, maximum=1);

    # Load DATA
    f = hdrs[0]['ORIGNAME'];
    log.info ('Load PREPROC file (copy) %s'%f);
    hdr = pyfits.getheader (f);
    fringe = pyfits.getdata (f).astype(float);
    photo  = pyfits.getdata (f, 'PHOTOMETRY_PREPROC').astype(float);
    nr,nf,ny,nx = fringe.shape

    # Some verbose
    log.info ('fringe.shape = %s'%str(fringe.shape));
    log.info ('mean(fringe) = %f adu/pix/frame'%np.mean(fringe,axis=(0,1,2,3)));

    # Saturation checks
    fsat  = 1.0 * np.sum (np.mean (np.sum (fringe,axis=1),axis=0)>40000) / (ny*nx);
    log.info (HMQ+'FRAC_SAT = %.3f'%rep_nan (fsat));
    hdr[HMQ+'FRAC_SAT'] = (rep_nan (fsat), 'fraction of saturated pixel');

    # Load the wavelength table
    f = speccal[0]['ORIGNAME'];
    log.info ('Load SPEC_CAL file %s'%f);
    lbd = pyfits.getdata (f);

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
        ax.plot (val / (np.mean (val)+1e-20), label='photo');
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

    # Plot photometry versus time
    log.info ('Plot photometry');
    fig,axes = plt.subplots (3,2,sharex=True);
    fig.suptitle (headers.summary (hdr));
    plot.compact (axes);
    for b in range (6):
        data = np.mean (photo[b,:,:,:], axis=(1,2));
        ax = axes.flatten()[b];
        ax.plot (data);
        ax.set_ylim (np.minimum (np.min (data), 0.0));
    ax.set_ylabel ('Xchan flux (adu)');
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
    log.info ('Build kappa-matrix with profile, filtering and registration');
    upper = np.sum (medfilt (fringe_kappa,[1,1,1,1,11]), axis=-1);
    lower = np.sum (medfilt (photo_kappa,[1,1,1,1,1]) * profile, axis=-1);
    for b in range(6):
        lower[b,:,:,:] = subpix_shift (lower[b,:,:,:], [0,0,-shifty[b]]);

    kappa = upper / (lower + 1e-20);

    # Filter the kappa-matrix and the data to avoid
    # craps on the edges. treshold(ny) is defined
    # based on fringe_map 
    log.info ('Compute threshold');
    threshold = np.mean (medfilt (fringe_map,[1,1,1,1,11]), axis = (0,1,2,-1));
    threshold /= np.max (medfilt (threshold,3)) + 1e-20;
    threshold = threshold > 0.25;

    log.info ('Apply threshold:');
    log.info (str(1*threshold));
    fringe[:,:,~threshold,:] = 0.0;
    photo[:,:,:,~threshold]  = 0.0;
    kappa[:,:,:,~threshold]  = 0.0;
        
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
    fringe_map  = medfilt (fringe_map, [1,1,1,1,11]);
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

    # Save integrated spectra before
    # subtracting the continuum
    fringe_spectra = np.mean (fringe,axis=(0,1,3));

    # Subtract continuum
    log.info ('Subtract dc');
    fringe -= cont;
    cont = [];

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
    # into integer pixels (max freq is 40)
    scale0 = 40. / np.abs (freqs * nx).max();
    ifreqs = np.round (freqs * scale0 * nx).astype(int);

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

    # DFT at fringe frequencies
    log.info ('Extract fringe frequency');
    base_dft  = cf[:,:,:,np.abs(ifreqs)];

    # Take complex conjugated for negative frequencies
    idx = ifreqs < 0.0;
    base_dft[:,:,:,idx] = np.conj(base_dft[:,:,:,idx]);

    # DFT at bias frequencies
    ibias = np.abs (ifreqs).max() + 4 + np.arange (24);
    bias_dft  = cf[:,:,:,ibias];

    # Compute unbiased PSD for plots (without coherent average
    # thus the bias is larger than in the base data).
    cf_upsd  = np.abs(cf[:,:,:,0:int(nx/2)])**2;
    cf_upsd -= np.mean (cf_upsd[:,:,:,ibias],axis=-1,keepdims=True);

    # Figures
    log.info ('Figures');

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
    ax[0].imshow (np.mean (cf_upsd, axis=(0,1)));
    for f in ifreqs: ax[0].axvline (np.abs(f), color='k', linestyle='--', alpha=0.5);
    ax[0].axvline (np.abs(ibias[0]), color='r', linestyle='--', alpha=0.3);
    ax[0].axvline (np.abs(ibias[-1]), color='r', linestyle='--', alpha=0.3);
    ax[1].plot (np.mean (cf_upsd, axis=(0,1))[int(ny/2),:]);
    ax[1].set_xlim (0,cf_upsd.shape[-1]);
    files.write (fig,output+'_psd.png');

    # File
    log.info ('Create file');

    # First HDU
    hdu0 = pyfits.PrimaryHDU ([]);
    hdu0.header = hdr;
    hdu0.header['FILETYPE'] = 'RTS';
    hdu0.header[HMP+'PREPROC'] = os.path.basename (hdrs[0]['ORIGNAME'])[-50:];

    # Set the input calibration file
    for pro in profiles:
        name = pro['FILETYPE'].split('_')[0]+'_PROFILE';
        hdu0.header[HMP+name] = os.path.basename (pro['ORIGNAME'])[-50:];

    # Set the input calibration file
    for kap in kappas:
        name = kap['FILETYPE'].split('_')[0]+'_KAPPA';
        hdu0.header[HMP+name] = os.path.basename (kap['ORIGNAME'])[-50:];

    # Set DFT of fringes, bias, photometry and lbd
    hdu1 = pyfits.ImageHDU (base_dft.real.astype('float32'));
    hdu1.header['EXTNAME'] = ('BASE_DFT_REAL','total flux in the fringe envelope');
    hdu1.header['BUNIT'] = 'adu';
    hdu1.header['SHAPE'] = '(nr,nf,ny,nb)';
    
    hdu2 = pyfits.ImageHDU (base_dft.imag.astype('float32'));
    hdu2.header['EXTNAME'] = ('BASE_DFT_IMAG','total flux in the fringe envelope');
    hdu2.header['BUNIT'] = 'adu'
    hdu2.header['SHAPE'] = '(nr,nf,ny,nb)';
    
    hdu3 = pyfits.ImageHDU (bias_dft.real.astype('float32'));
    hdu3.header['EXTNAME'] = ('BIAS_DFT_REAL','total flux in the fringe envelope');
    hdu3.header['BUNIT'] = 'adu';
    hdu3.header['SHAPE'] = '(nr,nf,ny,nbias)';
    
    hdu4 = pyfits.ImageHDU (bias_dft.imag.astype('float32'));
    hdu4.header['EXTNAME'] = ('BIAS_DFT_IMAG','total flux in the fringe envelope');
    hdu4.header['BUNIT'] = 'adu';
    hdu4.header['SHAPE'] = '(nr,nf,ny,nbias)';
    
    hdu5 = pyfits.ImageHDU (np.transpose (photok0,axes=(1,2,3,0)).astype('float32'));
    hdu5.header['EXTNAME'] = ('PHOTOMETRY','total flux in the fringe envelope');
    hdu5.header['BUNIT'] = 'adu'
    hdu5.header['SHAPE'] = '(nr,nf,ny,nt)';
    
    hdu6 = pyfits.ImageHDU (lbd);
    hdu6.header['EXTNAME'] = ('WAVELENGTH','effective wavelength');
    hdu6.header['BUNIT'] = 'm';
    hdu6.header['SHAPE'] = '(ny)';

    hdu7 = pyfits.ImageHDU (np.transpose (kappa,axes=(1,2,3,0)));
    hdu7.header['EXTNAME'] = ('KAPPA','ratio total_fringe/total_photo');
    hdu7.header['SHAPE'] = '(nr,nf,ny,nt)';

    # Write file
    hdulist = pyfits.HDUList ([hdu0,hdu1,hdu2,hdu3,hdu4,hdu5,hdu6,hdu7]);
    files.write (hdulist, output+'.fits');

    plt.close("all");
    return hdulist;

def compute_vis (hdrs, output='output_vis', ncoher=3.0, threshold=3.0, avgphot=True):
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
    base_dft  = pyfits.getdata (f, 'BASE_DFT_IMAG').astype(float) * 1.j;
    base_dft += pyfits.getdata (f, 'BASE_DFT_REAL').astype(float);
    bias_dft  = pyfits.getdata (f, 'BIAS_DFT_IMAG').astype(float) * 1.j;
    bias_dft += pyfits.getdata (f, 'BIAS_DFT_REAL').astype(float);
    photo     = pyfits.getdata (f, 'PHOTOMETRY').astype(float);
    lbd       = pyfits.getdata (f, 'WAVELENGTH').astype(float);

    # Dimensions
    nr,nf,ny,nb = base_dft.shape;
    log.info ('Data size: '+str(base_dft.shape));

    # Compute lbd0 and dlbd
    lbd0 = np.mean (lbd[4:-4]);
    dlbd = np.mean (np.diff (lbd[4:-4]));

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
    else:
        log.info ('No spectro-temporal averaging of photometry');
        hdr[HMP+'AVGPHOT'] = (False,'spectro-temporal averaging of photometry');
        
    # Do coherent integration
    log.info ('Coherent integration over %.1f frames'%ncoher);
    hdr[HMP+'NFRAME_COHER'] = (ncoher,'nb. of frames integrated coherently');
    base_dft = signal.gaussian_filter_cpx (base_dft,(0,ncoher,0,0),mode='constant',truncate=2.0);
    bias_dft = signal.gaussian_filter_cpx (bias_dft,(0,ncoher,0,0),mode='constant',truncate=2.0);

    # Smooth photometry over the same amount (FIXME: be be discussed)
    log.info ('Smoothing of photometry over %.1f frames'%ncoher);
    photo = gaussian_filter (photo,(0,ncoher,0,0),mode='constant',truncate=2.0);

    # Temporal/Spectral averaging of photometry
    # to be discussed

    log.info ('Compute 2d FFT');

    # Compute FFT for display, average over frames in ramp
    nscan = 128;
    base_fft  = np.fft.fftshift (np.fft.fft (base_dft, n=nscan, axis=2), axes=2);
    base_scan = np.mean (np.abs(base_fft),axis=1, keepdims=True);

    bias_fft  = np.fft.fftshift (np.fft.fft (bias_dft, n=nscan, axis=2), axes=2);
    bias_scan = np.mean (np.abs(bias_fft),axis=1, keepdims=True);

    # Plot the 'opd-scan'
    fig,axes = plt.subplots (5,3, sharex=True);
    fig.suptitle (headers.summary (hdr));
    plot.base_name (axes);
    plot.compact (axes);
    for i,ax in enumerate (axes.flatten()): ax.imshow (base_scan[:,0,:,i].T);
    files.write (fig,output+'_base_trend.png');

    # Plot the trend
    fig,axes = plt.subplots (5,3, sharex=True);
    fig.suptitle (headers.summary (hdr));
    plot.base_name (axes);
    plot.compact (axes);
    for i,ax in enumerate (axes.flatten()): ax.imshow (bias_scan[:,0,:,i].T);
    files.write (fig,output+'_bias_trend.png');

    log.info ('Compute GD');

    # Compute SNR and GD from this opd-scan
    base_powerbb    = np.max (base_scan, axis=2, keepdims=True);
    base_powerbb_np = base_scan[:,:,int(nscan/2),:][:,:,None,:];
    # bias_powerbb    = np.median (base_scan, axis=2, keepdims=True);

    # TODO
    log.warning ('TODO: get noise power on bias frequencies');
    bias_powerbb = np.mean (np.max (bias_scan, axis=2, keepdims=True), axis=-1, keepdims=True);
    
    # Scale for gd
    scale_gd = 1. / (lbd0**-1 - (lbd0+dlbd)**-1) / nscan;
    base_gd  = (np.argmax (base_scan, axis=2)[:,:,None,:] - int(nscan/2)) * scale_gd;
    gd_range = scale_gd * nscan / 2;

#    # Compute bandwidth
#    log.info ('Compute GD');
#    scale_gd = 1. / ( (1./(lbd0) - 1./(lbd0+dlbd)) * (2*np.pi));
#
#    # Compute group-delay in [m] and broad-band power
#    base_gd  = np.angle (np.sum (base_dft[:,:,1:,:] * np.conj (base_dft[:,:,:-1,:]), axis=2, keepdims=True));
#    phasor = np.exp (-1.j * np.arange(17)[None,None,:,None] * base_gd);
#
#    base_gd *= scale_gd;
#    # phasor = np.exp (2.j*np.pi * base_gd / lbd[None,None,:,None]);
#    base_powerbb    = np.abs (np.sum (base_dft * phasor, axis=2, keepdims=True))**2;
#
#    log.warning ('Test with no phasor');
#    base_powerbb_np = np.abs (np.sum (base_dft, axis=2, keepdims=True))**2;
#
#    # Compute group-delay and broad-band power for bias
#    bias_gd  = np.angle (np.sum (bias_dft[:,:,1:,:] * np.conj (bias_dft[:,:,:-1,:]), axis=2,keepdims=True));
#    phasor = np.exp (-1.j * np.arange(17)[None,None,:,None] * bias_gd);
#
#    bias_gd *= scale_gd;
#    # phasor = np.exp (2.j*np.pi * bias_gd / lbd[None,None,:,None]);
#    bias_powerbb = np.mean (np.abs (np.sum (bias_dft * phasor, axis=2,keepdims=True))**2,axis=-1,keepdims=True);
    
    # Broad-band SNR
    base_snr = base_powerbb / bias_powerbb;
    base_snr[~np.isfinite (base_snr)] = 0.0;

    # Compute power per spectral channels
    base_power = np.abs (base_dft)**2;
    bias_power = np.abs (bias_dft)**2;
    bias_power_mean = np.mean (bias_power,axis=-1,keepdims=True);

    # Compute norm power
    log.info ('Compute norm power');
    bbeam = setup.base_beam ();
    norm_power = 4. * photo[:,:,:,bbeam[:,0]] * photo[:,:,:,bbeam[:,1]];
    
    # QC for power
    log.info ('Compute QC for beam');
    for t in range(6):
        val = np.mean (photo[:,:,int(ny/2),t], axis=(0,1));
        hdr[HMQ+'FLUX%i MEAN'%t] = (val,'flux at lbd0');
        
    # QC for power
    log.info ('Compute QC for base');
    for b,name in enumerate (setup.base_name ()):
        val = rep_nan (np.mean (norm_power[:,:,int(ny/2),b], axis=(0,1)));
        hdr[HMQ+'NORM'+name+' MEAN'] = (val,'Norm Power at lbd0');
        val = rep_nan (np.mean (base_power[:,:,int(ny/2),b], axis=(0,1)));
        hdr[HMQ+'POWER'+name+' MEAN'] = (val,'Fringe Power at lbd0');
        val = rep_nan (np.std (base_power[:,:,int(ny/2),b], axis=(0,1)));
        hdr[HMQ+'POWER'+name+' STD'] = (val,'Fringe Power at lbd0');
        val = rep_nan (np.mean (base_snr[:,:,:,b]));
        hdr[HMQ+'SNR'+name+' MEAN'] = (val,'Broad-band SNR');
        val = rep_nan (np.std (base_snr[:,:,:,b]));
        hdr[HMQ+'SNR'+name+' STD'] = (val,'Broad-band SNR');

    # QC for bias
    log.info ('Compute QC for bias');
    qc_power = np.mean (bias_power[:,:,int(ny/2),:], axis=(0,1));
    hdr[HMQ+'BIASMEAN MEAN'] = (np.mean (qc_power),'Bias Power at lbd0');
    hdr[HMQ+'BIASMEAN STD'] = (np.std (qc_power),'Bias Power at lbd0');
    hdr[HMQ+'BIASMEAN MED'] = (np.median (qc_power),'Bias Power at lbd0');

    # Smooth SNR along the ramp
    log.info ('Smooth SNR over one ramp');
    base_snr = np.mean (base_snr,axis=1,keepdims=True);
    base_gd  = np.mean (base_gd,axis=1,keepdims=True);

    # Bootstrap over baseline. Maybe the GD should be
    # boostraped and averaged as a phasor
    base_snr0 = base_snr.copy ();
    base_gd0 = base_gd.copy ();
    base_snr, base_gd = signal.bootstrap_triangles (base_snr, base_gd);

    # Compute selection flag from averaged SNR over the ramp
    log.info ('SNR selection > %.2f'%threshold);
    hdr[HMQ+'SNR_THRESHOLD'] = (threshold, 'to accept fringe');
    base_flag = 1. * (base_snr > threshold);

    # TODO: Add selection on mean flux in ramp, gd...

    # Morphological operation
    log.info ('Closing/opening of selection');
    structure = np.ones ((2,1,1,1));

    base_flag0 = base_flag.copy ();
    base_flag = 1.0 * binary_closing (base_flag, structure=structure);
    base_flag = 1.0 * binary_opening (base_flag, structure=structure);

    # Plot this fringe selection
    fig,axes = plt.subplots (2,1);
    axes[0].imshow (base_flag0[:,0,0,:].T);
    axes[1].imshow (base_flag[:,0,0,:].T);
    files.write (fig,output+'_morpho.png');

    # Replace 0 by nan to perform nanmean and nanstd
    base_flag[base_flag == 0.0] = np.nan;

    # Compute the time stamp of each ramp
    time = np.ones (base_dft.shape[0]) * hdr['MJD-OBS'];
    
    # Create the file
    hdulist = oifits.create (hdr, lbd);

    # Compute OI_VIS2
    u_power = np.nanmean ((base_power - bias_power_mean)*base_flag, axis=1);
    l_power = np.nanmean (norm_power*base_flag, axis=1);
    
    oifits.add_vis2 (hdulist, time, u_power, l_power, output=output);

    # Compute OI_T3
    t_cpx = (base_dft*base_flag)[:,:,:,setup.triplet_base()];
    t_cpx = np.nanmean (t_cpx[:,:,:,:,0] * t_cpx[:,:,:,:,1] * np.conj (t_cpx[:,:,:,:,2]), axis=1);
    t_norm = photo[:,:,:,setup.triplet_beam()];
    t_norm = np.nanmean (t_norm[:,:,:,:,0] * t_norm[:,:,:,:,1] * t_norm[:,:,:,:,2], axis=1);

    oifits.add_t3 (hdulist, time, t_cpx, t_norm, output=output);

    # Figures
    log.info ('Figures');

    # Pseudo PSD
    fig,ax = plt.subplots (2,2, sharey='row',sharex='col');
    fig.suptitle (headers.summary (hdr));
    ax[0,0].imshow (np.mean (np.abs(base_dft)**2, axis=(0,1)));
    ax[0,1].imshow (np.mean (np.abs(bias_dft)**2, axis=(0,1)));
    ax[1,0].plot (np.mean (np.abs(base_dft)**2, axis=(0,1)).T);
    ax[1,1].plot (np.mean (np.abs(bias_dft)**2, axis=(0,1)).T);
    ax[0,0].set_title ('Fringe frequencies');
    ax[0,1].set_title ('Bias frequencies');
    files.write (fig,output+'_psd.png');
    
    # SNR
    fig,axes = plt.subplots (5,3, sharex=True);
    fig.suptitle (headers.summary (hdr));
    plot.base_name (axes);
    plot.compact (axes);
    d0 = np.log10 (np.mean (base_snr0,axis=(1,2)));
    d1 = np.log10 (np.mean (base_snr,axis=(1,2)));
    for b in range (15): axes.flatten()[b].axhline (np.log10(threshold),color='r', alpha=0.2);
    for b in range (15): axes.flatten()[b].plot (d1[:,b]);
    for b in range (15): axes.flatten()[b].plot (d0[:,b],'--', alpha=0.5);
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
        # lim = 1.05 * np.max (np.abs (d0[:,b]));
        axes.flatten()[b].plot (d1[:,b]);
        axes.flatten()[b].plot (d0[:,b],'--', alpha=0.5);
        axes.flatten()[b].set_ylim (-lim,+lim);
    files.write (fig,output+'_gd.png');

    # SNR versus GD
    fig,axes = plt.subplots (5,3, sharex=True);
    fig.suptitle (headers.summary (hdr));
    plot.base_name (axes);
    plot.compact (axes);
    d0 = np.mean (base_gd,axis=(1,2)) * 1e6;
    d1 = np.mean (base_snr0,axis=(1,2));
    lim = 1.05 * gd_range * 1e6;
    for b in range (15):
        axes.flatten()[b].axhline (threshold,color='r', alpha=0.2);
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
    hdulist[0].header['FILETYPE'] = 'VIS';
    hdulist[0].header[HMP+'RTS'] = os.path.basename (hdrs[0]['ORIGNAME'])[-50:];
    
    # Write file
    files.write (hdulist, output+'.fits');
            
    plt.close("all");
    return hdulist;
    
