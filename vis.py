import numpy as np;

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
        log.info ('Data size'+str(fringe.shape));

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

        log.info ('Accumulate auto-correlation');
        data = fringe.reshape (nr*nf,ny,nx);
        for y in range(ny):
            for s in range(nr*nf):
                tmp = np.correlate (data[s,y,:],data[s,y,:],mode='full');
                correl[y,:] += tmp;

    # Get center of spectrum
    fyc,fyw = signal.getwidth (spectrum);
    log.info ('Expect center of spectrum (lbd0) on %f'%fyc);

    # Build expected wavelength (should use some
    # width of the spectra for lbd0)
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
    for y in range (ny):
        ax = axes.flatten()[y];
        ax.plot (freq,signal.psd_projection (res[y].x[0], freq, freq0, delta0, None));
        ax.plot (freq,psd[y,:]);
        ax.set_xlim (0,1.3*np.max(freq0));
        ax.set_ylim (0,1.1);
        ax.grid();
    axes.flatten()[0].set_title ('Observed PSD (orange) and scaled template (blue)');
    files.write (fig,output+'_psdmodel.png');

    # Effective wavelength
    fig,ax = plt.subplots ();
    ax.plot (yfit,lbdfit * 1e6,'o-');
    ax.plot (yfit,lbd * 1e6,'o-');
    ax.set_ylabel ('lbd (um)');
    ax.set_xlabel ('Detector line (python-def)');
    ax.set_title ('Guess calib. (orange) and Fitted calib, (blue)');
    ax.set_ylim (1.5,1.825);
    ax.grid();
    files.write (fig,output+'_lbd.png');

    # PSD
    fig,ax = plt.subplots (2,1);
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
        hdr['HIERARCH MIRC PRO PREPROC%i'%(npp+1,)] = h['ORIGNAME'];

    # Write file
    hdulist = pyfits.HDUList ([hdu0]);
    files.write (hdulist, output+'.fits');
    
    plt.close ("all");
    return hdulist;

def compute_rts (hdrs, bmaps, speccal, output='output_rts'):
    '''
    Compute the RTS
    '''
    elog = log.trace ('compute_rts');

    # Check inputs
    headers.check_input (hdrs,  required=1, maximum=1);
    headers.check_input (bmaps, required=1, maximum=6);
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

    # Get fringe and photo maps
    fringe_map, photo_map = extract_maps (hdr, bmaps);

    # Compute the expected position of lbd0    
    fcy = np.mean ([h[HMW+'FRINGE CENTERY'] for h in bmaps]) - hdr[HMW+'FRINGE STARTY'];
    
    # Build wavelength
    lbd0,dlbd = setup.lbd0 (hdr);
    
    # Load the wavelength table
    f = speccal[0]['ORIGNAME'];
    log.info ('Load SPEC_CAL file %s'%f);
    lbd = pyfits.getdata (f);

    # Define profile for optimal extraction of photometry
    # The same profile is used for all spectral channels
    profile = np.mean (photo_map, axis=3, keepdims=True);

    # Remove edge of the profile
    profile /= np.sum (profile,axis=-1, keepdims=True) + 1e-20;
    profile[profile<0.01] = 0.0;

    # Profile is normalised to be flux-conservative
    profile *= np.sum (profile,axis=-1, keepdims=True) / \
               (np.sum (profile**2,axis=-1, keepdims=True)+1e-20);
    
    # Profile extraction
    fig,axes = plt.subplots (3,2);
    for b in range(6):
        ax = axes.flatten()[b];
        val = np.mean (profile[b,:,:,:,:],axis=(0,1,2));
        ax.plot (val / (np.mean (val)+1e-20), label='profile');
        val = np.mean (photo[b,:,:,:,:],axis=(0,1,2));
        ax.plot (val / (np.mean (val)+1e-20), label='photo');
        ax.legend(); ax.grid();
    files.write (fig,output+'_profile.png');

    # Optimal extraction of photometry with profile
    log.info ('Extract photometry with profile');
    photo = np.sum (photo * profile, axis=-1);

    # Shift between photo and fringes in spectral direction
    log.info ('Register photometry to fringe');
    shifty = np.zeros (6);
    upper = np.sum (medfilt (fringe_map,[1,1,1,1,11]), axis=(1,2,4));
    lower = np.sum (medfilt (photo_map,[1,1,1,1,1]) * profile, axis=(1,2,4));
    for b in range (6):
        shifty[b] = register_translation (lower[b,:,None],upper[b,:,None],
                                              upsample_factor=100)[0][0];

    # Re-align photometry (all with the same)
    log.info ('Register photo');
    for b in range(6):
        photo[b,:,:,:] = subpix_shift (photo[b,:,:,:], [0,0,-shifty[b]]);

    # Build kappa from the bmaps.
    # kappa(nb,nr,nf,ny)
    log.info ('Build kappa-matrix with profile, filtering and registration');
    upper = np.sum (medfilt (fringe_map,[1,1,1,1,11]), axis=-1);
    lower = np.sum (medfilt (photo_map,[1,1,1,1,1]) * profile, axis=-1);
    for b in range(6):
        lower[b,:,:,:] = subpix_shift (lower[b,:,:,:], [0,0,-shifty[b]]);

    kappa = upper / (lower + 1e-20);

    # Filter the kappa-matrix and the data to avoid
    # craps on the edges. treshold(ny)
    log.info ('Compute threshold');
    threshold = np.mean (medfilt (fringe_map,[1,1,1,1,11]), axis = (0,1,2,-1));
    threshold /= np.max (medfilt (threshold,3)) + 1e-20;

    log.info ('Apply threshold:');
    threshold = threshold > 0.25;
    log.info (str(1*threshold));
    fringe[:,:,~threshold,:] = 0.0;
    photo[:,:,:,~threshold]  = 0.0;
    kappa[:,:,:,~threshold]  = 0.0;
        
    # Kappa-matrix as spectrum
    log.info ('Plot kappa');
    fig,axes = plt.subplots (3,2);
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
        ax.legend(); ax.grid();
        ax.set_ylim ((0.1,1.5));
        ax.set_ylabel ('normalized');
    files.write (fig,output+'_kappa.png');

    # Use a supplementary kappa-matrix if provided
    
    # Kappa-matrix
    fig,ax = plt.subplots (1);
    ax.imshow (np.mean (kappa,axis=(1,2)));
    files.write (fig,output+'_kappaimg.png');

    # kappa is defined so that photok is the
    # total number of adu in the fringe
    log.info ('Compute kappak');
    photok = photo * kappa;

    # Smooth photometry
    log.info ('Smooth photometry by sigma=2 frames');
    photok = gaussian_filter (photok,(0,0,2,0));

    # Temporal / Spectral averaging of photometry
    # to be discussed
    log.info ('Temporal / Spectral averaging of photometry');
    spectra  = np.mean (photok, axis=(1,2), keepdims=True);
    spectra /= np.sum (spectra, axis=3, keepdims=True) + 1e-20;
    injection = np.sum (photok, axis=3, keepdims=True);
    photok = spectra*injection;
    
    # Compute flux in fringes
    log.info ('Compute dc in fringes');
    fringe_map  = medfilt (fringe_map, [1,1,1,1,11]);
    fringe_map /= np.sum (fringe_map, axis=-1, keepdims=True) + 1e-20;
    cont = np.einsum ('Brfy,Brfyx->rfyx', photok, fringe_map);
        
    # QC about the fringe dc
    log.info ('Compute QC about dc');
    photodc_mean  = np.mean (cont,axis=(2,3));
    fringedc_mean = np.mean (fringe,axis=(2,3));
    hdr[HMQ+'DC MEAN'] = rep_nan (np.sum (fringedc_mean) / np.sum (photodc_mean));

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
    freqs = setup.base_freq (hdr);
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
    
    # Check dc
    fig,ax = plt.subplots ();
    ax.hist2d (photodc_mean.flatten(), fringedc_mean.flatten(),
               bins=40, norm=mcolors.LogNorm());
    plt.plot (photodc_mean.flatten(),np.poly1d(poly_dc)(photodc_mean.flatten()),'--');
    plt.plot (photodc_mean.flatten(),photodc_mean.flatten(),'-');
    ax.set_xlabel('fringe dc'); ax.set_ylabel('sum of photo');
    ax.grid();
    files.write (fig,output+'_dccorr.png');

    # Integrated spectra
    fig,ax = plt.subplots (2,1);
    val = np.mean (fringe,axis=(0,1,3));
    val /= np.max (medfilt (val,3), keepdims=True) + 1e-20;
    ax[0].plot (lbd*1e6,val,'--', label='fringes and photo');
    val = np.mean (photo, axis=(1,2));
    val /= np.max (medfilt (val,(1,3)), axis=1, keepdims=True) + 1e-20;
    ax[0].plot (lbd*1e6,val.T);
    ax[0].legend(); ax[0].grid();
    ax[0].set_ylabel ('normalized');
    
    val = np.mean (fringe,axis=(0,1,3));
    val /= np.max (medfilt (val,3), keepdims=True) + 1e-20;
    ax[1].plot (lbd*1e6,val,'--', label='fringes and photo * kappa * map');
    val = np.mean (photok, axis=(1,2));
    val /= np.max (medfilt (val,(1,3)), axis=1, keepdims=True) + 1e-20;
    ax[1].plot (lbd*1e6,val.T);
    ax[1].legend(); ax[1].grid();
    ax[1].set_ylabel ('normalized');
    ax[1].set_xlabel ('lbd (um)');
    files.write (fig,output+'_spectra.png');
    
    # Power densities
    fig,ax = plt.subplots (2,1);
    ax[0].imshow (np.mean (cf_upsd, axis=(0,1)));
    for f in ifreqs: ax[0].axvline (np.abs(f), color='k', linestyle='--');
    ax[1].plot (np.mean (cf_upsd, axis=(0,1))[int(ny/2),:]);
    ax[1].set_xlim (0,cf_upsd.shape[-1]);
    ax[1].grid();
    files.write (fig,output+'_psd.png');

    # File
    log.info ('Create file');

    # First HDU
    hdu0 = pyfits.PrimaryHDU ([]);
    hdu0.header = hdr;
    hdu0.header['FILETYPE'] = 'RTS';
    hdu0.header[HMP+'PREPROC'] = hdrs[0]['ORIGNAME'];

    # Set the input calibration file
    for bmap in bmaps:
        hdu0.header[HMP+bmap['FILETYPE']] = bmap['ORIGNAME'];

    # Set DFT of fringes, bias, photometry and lbd
    hdu1 = pyfits.ImageHDU (base_dft.real.astype('float32'));
    hdu1.header['EXTNAME'] = 'BASE_DFT_REAL';
    hdu1.header['BUNIT'] = ('adu','adu in the fringe envelope');
    
    hdu2 = pyfits.ImageHDU (base_dft.imag.astype('float32'));
    hdu2.header['EXTNAME'] = 'BASE_DFT_IMAG';
    hdu1.header['BUNIT'] = ('adu','adu in the fringe envelope');
    
    hdu3 = pyfits.ImageHDU (bias_dft.real.astype('float32'));
    hdu3.header['EXTNAME'] = 'BIAS_DFT_REAL';
    hdu1.header['BUNIT'] = ('adu','adu in the fringe envelope');
    
    hdu4 = pyfits.ImageHDU (bias_dft.imag.astype('float32'));
    hdu4.header['EXTNAME'] = 'BIAS_DFT_IMAG';
    hdu1.header['BUNIT'] = ('adu','adu in the fringe envelope');
    
    hdu5 = pyfits.ImageHDU (np.transpose (photo,axes=(1,2,3,0)).astype('float32'));
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
    base_dft  = pyfits.getdata (f, 'BASE_DFT_IMAG').astype(float) * 1.j;
    base_dft += pyfits.getdata (f, 'BASE_DFT_REAL').astype(float);
    bias_dft  = pyfits.getdata (f, 'BIAS_DFT_IMAG').astype(float) * 1.j;
    bias_dft += pyfits.getdata (f, 'BIAS_DFT_REAL').astype(float);
    photo     = pyfits.getdata (f, 'PHOTOMETRY').astype(float);
    lbd       = pyfits.getdata (f, 'WAVELENGTH').astype(float);

    # Dimensions
    nr,nf,ny,nb = base_dft.shape;
    log.info ('Data size nramp=%i nframe=%i nlbd=%i nbase=%i'%base_dft.shape);

    # Compute lbd0 and dlbd
    lbd0 = np.mean (lbd);
    dlbd = np.mean (np.diff (lbd));

    # Do coherent integration
    log.info ('Coherent integration over %.1f frames'%ncoher);
    hdr[HMP+'NFRAME_COHER'] = (ncoher,'nb. of frames integrated coherently');
    base_dft = signal.gaussian_filter_cpx (base_dft,(0,ncoher,0,0),mode='constant',truncate=2.0);
    bias_dft = signal.gaussian_filter_cpx (bias_dft,(0,ncoher,0,0),mode='constant',truncate=2.0);
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
    bbeam = setup.base_beam ();
    norm_power = photo[:,:,:,bbeam[:,0]] * photo[:,:,:,bbeam[:,1]];
    
    # QC for power
    log.info ('Compute QC for power');
    for b,name in enumerate (setup.base_name ()):
        val = np.mean (norm_power[:,:,int(ny/2),b], axis=(0,1));
        hdr[HMQ+'NORM'+name+' MEAN'] = (val,'Norm Power at lbd0');
        val = np.mean (base_power[:,:,int(ny/2),b], axis=(0,1));
        hdr[HMQ+'POWER'+name+' MEAN'] = (val,'Fringe Power at lbd0');
        val = np.std (base_power[:,:,int(ny/2),b], axis=(0,1));
        hdr[HMQ+'POWER'+name+' STD'] = (val,'Fringe Power at lbd0');
        val = np.mean (base_snrbb[:,:,:,b]);
        hdr[HMQ+'SNR'+name+' MEAN'] = (val,'Broad-band SNR');
        val = np.std (base_snrbb[:,:,:,b]);
        hdr[HMQ+'SNR'+name+' STD'] = (val,'Broad-band SNR');
    val = np.mean (bias_power[:,:,int(ny/2),:], axis=(0,1,-1));
    hdr[HMQ+'BIAS MEAN'] = (val,'Bias Power at lbd0');

    # Smooth SNR (should do the same for GD actually)
    log.info ('Smooth SNR over %.1f frames'%(ncoher*10));
    base_snr = gaussian_filter (base_snrbb, (0,ncoher*10,0,0));

    # Bootstrap over baseline
    base_snr *= base_snr > 3.0;
    base_snr, base_gd = signal.bootstrap (base_snr, base_gd);

    # Compute selection flag from averaged SNR over the ramp
    base_flag = 1.0 * (np.mean (base_snr,axis=1,keepdims=True) > 3.0);
    base_flag[base_flag == 0.0] = np.nan;

    # Compute the time stamp of each ramp
    time = np.ones (base_dft.shape[0]) * hdr['MJD-OBS'];
    
    # Create the file
    hdulist = oifits.create (hdr, lbd);

    # Compute OI_VIS2
    u_power = np.nanmean ((base_power - bias_power)*base_flag, axis=1);
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
    ax[0,0].imshow (np.mean (np.abs(base_dft)**2, axis=(0,1)));
    ax[0,1].imshow (np.mean (np.abs(bias_dft)**2, axis=(0,1)));
    ax[1,0].plot (np.mean (np.abs(base_dft)**2, axis=(0,1)).T);
    ax[1,1].plot (np.mean (np.abs(bias_dft)**2, axis=(0,1)).T);
    ax[0,0].set_title ('Fringe frequencies');
    ax[0,1].set_title ('Bias frequencies');
    files.write (fig,output+'_psd.png');
    
    # SNR, GD and FLAGs
    fig,ax = plt.subplots (3,1,sharex=True);
    ax[0].plot (np.log10 (np.mean (base_snrbb,axis=(1,2))));
    ax[0].grid(); ax[0].set_ylabel ('log10 (SNR_bb)');
    ax[1].plot (np.mean (base_gd,axis=(1,2)) * 1e6);
    ax[1].grid(); ax[1].set_ylabel ('gdelay (um)');
    ax[1].set_xlabel ('ramp');
    ax[2].imshow (np.mean (base_flag,axis=(1,2)).T);
    ax[2].grid(); ax[1].set_ylabel ('gdelay (um)');
    ax[2].set_xlabel ('flag');
    files.write (fig,output+'_snr_gd.png');

    # File
    log.info ('Create file');

    # First HDU
    hdulist[0].header['FILETYPE'] = 'VIS';
    hdulist[0].header[HMP+'RTS'] = hdrs[0]['ORIGNAME'];
    
    # Write file
    files.write (hdulist, output+'.fits');
            
    plt.close("all");
    return hdulist;
    
