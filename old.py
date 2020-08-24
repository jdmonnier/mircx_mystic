def compute_fringemap (hdrs,bkg,output='output_fringemap'):
    '''
    Find the location of the fringe on the detector.
    '''
    elog = log.trace ('compute_fringemap');
    
    # Check inputs
    check_hdrs_input (hdrs, required=1);
    check_hdrs_input (bkg, required=1);
    
    # Load files
    hdr,cube = files.load_raw (hdrs, coaddRamp=True);

    # Remove background
    remove_background (cube, bkg[0]);

    # Compute the sum
    log.info ('Compute mean over ramps and frames');
    fmean = np.mean (cube, axis=(0,1));

    # Figures
    fig,ax = plt.subplots();
    ax.imshow (fmean, interpolation='none');
    fig.savefig (output+'_sum.png');

    # Keep only fringes (supposedly smoothed in x)
    fmap = medfilt (fmean, [1,11]);
    
    # Get spectral limits of profile
    fcut = np.mean (fmap,axis=1);
    fcut /= np.max (fcut);

    idy_s = np.argmax (fcut>0.25);
    idy_e = len(fcut) - np.argmax(fcut[::-1]>0.25);
    
    log.info ('Found limit in spectral direction: %i:%i'%(idy_s,idy_e));

    # Get spatial limits of profile
    fcut = np.mean (fmap[idy_s:idy_e,:],axis=0);
    fcut /= np.max (fcut);
    
    idx_s = np.argmax (fcut>0.25);
    idx_e = len(fcut) - np.argmax(fcut[::-1]>0.25);
    
    log.info ('Found limit in spatial direction: %i:%i'%(idx_s,idx_e));

    # Cut
    fmeancut = fmean[idy_s:idy_e,idx_s:idx_e];

    # Figures
    fig,ax = plt.subplots();
    ax.imshow (fmeancut, interpolation='none');
    fig.savefig (output+'_cut.png');

    # Add QC parameters
    hdr.set ('HIERARCH MIRC QC FRINGE_WIN STARTX',idx_s,'[pix]');
    hdr.set ('HIERARCH MIRC QC FRINGE_WIN NX',idx_e-idx_s,'[pix]');
    hdr.set ('HIERARCH MIRC QC FRINGE_WIN STARTY',idy_s,'[pix]');
    hdr.set ('HIERARCH MIRC QC FRINGE_WIN NY',idy_e-idy_s,'[pix]');

    # Add QC parameters
    hdr.set ('HIERARCH MIRC QC EMPTY_WIN STARTX',200,'[pix]');
    hdr.set ('HIERARCH MIRC QC EMPTY_WIN NX',80,'[pix]');
    hdr.set ('HIERARCH MIRC QC EMPTY_WIN STARTY',idy_e+10,'[pix]');
    hdr.set ('HIERARCH MIRC QC EMPTY_WIN NY',15,'[pix]');

    # Check background subtraction in empty region
    check_empty_window (cube, hdr, hdr);
    
    # Create output HDU
    hdu1 = pyfits.PrimaryHDU (fmeancut);
    hdu1.header = hdr;

    # Update header
    hdu1.header['BZERO'] = 0;
    hdu1.header['FILETYPE'] = 'FRINGE_MAP';

    # Set files
    hdu1.header['HIERARCH MIRC PRO BACKGROUND'] = bkg[0]['ORIGNAME'];

    # Write output file
    hdulist = pyfits.HDUList (hdu1);
    files.write (hdulist, output+'.fits');
        
    plt.close("all");
    return fmean;


#
# Compute FRINGE_MAP
#

if argopt.fmap != 'FALSE':

    # Read all calibration products
    hdrs_calib = mrx.headers.loaddir (outputDir);

    # Group all FOREGROUND
    gps = mrx.headers.group (hdrs_raw, 'FOREGROUND', delta=dTime);
    overwrite = (argopt.fmap == 'OVERWRITE');

    # Compute all
    for i,gp in enumerate(gps):
        try:
            mrx.log.info ('Compute FRINGE_MAP {0} over {1} '.format(i+1,len(gps)));

            output = mrx.files.output (outputDir, gp[0], 'fmap');
            if os.path.exists (output+'.fits') and overwrite is False:
                mrx.log.info ('Product already exists');
                continue;
            
            mrx.log.setFile (output+'.log');
            
            bkg = mrx.headers.assoc (gp[0], hdrs_calib, 'BACKGROUND_MEAN',
                                     keys, which='closest', required=1);
            
            mrx.compute_fringemap (gp[0:mf], bkg, output=output);
            
        except Exception as exc:
            mrx.log.error ('Cannot compute FRINGE_MAP: '+str(exc));
            if argopt.debug == 'TRUE': raise;
        finally:
            mrx.log.closeFile ();


                
def compute_snr (hdrs,output=None,overwrite=True):

    nr,nf,nx,ny = fringe.shape;
    ny2 = int(ny/2);

    # Check inputs
    check_hdrs_input (hdrs, required=1);
    
    # Compute fft
    log.info ('Compute FFT');
    fringe_ft = fft (fringe, axis=-1);

    # Compute integrated PSD (no coherent integration)
    log.info ('Compute PSD');
    mean_psd = np.mean (np.abs(fringe_ft)**2, (0,1));

    # Figures
    (mean,med,std) = sigma_clipped_stats (mean_psd);
    fig,ax = plt.subplots(3,1);
    ax[0].imshow (mean_psd[:,0:ny2],vmin=med-5*std,vmax=med+5*std, interpolation='none');
    ax[1].plot (mean_psd[:,0:ny2].T);
    ax[2].plot (mean_psd[:,0:ny2].T); ax[2].set_ylim (med-3*std, med+3*std);
    fig.savefig (output+'_psd.png');

    # Compute cross-spectra
    log.info ('Compute CSP');
    csd = 0.5 * fringe_ft[:,0:nf-3:4] * np.conj (fringe_ft[:,2:nf-1:4]) + \
          0.5 * fringe_ft[:,1:nf-2:4] * np.conj (fringe_ft[:,3:nf-0:4]);
    mean_psd = np.mean (csd, (0,1)).real;

    # Figures
    (mean,med,std) = sigma_clipped_stats (mean_psd);
    fig,ax = plt.subplots(3,1);
    ax[0].imshow (mean_psd[:,0:ny2],vmin=med-5*std,vmax=med+5*std, interpolation='none');
    ax[1].plot (mean_psd[:,0:ny2].T);
    ax[2].plot (mean_psd[:,0:ny2].T); ax[2].set_ylim (med-3*std, med+3*std);
    fig.savefig (output+'_csd.png');
    
    
    # Create output HDU
    hdu1 = pyfits.PrimaryHDU (fringe);
    hdu1.header = hdr;

    # Update header
    hdu1.header['BZERO'] = 0;
    hdu1.header['BUNIT'] = 'ADU';
    hdu1.header['FILETYPE'] = 'SNR';

    # Set files
    hdu1.header['HIERARCH MIRC PRO BACKGROUND_MEAN'] = bkg['ORIGNAME'];

    # Write output file
    hdulist = pyfits.HDUList (hdu1);
    files.write (hdulist, output+'.fits');
        
    plt.close("all");
    return fringe;
        photos[beam,:,:,:,:] = subpix_shift (photo, [0,0,-shifty,0]);

def triplet_beam ():
    '''
    Return the MIRC beam numbering for each base
    beam[15,2]
    '''
    tmp = np.array ([[0,1,2],[0,1,3][0,1,4],[0,1,5],
                     [0,2,3],[0,2,4][0,2,5],
                     [0,3,4],[0,3,5],
                     [0,4,5],
                     [1,2,3],[1,2,4],[1,2,5],
                     
                         ]);
    return tmp;



#    # Load the sciences
#    hdusci = [];
#    for sci in scis:
#        log.info ('Load SCI %s'%(sci['ORIGNAME']));
#        hdusci.append (pyfits.open (sci['ORIGNAME']));
#
#    # Compute the TF for each science
#    hdutfs = tfs_time_weight (hdusci, hdutf, delta);
#
#    # Loop on science to calibrate and write results
#    for hdus,hduc in zip (hdusci,hdutfs):
#
#        # Calibrate
#        hdulist = tf_divide (hdus, hduc);
#    
#        # Write file
#        output = files.output (outputDir,sci,'viscal');
#        files.write (hdulist, output+'.fits');



def tfs_time_weight (hdus, hdutf, delta):
    '''
    Compute a Transfer function file with
    time weighted interpolation.

    delta is in [days]
    '''
    log.info ('Interpolate %i TF with time_weight'%len(hdutf));

    # Copy VIS_SCI to build VIS_TF
    hdutfs = [];
    for hdu in hdus:
        hdutfs.append (pyfits.HDUList([h.copy() for h in hdu]));
        hdutfs[-1][0].header['FILETYPE'] = 'VIS_SCI_TF';

    obs = [['OI_VIS2','VIS2DATA','VIS2ERR',False],
           ['OI_T3','T3AMP','T3AMPERR',False],
           ['OI_T3','T3PHI','T3PHIERR',True]];

    for o in obs:

        # Get calibration data
        mjd = np.array ([h[o[0]].data['MJD'] for h in hdutf]);
        val = np.array ([h[o[0]].data[o[1]] for h in hdutf]);
        err = np.array ([h[o[0]].data[o[2]] for h in hdutf]);

        # Set nan to flagged data and to data without error
        flg  = np.array([h[o[0]].data['FLAG'] for h in hdutf]);
        flg += ~np.isfinite (val) + ~np.isfinite (err) + (err<=0);
        val[flg] = np.nan;
        err[flg] = np.nan;

        # Replace by phasor
        if o[3] is True: val = phasor (val);

        # Loop on observations
        for hdu in hdutfs:
            
            # Compute the weighted mean
            mjd0 = hdu[o[0]].data['MJD'];
            weight = np.exp (-(mjd0[None,:,None]-mjd[:,:,None])**2/delta**2) / err**2;
            tf = np.nansum (val * weight, axis=0) / np.nansum (weight, axis=0);

            # Replace by phasor
            if o[3] is True: tf = np.angle (tf, deg=True);

            # Set data
            hdu[o[0]].data[o[1]] = tf;
            hdu[o[0]].data['FLAG'] += ~np.isfinite (tf);
            
    return hdutfs;


def linux_to_mjd (hdr, default=0.0):
    '''
    Return the MJD-OBS as computed by Linux time
    TIME_S + 1e-9 * TIME_US
    '''
    try:    
        return Time (hdr['TIME_S']+hdr['TIME_US']*1e-9,format='unix').mjd;
    except:
        return default;
    
def utc_to_mjd (hdr, default=0.0):
    '''
    Return the MJD-OBS as computed by CHARA GPS time
    DATE-OBS + UTC-OBS
    '''
    try:    
        return Time (hdr['DATE-OBS'] + 'T'+ hdr['UTC-OBS'], format='isot', scale='utc').mjd;
    except:
        return default;


'HIERARCH MIRC REF FRAME START'
'HIERARCH MIRC REF FRAME END'
'HIERARCH MIRC REF TIME_S START'
'HIERARCH MIRC REF TIME_S END'
'HIERARCH MIRC REF TIME_NS START'
'HIERARCH MIRC REF TIME_NS END'

##### below is some code fragments from JDM. will not compile.

# code to estimate mean phase based on differential phase then integrating.
# used in vis.py
dphase_method=1 # 0=differential 1=absolute
        if (dphase_method==0):
            log.info ('Using dphase method 0 (dphase/dlambda)');
            ## Remove differential phase:
            c_cpx1 = base_dft[:,:,1:,:] * np.conj(base_dft[:,:,:-1,:]);
            c_cpx1 = np.nanmean (c_cpx1, axis=(0,1));
            true_phi1 = np.angle(c_cpx1)
            true_phi1 = np.insert(true_phi1,0,np.nan,axis=0);
            new_phi1 = np.nancumsum(true_phi1,axis=0)
            ## set 0 phase in middle channel
            ref_phi1 = new_phi1[int(ny/2),:]
            new_phi1 -= np.repeat(ref_phi1[np.newaxis,:],ny,axis=0)
            new_phi1 = np.repeat(new_phi1[np.newaxis,:,:],nf,axis=0)
            new_phi1 = np.repeat(new_phi1[np.newaxis,:,:,:],nr,axis=0)
#######################
###JDM. Big chunk of code that tries to linearize the spectrograph K-values before finding gdt.
#did not really improve but the code to the fringe stretching, etc. might be useful one day.
# also from vis.py

     # ccpx shpae [scans?, frames, waves, baselines]
            # Compute FFT over the lbd direction, thus OPD-scan

            c_cpx_base = c_cpx[:,:,1:,:] * np.conj(c_cpx[:,:,:-1,:]);
            #c_cpx_base1d=np.nanmean (c_cpx_base, axis=(1),keepdims=True);
            c_cpx_base = np.nanmean (c_cpx_base, axis=(1,2),keepdims=True);
            base_amp =  np.abs(c_cpx_base)
            base_phi = np.angle(c_cpx_base)
            c_cpx_bias = bias_dft[:,:,1:,:] * np.conj(bias_dft[:,:,:-1,:]);
            c_cpx_bias = np.nanmean (c_cpx_bias, axis=(1,2), keepdims=True);
            bias_amp =  np.abs(c_cpx_bias)
            bias_phi = np.angle(c_cpx_bias)


            true_phi1 = np.angle(c_cpx1)

            # JDM: Use FFT Method but STRETCH first to account for non-linear spectrograph 

            c_cpx_stretch =c_cpx.copy()
            c_cpx_stretch_real = c_cpx_stretch.real
            c_cpx_stretch_imag = c_cpx_stretch.imag

            klambda = 1e-6/lbd;
            lindex=np.arange(ny*1.0)            
            #k_par= np.polyfit (lindex[1:-2], klambda[1:-2], deg=2);# ignore edge channels
            #kequal_deg2 = k_par[0]*lindex*lindex+k_par[1]*lindex+k_par[2]
            kequal = np.linspace(klambda.max()-1e-5,klambda.min()+1e-5,ny)
            linf = interp1d(klambda,c_cpx_stretch_real,axis=2,kind='linear',assume_sorted=False)
            new_real = linf(kequal)
            linf2 = interp1d(klambda,c_cpx_stretch_imag,axis=2,kind='linear',assume_sorted=False)
            new_imag = linf2(kequal)

            c_cpx_stretch = new_real + 1.j*new_imag

            base_scan  = np.fft.fftshift (np.fft.fft (c_cpx, n=nscan, axis=2), axes=2); 
            base_scan_stretch  = np.fft.fftshift (np.fft.fft (c_cpx_stretch, n=nscan, axis=2), axes=2); 

            bias_scan  = np.fft.fftshift (np.fft.fft (bias_dft, n=nscan, axis=2), axes=2);

    # JDM Note this method will likely fail over wide bandwidth (J+H) since lambda spacing
    # not uniform.  solution is slower than fft.
    #

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

            # Incoherent integration over several ramp. default 5?
            nincoher=1
            if nincoher > 0:
                log.info ('Incoherent integration over %i ramps'%nincoher);
                base_scan = signal.uniform_filter (base_scan,(nincoher,0,0,0),mode='constant');
                bias_scan = signal.uniform_filter (bias_scan,(nincoher,0,0,0),mode='constant');
            else:
                log.info ('Incoherent integration over 1 ramp');


            scale_gd = 1. / (lbd0**-1 - (lbd0+dlbd)**-1) / nscan;
            scale_gd_stretch = 1e-6/(kequal[1]-kequal[2])/nscan

            # Observed noise, whose statistic is independent of averaging
            base_scan -= np.median (base_scan, axis=2, keepdims=True); 
            bias_scan -= np.median (bias_scan, axis=2, keepdims=True);
            base_powerbb_np = base_scan[:,:,int(nscan/2),:][:,:,None,:];

            gdt_attenuation_correction = False # doesn't work so well and amplifies noise :(
            if (gdt_attenuation_correction == False):
                log.info ('gdt attenutation correction False');
                base_powerbb    = np.max (base_scan, axis=2, keepdims=True);
                bias_powerbb    = np.mean (np.max (bias_scan, axis=2, keepdims=True), axis=-1, keepdims=True);
            elif gdt_attenuation_correction == True:
                log.info ('gdt attenutation correction True');
                # apply factor to account for attenuation.i hope will help in case of near-equal binaries!!
                temp1d = (np.arange(nscan)-int(nscan/2))*scale_gd
                attenuation_factor1d = np.exp (-(np.pi * temp1d / (coherence_length*2))**2); #approx. should be sinc
                attenutation_factor2d = attenuation_factor1d[None,None,:,None]
                test_broadcast_rules=base_scan*attenutation_factor2d


                base_powerbb    = np.max (base_scan, axis=2, keepdims=True);
                bias_powerbb    = np.mean (np.max (bias_scan, axis=2, keepdims=True), axis=-1, keepdims=True);
            else:
                raise ValueError("gdt attenuation_correction is unknown");

            # Scale for gd in [um]
            log.info ('Compute GD test');
    

            base_gd  = (np.argmax (base_scan, axis=2)[:,:,None,:] - int(nscan/2)) * scale_gd;
            gd_range = scale_gd * nscan / 2;

            # Broad-band SNR
            log.info ('Compute SNR');
            base_snr = base_powerbb / bias_powerbb;

            #JDM version
            #base_snr=base_amp/bias_amp
            #base_gd = base_phi*6.86e-5

            base_snr_jdm, base_gd_jdm = signal.bootstrap_triangles_jdm (base_snr, base_gd);
            base_snr_jb, base_gd_jb = signal.bootstrap_triangles (base_snr, base_gd);

            base_snr=base_snr_jdm
            base_gd=base_gd_jdm

            base_snr[~np.isfinite (base_snr)] = 0.0;

#######################
#JDM. The following code was to correct for attenuation of gdt peaks near edg eof opd so that gdt 
# will consistenlty find the 'same' brightes tpeak even if it wanders to edge.
# maybe not a bad idea but didn't fix the data I was working with and pursuing a more 
# fanyc method...
# from vis.py

        gdt_attenuation_correction = False;
        if (gdt_attenuation_correction == False):
            log.info ('gdt attenutation correction False');
            base_powerbb    = np.max (base_scan, axis=2, keepdims=True);
            bias_powerbb    = np.mean (np.max (bias_scan, axis=2, keepdims=True), axis=-1, keepdims=True);
            base_gd  = (np.argmax (base_scan, axis=2)[:,:,None,:] - int(nscan/2)) * scale_gd;

        elif gdt_attenuation_correction == True:
            log.info ('gdt attenutation correction True');
            # apply factor to account for attenuation.i hope will help in case of near-equal binaries!!
            temp1d = (np.arange(nscan)-int(nscan/2))*scale_gd
            attenuation_factor1d = np.exp (-(np.pi * temp1d / (coherence_length*2.))**2); #approx. should be sinc
            attenuation_factor2d = attenuation_factor1d[None,None,:,None]
            base_scan /=attenuation_factor2d
            bias_scan /=attenuation_factor2d
            base_powerbb    = np.max (base_scan, axis=2, keepdims=True);
            bias_powerbb    = np.mean (np.max (bias_scan, axis=2, keepdims=True), axis=-1, keepdims=True);
            base_gd  = (np.argmax (base_scan, axis=2)[:,:,None,:] - int(nscan/2)) * scale_gd;

        else:
            raise ValueError("gdt attenuation_correction is unknown");

#######################
# more crazy jdm ideas about GDT
# actually good dieas but too slow to calculate.

    i_ramp =0  # will need to loop!
    num_combinations = 3**15
    counter = np.arange(num_combinations)
    test_gd = np.zeros( (num_combinations,nb))[:,None,None,:]
    test_snr = np.zeros( (num_combinations,nb))[:,None,None,:]
    for i_test in range(15):
        log.info ('Filling in arrays for baseline %i '%i_test);
        test_gd[:,0,0,i_test] = gd_key[ bestsnr_indices[i_ramp,0,(counter // 3**i_test) % 3, i_test]]
        test_snr[:,0,0,i_test] = bestsnr_snrs[ i_ramp,0,(counter // 3**i_test) %3 , i_test]
    log.info ('Starting mega bootstrap for  %i  combinations'%num_combinations);
    for i_test in range(num_combinations//1000):
        log.info ('bootstrap progress %i '%i_test);
        test_snr_jdm, test_gd_jdm, test_results_jdm = signal.bootstrap_triangles_jdm (test_snr[i_test*1000:(i_test+1)*1000,:,:,:], test_gd[i_test*1000:(i_test+1)*1000,:,:,:]);

    test_snr_jdm, test_gd_jdm, test_results_jdm = signal.bootstrap_triangles_jdm (test_snr, test_gd);
    log.info ('Finished! mega bootstrap for  %i  combinations'%num_combinations);

    log.info ('Stopped');




    for i_ramp in range(nr):
        log.info ('Minimizing opds for frame %i '%i_ramp);
        input_snr = np.squeeze(base_scan[i_ramp,0,:,:]/bias_powerbb[i_ramp] )
        input_snr = np.maximum(input_snr , snr_threshold)
        opds_trial=np.squeeze(results_jdm[i_ramp,0,0,:])*1e6
        eps_deriv = .5*scale_gd*1e6 # our curve is already interpolated !
        test = signal.gd_tracker(opds_trial,input_snr,gd_key)
        gd_jdm,snr_jdm = signal.get_gds(opds_trial,input_snr,gd_key)

        #temp=minimize(signal.gd_tracker,opds_trial,args=(input_snr,gd_key),method='L-BFGS-B',options={'eps':eps_deriv})
        temp=minimize(signal.gd_tracker,opds_trial,args=(input_snr,gd_key),method='Nelder-Mead',options={'disp':True})

        gd_jdm,snr_jdm = signal.get_gds(temp.x,input_snr,gd_key)
        topds[i_ramp,:]=temp.x
        new_opds[i_ramp,:]=gd_jdm
        new_snrs[i_ramp,:]=snr_jdm
        #opds_metric[i_ramp],,base_opds[i_ramp,:],base_snrs[i_ramp,:] = signal.gd_tracker(temp.x,input_snr,gd_key)
        #test_base_snr_jdm, test_base_gd_jdm, test_results_jdm = signal.bootstrap_triangles_jdm (base_snrs, base_opds);
 
        #temp
#######
# Code to try to figure which tels have fringes and compnainons:
# 
    primary_flag = np.squeeze(ref_snr_raw > (snr_threshold*3.)) # may not be correct way to detect fringes .
    pkeep=np.nonzero(primary_flag)
    skeep=np.nonzero(binary_flag)
    beaminfo=setup.base_beam()

    if pkeep == 0:
        tels_w_pri=np.zeros(6)
    else:
        tels_w_pri=np.bincount((primary_flag[pkeep,None]*beaminfo[[pkeep],:]).ravel(),None,6)
    if skeep == 0:
        tels_w_sec=np.zeros(6)
    else:
        tels_w_sec=np.bincount((binary_flag[skeep,None]*beaminfo[[skeep],:]).ravel(),None,6)

        tels_w_sec=np.zeros(6)
    beaminfo=setup.base_beam()
    test1=(primary_flag[:,None]*beaminfo).ravel()
    test2=(binary_flag[:,None]*beaminfo).ravel()

    tels_w_pri=np.bincount(test1,None,6)
    tels_w_sec=np.bincount(test2,None,6)

#################    
