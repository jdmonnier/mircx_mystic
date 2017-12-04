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
