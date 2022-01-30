import numpy as np;
import os;
import copy;

from astropy.io import fits as pyfits

import matplotlib.pyplot as plt;
import matplotlib.colors as mcolors;

from . import files, headers, mircx_mystic_log, setup, oifits, signal, plot;
from .headers import HM, HMQ, HMP, HMW, rep_nan;

from .setup import visparam;

def phasor (val):
    return np.exp (2.j * np.pi * val / 360);

def wrap (val):
    return np.angle (np.exp (2.j * np.pi * val / 360), deg=True);

def get_spfreq (hdulist,name):
    '''
    Return the spatial frequency B/lbd in [rad-1]
    '''
    lbd = hdulist['OI_WAVELENGTH'].data['EFF_WAVE'];

    if name == 'OI_VIS':
        u = hdulist['OI_VIS'].data['UCOORD'];
        v = hdulist['OI_VIS'].data['VCOORD'];
        return np.sqrt (u**2 + v**2)[:,None] / lbd[None,:];
 
    if name == 'OI_VIS2':
        u = hdulist['OI_VIS2'].data['UCOORD'];
        v = hdulist['OI_VIS2'].data['VCOORD'];
        return np.sqrt (u**2 + v**2)[:,None] / lbd[None,:];

    if name == 'OI_T3':
        u1 = hdulist['OI_T3'].data['U1COORD'];
        v1 = hdulist['OI_T3'].data['V1COORD'];
        u2 = hdulist['OI_T3'].data['U2COORD'];
        v2 = hdulist['OI_T3'].data['V2COORD'];
        u = np.array([u1,u2,u1+u2]);
        v = np.array([v1,v2,v1+v2]);
        return np.sqrt (u**2 + v**2)[:,:,None] / lbd[None,None,:];

def tf_time_weight (hdus, hdutf, delta):
    '''
    Average the Transfer Functions in hdutf (a list of
    fits handlers) at the time of a science
    observation (hdus) with time weighted interpolation.
    delta is in [days]
    The function assumes the baselines are ordered the same way.

    Return the OIFITS_SCI_TF, as a FITS handler.
    '''
    mircx_mystic_log.info ('Interpolate %i TF with time_weight'%len(hdutf));

    # Copy OIFITS_SCI to build OIFITS_TF
    hdutfs = pyfits.HDUList([hdu.copy() for hdu in hdus]);
    hdutfs[0].header['FILETYPE'] = 'OIFITS_SCI_TF';
    
    obs = [['OI_VIS2','VIS2DATA','VIS2ERR',False],
           ['OI_T3','T3AMP','T3AMPERR',False],
           ['OI_T3','T3PHI','T3PHIERR',True],
           ['OI_VIS','VISAMP','VISAMPERR',False],
           ['OI_VIS','VISPHI','VISPHIERR',True]]

    # Loop on observables
    for o in obs:
        
        # Get TF data
        mjd = np.array ([h[o[0]].data['MJD'] for h in hdutf]);
        val = np.array ([h[o[0]].data[o[1]] for h in hdutf]);
        err = np.array ([h[o[0]].data[o[2]] for h in hdutf]);

        # Set nan to flagged data and to data without error
        flg  = np.array([h[o[0]].data['FLAG'] for h in hdutf]);
        flg += ~np.isfinite (val) + ~np.isfinite (err) + (err<=0);
        val[flg] = np.nan;
        err[flg] = np.nan;

        # Verbose 
        mircx_mystic_log.info (o[1]+": %i valid TF points over %i"%(np.sum(np.isfinite(val)),val.size));

        # Don't give added advantage for <2% percent error
        #  or 0.1deg for phase, Idea from John Monnier
        # FIXME: to be done for vis2
        if o[3] is True:  we = np.maximum (err, 0.1)**-2;
        if o[3] is False: we = np.maximum (err, np.abs (0.02*val)+1e-4)**-2;

        # Replace by phasor
        if o[3] is True: val = phasor (val);

        # When we want to interpolate
        mjd0 = hdus[o[0]].data['MJD'];

        # Check date
        mjd0[~np.isfinite(mjd0)] = 0.0;
        mjd[~np.isfinite(mjd)]   = 0.0;
        if np.sum(mjd0 <= 0): mircx_mystic_log.warning ('Invalid MJDs in SCI !!');
        if np.sum(mjd  <= 0): mircx_mystic_log.warning ('Invalid MJDs in TFs !!');

        # JDM recommends also adding weight from distance in sky for CHARA... esp. for visphi.
        # Compute the weights at science time (ntf,nb,nc)
        ws = np.exp (-(mjd0[None,:,None]-mjd[:,:,None])**2/delta**2);

        # Compute the weighted mean (nb,nc)
        tf = np.nansum (val * ws * we, axis=0) / np.nansum (ws * we, axis=0);
        
        # Compute the model at TF times (ntf, nb, nc)
        wtf = np.exp (-(mjd[None,:,:,None]-mjd[:,None,:,None])**2/delta**2);
        model = np.nansum (wtf * we[:,None,:,:] * val[:,None,:,:], axis=0) / np.nansum (wtf * we[:,None,:,:], axis=0);
        
        # Compute the residuals
        if o[3] is False:  
            res = tf - model;
        else:
            res = np.angle (tf * np.conj (model), deg=True);

        # Compute the variance of residual and chi2
        varm = np.nansum (ws * we, axis=0)**-1;
        chi2 = np.nansum (res**2 * we * ws, axis=0) / np.nansum (ws, axis=0);

        # Compute errors of interpolated TF (non-standard):
        # error = sqrt(chi2 * variance * correctif)
        # Because of last therm, if chi2>>1, this tends to:
        # error = RMS(tf)
        chi2  = np.maximum (chi2, 1.01);
        tfErr = np.sqrt (chi2 * varm * np.nansum (ws, axis=0)**(1.-1./chi2**2));

        # Replace TF by phasor if needed
        if o[3] is True: tf = np.angle (tf, deg=True);

        # Ensure error is positive
        tfErr = np.maximum (tfErr, 0.0);

        # Set data and error
        hdutfs[o[0]].data[o[1]] = tf;
        hdutfs[o[0]].data[o[2]] = tfErr;

        # FLAG. Note that flag is not updated for T3AMP and VISAMP
        # since the FLAG is only comming from T3PHI and VISPHI
        if o[1] != 'T3AMP' and o[1] != 'VISAMP':
            hdutfs[o[0]].data['FLAG']  = ~np.isfinite (tf);
            hdutfs[o[0]].data['FLAG'] += ~np.isfinite (tfErr);
            hdutfs[o[0]].data['FLAG'] += tfErr <= 0.0;

            # Flag huge errors
            if o[3] is True: 
                hdutfs[o[0]].data['FLAG'] += (tfErr > 50);
            else:
                hdutfs[o[0]].data['FLAG'] += (tfErr > 0.6);

        # Verbose
        valid = (~hdutfs[o[0]].data['FLAG']) & np.isfinite (hdutfs[o[0]].data[o[1]]);
        mircx_mystic_log.info (o[1]+": %i valid interpolated points over %i"%(np.sum(valid),valid.size));
        # log.info ("(%i un-flagged points)"%(np.sum(~hdutfs[o[0]].data['FLAG'])));
        # log.info ("(%i finite points)"%(np.sum(np.isfinite(hdutfs[o[0]].data[o[1]]))));
            
    return hdutfs;

def tf_divide (hdus, hdutf):
    '''
    Calibrate the SCI hdus (a FITS handler) by the TF hdutf (another
    FITS handler). The TF which may come from the averaging of many TF.
    The function assumes the baselines are ordered the same way.

    Return the calibrated OIFITS_CALIBRATED (a FITS handler).
    '''

    mircx_mystic_log.info ('Divide data by interpolated TF');

    # Copy OIFITS_SCI to build OIFITS_CALIBRATED
    hdusc = pyfits.HDUList([hdu.copy() for hdu in hdus]);
    hdusc[0].header['FILETYPE'] = 'OIFITS_CALIBRATED';

    obs = [['OI_VIS2','VIS2DATA','VIS2ERR',False],
           ['OI_T3','T3AMP','T3AMPERR',False],
           ['OI_T3','T3PHI','T3PHIERR',True],
           ['OI_VIS','VISAMP','VISAMPERR',False],
           ['OI_VIS','VISPHI','VISPHIERR',True]]

    for o in obs:
        # Verbose
        valid = (~hdusc[o[0]].data['FLAG']) & np.isfinite(hdusc[o[0]].data[o[1]]);
        mircx_mystic_log.info (o[1]+": %i valid raw points over %i"%(np.sum(valid),valid.size));
        # log.info ("(%i un-flagged points)"%(np.sum(~hdusc[o[0]].data['FLAG'])));
        # log.info ("(%i finite points)"%(np.sum(np.isfinite(hdusc[o[0]].data[o[1]]))));
        
        if o[3] is True:
            # Correct phase from TF
            hdusc[o[0]].data[o[1]] -= hdutf[o[0]].data[o[1]];
            hdusc[o[0]].data[o[1]]  = wrap (hdusc[o[0]].data[o[1]]);
            # Add errors
            hdusc[o[0]].data[o[2]] = np.sqrt (hdusc[o[0]].data[o[2]]**2 + hdutf[o[0]].data[o[2]]**2);
        else:
            # Get values
            Raw = copy.copy (hdusc[o[0]].data[o[1]]);
            Tfi = copy.copy (hdutf[o[0]].data[o[1]]);
            dRaw = copy.copy (hdusc[o[0]].data[o[2]]);
            dTfi = copy.copy (hdutf[o[0]].data[o[2]]);
            # Set errors and value
            Cal = Raw / Tfi;
            hdusc[o[0]].data[o[1]] = Cal;
            hdusc[o[0]].data[o[2]] = Cal * np.sqrt ((dRaw/Raw)**2 + (dTfi/Tfi)**2);

        # (deprecated by JDM): FLAG. Note that flag is not updated for T3AMP and VISAMP 
        # since the FLAG is only comming from T3PHI and VISPHI
        # JDM NOTE. I switched thie logic here. since large phase errors are natural if the 
        #   amplitudes are small, i think its better to keep data as long as amps are not noisy.
        #   ideally we could do a check that the snr of visamp <1 or something better  but
        #   the loop structure here meas we don't have access to all all data only one data type I think.

        if o[1] != 'T3PHI' and o[1] != 'VISPHI':
            
            hdusc[o[0]].data['FLAG'] += ~np.isfinite (hdusc[o[0]].data[o[1]]);
            hdusc[o[0]].data['FLAG'] += ~np.isfinite (hdusc[o[0]].data[o[2]]);
            hdusc[o[0]].data['FLAG'] += hdusc[o[0]].data[o[2]] <= 0.0;

            # Flag huge errors
            if o[3] is True: 
                hdusc[o[0]].data['FLAG'] += (hdusc[o[0]].data[o[2]] > 60);
            else:
                hdusc[o[0]].data['FLAG'] += (hdusc[o[0]].data[o[2]] > 0.4);


        # Verbose
        valid = (~hdusc[o[0]].data['FLAG']) & np.isfinite (hdusc[o[0]].data[o[1]]);
        mircx_mystic_log.info (o[1]+": %i valid calibrated points over %i"%(np.sum(valid),valid.size));
        # log.info ("(%i un-flagged points)"%(np.sum(~hdusc[o[0]].data['FLAG'])));
        # log.info ("(%i finite points)"%(np.sum(np.isfinite(hdusc[o[0]].data[o[1]]))));
        
    return hdusc;
    
def compute_all_viscalib (hdrs, catalog, deltaTf=0.05,
                          outputDir='viscal/',
                          outputSetup='calibration_setup',
                          overwrite=True,
                          lbdMin=1.0, lbdMax=2.0,
                          flagEdges=False,
                          keys=visparam):
    '''
    Cross-calibrate the OIFITS in hdrs. The choice of SCI and CAL, and the diameter
    of the calibration stars, are specified with the catalog. Catalog should be
    of the form [('name1',diam1,err1),('name2',diam2,err2),...] where the diam
    and err are in [mas]. The input hdrs shall be a list of FITS headers.
    '''
    elog = mircx_mystic_log.trace ('compute_all_viscalib');

    # Check inputs
    headers.check_input (hdrs, required=1);

    # Get setup name (assume the same for all file)
    setup_name = '/'.join([str(hdrs[0].get(k,'--')) for k in keys]);
    elog = mircx_mystic_log.trace ('setup: '+setup_name);

    # Get OIFITS_SCI and OIFITS_CAL from input catalog
    scis, calibs = headers.get_sci_cal (hdrs, catalog);

    # List of measured Transfert Functions
    # (OIFITS_CAL / diameter)
    hdutf = [];
    for calib in calibs:
        f = calib['ORIGNAME'];

        mircx_mystic_log.info ('Load %s (%s)'%(f,calib['FILETYPE']));
        hdulist = pyfits.open (f);

        if calib[HMP+'CALIB MODEL_NAME'] != 'UD_H':
           mircx_mystic_log.warning ('MODEL_NAME is not supported');
           continue;

        # Get diameter in [rad]
        diam = calib[HMP+'CALIB PARAM1'] * 4.84813681109536e-09;
        diamErr = calib[HMP+'CALIB PARAM2'] * 4.84813681109536e-09;

        # Compute the VIS TF
        mircx_mystic_log.info ('Compute vis amp TF');
        spf = get_spfreq (hdulist,'OI_VIS');
        visamp = signal.airy (diam * spf); # JDM no error analysis added for diameters?
        hdulist['OI_VIS'].data['VISAMP'] /= visamp;
        hdulist['OI_VIS'].data['VISAMPERR'] /= visamp;

        # Compute the VIS2 TF
        mircx_mystic_log.info ('Compute vis2 TF');
        spf = get_spfreq (hdulist,'OI_VIS2');
        v2 = signal.airy (diam * spf)**2;
        hdulist['OI_VIS2'].data['VIS2DATA'] /= v2;
        hdulist['OI_VIS2'].data['VIS2ERR'] /= v2;

        # Compute the T3AMP TF
        mircx_mystic_log.info ('Compute t3amp TF');
        spf = get_spfreq (hdulist,'OI_T3');
        v123 = signal.airy (diam * spf);
        v123 = v123[0,:,:] * v123[1,:,:] * v123[2,:,:];
        hdulist['OI_T3'].data['T3AMP'] /= v123;
        hdulist['OI_T3'].data['T3AMPERR'] /= v123;

        # These are OIFITS_CAL_TF
        hdulist[0].header['FILETYPE'] = 'OIFITS_CAL_TF';
        hdutf.append (hdulist);

    # Number of TF
    ntf = len (hdutf);
    
    # Loop on OIFITS_SCI to calibrate them
    hdusci, hdutfs = [], [];
    for sci in scis:
    
        mircx_mystic_log.info ('Load SCI %s'%(sci['ORIGNAME']));
        hdus = pyfits.open (sci['ORIGNAME']);

        # Check
        if ntf == 0: continue;

        # Define output name
        output = files.output (outputDir,sci,'viscal');
        
        # Compute interpolation at the time of science and divide
        hdutfsi = tf_time_weight (hdus, hdutf, deltaTf);
        hdulist = tf_divide (hdus, hdutfsi);

        # Flag wavelengths
        lbd = hdulist['OI_WAVELENGTH'].data['EFF_WAVE'] * 1e6;
        flag = (lbd < lbdMin) + (lbd > lbdMax);
        hdulist['OI_VIS2'].data['FLAG'] += flag[None,:];
        hdulist['OI_VIS'].data['FLAG'] += flag[None,:];
        hdulist['OI_T3'].data['FLAG'] += flag[None,:];

        # Flag edges
        if flagEdges:
            hdulist['OI_VIS2'].data['FLAG'][:,[0,-1]] += True;
            hdulist['OI_VIS'].data['FLAG'][:,[0,-1]] += True;
            hdulist['OI_T3'].data['FLAG'][:,[0,-1]] += True;

        # First HDU
        hdulist[0].header['FILETYPE'] = 'OIFITS_CALIBRATED';
        hdulist[0].header[HMP+'OIFITS_SCI'] = os.path.basename (sci['ORIGNAME']);
        hdulist[0].header[HMP+'DELTA_INTERP'] = (deltaTf,'[days] delta for interpolation');

        # Write file
        files.write (hdulist, output+'.fits');
    
        # Append OIFITS_SCI OIFITS_SCI_TF for the trend
        hdutfs.append (hdutfsi);
        hdusci.append (hdus);
        
        calplots=True;
        if calplots==True:
            # VIS
            fig,axes = plt.subplots ();
            fig.suptitle (headers.summary (sci));
            x  = get_spfreq (hdulist,'OI_VIS')
            y  = hdulist['OI_VIS'].data['VISPHI'];
            dy = hdulist['OI_VIS'].data['VISPHIERR'];
            y[hdulist['OI_VIS'].data['FLAG']] = np.nan;
            for b in range (15):
                bars = axes.errorbar (1e-6*x[b,:],y[b,:],yerr=dy[b,:],fmt='o-',ms=2)[2];
                for bar in bars: bar.set_alpha(0.15);
            axes.set_xlim (0);
            axes.set_xlabel ('sp. freq. (M$\lambda$)');
            axes.set_ylabel ('VISPHI (deg)');
            files.write (fig,output+'_visphi.png');
            
            plt.close ("all");

            fig,axes = plt.subplots ();
            fig.suptitle (headers.summary (sci));
            x  = get_spfreq (hdulist,'OI_VIS');
            y  = hdulist['OI_VIS'].data['VISAMP'];
            dy = hdulist['OI_VIS'].data['VISAMPERR'];
            y[hdulist['OI_VIS'].data['FLAG']] = np.nan;
            for b in range (15):
                bars = axes.errorbar (1e-6*x[b,:],y[b,:],yerr=dy[b,:],fmt='o-',ms=2)[2];
                for bar in bars: bar.set_alpha(0.15);
            axes.set_ylim (-0.1,1.2);
            axes.set_xlim (0);
            axes.set_xlabel ('sp. freq. (Mlbd)');
            axes.set_ylabel ('visamp');
            files.write (fig,output+'_visamp.png');
            
            plt.close ("all");
        


            # VIS2
            fig,axes = plt.subplots ();
            fig.suptitle (headers.summary (sci));
            x  = get_spfreq (hdulist,'OI_VIS2');
            y  = hdulist['OI_VIS2'].data['VIS2DATA'];
            dy = hdulist['OI_VIS2'].data['VIS2ERR'];
            y[hdulist['OI_VIS2'].data['FLAG']] = np.nan;
            for b in range (15):
                bars = axes.errorbar (1e-6*x[b,:],y[b,:],yerr=dy[b,:],fmt='o-',ms=2)[2];
                for bar in bars: bar.set_alpha(0.15);
            axes.set_ylim (-0.1,1.2);
            axes.set_xlim (0);
            axes.set_xlabel ('sp. freq. (Mlbd)');
            axes.set_ylabel ('vis2');
            files.write (fig,output+'_vis2.png');
            
            plt.close ("all");
            
            # CP
            fig,axes = plt.subplots ();
            fig.suptitle (headers.summary (sci));
            x  = np.max (get_spfreq (hdulist,'OI_T3'), axis=0);
            y  = hdulist['OI_T3'].data['T3PHI'];
            dy = hdulist['OI_T3'].data['T3PHIERR'];
            y[hdulist['OI_T3'].data['FLAG']] = np.nan;
            for b in range (20):
                bars = axes.errorbar (1e-6*x[b,:],y[b,:],yerr=dy[b,:],fmt='o-',ms=2)[2];
                for bar in bars: bar.set_alpha(0.15);
            axes.set_xlim (0);
            axes.set_xlabel ('max sp. freq. (M$\lambda$)');
            axes.set_ylabel ('T3PHI (deg)');
            files.write (fig,output+'_t3phi.png');
            
            plt.close ("all");

            fig,axes = plt.subplots ();
            fig.suptitle (headers.summary (sci));
            x  = np.max (get_spfreq (hdulist,'OI_T3'), axis=0);
            y  = hdulist['OI_T3'].data['T3AMP'];
            dy = hdulist['OI_T3'].data['T3AMPERR'];
            y[hdulist['OI_T3'].data['FLAG']] = np.nan;
            for b in range (20):
                bars = axes.errorbar (1e-6*x[b,:],y[b,:],yerr=dy[b,:],fmt='o-',ms=2)[2];
                for bar in bars: bar.set_alpha(0.15);
            axes.set_ylim (-0.1,1.2);
            axes.set_xlim (0);
            axes.set_xlabel ('max sp. freq. (Mlbd)');
            axes.set_ylabel ('t3amp');
            files.write (fig,output+'_t3amp.png');
            plt.close ("all");
        else:
            mircx_mystic_log.info("Skipping individual plots:")

        # JDM unix comment: ulimit -n 5000 to incraes elimit or
        # import resource
        # soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        # resource.setrlimit(resource.RLIMIT_NOFILE, (5000, hard))
        # but requires root.
        
        # other closing of files. getting an error.
        # del hdulist
        # del hdulist.data 
        #JDM to avoid the too many files open bug.
        #hdus.close()
    
    #JDM other diagnostic figs? liek visamp vs vis2data, or t3amp vs product of vis2? 

    mircx_mystic_log.info ('Figures for the trends');
    
    # Check of amount of files to plot
    mircx_mystic_log.info ('Number of transfer function %i'%ntf);
    if ntf == 0:
        raise ValueError ('No calibrator for this setup');

    nsci = len (hdusci);
    mircx_mystic_log.info ('Number of calibrated science %i'%nsci);
    if nsci == 0:
        raise ValueError ('No calibrated science for this setup');
        
    # Get the number of spectral channels (assume the same for all)
    lbd = hdusci[0]['OI_WAVELENGTH'].data['EFF_WAVE'];
    nc = len (lbd);

    # VIS2
    names = ['MJD','VIS2DATA','VIS2ERR'];
    xtf, ytf, dytf = oifits.getdata (hdutf,'OI_VIS2',names);
    xts, yts, dyts = oifits.getdata (hdutfs,'OI_VIS2',names);
    xsc, ysc, dysc = oifits.getdata (hdusci,'OI_VIS2',names);
    

    # Get station name for labels (assume the same for all files,
    # and assume the OI_ARRAY is ordered starting with 1)
    idx = hdutf[0]['OI_VIS2'].data['STA_INDEX'];
    sta = hdutf[0]['OI_ARRAY'].data['STA_NAME'][idx-1];
    bname = np.array ([s[0]+s[1]+' '+n for s,n in zip(sta,setup.base_name())]);

    for f in range (5):
        for c in range (nc):
            fig,axes = plt.subplots (3,1, sharex=True);
            plot.base_name (axes, names=bname[f*3:f*3+3]);
            plot.compact (axes);
            plt.subplots_adjust (hspace=0.03);
            fig.suptitle (setup_name + ' c%i'%c);

            for bb in range (3):
                ax = axes.flatten()[bb];
                b = f*3+bb;
                ax.errorbar (xtf[:,b],ytf[:,b,c],fmt='o',yerr=dytf[:,b,c],color='k',ms=1);
                ax.errorbar (xts[:,b],yts[:,b,c],fmt='o',yerr=dyts[:,b,c],color='k',ms=1,alpha=0.25);
                ax.errorbar (xsc[:,b],ysc[:,b,c],fmt='o',yerr=dysc[:,b,c],color='g',ms=1);
                ylim = ax.get_ylim ();
                ax.set_ylim (0.0,np.minimum (ylim[1],1.1));
    
            files.write (fig,outputDir+'/'+outputSetup+'_vis2_%i_c%02i.png'%(f,c));
            plt.close ("all");


    # VISAMP
    names = ['MJD','VISAMP','VISAMPERR'];
    xtf, ytf, dytf = oifits.getdata (hdutf,'OI_VIS',names);
    xts, yts, dyts = oifits.getdata (hdutfs,'OI_VIS',names);
    xsc, ysc, dysc = oifits.getdata (hdusci,'OI_VIS',names);
    
    # Get station name for labels (assume the same for all files,
    # and assume the OI_ARRAY is ordered starting with 1)
    idx = hdutf[0]['OI_VIS'].data['STA_INDEX'];
    sta = hdutf[0]['OI_ARRAY'].data['STA_NAME'][idx-1];
    bname = np.array ([s[0]+s[1]+' '+n for s,n in zip(sta,setup.base_name())]);

    for f in range (5):
        for c in range (nc):
            fig,axes = plt.subplots (3,1, sharex=True);
            plot.base_name (axes, names=bname[f*3:f*3+3]);
            plot.compact (axes);
            plt.subplots_adjust (hspace=0.03);
            fig.suptitle (setup_name + ' c%i'%c);

            for bb in range (3):
                ax = axes.flatten()[bb];
                b = f*3+bb;
                ax.errorbar (xtf[:,b],ytf[:,b,c],fmt='o',yerr=dytf[:,b,c],color='k',ms=1);
                ax.errorbar (xts[:,b],yts[:,b,c],fmt='o',yerr=dyts[:,b,c],color='k',ms=1,alpha=0.25);
                ax.errorbar (xsc[:,b],ysc[:,b,c],fmt='o',yerr=dysc[:,b,c],color='g',ms=1);
                ylim = ax.get_ylim ();
                ax.set_ylim (0.0,np.minimum (ylim[1],1.1));
    
            files.write (fig,outputDir+'/'+outputSetup+'_visamp_%i_c%02i.png'%(f,c));
            plt.close ("all");

    # VISPHI
    names = ['MJD','VISPHI','VISPHIERR'];
    xtf, ytf, dytf = oifits.getdata (hdutf,'OI_VIS',names);
    xts, yts, dyts = oifits.getdata (hdutfs,'OI_VIS',names);
    xsc, ysc, dysc = oifits.getdata (hdusci,'OI_VIS',names);
    
    # Get station name for labels (assume the same for all files,
    # and assume the OI_ARRAY is ordered starting with 1)
    idx = hdutf[0]['OI_VIS'].data['STA_INDEX'];
    sta = hdutf[0]['OI_ARRAY'].data['STA_NAME'][idx-1];
    bname = np.array ([s[0]+s[1]+' '+n for s,n in zip(sta,setup.base_name())]);

    for f in range (5):
        for c in range (nc):
            fig,axes = plt.subplots (3,1, sharex=True);
            plot.base_name (axes, names=bname[f*3:f*3+3]);
            plot.compact (axes);
            plt.subplots_adjust (hspace=0.03);
            fig.suptitle (setup_name + ' c%i'%c);

            for bb in range (3):
                ax = axes.flatten()[bb];
                b = f*3+bb;
                ax.errorbar (xtf[:,b],ytf[:,b,c],fmt='o',yerr=dytf[:,b,c],color='k',ms=1);
                ax.errorbar (xts[:,b],yts[:,b,c],fmt='o',yerr=dyts[:,b,c],color='k',ms=1,alpha=0.25);
                ax.errorbar (xsc[:,b],ysc[:,b,c],fmt='o',yerr=dysc[:,b,c],color='g',ms=1);
                ylim = ax.get_ylim ();
                ax.set_ylim (np.maximum (ylim[0],-200),np.minimum (ylim[1],200));
    
            files.write (fig,outputDir+'/'+outputSetup+'_visphi_%i_c%02i.png'%(f,c));
            plt.close ("all");

    # T3PHI
    names = ['MJD','T3PHI','T3PHIERR'];
    xtf, ytf, dytf = oifits.getdata (hdutf, 'OI_T3',names);
    xts, yts, dyts = oifits.getdata (hdutfs,'OI_T3',names);
    xsc, ysc, dysc = oifits.getdata (hdusci,'OI_T3',names);

    # Get triplet name for labels (assume the same for all files,
    # and assume the OI_ARRAY is ordered starting with 1)
    idx = hdutf[0]['OI_T3'].data['STA_INDEX'];
    sta = hdutf[0]['OI_ARRAY'].data['STA_NAME'][idx-1];
    tname = np.array ([s[0]+s[1]+s[2]+' '+n for s,n in zip(sta,setup.triplet_name())]);

    for f in range (5):
        for c in range (nc):
            fig,axes = plt.subplots (4,1, sharex=True);
            plot.base_name (axes, names=tname[f*4:f*4+4]);
            plot.compact (axes);
            plt.subplots_adjust (hspace=0.03);
            fig.suptitle (setup_name + ' c%i'%c);

            for bb in range (4):
                b = f*4+bb;
                ax = axes.flatten()[bb];
                ax.errorbar (xtf[:,b],ytf[:,b,c],fmt='o',yerr=dytf[:,b,c],color='k',ms=1);
                ax.errorbar (xts[:,b],yts[:,b,c],fmt='o',yerr=dyts[:,b,c],color='k',ms=1,alpha=0.25);
                ax.errorbar (xsc[:,b],ysc[:,b,c],fmt='o',yerr=dysc[:,b,c],color='g',ms=1);
                ylim = ax.get_ylim ();
                ax.set_ylim (np.maximum (ylim[0],-200),np.minimum (ylim[1],200));
                
            files.write (fig,outputDir+'/'+outputSetup+'_t3phi_%i_c%02i.png'%(f,c));
            plt.close ("all");
    
    # T3AMP
    names = ['MJD','T3AMP','T3AMPERR'];
    xtf, ytf, dytf = oifits.getdata (hdutf, 'OI_T3',names);
    xts, yts, dyts = oifits.getdata (hdutfs,'OI_T3',names);
    xsc, ysc, dysc = oifits.getdata (hdusci,'OI_T3',names);

    # Get triplet name for labels (assume the same for all files,
    # and assume the OI_ARRAY is ordered starting with 1)
    idx = hdutf[0]['OI_T3'].data['STA_INDEX'];
    sta = hdutf[0]['OI_ARRAY'].data['STA_NAME'][idx-1];
    tname = np.array ([s[0]+s[1]+s[2]+' '+n for s,n in zip(sta,setup.triplet_name())]);

    for f in range (5):
        for c in range (nc):
            fig,axes = plt.subplots (4,1, sharex=True);
            plot.base_name (axes, names=tname[f*4:f*4+4]);
            plot.compact (axes);
            plt.subplots_adjust (hspace=0.03);
            fig.suptitle (setup_name + ' c%i'%c);

            for bb in range (4):
                b = f*4+bb;
                ax = axes.flatten()[bb];
                ax.errorbar (xtf[:,b],ytf[:,b,c],fmt='o',yerr=dytf[:,b,c],color='k',ms=1);
                ax.errorbar (xts[:,b],yts[:,b,c],fmt='o',yerr=dyts[:,b,c],color='k',ms=1,alpha=0.25);
                ax.errorbar (xsc[:,b],ysc[:,b,c],fmt='o',yerr=dysc[:,b,c],color='g',ms=1);
                ylim = ax.get_ylim ();
                #ax.set_ylim (np.maximum (ylim[0],-200),np.minimum (ylim[1],200));
                ax.set_yscale('log')
                ax.set_ylim(np.maximum(1e-5,ylim[0]),1.5)
                
            files.write (fig,outputDir+'/'+outputSetup+'_t3amp_%i_c%02i.png'%(f,c));
            plt.close ("all");

    """ # T3AMP
    fig,axes = plt.subplots (5,4, sharex=True);
    plot.base_name (axes);
    plot.compact (axes);
    fig.suptitle (setup_name);

    for b in range (20):
        ax = axes.flatten()[b];
        x  = [h['OI_T3'].data['MJD'][b] for h in hdutf];
        y  = [h['OI_T3'].data['T3AMP'][b,6] for h in hdutf];
        dy = [h['OI_T3'].data['T3AMPERR'][b,6] for h in hdutf];
        ax.errorbar (x,y,fmt='o',yerr=dy,color='k',ms=1);
        x  = [h['OI_T3'].data['MJD'][b] for h in hdutfs];
        y  = [h['OI_T3'].data['T3AMP'][b,6] for h in hdutfs];
        dy = [h['OI_T3'].data['T3AMPERR'][b,6] for h in hdutfs];
        ax.errorbar (x,y,fmt='o',yerr=dy,color='k',ms=1,alpha=0.25);
        x  = [h['OI_T3'].data['MJD'][b] for h in hdusci];
        y  = [h['OI_T3'].data['T3AMP'][b,6] for h in hdusci];
        dy = [h['OI_T3'].data['T3AMPERR'][b,6] for h in hdusci];
        ax.errorbar (x,y,fmt='o',yerr=dy,color='g',ms=1);
    
    files.write (fig,outputDir+'/'+outputSetup+'_t3amp.png'); """
    
    plt.close ("all");
