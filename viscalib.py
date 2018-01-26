import numpy as np;
import os;
import copy;

from astropy.io import fits as pyfits

import matplotlib.pyplot as plt;
import matplotlib.colors as mcolors;

from . import log, files, headers, setup, oifits, signal, plot;
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
    u = hdulist[name].data['UCOORD'];
    v = hdulist[name].data['VCOORD'];
    lbd = hdulist['OI_WAVELENGTH'].data['EFF_WAVE'];
    return np.sqrt (u**2 + v**2)[:,None] / lbd[None,:];

def tf_time_weight (hdus, hdutf, delta):
    '''
    Average the Transfer Functions in hdutf (a list of
    fits handlers) at the time of a science
    observation (hdus) with time weighted interpolation.
    delta is in [days]
    The function assumes the baselines are ordered the same way.

    Return the VIS_SCI_TF, as a FITS handler.
    '''
    log.info ('Interpolate %i TF with time_weight'%len(hdutf));

    # Copy VIS_SCI to build VIS_TF
    hdutfs = pyfits.HDUList([hdu.copy() for hdu in hdus]);
    hdutfs[0].header['FILETYPE'] = 'VIS_SCI_TF';
    
    obs = [['OI_VIS2','VIS2DATA','VIS2ERR',False],
           ['OI_T3','T3AMP','T3AMPERR',False],
           ['OI_T3','T3PHI','T3PHIERR',True]];

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

        # Don't give added advantage for <2% percent error
        #  or 0.1deg for phase, Idea from John Monnier
        # FIXME: to be done for vis2
        if o[3] is True:  we = np.maximum (err, 0.1)**-2;
        if o[3] is False: we = np.maximum (err, np.abs (0.02*val)+1e-4)**-2;

        # Replace by phasor
        if o[3] is True: val = phasor (val);

        # When we want to interpolate
        mjd0 = hdus[o[0]].data['MJD'];

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
        chi2  = np.maximum (chi2, 1.0);
        tfErr = np.sqrt (chi2 * varm * np.nansum (ws, axis=0)**(1.-1./chi2**2));

        # Replace TF by phasor if needed
        if o[3] is True: tf = np.angle (tf, deg=True);

        # Set data and error
        hdutfs[o[0]].data[o[1]] = tf;
        hdutfs[o[0]].data[o[2]] = tfErr;
        hdutfs[o[0]].data['FLAG'] += ~np.isfinite (tf);

            
    return hdutfs;

def tf_divide (hdus, hdutf):
    '''
    Calibrate the SCI hdus (a FITS handler) by the TF hdutf (another
    FITS handler). The TF which may come from the averaging of many TF.
    The function assumes the baselines are ordered the same way.

    Return the calibrated VIS_CALIBRATED (a FITS handler).
    '''

    # Copy VIS_SCI to build VIS_CALIBRATED
    hdusc = pyfits.HDUList([hdu.copy() for hdu in hdus]);
    hdusc[0].header['FILETYPE'] = 'VIS_CALIBRATED';

    obs = [['OI_VIS2','VIS2DATA','VIS2ERR',False],
           ['OI_T3','T3AMP','T3AMPERR',False],
           ['OI_T3','T3PHI','T3PHIERR',True]];

    for o in obs:
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
            
        hdusc[o[0]].data['FLAG'] += ~np.isfinite (hdusc[o[0]].data[o[1]]);

    return hdusc;
    
def compute_all_viscalib (hdrs, catalog, delta=0.05,
                          outputDir='viscal/',
                          outputSetup='calibration_setup',
                          overwrite=True,
                          keys=visparam):
    '''
    Cross-calibrate the VIS in hdrs. The choice of SCI and CAL, and the diameter
    of the calibration stars, are specified with the catalog. Catalog should be
    of the form [('name1',diam1,err1),('name2',diam2,err2),...] where the diam
    and err are in [mas]. The input hdrs shall be a list of FITS headers.
    '''
    elog = log.trace ('compute_all_viscal');

    headers.check_input (hdrs, required=1);

    # Get VIS_SCI and VIS_CAL from input catalog
    scis, calibs = headers.get_sci_cal (hdrs, catalog);
    
    # List of measured Transfert Functions
    # (VIS_CAL / diameter)
    hdutf = [];
    for calib in calibs:
        f = calib['ORIGNAME'];

        log.info ('Load %s (%s)'%(f,calib['FILETYPE']));
        hdulist = pyfits.open (f);

        # Get diameter in [rad]
        diam = calib[HMP+'CALIB DIAM'] * 4.84813681109536e-09;
        diamErr = calib[HMP+'CALIB DIAMERR'] * 4.84813681109536e-09;

        # Compute the TF
        spf = get_spfreq (hdulist,'OI_VIS2');
        v2 = signal.airy (diam * spf)**2;
        hdulist['OI_VIS2'].data['VIS2DATA'] /= v2;
        hdulist['OI_VIS2'].data['VIS2ERR'] /= v2;

        # These are VIS_CAL_TF
        hdulist[0].header['FILETYPE'] = 'VIS_CAL_TF';
        hdutf.append (hdulist);

    # Loop on VIS_SCI to calibrate them
    hdusci, hdutfs = [], [];
    for sci in scis:
    
        log.info ('Load SCI %s'%(sci['ORIGNAME']));
        hdus = pyfits.open (sci['ORIGNAME']);

        # Define output name
        output = files.output (outputDir,sci,'viscal');
        
        # Compute interpolation at the time of science and divide
        hdutfsi = tf_time_weight (hdus, hdutf, delta);
        hdulist = tf_divide (hdus, hdutfsi);

        # First HDU
        hdulist[0].header['FILETYPE'] = 'VIS_CALIBRATED';
        hdulist[0].header[HMP+'VIS_SCI'] = os.path.basename (sci['ORIGNAME']);
        hdulist[0].header[HMP+'DELTA_INTERP'] = (delta,'[days] delta for interpolation');

        # Write file
        files.write (hdulist, output+'.fits');
    
        # Append VIS_SCI and VIS_SCI_TF, to allow a plot
        # of a trend over the night for this setup
        hdusci.append (hdus);
        hdutfs.append (hdutfsi);
        
        # VIS2
        fig,axes = plt.subplots ();
        fig.suptitle (headers.summary (sci));
        x  = get_spfreq (hdulist,'OI_VIS2');
        y  = hdulist['OI_VIS2'].data['VIS2DATA'];
        dy = hdulist['OI_VIS2'].data['VIS2ERR'];
        for b in range (15):
            axes.errorbar (1e-6*x[b,:],y[b,:],yerr=dy[b,:],fmt='o',ms=1);
        axes.set_ylim (-0.1,1.2);
        axes.set_xlim (0);
        axes.set_xlabel ('sp. freq. (Mlbd)');
        axes.set_ylabel ('vis2');
        files.write (fig,output+'_vis2.png');
         
        plt.close ("all");
        
        
    log.info ('Figures for the trends');

    nc = int(hdrs[0]['HIERARCH MIRC QC WIN FRINGE NY']);

    # VIS2
    names = ['MJD','VIS2DATA','VIS2ERR'];
    xtf, ytf, dytf = oifits.getdata (hdutf,'OI_VIS2',names);
    xts, yts, dyts = oifits.getdata (hdutfs,'OI_VIS2',names);
    xsc, ysc, dysc = oifits.getdata (hdusci,'OI_VIS2',names);

    for f in range (5):
        for c in range (nc):
            fig,axes = plt.subplots (3,1, sharex=True);
            plot.base_name (axes, bstart=f*3);
            plot.compact (axes);
            plt.subplots_adjust (hspace=0.03);
            fig.suptitle (' / '.join([str(hdrs[0].get(k,'--')) for k in keys]) + ' c%i'%c);

            for bb in range (3):
                ax = axes.flatten()[bb];
                b = f*3+bb;
                ax.errorbar (xtf[:,b],ytf[:,b,c],fmt='o',yerr=dytf[:,b,c],color='k',ms=1);
                ax.errorbar (xts[:,b],yts[:,b,c],fmt='o',yerr=dyts[:,b,c],color='k',ms=1,alpha=0.25);
                ax.errorbar (xsc[:,b],ysc[:,b,c],fmt='o',yerr=dysc[:,b,c],color='g',ms=1);
                ylim = ax.get_ylim ();
                ax.set_ylim (np.maximum (ylim[0],0),np.minimum (ylim[1],1.1));
    
            files.write (fig,outputDir+'/'+outputSetup+'_vis2_%i_c%02i.png'%(f,c));
            plt.close ("all");

    # T3PHI
    names = ['MJD','T3PHI','T3PHIERR'];
    xtf, ytf, dytf = oifits.getdata (hdutf, 'OI_T3',names);
    xts, yts, dyts = oifits.getdata (hdutfs,'OI_T3',names);
    xsc, ysc, dysc = oifits.getdata (hdusci,'OI_T3',names);

    for f in range (5):
        for c in range (nc):
            fig,axes = plt.subplots (4,1, sharex=True);
            plot.base_name (axes, tstart=f*4);
            plot.compact (axes);
            plt.subplots_adjust (hspace=0.03);
            fig.suptitle (' / '.join([str(hdrs[0].get(k,'--')) for k in keys]) + ' c%i'%c);

            for bb in range (4):
                b = f*4+bb;
                ax = axes.flatten()[bb];
                ax.errorbar (xtf[:,b],ytf[:,b,c],fmt='o',yerr=dytf[:,b,c],color='k',ms=1);
                ax.errorbar (xts[:,b],yts[:,b,c],fmt='o',yerr=dyts[:,b,c],color='k',ms=1,alpha=0.25);
                ax.errorbar (xsc[:,b],ysc[:,b,c],fmt='o',yerr=dysc[:,b,c],color='g',ms=1);
            
            files.write (fig,outputDir+'/'+outputSetup+'_t3phi_%i_c%02i.png'%(f,c));
            plt.close ("all");
    
    # T3AMP
    fig,axes = plt.subplots (5,4, sharex=True);
    plot.base_name (axes);
    plot.compact (axes);
    fig.suptitle (' / '.join([str(hdrs[0].get(k,'--')) for k in keys]));

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
    
    files.write (fig,outputDir+'/'+outputSetup+'_t3amp.png');
    
    plt.close ("all");
