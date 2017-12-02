import numpy as np;
import os;

from astropy.io import fits as pyfits

import matplotlib.pyplot as plt;
import matplotlib.colors as mcolors;

from . import log, files, headers, setup, oifits, signal, plot;
from .headers import HM, HMQ, HMP, HMW, rep_nan;

def phasor (val):
    return np.exp (2.j * np.pi * val / 360);

def wrap (val):
    return np.angle (np.exp (2.j * np.pi * val / 360), deg=True);


def tf_time_weight (hdus, hdutf, delta):
    '''
    Average the Transfer function at the time of a science
    observation with time weighted interpolation.
    delta is in [days]
    '''
    log.info ('Interpolate %i TF with time_weight'%len(hdutf));

    # Copy VIS_SCI to build VIS_TF
    hdutfs = pyfits.HDUList([hdu.copy() for hdu in hdus]);
    hdutfs[0].header['FILETYPE'] = 'VIS_SCI_TF';
    
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

        # When we want to interpolate
        mjd0 = hdus[o[0]].data['MJD'];

        # Compute the weighted mean
        weight = np.exp (-(mjd0[None,:,None]-mjd[:,:,None])**2/delta**2) / err**2;
        tf = np.nansum (val * weight, axis=0) / np.nansum (weight, axis=0);

        # Replace by phasor
        if o[3] is True: tf = np.angle (tf, deg=True);

        # Set data
        hdutfs[o[0]].data[o[1]] = tf;
        hdutfs[o[0]].data['FLAG'] += ~np.isfinite (tf);
            
    return hdutfs;

def tf_divide (hdus, hdutf):
    '''
    Calibrate a SCI by a TF (which may come from the
    interpoloation of many TF).
    '''

    # Copy VIS_SCI to build VIS_CALIBRATED
    hdusc = pyfits.HDUList([hdu.copy() for hdu in hdus]);
    hdusc[0].header['FILETYPE'] = 'VIS_CALIBRATED';

    obs = [['OI_VIS2','VIS2DATA','VIS2ERR',False],
           ['OI_T3','T3AMP','T3AMPERR',False],
           ['OI_T3','T3PHI','T3PHIERR',True]];

    for o in obs:
        if o[3] is True:
            hdusc[o[0]].data[o[1]] -= hdutf[o[0]].data[o[1]];
            hdusc[o[0]].data[o[1]]  = wrap (hdusc[o[0]].data[o[1]]);
        else:
            hdusc[o[0]].data[o[1]] /= hdutf[o[0]].data[o[1]];
            hdusc[o[0]].data[o[2]] /= hdutf[o[0]].data[o[1]];
            
        hdusc[o[0]].data['FLAG'] += ~np.isfinite (hdusc[o[0]].data[o[1]]);

    return hdusc;
    
def compute_all_viscalib (hdrs, catalog, delta=0.05,
                          outputDir='viscal/', overwrite=True):
    '''
    Cross-calibrate the VIS in hdrs. The choice of SCI and CAL, and the diameter
    of the calibration stars, are specified with the catalog. Catalog should be
    of the form [('name1',diam1,err1),('name2',diam2,err2),...] where the diam
    and err are in [mas].
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

        # Get spatial frequencies in [rad-1]
        lbd = hdulist['OI_WAVELENGTH'].data['EFF_WAVE'];
        fu = hdulist['OI_VIS2'].data['UCOORD'][:,None] / lbd[None,:];
        fv = hdulist['OI_VIS2'].data['VCOORD'][:,None] / lbd[None,:];

        # Compute the TF
        v2 = signal.airy (diam * np.sqrt(fu*fu+fv*fv))**2;
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
        hdulist[0].header[HMP+'DELTA_INTERP'] = (delta,'[days] delta for weighted interpolation');

        # Write file
        files.write (hdulist, output+'.fits');

        log.info ('Figures');
    
        # VIS2
        fig,axes = plt.subplots ();
        fig.suptitle (headers.summary (sci));
        x  = np.sqrt (hdulist['OI_VIS2'].data['UCOORD']**2 + hdulist['OI_VIS2'].data['VCOORD']**2)[:,None] / hdulist['OI_WAVELENGTH'].data['EFF_WAVE'][None,:];
        y  = hdulist['OI_VIS2'].data['VIS2DATA'];
        dy = hdulist['OI_VIS2'].data['VIS2ERR'];
        for b in range (15):
            axes.errorbar (1e-6*x[b,:],y[b,:],yerr=dy[b,:],fmt='o',ms=1);
        axes.set_ylim (-0.1,1.2);
        axes.set_xlim (0);
        axes.set_xlabel ('sp. freq. (Mlbd)');
        axes.set_ylabel ('vis2');
        files.write (fig,output+'_vis2.png');
         
        # Append VIS_SCI and VIS_SCI_TF, to allow a plot
        # of a trend over the night for this setup
        hdusci.append (hdus);
        hdutfs.append (hdutfsi);
        
    log.info ('Figures for the trends');

    # VIS2
    fig,axes = plt.subplots (5,3, sharex=True);
    # fig.suptitle (headers.summary (sci));
    plot.base_name (axes);
    plot.compact (axes);
    for b in range (15):
        ax = axes.flatten()[b];
        x  = [h['OI_VIS2'].data['MJD'][b] for h in hdutf];
        y  = [h['OI_VIS2'].data['VIS2DATA'][b,6] for h in hdutf];
        dy = [h['OI_VIS2'].data['VIS2ERR'][b,6] for h in hdutf];
        ax.errorbar (x,y,fmt='o',yerr=dy,color='k',ms=1);
        x  = [h['OI_VIS2'].data['MJD'][b] for h in hdutfs];
        y  = [h['OI_VIS2'].data['VIS2DATA'][b,6] for h in hdutfs];
        dy = [h['OI_VIS2'].data['VIS2ERR'][b,6] for h in hdutfs];
        ax.errorbar (x,y,fmt='o',yerr=dy,color='k',ms=1,alpha=0.25);
        x  = [h['OI_VIS2'].data['MJD'][b] for h in hdusci];
        y  = [h['OI_VIS2'].data['VIS2DATA'][b,6] for h in hdusci];
        dy = [h['OI_VIS2'].data['VIS2ERR'][b,6] for h in hdusci];
        ax.errorbar (x,y,fmt='o',yerr=dy,color='g',ms=1);
    
    files.write (fig,output+'_all_vis2.png');

    # T3PHI
    fig,axes = plt.subplots (5,4, sharex=True);
    # fig.suptitle (headers.summary (hdr));
    plot.base_name (axes);
    plot.compact (axes);
    for b in range (20):
        ax = axes.flatten()[b];
        x  = [h['OI_T3'].data['MJD'][b] for h in hdutf];
        y  = [h['OI_T3'].data['T3PHI'][b,6] for h in hdutf];
        dy = [h['OI_T3'].data['T3PHIERR'][b,6] for h in hdutf];
        ax.errorbar (x,y,fmt='o',yerr=dy,color='k',ms=1);
        x  = [h['OI_T3'].data['MJD'][b] for h in hdutfs];
        y  = [h['OI_T3'].data['T3PHI'][b,6] for h in hdutfs];
        dy = [h['OI_T3'].data['T3PHIERR'][b,6] for h in hdutfs];
        ax.errorbar (x,y,fmt='o',yerr=dy,color='k',ms=1,alpha=0.25);
        x  = [h['OI_T3'].data['MJD'][b] for h in hdusci];
        y  = [h['OI_T3'].data['T3PHI'][b,6] for h in hdusci];
        dy = [h['OI_T3'].data['T3PHIERR'][b,6] for h in hdusci];
        ax.errorbar (x,y,fmt='o',yerr=dy,color='g',ms=1);
    
    files.write (fig,output+'_all_t3phi.png');
    
    # T3AMP
    fig,axes = plt.subplots (5,4, sharex=True);
    # fig.suptitle (headers.summary (hdr));
    plot.base_name (axes);
    plot.compact (axes);
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
    
    files.write (fig,output+'_all_t3amp.png');
    
    plt.close ("all");
    return hdulist;
