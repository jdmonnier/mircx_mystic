import numpy as np;
import os;

from astropy.io import fits as pyfits

import matplotlib.pyplot as plt;
import matplotlib.colors as mcolors;

from . import log, files, headers, setup, oifits, signal, plot;
from .headers import HM, HMQ, HMP, HMW, rep_nan;

def phasor (val):
    return np.exp (2.j * val / 360);

def wrap (val):
    return np.angle (np.exp (2.j * val / 360), deg=True);


def tf_time_weight (hdus, hdutf, delta):
    '''
    Compute a Transfer function file with
    time weighted interpolation.

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

        # Set nan to flagged data
        flg = np.array([h[o[0]].data['FLAG'] for h in hdutf]);
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
            
    return hdus;

def tf_divide (hdus, hdutf):
    '''
    Calibrate a SCI by a TF (which may come from the
    interpoloation of many TF).
    '''
    log.info ('Calibrate');

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

    return hdusc;

def compute_viscalib (hdrs, calibs, output='output_viscalib', delta=0.05):
    '''
    Compute the VISCAL
    Assume the baseline are ordered
    the same way in all files
    '''
    elog = log.trace ('compute_viscal');

    # Check inputs
    headers.check_input (hdrs,   required=1, maximum=1);
    headers.check_input (calibs, required=1);

    # Read transfert functions
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

        # These are not VIS_CAL_TF
        hdulist[0].header['FILETYPE'] = 'VIS_CAL_TF';
        hdutf.append (hdulist);

    # Read data
    f = hdrs[0]['ORIGNAME'];
    
    log.info ('Load SCI %s'%(f));
    hdus = pyfits.open (f);

    # Compute interpolation at the time of science
    hdutfs = tf_time_weight (hdus, hdutf, delta);

    # Divide
    hdulist = tf_divide (hdus, hdutfs);

    log.info ('Figures');

    log.info ('Create file');
    
    # First HDU
    hdulist[0].header['FILETYPE'] = 'VIS_CALIBRATED';
    hdulist[0].header[HMP+'VIS_SCI'] = os.path.basename (f);
    hdulist[0].header[HMP+'DELTA_INTERP'] = (delta,'[days] delta for weighted interpolation');

    # Write file
    files.write (hdulist, output+'.fits');
    
    plt.close ("all");
    return hdulist;
