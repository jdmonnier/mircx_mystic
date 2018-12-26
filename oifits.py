import matplotlib.pyplot as plt;

from astropy.time import Time;
from astropy.io import fits as pyfits;
from astropy.stats import sigma_clipped_stats;
from astropy import units as units
from astropy.coordinates import SkyCoord

import numpy as np
import os

from . import log, setup, files, plot, headers, version;
from .headers import HM, HMQ, HMP, HMW, rep_nan;
from .version import revision;

def create (hdr,lbd,y0=None):
    '''
    Create an OIFITS file handler (FITS) wit the 
    OI_WAVELENGTH, OI_ARRAY and OI_TARGET.
    '''

    # Spectral channel for QC
    if y0 is None: y0 = int(ny/2) - 2;
    
    # Create primary HDU
    hdu0 = pyfits.PrimaryHDU ([]);
    hdu0.header = hdr;
    hdu0.header['CONTENT']  = 'OIFITS2';
    hdu0.header['TELESCOP'] = 'CHARA';
    hdu0.header['INSTRUME'] = 'MIRCX';
    hdu0.header['INSMODE']  = hdr['CONF_NA'];
    hdu0.header['PROCSOFT'] = 'mircx_pipeline '+version.revision;
    hdu0.header['EFF_WAVE'] = (lbd[y0],'[m] central wavelength');

    # Create file
    hdulist = pyfits.HDUList ([hdu0]);

    # Create OI_WAVELENGTH table
    log.info ('Create OI_WAVELENGTH table');
    dlbd = np.abs (lbd * 0 + np.mean (np.diff(lbd)));
    tbhdu = pyfits.BinTableHDU.from_columns ( \
            [pyfits.Column (name='EFF_WAVE', format='1E', array=lbd, unit='m'), \
             pyfits.Column (name='EFF_BAND', format='1E', array=dlbd, unit='m')]);

    tbhdu.header['EXTNAME'] = 'OI_WAVELENGTH';
    tbhdu.header['INSNAME'] = 'MIRCX';
    tbhdu.header['OI_REVN'] = 2;
    hdulist.append(tbhdu);

    # Create OI_TARGET table
    log.info ('Create OI_TARGET table');
    name = hdr['OBJECT'];
    coord = SkyCoord (hdr['RA'],hdr['DEC'], unit=(units.hourangle, units.deg));
    ra0  = coord.ra.to('deg').value;
    dec0 = coord.dec.to('deg').value;
    parallax = hdr['PARALLAX'] * units.arcsec.to('deg');
    spectype = hdr['SPECTYPE'];
    pmra = hdr['PM_RA'] * units.rad.to('deg');
    pmdec = hdr['PM_DEC'] * units.rad.to('deg');
    
    tbhdu = pyfits.BinTableHDU.from_columns ([\
             pyfits.Column (name='TARGET_ID', format='I', array=[1]), \
             pyfits.Column (name='TARGET', format='A16', array=[name]), \
             pyfits.Column (name='RAEP0', format='D', array=[ra0], unit='deg'), \
             pyfits.Column (name='DECEP0', format='D', array=[dec0], unit='deg'), \
             pyfits.Column (name='EQUINOX', format='E', array=[2000], unit='year'), \
             pyfits.Column (name='RA_ERR', format='D', array=[0.0], unit='deg'), \
             pyfits.Column (name='DEC_ERR', format='D', array=[0.0], unit='deg'), \
             pyfits.Column (name='SYSVEL', format='D', array=[0.0], unit='m/s'), \
             pyfits.Column (name='VELTYP', format='A8', array=['LSR']), \
             pyfits.Column (name='VELDEF', format='A8', array=['OPTICAL']), \
             pyfits.Column (name='PMRA', format='D', array=[pmra], unit='deg/yr'), \
             pyfits.Column (name='PMDEC', format='D', array=[pmdec], unit='deg/yr'), \
             pyfits.Column (name='PMRA_ERR', format='D', array=[0.0], unit='deg/yr'), \
             pyfits.Column (name='PMDEC_ERR', format='D', array=[0.0], unit='deg/yr'), \
             pyfits.Column (name='PARALLAX', format='E', array=[parallax], unit='deg'), \
             pyfits.Column (name='PARA_ERR', format='E', array=[0.0], unit='deg'), \
             pyfits.Column (name='SPECTYP', format='A16', array=[spectype]) ]);

    tbhdu.header['EXTNAME'] = 'OI_TARGET';
    tbhdu.header['OI_REVN'] = 2;
    hdulist.append(tbhdu);

    # Create OI_ARRAY table
    log.info ('Create OI_ARRAY table');
    diameter = np.ones (6) * 1.0;
    staindex = range (1,7);
    telname = ['S1','S2','E1','E2','W1','W2'];
    staname = telname;
    fov     = np.ones (6) * 0.320;
    fovtype = ['FWHM','FWHM','FWHM','FWHM','FWHM','FWHM'];

    # Get staxyz from header or default
    telpos = setup.tel_xyz (hdr);
    staxyz = np.array ([telpos[t] for t in telname]);

    tbhdu = pyfits.BinTableHDU.from_columns ([\
             pyfits.Column (name='TEL_NAME',  format='A16', array=telname), \
             pyfits.Column (name='STA_NAME',  format='A16', array=staname), \
             pyfits.Column (name='STA_INDEX', format='I', array=staindex), \
             pyfits.Column (name='DIAMETER', format='E', array=diameter, unit='m'), \
             pyfits.Column (name='STAXYZ', format='3D', dim='(3)', array=staxyz, unit='m'), \
             pyfits.Column (name='FOV', format='D', array=fov, unit='arc sec'), \
             pyfits.Column (name='FOVTYPE', format='A6', array=fovtype)]);
    
    tbhdu.header['EXTNAME'] = 'OI_ARRAY';
    tbhdu.header['ARRNAME'] = 'CHARA';
    tbhdu.header['OI_REVN'] = 2;
    tbhdu.header['FRAME'] = 'GEOCENTRIC';
    tbhdu.header['ARRAYX'] = (-2476998.047780,'[m]');
    tbhdu.header['ARRAYY'] = (-4647390.089884,'[m]');
    tbhdu.header['ARRAYZ'] = (3582240.612296,'[m]');
    
    hdulist.append(tbhdu);
    
    return hdulist;


def add_vis2 (hdulist,mjd0,u_power,b_power,l_power,output='output',y0=None):
    '''
    Compute the OI_VIS2 table from a sample of observations
    u_power, b_power, l_power shall be (sample, lbd, base)
    mjd shall be (sample)
    '''

    log.info ('Compute OI_VIS2');
    hdr = hdulist[0].header;
    ns,ny,nb = u_power.shape;

    # Spectral channel for QC
    if y0 is None: y0 = int(ny/2) - 2;
        
    # Remove warning for invalid
    old_np_setting = np.seterr (divide='ignore',invalid='ignore');

    # How many valid frame
    valid = np.isfinite (u_power) * np.isfinite (l_power) * np.isfinite (b_power);
    nvalid = np.nansum (1. * valid, axis=0);

    # Compute bootstrap sample
    boot = (np.random.random ((ns,20)) * ns).astype(int);
    boot[:,0] = range (ns);

    # Compute mean bias over spatial frequencies
    bm_power = np.mean (b_power, axis=-1, keepdims=True);
    ub_power = u_power - bm_power;

    # Compute mean vis2
    vis2 = np.nanmean (ub_power[boot,:,:], axis=0) / np.nanmean (l_power[boot,:,:], axis=0);
    vis2err = np.nanstd (vis2, axis=0);
    vis2 = vis2[0,:,:];

    # Systematic due to bias spatial structure
    vis2sys = np.std (np.nanmean (b_power, axis=0), axis=-1, keepdims=True) \
            / np.nanmean (l_power, axis=0);

    # Enlarge error with systematics. FIXME: this somewhat conservative
    # as we estimate the bias with several frequencies.
    log.info ('Enlarge statistical errors with systematics');
    vis2err = np.sqrt (vis2err**2 + vis2sys**2);

    # Construct mjd[ns,ny,nb]
    mjd = mjd0[:,None,None] * np.ones (valid.shape);
    mjd[~valid] = np.nan;
    
    # Average MJD per baseline
    int_time = (np.nanmax (mjd, axis=(0,1)) - np.nanmin (mjd, axis=(0,1))) * 24 * 3600;
    mjd = np.nanmean (mjd, axis=(0,1));

    # Baseline fully rejected should have a MJD anyway
    mjd[~np.isfinite(mjd)] = hdr['MJD-OBS'];
    
    # Create OI_VIS table
    target_id = np.ones (nb).astype(int);
    time = mjd * 0.0;
    staindex = setup.beam_index(hdr)[setup.base_beam()];
    # ucoord, vcoord = setup.base_uv (hdr);
    ucoord, vcoord = setup.compute_base_uv (hdr, mjd=mjd);
    
    # Flag data
    flag = ~np.isfinite (vis2) + ~np.isfinite (vis2err);

    tbhdu = pyfits.BinTableHDU.from_columns ([\
             pyfits.Column (name='TARGET_ID', format='I', array=target_id), \
             pyfits.Column (name='TIME', format='D', array=time, unit='s'), \
             pyfits.Column (name='MJD', format='D', array=mjd,unit='day'), \
             pyfits.Column (name='INT_TIME', format='D', array=int_time, unit='s'), \
             pyfits.Column (name='VIS2DATA', format='%iD'%ny, array=vis2.T), \
             pyfits.Column (name='VIS2ERR', format='%iD'%ny, array=vis2err.T), \
             pyfits.Column (name='UCOORD', format='D', array=ucoord, unit='m'), \
             pyfits.Column (name='VCOORD', format='D', array=vcoord, unit='m'), \
             pyfits.Column (name='STA_INDEX', format='2I', array=staindex), \
             pyfits.Column (name='FLAG', format='%iL'%ny, array=flag.T)
             ]);
    
    tbhdu.header['EXTNAME'] = 'OI_VIS2';
    tbhdu.header['INSNAME'] = 'MIRCX';
    tbhdu.header['ARRNAME'] = 'CHARA';
    tbhdu.header['OI_REVN'] = 2;
    tbhdu.header['DATE-OBS'] = hdr['DATE-OBS'];
    hdulist.append(tbhdu);

    # QC for VIS2
    for b,name in enumerate (setup.base_name ()):
        hdr[HMQ+'REJECTED'+name] = (1.0*(ns - nvalid[y0,b])/ns,'fraction of rejected');
        val = rep_nan (np.nanmean (ub_power[:,y0,b]));
        hdr[HMQ+'UPOWER'+name+' MEAN'] = (val,'unbiased power at lbd0');
        val = rep_nan (np.nanstd (ub_power[:,y0,b]));
        hdr[HMQ+'UPOWER'+name+' STD'] = (val,'unbiased power at lbd0');
        val = rep_nan (np.nanmean (l_power[:,y0,b]));
        hdr[HMQ+'LPOWER'+name+' MEAN'] = (val,'unbiased power at lbd0');
        val = rep_nan (np.nanstd (l_power[:,y0,b]));
        hdr[HMQ+'LPOWER'+name+' STD'] = (val,'unbiased power at lbd0');
        val = rep_nan (vis2[y0,b]);
        hdr[HMQ+'VISS'+name+' MEAN'] = (val,'visibility at lbd0');
        val = rep_nan (vis2err[y0,b]);
        hdr[HMQ+'VISS'+name+' ERR'] = (val,'visibility at lbd0');
        val = rep_nan (ucoord[b]);
        hdr[HMQ+'UCOORD'+name] = (val,'[m] u coordinate');
        val = rep_nan (vcoord[b]);
        hdr[HMQ+'VCOORD'+name] = (val,'[m] v coordinate');
        val = rep_nan (np.sqrt(ucoord[b]**2 + vcoord[b]**2));
        hdr[HMQ+'BASELENGTH'+name] = (val,'[m] uv coordinate');
        
    # Correlation plot
    log.info ('Correlation plots');
    fig,axes = plt.subplots (5,3, sharex=True);
    fig.subplots_adjust (wspace=0.3, hspace=0.1);
    fig.suptitle (headers.summary (hdr));
    plot.base_name (axes);
    plot.compact (axes);
    
    for b,ax in enumerate(axes.flatten()):
        
        datax = l_power[:,y0,b];
        datay = ub_power[:,y0,b];
    
        scalex = rep_nan (np.abs (np.nanmax (datax)), 1.);
        scaley = rep_nan (np.abs (np.nanmax (datay)), 1.);
        scale  = rep_nan (scaley/scalex, 1.);
        ax.plot (datax/scalex, datay/scalex, '+', alpha=0.75, ms=4);

        ax.set_xlim (-0.1,1.1);
        ax.set_ylim (-0.1*scale,1.1*scale);
        plot.scale (ax, scalex);
        
        ax.plot ([0], [0], '+r', alpha=0.75, ms=4);
        ax.plot ([0,2.0], [0,2.*vis2[y0,b]],
                  '-r', alpha=0.5);
        ax.plot ([0,2.0], [0,2.*(vis2+vis2err)[y0,b]],
                  '--r', alpha=0.5);
        ax.plot ([0,2.0], [0,2.*(vis2-vis2err)[y0,b]],
                  '--r', alpha=0.5);

    files.write (fig,output+'_norm_power.png');

    # Pseudo PSD
    fig,ax = plt.subplots (2,2, sharey='row',sharex='col');
    fig.suptitle (headers.summary (hdr));
    ax[0,0].imshow (np.nanmean (u_power, axis=0),aspect='auto');
    ax[0,1].imshow (np.nanmean (b_power, axis=0),aspect='auto');
    ax[1,0].plot (np.nanmean (u_power, axis=0).T);
    ax[1,1].plot (np.nanmean (b_power, axis=0).T);
    ax[0,0].set_title ('Fringe frequencies');
    ax[0,1].set_title ('Bias frequencies');
    files.write (fig,output+'_psd.png');

    # Reset warning
    np.seterr (**old_np_setting);

def add_vis (hdulist,mjd0, c_cpx, c_norm, output='output',y0=None):
    '''
    Compute the OI_VIS table from a sample of observations
    c_power shall be of size (sample, lbd, base)
    mjd shall be (sample)
    '''

    log.info ('Compute OI_VIS');
    hdr = hdulist[0].header;
    ns,ny,nb = c_cpx.shape;

    # Spectral channel for QC
    if y0 is None: y0 = int(ny/2) - 2;
        
    # Remove warning for invalid
    old_np_setting = np.seterr (divide='ignore',invalid='ignore');

    # How many valid frame
    valid = np.isfinite (c_cpx) * np.isfinite (c_norm);
    nvalid = np.nansum (1. * valid, axis=0);

    # Compute bootstrap sample
    boot = (np.random.random ((ns,20)) * ns).astype(int);
    boot[:,0] = range (ns);

    # Compute mean vis
    vis = np.nanmean (c_cpx[boot,:,:], axis=0) / np.nanmean (c_norm[boot,:,:], axis=0);
    visAmp = np.abs (vis[0,:,:]);
    visAmperr = np.nanstd (np.abs (vis), axis=0);
    visPhi = np.angle (vis[0,:,:], deg=True);
    visPhierr = np.nanstd (np.angle (vis, deg=True), axis=0);
    
    # Construct mjd[ns,ny,nb]
    mjd = mjd0[:,None,None] * np.ones (valid.shape);
    mjd[~valid] = np.nan;
    
    # Average MJD per baseline
    int_time = (np.nanmax (mjd, axis=(0,1)) - np.nanmin (mjd, axis=(0,1))) * 24 * 3600;
    mjd = np.nanmean (mjd, axis=(0,1));

    # Baseline fully rejected should have a MJD anyway
    mjd[~np.isfinite(mjd)] = hdr['MJD-OBS'];
    
    # Create OI_VIS table
    target_id = np.ones (nb).astype(int);
    time = mjd * 0.0;
    staindex = setup.beam_index(hdr)[setup.base_beam()];
    # ucoord, vcoord = setup.base_uv (hdr);
    ucoord, vcoord = setup.compute_base_uv (hdr, mjd=mjd);
    
    # Flag data
    flag = ~np.isfinite (visPhi) + ~np.isfinite (visPhierr);

    tbhdu = pyfits.BinTableHDU.from_columns ([\
             pyfits.Column (name='TARGET_ID', format='I', array=target_id), \
             pyfits.Column (name='TIME', format='D', array=time, unit='s'), \
             pyfits.Column (name='MJD', format='D', array=mjd,unit='day'), \
             pyfits.Column (name='INT_TIME', format='D', array=int_time, unit='s'), \
             pyfits.Column (name='VISPHI', format='%iD'%ny, array=visPhi.T,unit='deg'), \
             pyfits.Column (name='VISPHIERR', format='%iD'%ny, array=visPhierr.T,unit='deg'), \
             pyfits.Column (name='VISAMP', format='%iD'%ny, array=visAmp.T), \
             pyfits.Column (name='VISAMPERR', format='%iD'%ny, array=visAmperr.T), \
             pyfits.Column (name='UCOORD', format='D', array=ucoord, unit='m'), \
             pyfits.Column (name='VCOORD', format='D', array=vcoord, unit='m'), \
             pyfits.Column (name='STA_INDEX', format='2I', array=staindex), \
             pyfits.Column (name='FLAG', format='%iL'%ny, array=flag.T)
             ]);
    
    tbhdu.header['EXTNAME'] = 'OI_VIS';
    tbhdu.header['INSNAME'] = 'MIRCX';
    tbhdu.header['ARRNAME'] = 'CHARA';
    tbhdu.header['OI_REVN'] = 2;
    tbhdu.header['DATE-OBS'] = hdr['DATE-OBS'];
    hdulist.append(tbhdu);

    # Reset warning
    np.seterr (**old_np_setting);

    
def add_flux (hdulist,mjd0,p_flux,output='output',y0=None):
    '''
    Compute the OI_FLUX table from a sample of observations
    t_flux shall be (sample, lbd, tel)
    mjd shall be (sample)
    '''
    
    log.info ('Compute OI_FLUX');
    hdr = hdulist[0].header;
    ns,ny,nt = p_flux.shape;

    # Spectral channel for QC
    if y0 is None: y0 = int(ny/2) - 2;
        
    # Remove warning for invalid
    old_np_setting = np.seterr (divide='ignore',invalid='ignore');
        
    # How many valid frame.
    valid = np.isfinite (p_flux);
    nvalid = np.nansum (1. * valid, axis=0);
    
    # Compute bootstrap sample
    boot = (np.random.random ((ns,20)) * ns).astype(int);
    boot[:,0] = range (ns);

    # Compute mean flux
    flux = np.nanmean (p_flux[boot,:,:], axis=0);
    fluxerr = np.nanstd (flux, axis=0);
    flux = flux[0,:,:];

    # Construct mjd[ns,ny,nt]
    mjd = mjd0[:,None,None] * np.ones (valid.shape);
    mjd[~valid] = np.nan;
    
    # Average MJD per baseline
    int_time = (np.nanmax (mjd, axis=(0,1)) - np.nanmin (mjd, axis=(0,1))) * 24 * 3600;
    mjd = np.nanmean (mjd, axis=(0,1));

    # Baseline fully rejected should have a MJD anyway
    mjd[~np.isfinite(mjd)] = hdr['MJD-OBS'];
    
    # Create OI_FLUX table
    target_id = np.ones (nt).astype(int);
    time = mjd * 0.0;
    staindex = setup.beam_index(hdr);
    
    # Flag data
    flag = ~np.isfinite (flux) + ~np.isfinite (fluxerr);

    tbhdu = pyfits.BinTableHDU.from_columns ([\
             pyfits.Column (name='TARGET_ID', format='I', array=target_id), \
             pyfits.Column (name='TIME', format='D', array=time, unit='s'), \
             pyfits.Column (name='MJD', format='D', array=mjd,unit='day'), \
             pyfits.Column (name='INT_TIME', format='D', array=int_time, unit='s'), \
             pyfits.Column (name='FLUXDATA', format='%iD'%ny, array=flux.T, unit='adu'), \
             pyfits.Column (name='FLUXERR', format='%iD'%ny, array=fluxerr.T, unit='adu'), \
             pyfits.Column (name='STA_INDEX', format='1I', array=staindex), \
             pyfits.Column (name='FLAG', format='%iL'%ny, array=flag.T)
             ]);
    
    tbhdu.header['EXTNAME'] = 'OI_FLUX';
    tbhdu.header['INSNAME'] = 'MIRCX';
    tbhdu.header['ARRNAME'] = 'CHARA';
    tbhdu.header['OI_REVN'] = 1;
    tbhdu.header['CALSTAT'] = 'U';
    tbhdu.header['DATE-OBS'] = hdr['DATE-OBS'];
    hdulist.append(tbhdu);
    
    # Reset warning
    np.seterr (**old_np_setting);

def add_t3 (hdulist,mjd0,t_product,t_norm,output='output',y0=None):
    '''
    Compute the OI_T3 table from a sample of observations
    t_product and t_norm shall be (sample, lbd, base)
    mjd shall be (sample)
    '''
    
    log.info ('Compute OI_T3');
    hdr = hdulist[0].header;
    ns,ny,nt = t_product.shape;

    # Spectral channel for QC
    if y0 is None: y0 = int(ny/2) - 2;
        
    # Remove warning for invalid
    old_np_setting = np.seterr (divide='ignore',invalid='ignore');

    # Discard triple product without amplitude
    t_product[t_product==0] = np.nan;
    t_norm[t_norm==0] = np.nan;
    
    # How many valid frame. Note that valid is defined for T3PHI
    # So the mean MJD is the one of the T3
    # valid = np.isfinite (t_product) * np.isfinite (t_norm);
    valid = np.isfinite (t_product);
    nvalid = np.nansum (1. * valid, axis=0);

    # Compute bootstrap sample
    boot = (np.random.random ((ns,20)) * ns).astype(int);
    boot[:,0] = range (ns);

    # Compute mean phase
    t3phi = np.angle (np.nanmean (t_product[boot,:,:], axis=0));
    t3phiErr = np.nanstd (t3phi, axis=0);
    t3phi = t3phi[0,:,:];

    # Compute mean norm
    t3amp = np.abs (np.nanmean (t_product[boot,:,:], axis=0)) / np.nanmean (t_norm[boot,:,:], axis=0);
    t3ampErr = np.nanstd (t3amp, axis=0);
    t3amp = t3amp[0,:,:];
    
    # Construct mjd[ns,ny,nb]
    mjd = mjd0[:,None,None] * np.ones (valid.shape);
    mjd[~valid] = np.nan;
    
    # Average MJD per baseline
    int_time = (np.nanmax (mjd, axis=(0,1)) - np.nanmin (mjd, axis=(0,1))) * 24 * 3600;
    mjd  = np.nanmean (mjd, axis=(0,1));

    # Baseline fully rejected should have a MJD anyway
    mjd[~np.isfinite(mjd)] = hdr['MJD-OBS'];
    
    # Create OI_T3 table
    target_id = np.ones (nt).astype(int);
    time = mjd * 0.0;
    staindex = setup.beam_index (hdr)[setup.triplet_beam()];
    # ucoord, vcoord = setup.base_uv (hdr);
    # u1coord = ucoord[setup.triplet_base()[:,0]];
    # v1coord = vcoord[setup.triplet_base()[:,0]];
    # u2coord = ucoord[setup.triplet_base()[:,1]];
    # v2coord = vcoord[setup.triplet_base()[:,1]];
    u1coord, v1coord = setup.compute_base_uv (hdr, mjd=mjd, baseid='base1');
    u2coord, v2coord = setup.compute_base_uv (hdr, mjd=mjd, baseid='base2');

    # Flag data
    flag = ~np.isfinite (t3phi) + ~np.isfinite (t3phiErr);
    
    tbhdu = pyfits.BinTableHDU.from_columns ([\
             pyfits.Column (name='TARGET_ID', format='I', array=target_id), \
             pyfits.Column (name='TIME', format='D', array=time, unit='s'), \
             pyfits.Column (name='MJD', format='D', array=mjd,unit='day'), \
             pyfits.Column (name='INT_TIME', format='D', array=int_time, unit='s'), \
             pyfits.Column (name='T3PHI', format='%iD'%ny, array=t3phi.T*180/np.pi, unit='deg'), \
             pyfits.Column (name='T3PHIERR', format='%iD'%ny, array=t3phiErr.T*180/np.pi, unit='deg'), \
             pyfits.Column (name='T3AMP', format='%iD'%ny, array=t3amp.T), \
             pyfits.Column (name='T3AMPERR', format='%iD'%ny, array=t3ampErr.T), \
             pyfits.Column (name='U1COORD', format='D', array=u1coord, unit='m'), \
             pyfits.Column (name='V1COORD', format='D', array=v1coord, unit='m'), \
             pyfits.Column (name='U2COORD', format='D', array=u2coord, unit='m'), \
             pyfits.Column (name='V2COORD', format='D', array=v2coord, unit='m'), \
             pyfits.Column (name='STA_INDEX', format='3I', array=staindex), \
             pyfits.Column (name='FLAG', format='%iL'%ny, array=flag.T)
             ]);
    
    tbhdu.header['EXTNAME'] = 'OI_T3';
    tbhdu.header['INSNAME'] = 'MIRCX';
    tbhdu.header['ARRNAME'] = 'CHARA';
    tbhdu.header['OI_REVN'] = 2;
    tbhdu.header['DATE-OBS'] = hdr['DATE-OBS'];
    hdulist.append (tbhdu);
    
    # QC for T3
    for t,name in enumerate (setup.triplet_name()):
        hdr[HMQ+'REJECTED'+name] = (1.0*(ns - nvalid[y0,t])/ns,'fraction of rejected');
        val = rep_nan (t3phi[y0,t])*180/np.pi;
        hdr[HMQ+'T3PHI'+name+' MEAN'] = (val,'[deg] tphi at lbd0');
        val = rep_nan (t3phiErr[y0,t])*180/np.pi;
        hdr[HMQ+'T3PHI'+name+' ERR'] = (val,'[deg] visibility at lbd0');
    
    # Correlation plot
    log.info ('Correlation plots');
    fig,axes = plt.subplots (5,4, sharex=True, sharey=True);
    fig.subplots_adjust (wspace=0.2, hspace=0.1);
    fig.suptitle (headers.summary (hdr));
    plot.base_name (axes);
    plot.compact (axes);
    
    for t,ax in enumerate(axes.flatten()):
        data = t_product[:,y0,t];
        scale = np.nanmax (np.abs (data));
        
        ax.plot (data.real/scale, data.imag/scale, '+', alpha=0.75, ms=4);
        
        ax.set_xlim (-1.05, +1.05);
        ax.set_ylim (-1.05, +1.05);
        plot.scale (ax, scale);
        
        ax.plot ([0], [0], '+r', alpha=0.75, ms=4);
        ax.plot ([0,2.*np.cos(t3phi[y0,t])], \
                 [0,2.*np.sin(t3phi[y0,t])], \
                 '-r', alpha=0.5);
        ax.plot ([0,2.*np.cos((t3phi+t3phiErr)[y0,t])], \
                 [0,2.*np.sin((t3phi+t3phiErr)[y0,t])], \
                 '--r', alpha=0.5);
        ax.plot ([0,2.*np.cos((t3phi-t3phiErr)[y0,t])], \
                 [0,2.*np.sin((t3phi-t3phiErr)[y0,t])], \
                 '--r', alpha=0.5);

    files.write (fig,output+'_bispec.png');

    # Reset warning
    np.seterr (**old_np_setting);


def getdata (hdus,ext,names,flag=True):
    '''
    Get the data of many OIFITS handler and return as an array
    (actually a tupple of array since names is multiple entries).

    If the option flag is True, the flagged data are replaced by nan.
    '''
    out = [];

    # Read flag if needed
    if flag:
        ff = np.array ([h[ext].data['FLAG'] for h in hdus]);

    # Loop on variables to be read
    for n in names:
        d = np.array ([h[ext].data[n] for h in hdus]);
        if flag and d.shape == ff.shape: d[ff] = np.nan;
        out.append (d);
        
    return out;

    # Old code
    # return [np.array([h[ext].data[n] for h in hdus]) for n in names];

