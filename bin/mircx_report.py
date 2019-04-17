#! /usr/bin/env python                                                          
# -*- coding: iso-8859-15 -*-                                                   

# Updates:
# 2019 - 04 - 10: CLD incorporated instrument sensitivity assessment 
#                 to convert transmission to units of % of expected stellar flux

import mircx_pipeline as mrx;
import argparse, glob, os;
import numpy as np;
from astropy.io import fits as pyfits;
import matplotlib.pyplot as plt;

from mircx_pipeline import log, setup, plot, files, signal, headers, qc;
from mircx_pipeline.headers import HM, HMQ, HMP;

#
# Implement options
#

# Describe the script
description = \
"""
description:
 Plot a report of data reduced by the pipeline.
 Should be run in a directory where the OIFITS
 are stored or use the option --oifits-dir

"""

epilog = \
"""
examples:

   cd /my/reduced/data/oifits/
  mircx_report.py


"""

TrueFalse = ['TRUE','FALSE'];

parser = argparse.ArgumentParser (description=description, epilog=epilog,
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 add_help=True);

parser.add_argument ("--oifits-dir", dest="oifits_dir",default='./',type=str,
                     help="directory of products [%(default)s]");

parser.add_argument ("--snr-threshold", dest="snr_threshold", type=float,
                     default=5.0, help="SNR threshold for plotting value [%(default)s]");

parser.add_argument ("--vis2-threshold", dest="vis2_threshold", type=float,
                     default=0.1, help="Vis2 threshold for plotting TF value [%(default)s]");

parser.add_argument ("--only-reference", dest="only_reference",default='FALSE',
                     choices=TrueFalse,
                     help="Use only REFERENCE stars [%(default)s]");

#
# Initialisation
#

# Matplotlib
import matplotlib as mpl;
mpl.rcParams['lines.markersize'] = 2;
     
# Remove warning for invalid
np.seterr (divide='ignore',invalid='ignore');

# Parse argument
argopt = parser.parse_args ();

# Verbose
elog = log.trace ('mircx_report');

# List of basename
bname = setup.base_name ();

# Load all the headers
hdrs = mrx.headers.loaddir (argopt.oifits_dir);

# Sort the headers by time
ids = np.argsort ([h['MJD-OBS'] for h in hdrs]);
hdrs = [hdrs[i] for i in ids];

# Keep only reference stars
if argopt.only_reference == 'TRUE':
    hdrs = [h for h in hdrs if 'OBJECT_TYPE' in h and h['OBJECT_TYPE'] == 'REFERENCE'];


#
# Query CDS to build a catalog of object information
# This is protected since it may fail
#

# Load astroquery
try:
    from astroquery.vizier import Vizier;
    log.info ('Load astroquery.vizier');
except:
    log.warning ('Cannot load astroquery.vizier, try:');
    log.warning ('sudo conda install -c astropy astroquery');
    

# List of object
objlist = list(set([h['OBJECT'] for h in hdrs]));
objcat = dict();

for obj in objlist:
    try:
        cat = Vizier.query_object (obj, catalog='JSDC')[0];
        log.info ('Find JSDC for '+obj);
        log.info ("diam = %.3f mas"%cat['UDDH'][0]);
        log.info ("Hmag = %.3f mas"%cat['Hmag'][0]);
        objcat[obj] = cat;
    except:
        log.info ('Cannot find JSDC for '+obj);


#
# Compute the transmission and instrumental visibility
#
# Zero point of 2MASS:H from Cohen et al. (2003, AJ 126, 1090):
Hzp = 9.464537e6 # [photons/millisec/m2/mircons]
# internal transmission * quantum efficiency from Cyprien [dimensionless]:
iTQE    = 0.5
# collecting area of 1 telescope (assuming circular aperture) [m2]:
telArea = np.pi * 0.5*0.5

kl = 0 # dummy character to avoid error message being output multiple times
for h in hdrs:
    expT = h['EXPOSURE']   # exposure time in             [millisec]
    bWid = h['BANDWID']    # spectral bandwidth           [microns]
    gain = 0.5 * h['GAIN'] # conversion gain from Cyprien [ADU/e]
    
    # If we have the info about this star
    try:
        Hmag = float(objcat[h['OBJECT']]['Hmag'][0]) #    [mag]
        fH   = Hzp * 10**(-Hmag/2.5)
        # expected flux based on instrument sensitivity [photon/frame]:
        fExpect = fH * expT * bWid * telArea * iTQE
    
        # loop over the beams:
        for b in range (6):
            # measured flux integrated over spectral band [photon/frame]:
            fMeas = h[HMQ+'BANDFLUX%i MEAN'%b] / gain
            # transmission [%]:
            h[HMQ+'TRANS%i'%b] = 100.* (fMeas / fExpect)
    
        diam    = objcat[h['OBJECT']]['UDDH'][0]
        # Loop on baseline 
        for b in bname:
            vis2 = h[HMQ+'VISS'+b+' MEAN'];
            spf  = h[HMQ+'BASELENGTH'+b] / h['EFF_WAVE'];
            vis2m = signal.airy (diam * spf * 4.84813681109536e-09)**2;
            h[HMQ+'TF'+b+' MEAN'] = vis2/vis2m;
            h[HMQ+'VISSM'+b+' MEAN'] = vis2m;
    
    # If we don't have the info about this star
    except NameError:
        for b in range (6):
            h[HMQ+'TRANS%i'%b] = -1.0;
        for b in bname:
            h[HMQ+'TF'+b+' MEAN'] = -1.0;
    except KeyError:
        for b in range(6):
            h[HMQ+'TRANS%i'%b] = -1.0
        if kl == 0:
            log.info('QC parameter BANDFLUX missing from header.')
            log.info('Re-running the reduction is recommended.')
            kl += 1

#
# Plots
#

# Plot coherence
fig,axes = plt.subplots (5,3,sharex=True);
fig.suptitle ('Decoherence Half Time [ms]');
plot.base_name (axes);
plot.compact (axes);

for b in range (15):
    data = headers.getval (hdrs, HMQ+'DECOHER'+bname[b]+'_HALF');
    snr  = headers.getval (hdrs, HMQ+'SNRB'+bname[b]+' MEAN');
    data /= (snr>argopt.snr_threshold);
    axes.flatten()[b].plot (data, 'o');
    axes.flatten()[b].set_ylim (0);
    
files.write (fig,'report_decoher.png');

# Plot SNR
fig,axes = plt.subplots (5,3,sharex=True);
fig.suptitle ('SNR');
plot.base_name (axes);
plot.compact (axes);

for b in range (15):
    data = headers.getval (hdrs, HMQ+'SNR'+bname[b]+' MEAN');
    data /= (data>0);
    axes.flatten()[b].plot (data, 'o');
    axes.flatten()[b].set_yscale ('log');
    
files.write (fig,'report_snr.png');

# Plot TF
fig,axes = plt.subplots (5,3,sharex=True);
fig.suptitle ('Transfer Function');
plot.base_name (axes);
plot.compact (axes);

for b in range (15):
    data = headers.getval (hdrs, HMQ+'TF'+bname[b]+' MEAN');
    vis2 = headers.getval (hdrs, HMQ+'VISS'+bname[b]+' MEAN');
    snr  = headers.getval (hdrs, HMQ+'SNRB'+bname[b]+' MEAN');
    data /= (snr>argopt.snr_threshold);
    data /= (vis2>argopt.vis2_threshold);
    data /= (data>0);
    axes.flatten()[b].plot (data, 'o');
    axes.flatten()[b].set_ylim (0,1.2);

files.write (fig,'report_tf2.png');

# Trans
fig,axes = plt.subplots (3,2,sharex=True);
fig.suptitle ('Transmission [$\%$ of expected $F_\star$]');
plot.compact (axes);

for b in range (6):
    data = headers.getval (hdrs, HMQ+'TRANS%i'%b);
    data /= (data>0);
    axes.flatten()[b].plot (data, 'o');
    
files.write (fig,'report_trans.png');

# Plot vis2
fig,axes = plt.subplots (5,3,sharex=True);
fig.suptitle ('Vis2');
plot.base_name (axes);
plot.compact (axes);

for b in range (15):
    data  = headers.getval (hdrs, HMQ+'VISS'+bname[b]+' MEAN');
    snr  = headers.getval (hdrs, HMQ+'SNRB'+bname[b]+' MEAN');
    data /= (snr>argopt.snr_threshold);
    data /= (data>0);
    axes.flatten()[b].plot (data,  'o');
    # data2 = headers.getval (hdrs, HMQ+'VISSM'+bname[b]+' MEAN');
    # axes.flatten()[b].plot (data2, 'o', alpha=0.1);
    
files.write (fig,'report_vis2.png');
