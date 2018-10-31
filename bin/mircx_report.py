#! /usr/bin/env python                                                          
# -*- coding: iso-8859-15 -*-                                                   

import mircx_pipeline as mrx;
import argparse, glob, os;
import numpy as np;
from astropy.io import fits as pyfits;
import matplotlib.pyplot as plt;

from mircx_pipeline import log, setup, plot, files, signal, headers;
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

# Zero point of Hband (arbitrary unit)
Hzp = 1e5;

# Load all the headers
hdrs = mrx.headers.loaddir (argopt.oifits_dir);

# Sort the headers by time
ids = np.argsort ([h['MJD-OBS'] for h in hdrs]);
hdrs = [hdrs[i] for i in ids];


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

for h in hdrs:
    
    # If we have the info about this star
    try:
        fluxm = Hzp * 10**(-objcat[h['OBJECT']]['Hmag'][0]/2.5);
        diam  = objcat[h['OBJECT']]['UDDH'][0];
        
        # Loop on beam 
        for b in range (6):
            flux = h['HIERARCH MIRC QC FLUX%i MEAN'%b];
            h['HIERARCH MIRC QC TRANS%i'%b] = flux / fluxm;

        # Loop on baseline 
        for b in bname:
            vis2 = h['HIERARCH MIRC QC VISS'+b+' MEAN'];
            spf  = h['HIERARCH MIRC QC BASELENGTH'+b] / h['EFF_WAVE'];
            vis2m = signal.airy (diam * spf * 4.84813681109536e-09)**2;
            h['HIERARCH MIRC QC TF'+b+' MEAN'] = vis2/vis2m;

    # If we don't have the info about this star
    except:
        for b in range (6):
            h['HIERARCH MIRC QC TRANS%i'%b] = -1.0;
        for b in bname:
            h['HIERARCH MIRC QC TF'+b+' MEAN'] = -1.0;
        

#
# Plots
#

# Flux
log.info ('Plot photometry');

fig,axes = plt.subplots (3,2,sharex=True);
fig.suptitle ('Transmission');
plot.compact (axes);

for b in range (6):
    data = headers.getval (hdrs, HMQ+'TRANS%i'%b);
    data /= (data>0);
    axes.flatten()[b].plot (data, 'o');
    
files.write (fig,'report_trans.png');

# Plot TF
log.info ('Plot TF');

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

# Plot vis2
log.info ('Plot vis2');

fig,axes = plt.subplots (5,3,sharex=True);
fig.suptitle ('Vis2');
plot.base_name (axes);
plot.compact (axes);

for b in range (15):
    data = headers.getval (hdrs, HMQ+'VISS'+bname[b]+' MEAN');
    snr  = headers.getval (hdrs, HMQ+'SNRB'+bname[b]+' MEAN');
    data /= (snr>argopt.snr_threshold);
    data /= (data>0);
    axes.flatten()[b].plot (data, 'o');
    
files.write (fig,'report_vis2.png');


# Plot coherence
log.info ('Plot decoherence');

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
log.info ('Plot SNR');

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
