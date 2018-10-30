#! /usr/bin/env python                                                          
# -*- coding: iso-8859-15 -*-                                                   

import mircx_pipeline as mrx;
import argparse, glob, os;
import numpy as np;
from astropy.io import fits as pyfits;
import matplotlib.pyplot as plt;

from mircx_pipeline import log, setup, plot, files, signal;
from mircx_pipeline.headers import HM, HMQ, HMP, HMW, rep_nan;


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

parser.add_argument ("--debug", dest="debug",default='FALSE',
                     choices=TrueFalse,
                     help="stop on error [%(default)s]");

#
# Initialisation
#

# Parse argument
argopt = parser.parse_args ();

# Verbose
elog = log.trace ('mircx_report');

# Load all the headers
hdrs = mrx.headers.loaddir (argopt.oifits_dir);

# List of basename
bname = setup.base_name ();

# Zero point of Hband
Hzp = 1.0;


#
# Query CDS to build a catalog of object information
# This is protected since it may fail
#

# List of object
objlist = list(set([h['OBJECT'] for h in hdrs]));
objcat = dict();

for obj in objlist:
    try:
        from astroquery.vizier import Vizier;
        objcat[obj] = Vizier.query_object (obj, catalog='JSDC')[0];
        log.info ('Find JSDC for '+obj);
    except:
        log.info ('Cannot find JSDC for '+obj);


#
# Compute the transmission and instrumental visibility
#

for h in hdrs:
    
    # If we have the info about this star
    if h['OBJECT'] in objcat:
        fluxm = Hzp * 10**(-objcat[h['OBJECT']]['Hmag'][0]/2.5);
        diam  = objcat[h['OBJECT']]['UDDH'][0];

        log.info ("diam = %.3f mas"%diam);

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
    else:
        for b in range (6):
            h['HIERARCH MIRC QC TRANS%i'%b] = 0.0;
        for b in bname:
            h['HIERARCH MIRC QC TF'+b+' MEAN'] = 0.0;
        

#
# Plots
#

# Flux
log.info ('Plot photometry');

fig,axes = plt.subplots (3,2,sharex=True);
plot.compact (axes);

for b in range (6):
    data = [h['HIERARCH MIRC QC TRANS%i'%b] for h in hdrs];
    axes.flatten()[b].plot (data, 'o');
    
files.write (fig,'report_flux.png');

# Plot TF
log.info ('Plot TF');

fig,axes = plt.subplots (5,3,sharex=True);
plot.base_name (axes);
plot.compact (axes);

for b in range (15):
    data = [h['HIERARCH MIRC QC TF'+bname[b]+' MEAN'] for h in hdrs];
    axes.flatten()[b].plot (data, 'o');
    
files.write (fig,'report_tf2.png');

# Plot vis2
log.info ('Plot vis2');

fig,axes = plt.subplots (5,3,sharex=True);
plot.base_name (axes);
plot.compact (axes);

for b in range (15):
    data = [h['HIERARCH MIRC QC VISS'+bname[b]+' MEAN'] for h in hdrs];
    axes.flatten()[b].plot (data, 'o');
    
files.write (fig,'report_vis2.png');


# Plot coherence
log.info ('Plot decoherence');

fig,axes = plt.subplots (5,3,sharex=True);
plot.base_name (axes);
plot.compact (axes);

for b in range (15):
    data = [h['HIERARCH MIRC QC DECOHER'+bname[b]+'_HALF'] for h in hdrs];
    axes.flatten()[b].plot (data, 'o');
    
files.write (fig,'report_decoher.png');
