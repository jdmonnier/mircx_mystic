#! /usr/bin/env python                                                          
# -*- coding: iso-8859-15 -*-                                                   

import mircx_pipeline as mrx
import argparse
import glob
import os

from mircx_pipeline import log, setup, files;
from astropy.io import fits as pyfits;

#
# Implement options
#

# Describe the script
description = \
"""
description:

"""

epilog = \
"""
examples:



"""

parser = argparse.ArgumentParser (description=description, epilog=epilog,
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 add_help=False);


TrueFalse = ['TRUE','FALSE'];
TrueFalseOverwrite = ['TRUE','FALSE','OVERWRITE'];

parser.add_argument ("--lbd", dest="lbd", default=[1.,3.],
                     type=float, nargs=2,
                     help="interval of wavelength in um (default is all)");

parser.add_argument ("--mjd", dest="mjd", default=[50000,60000],
                     type=float, nargs=2,
                     help="time interval in modified julian day (default is all)");

parser.add_argument ("--target", dest="target", default=['INTERNAL'],
                     type=str, nargs="+",
                     help="list of target (default is none)");

parser.add_argument ("--base", dest="base",
                     type=str, nargs="+",
                     help="list of baseline (e.g S1S2, default is none)");

parser.add_argument ("--output-dir", dest="output_dir", default='./flagged/',
                     type=str,
                     help=" def: %(default)s");

parser.add_argument ('--help', action='help',
                     help=argparse.SUPPRESS);

parser.add_argument ('-h', action='help',
                     help=argparse.SUPPRESS);

# Parse argument
argopt = parser.parse_args ();

# Verbose
elog = log.trace ('mircx_flag');

# Checks
if len (argopt.lbd) % 2:
    raise ValueError ('lbd should have even number of values');

if len (argopt.mjd) % 2:
    raise ValueError ('mjd should have even number of values');

# Function to explore OIFITS
def get_wave (hdulist, hdu):
    n = hdu.header['INSNAME'];
    for h in hdulist:
        if h.header.get ('EXTNAME') == 'OI_WAVELENGTH':
            if h.header.get ('INSNAME') == n: return h.data['EFF_WAVE'];
        
def get_station (hdulist, hdu):
    n = hdu.header['ARRNAME'];
    for h in hdulist:
        if h.header.get ('EXTNAME') == 'OI_ARRAY':
            return dict([(d['STA_INDEX'],d['STA_NAME']) for d in h.data]);
                
def get_target (hdulist):
    for h in hdulist:
        if h.header.get ('EXTNAME') == 'OI_TARGET':
            return dict([(d['TARGET_ID'],d['TARGET']) for d in h.data]);

# Loop on list of files
inputs = glob.glob ('*.fits');
for file in inputs:

    # Open file
    log.info ('File ' + file);
    hdulist = pyfits.open (file);

    # Get the target dictionnary for this file
    trgdic = get_target (hdulist);

    # Loop on extensions
    for hdu in hdulist:
        if hdu.header.get('EXTNAME') not in ['OI_VIS2', 'OI_VIS', 'OI_T3']: continue;

        # Get the station dictionary and the lbd for this table
        stadic = get_station (hdulist, hdu);
        lbd = get_wave (hdulist, hdu) * 1e6;
        
        # Loop on data line by line
        for data in hdu.data:

            # Check time
            mjd = data['MJD'];
            if mjd < argopt.mjd[0] or mjd > argopt.mjd[1]: continue;
            
            # Check target
            trg = trgdic[data['TARGET_ID']];
            if argopt.target is not None and trg not in argopt.target: continue;

            # Check baseline
            base = ''.join([stadic[i] for i in data['STA_INDEX']]);
            if argopt.base is not None and base not in argopt.base: continue;
            
            # Check wavelength
            ids = (lbd >= argopt.lbd[0]) * (lbd <= argopt.lbd[1]);
            data['FLAG'] += ids;

            log.info ('Flag %-7s %.4f %s %s'%(hdu.header['EXTNAME'], mjd, trg, base));
        
    # Save
    files.write (hdulist, argopt.output_dir+file);
    hdulist.close ();
