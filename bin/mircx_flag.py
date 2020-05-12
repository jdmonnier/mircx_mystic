#! /usr/bin/env python                                                          
# -*- coding: iso-8859-15 -*-                                                   

import mircx_pipeline as mrx
import argparse
import glob
import os
import copy
from fnmatch import fnmatch

from mircx_pipeline import log, files;
from astropy.io import fits as pyfits;

#
# Function to associate tables in OIFITS
#

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
    
#
# Implement options
#

# Describe the script
description = \
"""
description:

Use this script to flag (=reject) data based on time,
baseline, target name, and wavelength. The script create
new files in the output-dir.

By default, the script parse all *.fits files in the current
directory, but a list of input file(s) can be specified instead.

TODO: I want to evolve the script to handle
to better handle the baseline (swap order)...

"""

epilog = \
"""
examples:

  mircx_flag.py --base S1S2 --target HD1234 --output-dir=all_flagged_data/

  mircx_flag.py --base *W1* --target HD1234 mircx00124_oifits.fits

  mircx_flag.py --base S1S2 --lbd 1.8 1.9 --mjd 55670.0 55670.1 --output-dir=all_flagged_data/

contact lebouquj@umich.edu for support.
"""

parser = argparse.ArgumentParser (formatter_class=argparse.RawDescriptionHelpFormatter,
                                  add_help=False);


TrueFalse = ['TRUE','FALSE'];
TrueFalseOverwrite = ['TRUE','FALSE','OVERWRITE'];

# Rules arguments
parser.add_argument ('--lbd', dest='lbd', default=[1.,3.],
                     type=float, nargs=2,
                     help='interval of wavelength in um (default is 1. 3.)');

parser.add_argument ('--mjd', dest='mjd', default=[50000,60000],
                     type=float, nargs=2,
                     help='time interval in modified julian day (default is 50000 60000)');

parser.add_argument ('--target', dest='target', default='*',
                     type=str, nargs='+',
                     help='list of target, with basic wildcard matching '
                     'such as "*HD_*" (default is "*")');

parser.add_argument ('--base', dest='base', default='*',
                     type=str, nargs='+',
                     help='list of baseline and/or triplet, with basic wildcard '
                     'matching such as "*S2*" (default is "*")');

# Copy the parser of the rules only
rparser = copy.deepcopy (parser);

# Other argumens
parser.add_argument ('--output-dir', dest='output_dir', default='./flagged/',
                     type=str,
                     help='Directory for output product. If INPLACE, then '
                          'FITS files are updated in-place (default is %(default)s)');

parser.add_argument ('--rules', dest='rules', type=argparse.FileType('r'), default=None,
                     help='Text files with list of rules entered as in-line arguments. '
                     'Each line in the file is a new set of rules.');

parser.add_argument ("--debug", dest="debug",default='FALSE',
                     choices=TrueFalse,
                     help="stop on error [%(default)s]");

parser.add_argument ('--help', action='help',
                     help=argparse.SUPPRESS);

parser.add_argument ('-h', action='help',
                     help=argparse.SUPPRESS);

# Positional argument
parser.add_argument ('input_files', type=str, nargs='*', default=['*_oifits.fits']);

# Add long help
parser.description = description;
parser.epilog = epilog;

#
# Parse argument
#

argopt = parser.parse_args ();

# Verbose
elog = log.trace ('mircx_flag');

# Load the list of rules from files or
# from command line
if argopt.rules is not None:
    log.info ('Load list of rules from file');
    rules = [rparser.parse_args(l.split()) for l in argopt.rules.readlines() if l.strip() is not ''];
else:
    log.info ('Load list of rules from command line');
    rules = [argopt];

# Print and check rules
for i,rule in enumerate(rules):
    log.info ('Rule %i: %s'%(1,str(rule)[10:-1]));

# Define input list of files
inputs = [];
for l in argopt.input_files: inputs += glob.glob(l);

# Create output directory
if argopt.output_dir == 'INPLACE':
    log.info ('FITS files will be modified inplace');
    open_mode = 'update';
else:
    log.info ('Create output directory');
    files.ensure_dir (argopt.output_dir);
    open_mode = 'readonly';

#
# Loop on list of files
#

for file in inputs:

    # Open file
    log.info ('File ' + file);
    try:
        hdulist = pyfits.open (file,mode=open_mode);
        
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

                # Get data
                base = ''.join([stadic[i] for i in data['STA_INDEX']]);
                mjd = data['MJD'];
                trg = trgdic[data['TARGET_ID']];

                # Loop on rules
                for rule in rules:
                    
                    # Check time
                    if mjd < rule.mjd[0] or mjd > rule.mjd[1]: continue;
                
                    # Check target
                    if any ([fnmatch (trg, t) for t in rule.target]) is False: continue;
                    
                    # Check baseline
                    if any ([fnmatch (base, b) for b in rule.base]) is False: continue;
                    
                    # Check wavelength
                    ids = (lbd >= rule.lbd[0]) * (lbd <= rule.lbd[1]);
                    data['FLAG'] += ids;
                    
                    log.info ('Flag %-7s %.4f %s %s'%(hdu.header['EXTNAME'], mjd, trg, base));
            
        # Save
        if argopt.output_dir == 'INPLACE':
            log.info ('FITS file saved in-place');
            hdulist.close ();
        else:
            files.write (hdulist, argopt.output_dir+file);
        
    except Exception as exc:
        log.error ('Cannot deal with file '+str(file));
        if argopt.debug == 'TRUE': raise;
