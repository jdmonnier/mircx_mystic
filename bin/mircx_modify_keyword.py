#! /usr/bin/env python                                                          
# -*- coding: iso-8859-15 -*-                                                   

import mircx_pipeline as mrx
import argparse
import glob
import os

from astropy.io import fits as fits
from mircx_pipeline import log;

# Describe the script
description = \
"""
description:

  Modify the value a KEYWORD in the FITS main header of MIRCX files. It works
  both with compressed (*.fits.fz) and uncompressed (*.fits) files.

  If the OLD value is specified in the common line, only file matching this value
  are updated. If the OLD value is not specified, then all files are updated
  with the new value.

  Files are not modified in-place, but are saved into an output directory.
"""

epilog = \
"""
examples:

  mircx_modify_keyword.py --old HD_40281 --new HD_40282 *.fits.fz

"""

parser = argparse.ArgumentParser (description=description, epilog=epilog,
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 add_help=False);

parser.add_argument ("--key", dest="key",default='OBJECT',type=str,
                     help="name of key to update");

parser.add_argument ("--old", dest="old",default=None,
                     help="current value to update");

parser.add_argument ("--new", dest="new",default=None,
                     help="new value to set");

parser.add_argument ("--output-dir", dest="output_dir",default='./modified/', type=str,
                     help="output directory");

parser.add_argument ('input_files', type=str, nargs='*');

# Parse argument
argopt = parser.parse_args ();

# Output directory
mrx.files.ensure_dir (argopt.output_dir);

# List files
files = argopt.input_files;

# Loop on files
for f in files:
    hdulist = fits.open (f);

    # Which HDU to change (compressed files)
    if f[-8:] == '.fits.fz':
        i = 1;
    else:
        i = 0;

    # Get object
    o = hdulist[i].header[argopt.key];
    if  argopt.old != None and (o != argopt.old):
        log.info ('Skip: '+f);
        continue;

    # Change
    hdulist[i].header[argopt.key] =  argopt.new;

    # New file
    name = argopt.output_dir + '/' + os.path.basename (f);
    
    # Write file
    log.info ('Write: '+name);
    hdulist.writeto (name);


