#! /usr/bin/env python                                                          
# -*- coding: iso-8859-15 -*-                                                   

import mircx_pipeline as mrx
import argparse
import glob
import os

from mircx_pipeline import log, setup, files;

from astropy.io import fits;
from astropy.time import Time;

#
# Implement options
#

# Describe the script
description = \
"""
description:

Recompute the UV plan of OIFITS file produced by the mircx_pipeline.
It makes use of the ERFA library to handle all known effect.

This script should be made more generic to handle any OIFITS file,
but as of now it will only handle files produced by the mircx_pipeline.

The script list all *.fits files in the input directory and will try
to fix them, dumping the resulting files in output-dir.

"""

epilog = \
"""
examples:

 mircx_fixuv.py

"""

parser = argparse.ArgumentParser (description=description, epilog=epilog,
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 add_help=False);


TrueFalse = ['TRUE','FALSE'];
TrueFalseOverwrite = ['TRUE','FALSE','OVERWRITE'];

parser.add_argument ("--input-dir", dest="input_dir",default='./',type=str,
                     help="directory of input data [%(default)s]");

parser.add_argument ("--output-dir", dest="output_dir",default='./fixuv/',type=str,
                     help="directory of output data [%(default)s]");

parser.add_argument ("--debug", dest="debug",default='FALSE',
                     choices=TrueFalse,
                     help="stop on error [%(default)s]");

elog = log.trace ('mircx_fixuv');

# Parse argument
argopt = parser.parse_args ();

# List files
log.info ('List FITS files in '+argopt.input_dir);
filenames = glob.glob (argopt.input_dir+'/*.fits');

# Loop on files
for filename in filenames:
    
    log.info ('Fix file '+filename);
    try:
        # Open file
        hdulist = fits.open (filename);
        hdr = hdulist[0].header;

        # Loop on tables
        for hdu in hdulist[1:]:
            ext = hdu.header['EXTNAME'];
            
            if ext == 'OI_VIS' or ext == 'OI_VIS2':
                log.info ('Fix uv of '+ext);
                
                # Re compute uv
                mjd = Time (hdu.data['MJD'], format='mjd');
                ucoord, vcoord = setup.compute_base_uv (hdr, mjd=mjd);

                # Check
                maxrel = setup.uv_maxrel_distance (ucoord,vcoord,hdu.data['UCOORD'],hdu.data['VCOORD']);
                log.check (maxrel>0.01, 'Maximum rel. distance %.3f'%maxrel);
                
                # Set uv
                hdu.data['UCOORD'] = ucoord;
                hdu.data['VCOORD'] = vcoord;

            elif ext == 'OI_T3':
                log.info ('Fix uv of '+ext);

                # Re compute uv
                mjd = Time (hdu.data['MJD'], format='mjd');
                u1coord, v1coord = setup.compute_base_uv (hdr, mjd=mjd, baseid='base1');
                u2coord, v2coord = setup.compute_base_uv (hdr, mjd=mjd, baseid='base2');

                # Check
                maxrel1 = setup.uv_maxrel_distance (u1coord,v1coord,hdu.data['U1COORD'],hdu.data['V1COORD']);
                maxrel2 = setup.uv_maxrel_distance (u2coord,v2coord,hdu.data['U2COORD'],hdu.data['V2COORD']);
                log.check (maxrel1>0.01, 'Maximum rel. distance %.3f'%maxrel1);
                log.check (maxrel2>0.01, 'Maximum rel. distance %.3f'%maxrel2);
                
                # Set uv
                hdu.data['U1COORD'] = u1coord;
                hdu.data['V1COORD'] = v1coord;
                hdu.data['U2COORD'] = u2coord;
                hdu.data['V2COORD'] = v2coord;
        
        # Set a keyword in header
        hdr.set ('HIERARCH MIRC PRO UV_EQUATION', 'ERFA', 'Fixed by mircx_fixuv');
        
        # Output filename
        output = mrx.files.output (argopt.output_dir, filename, 'uvfix');
        files.write (hdulist, output+'.fits');
        
    except Exception as exc:
        log.error ('Cannot  '+str(exc));
        if argopt.debug == 'TRUE': raise;
        
    log.info ('Cleanup memory');
    del hdr;


    
# Delete elog to have final
# pring of execution time
del elog;
    
    

