#! /usr/bin/env python                                                          
# -*- coding: iso-8859-15 -*-                                                   

import mircx_pipeline as mrx
import argparse
import glob
import os

from mircx_pipeline import log, setup;
import numpy as np;
import os;
import pdb;

import matplotlib.pyplot as plt;
import matplotlib.colors as mcolors;
from skimage import exposure #sci-kit imaging processing library


from astropy.stats import sigma_clipped_stats;
from astropy.io import fits as pyfits;
from astropy.modeling import models, fitting;

from skimage.feature import register_translation;

from scipy import fftpack;
from scipy.signal import medfilt;
from scipy.ndimage.interpolation import shift as subpix_shift;
from scipy.ndimage import gaussian_filter, uniform_filter, median_filter,shift;
from scipy.optimize import least_squares, curve_fit, brute;
from scipy.ndimage.morphology import binary_closing, binary_opening;
from scipy.ndimage.morphology import binary_dilation, binary_erosion;
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from itertools import permutations


#
# Implement options
#

# Describe the script
description = \
"""
description:
  This routine will offer an interactive display to mark bad files due to fringe or photometric
  similar to the working of MIRC IDL pipeline.  This will operate on the rts data so as not to 
  contaminate the oifits step later. [how to mark as bad not sure, may just write a txt file with 
  numbers for now]. 

"""

epilog = \
"""
examples:

  # Run the entire reduction
  
  cd /path/where/I/want/my/reduced/data/
  mircx_fringe_inspector.py --rts-dir = rts_2020Oct15_ncoh5_analysis

"""

parser = argparse.ArgumentParser (description=description, epilog=epilog,
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 add_help=False);


TrueFalse = ['TRUE','FALSE'];
TrueFalseOverwrite = ['TRUE','FALSE','OVERWRITE'];

rts = parser.add_argument_group ('(2) rts',
                  '\nCreates RTS intermediate products, which are the\n'
                  'coherent flux and the photometric flux in real time,\n'
                  'cleaned from the instrumental behavior.');

rts.add_argument ("--rts-dir", dest="rts_dir",default='./rts/',type=str,
                  help="directory of products [%(default)s]");

advanced = parser.add_argument_group ('advanced user arguments');
                                         
advanced.add_argument ("--debug", dest="debug",default='FALSE',
                     choices=TrueFalse,
                     help="stop on error [%(default)s]");

#
# Initialisation
#

# Parse argument
argopt = parser.parse_args ();

# Verbose
elog = log.trace ('mircx_fringe_inspector');

# Set debug
if argopt.debug == 'TRUE':
    log.info ('start debug mode')
    import pdb;



# List inputs
hdrs = mrx.headers.loaddir (argopt.rts_dir);

# Group all DATA.  only change groups if # of wavelength chnnels or mode .
#keys = setup.detwin + setup.detmode + setup.insmode + setup.fringewin; 
keys = setup.detwin + setup.insmode + setup.fringewin;

# gruop 
gps = mrx.headers.group (hdrs, 'DATA_RTS', delta=24*3600,
                            Delta=24*3600, keys=keys, continuous=False);

## Get rid of groups with low integration time
nloads = [len(g) for g in gps]
max_loads = max(nloads)
gps = [g for g in gps if len(g)>(max_loads/2)]

# Compute 
for i,gp in enumerate(gps):
    try:
        log.info ('INSPECT RTS {0} to {1} '.format(i+1,len(gps)));

        if 'DATA' in gp[0]['FILETYPE']:
            filetype = 'fringe_inspector';
            data = True;
        else:
            filetype = gp[0]['FILETYPE'].replace('_RTS','_OIFITS');
            data = False
            
        output = mrx.files.output (argopt.rts_dir, gp[0], filetype);

        print('Group ',i,':')
        print(output)
        
        
        if os.path.exists (output+'.fits') and overwrite is False:
            log.info ('Product already exists');
            continue;
        nrts = len(gp)
        log.info ('Number of RTS Files to Inspect: '+str(nrts));
        #for j in range(nrts):
        #    print(j,gp[j]['OBJECT'],gp[j]['FILETYPE'])
        #The first file will be used to setup arrays since we are choosing same configuration

        hdrs=gp
        log.setFile (output+'.log');
        f = hdrs[0]['ORIGNAME'];
        log.info ('Load RTS file %s'%f);
        hdr = pyfits.getheader (f);
        #base_dft  = pyfits.getdata (f, 'BASE_DFT_IMAG').astype(float) * 1.j;
        #base_dft += pyfits.getdata (f, 'BASE_DFT_REAL').astype(float);
        #bias_dft  = pyfits.getdata (f, 'BIAS_DFT_IMAG').astype(float) * 1.j;
        #bias_dft += pyfits.getdata (f, 'BIAS_DFT_REAL').astype(float);
        photo_avg     = pyfits.getdata (f, 'PHOTOMETRY_AVG').astype(float);
        gdt_avg       = pyfits.getdata (f,'GROUP_DELAY_AVG').astype(float);
        mjd       = pyfits.getdata (f, 'MJD');
        lbd       = pyfits.getdata (f, 'WAVELENGTH').astype(float);
        ny_zpad,nb = gdt_avg.shape;
        ny,nt = photo_avg.shape;
        all_fringes = np.zeros((ny_zpad,nrts,nb))
        all_photo = np.zeros((ny,nrts,nt))
        all_tfringes = np.zeros((ny_zpad,nrts,nt))

        log.info ('Loading fringe info:')
        bbeam = setup.base_beam ()
        for j in range(nrts):
            f = hdrs[j]['ORIGNAME'];
            #log.info ('Load RTS file %s'%f);
            #print(j,gp[j]['OBJECT'],gp[j]['FILETYPE'])
            #base_dft  = pyfits.getdata (f, 'BASE_DFT_IMAG').astype(float) * 1.j;
            #base_dft += pyfits.getdata (f, 'BASE_DFT_REAL').astype(float);
            photo_avg     = pyfits.getdata (f, 'PHOTOMETRY_AVG').astype(float);
            gdt_avg       = pyfits.getdata (f,'GROUP_DELAY_AVG').astype(float);
            #ALSO ADD UP TELS!
            gdt_avg = gdt_avg - np.median(gdt_avg,axis=0,keepdims=True)
            for bb in range(nb):
                all_tfringes[:,j,bbeam[bb,0]]+=gdt_avg[:,bb]
                all_tfringes[:,j,bbeam[bb,1]]+=gdt_avg[:,bb]
            all_fringes[:,j,:]=gdt_avg #/np.max(gdt_avg,axis=0,keepdims=True)
            all_photo[:,j,:]=photo_avg
        all_fringes /= np.max(all_fringes,axis=0,keepdims=True)
        all_tfringes /= np.max(all_tfringes,axis=0,keepdims=True)
        log.info ('Fringe info loaded:')

        #log.info ('Data size: '+str(base_dft.shape));j-

        #plt.imshow(exposure.equalize_hist(all_fringes[:,:,0]),aspect='auto')
        #plt.imshow(all_fringes[:,:,0],aspect='auto',cmap=plt.cm.Reds,vmin=0)

        newgdt = np.reshape(np.transpose(all_fringes,(0,2,1)),(ny_zpad*nb,nrts),order='F')
        newtfringe= np.reshape(np.transpose(all_tfringes,(0,2,1)),(ny_zpad*nt,nrts),order='F')
        
        
        plt.imshow(newgdt,aspect='auto',cmap=plt.cm.Reds,vmin=0)
        plt.show()
        plt.imshow(newtfringe,aspect='auto',cmap=plt.cm.Reds,vmin=0)
        plt.show()
        """         mrx.compute_vis (gp, coeff, output=output,
                            filetype=filetype,
                            ncoher=argopt.ncoherent,
                            gdt_tincoh=argopt.gdt_tincoh,
                            ncs=argopt.ncs, nbs=argopt.nbs,
                            snr_threshold=argopt.snr_threshold if data else -1*argopt.snr_threshold, #catch this!
                            flux_threshold=argopt.flux_threshold, #keep this even foreground.
                            gd_attenuation=argopt.gd_attenuation if data else False,
                            gd_threshold=argopt.gd_threshold if data else 1e10, #keep all frames.
                            vis_reference=argopt.vis_reference); """
        


    except Exception as exc:
        log.error ('Cannot compute OIFITS: '+str(exc));
        if argopt.debug == 'TRUE': pdb.post_mortem(); raise;
    finally:
        log.closeFile ();
        
log.info ('Cleanup memory');
del hdrs, gps;


    
# Delete elog to have final
# pring of execution time
del elog;
