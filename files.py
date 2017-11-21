import matplotlib.pyplot as plt
import matplotlib

from astropy.time import Time
from astropy.io import fits as pyfits
from astropy.stats import sigma_clipped_stats

from scipy.signal import medfilt;
from scipy.ndimage import gaussian_filter

import numpy as np
import os

from . import log
from .headers import HM, HMQ, HMP, HMW;
from .version import revision

def output (outputDir,hdr,suffix):
    '''
    Return a string like: ./outputDir/mircx00550_suffix
    '''
    
    # Build diretory if needed
    if not os.path.exists (outputDir):
        os.makedirs (outputDir);
        os.chmod (outputDir, 0o777);
        
    # Get filename 
    name = hdr['ORIGNAME'];
    name = os.path.splitext (os.path.basename(name))[0];
    if name[-5:] == '.fits':
        name = name[0:-5];

    # Clean from stuff added already
    for test in ['_vis','_rts','_preproc']:
        if len(name) < len(test): continue;
        if name[-len(test):] == test:
            name = name[:-len(test)];
    
    # Return
    output = outputDir + '/' + name + '_' + suffix;
    return output;

def write (hdulist,filename):
    '''
    Write file. The input shall be a hdulist or
    a matplotlib figure handler.
    '''

    # Use this function to save figure as well
    if type(hdulist) is matplotlib.figure.Figure:
        log.info ('Write %s'%filename);
        hdulist.savefig (filename);
        os.chmod (filename,0o666);
        return;
    
    # Get header
    hdr = hdulist[0].header;
    
    fileinfo = filename + ' ('+hdr['FILETYPE']+')';
    log.info ('Write %s'%fileinfo);

    # Add the pipeline version
    hdr[HMP+'REV'] = (revision,'Version of mircx_pipeline');

    # Remove if existing
    if os.path.exists (filename):
        os.remove (filename);

    # Write and make it writtable to all
    hdulist.writeto (filename);
    os.chmod (filename,0o666);

def load_raw (hdrs, coaddRamp=False, removeBias=True, differentiate=True):
    '''
    Load data and append into gigantic cube. The output cube is
    of shape: [nfile*nramp, nframes, ny, ny].
    
    If coaddRamp==True, the ramps inside each file are averaged together.
    Thus the resulting cube is of shape [nfile, nframes, ny, ny]    
    '''
    log.info ('Load RAW files in mode coaddRamp=%s'%str(coaddRamp));

    # Build header
    hdr = hdrs[0].copy();
    hdr['HIERARCH MIRC QC NRAMP'] = (0,'Total number of ramp loaded');
    hdr['HIERARCH MIRC QC NFILE'] = (0,'Total number of files loaded');
    hdr['BZERO'] = 0;
    
    cube = [];
    for h in hdrs:
        fileinfo = h['ORIGNAME'] + ' (' +h['FILETYPE']+')';
        log.info ('Load %s'%fileinfo);
        hdulist = pyfits.open(h['ORIGNAME']);

        # Read compressed data. 
        if h['ORIGNAME'][-7:] == 'fits.fz':
            # Manipulate the header to fake only one dimension
            nx = hdulist[1].header['NAXIS1'];
            ny = hdulist[1].header['NAXIS2'];
            nf = hdulist[1].header['NAXIS3'];
            nr = hdulist[1].header['NAXIS4'];
            hdulist[1].header['NAXIS'] = 1;
            hdulist[1].header['NAXIS1'] = nr*nf*ny*nx;
            # Uncompress and reshape data
            data = hdulist[1].data;
            data.shape = (nr,nf,ny,nx);
        # Read normal data. 
        else:
            data = hdulist[0].data;

        # Close file
        hdulist.close();

        # TODO: deal with non-linearity, saturation
        # static flat-field and bad-pixels.
        # frameSat = np.sum (data[:,:,5:-5,5:-5] > 60000, axis=(2,3));
        # plt.plot (frameSat.flatten());
        # plt.show ();

        # Take difference of consecutive frames
        if differentiate is True:
            data = np.diff (data.astype('float'),axis=1)[:,0:-1,:,:];
        
        # Remove bias. Note that the median should be taken
        # with an odd number of samples, to be unbiased.
        # WARNING: because of median, our best estimate of the bias
        # is to +/-1nph (convolved by the effect of gaussian_filter)
        if removeBias is True:
            ids = np.append (np.arange(15), data.shape[-1] - np.arange(1,15));
            bias = np.median (data[:,:,:,ids],axis=3);
            bias = gaussian_filter (bias,[0,0,1]);
            data = data - bias[:,:,:,None];

        # Append ramps or co-add them
        hdr['HIERARCH MIRC QC NRAMP'] += data.shape[0];
        if coaddRamp is True:
            cube.append (np.mean (data,axis=0,keepdims=True));
        else:
            cube.append (data);

        # Add this RAW file in hdr (not that the * matching
        # requires to avoid the HIERARCH)
        nraw = len (hdr['*MIRC PRO RAW*']);
        hdr['HIERARCH MIRC PRO RAW%i'%(nraw+1,)] = h['ORIGNAME'];
        hdr['HIERARCH MIRC QC NFILE'] += 1;

    # Allocate memory
    log.info ('Allocate memory');
    shape = cube[0].shape;
    nramp = sum ([c.shape[0] for c in cube]);
    cubenp = np.zeros ((nramp,shape[1],shape[2],shape[3]));

    # Set data in cube, and free initial memory in its way
    log.info ('Set data in cube');
    ramp = 0;
    for c in range (len(cube)):
        cubenp[ramp:ramp+cube[c].shape[0],:,:,:] = cube[c];
        ramp += cube[c].shape[0];
        cube[c] = None;

    plt.close('all');
    return hdr,cubenp;
