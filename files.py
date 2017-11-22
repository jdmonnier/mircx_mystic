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

def load_raw (hdrs, checkSaturation=True, differentiate=True,
              removeBias=True, background=None, coaddRamp=False):
    '''
    Load data and append into gigantic cube. The output cube is
    of shape: [nfile*nramp, nframes, ny, ny].

    If checkSaturation==True, the frame with >10 saturated pixels in
    the middle of the fringe window (read from header) are set to zero.

    If differentiate==True, the consecutive frames of a ramp are
    subtracted together.

    If removeBias==True, the detector bias interference is removed
    by using the median of the edges columns.

    If background is not None, thus background cube is subtracted
    from the data.

    If coaddRamp==True, the ramps inside each file are averaged together.
    Thus the resulting cube is of shape [nfile, nframes, ny, ny]    
    '''
    log.info ('Load RAW files in mode coaddRamp=%s'%str(coaddRamp));

    # Build header
    hdr = hdrs[0].copy();
    hdr[HMQ+'NFILE'] = (0,'total number of files loaded');
    hdr[HMQ+'NRAMP'] = (0,'total number of ramp loaded');
    hdr[HMQ+'NSAT']  = (0,'total number of saturated ramps');
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

        # Convert to float
        data = data.astype ('float32');

        # Close file
        hdulist.close();

        # Guessed fringe window
        nr,nf,ny,nx = data.shape;
        ys = ny - hdr['FR_ROW2'];
        ye = ny - hdr['FR_ROW1'];
        xc = int(nx - (hdr['FR_COL2'] + hdr['FR_COL1'])/2);
        xs = xc - 10;
        xe = xc + 10;

        # Frame is declared saturated if more than 10 pixels in
        # the center of the fringes are near saturation. flag is 0
        # if no saturation, or the id of the first saturated frame
        if checkSaturation is True:
            flag = np.sum (data[:,:,ys:ye,xs:xs]>60000, axis=(2,3));
            flag = np.argmax (flag > 10, axis=1);
            nsat = np.sum (flag>0);
            hdr[HMQ+'NSAT'] += nsat;

        # TODO: deal with non-linearity,
        # static flat-field and bad-pixels.

        # Take difference of consecutive frames
        if differentiate is True:
            data = np.diff (data,axis=1)[:,0:-1,:,:];
        
        # Remove bias. Note that the median should be taken
        # with an odd number of samples, to be unbiased.
        # WARNING: because of median, our best estimate of the bias
        # is to +/-1nph (convolved by the effect of gaussian_filter)
        if removeBias is True:
            ids = np.append (np.arange(15), data.shape[-1] - np.arange(1,15));
            bias = np.median (data[:,:,:,ids],axis=3);
            bias = gaussian_filter (bias,[0,0,1]);
            data = data - bias[:,:,:,None];

        if background is not None:
            data -= background;

        # Set the saturation fringes to zero
        if checkSaturation is True:
            for r in range(data.shape[0]):
                if flag[r] != 0:
                    data[r,flag[r]-3:,:,:] = 0.0;

        # Add this RAW file in hdr
        nraw = len (hdr['*MIRC PRO RAW*']);
        hdr['HIERARCH MIRC PRO RAW%i'%(nraw+1,)] = os.path.basename (h['ORIGNAME']);
        hdr['HIERARCH MIRC QC NFILE'] += 1;
        hdr['HIERARCH MIRC QC NRAMP'] += data.shape[0];

        # Co-add ramp if required
        if coaddRamp is True:
            data = np.mean (data,axis=0,keepdims=True);
        
        # Append the data in the final cube
        cube.append (data);

    # Allocate memory
    log.info ('Allocate memory');
    shape = cube[0].shape;
    nramp = sum ([c.shape[0] for c in cube]);
    cubenp = np.zeros ((nramp,shape[1],shape[2],shape[3]),dtype='float32');

    # Set data in cube, and free initial memory in its way
    log.info ('Set data in cube');
    ramp = 0;
    for c in range (len(cube)):
        cubenp[ramp:ramp+cube[c].shape[0],:,:,:] = cube[c];
        ramp += cube[c].shape[0];
        cube[c] = None;

    # Some verbose
    log.info ('Number of files loaded = %i'%hdr[HMQ+'NFILE']);
    log.info ('Number of ramp loaded = %i'%hdr[HMQ+'NRAMP']);
    log.info ('Number of saturated ramps = %i'%hdr[HMQ+'NSAT']);

    plt.close('all');
    return hdr,cubenp;
