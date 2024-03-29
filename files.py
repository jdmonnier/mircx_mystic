import matplotlib.pyplot as plt
import matplotlib

from astropy.time import Time
from astropy.io import fits as pyfits
from astropy.stats import sigma_clipped_stats

from scipy.signal import medfilt;
from scipy.ndimage import gaussian_filter

import numpy as np
import os

from . import log, headers, plot;
from .headers import HM, HMQ, HMP, HMW;
from .version import revision, git_hash, git_date, git_branch;
from .setup import coef_flat

def ensure_dir (outputDir):
    '''
    Build directory if needed. Set permission
    to all so that directory can be deleted by
    others (in orthanc for instance).
    '''
    if not os.path.exists (outputDir):
        log.info ('Create directory: %s'%outputDir);
        os.makedirs (outputDir);
        os.chmod (outputDir, 0o777);

def output (outputDir,hdr,suffix):
    '''
    Return a string like: ./outputDir/mircx00550_suffix
    '''
    
    # Build diretory if needed
    ensure_dir (outputDir);

    # Get filename
    name = hdr if type(hdr) is str else hdr['ORIGNAME'];
        
    # Get filename 
    name = os.path.splitext (os.path.basename(name))[0];
    if name[-5:] == '.fits':
        name = name[0:-5];

    # Clean as lower and continuous
    suffix = suffix.lower().replace('_','');

    # Clean from stuff added already
    for test in ['_datapreproc','_foregroundpreproc','_backgroundpreproc',
                 '_datarts','_foregroundrts','_backgroundrts',
                 '_beam1map','_beam2map','_beam3map','_beam4map','_beam5map','_beam6map',
                 '_beam1mean','_beam2mean','_beam3mean','_beam4mean','_beam5mean','_beam6mean',
                 '_beam1profile','_beam2profile','_beam3profile',
                 '_beam4profile','_beam5profile','_beam6profile',
                 '_vis','_rts','_preproc']:
        if len(name) < len(test): continue;
        if name[-len(test):] == test:
            name = name[:-len(test)];

    # Return
    output = outputDir + '/' + name + '_' + suffix;
    return output;


def blockoutput (outputDir,bnum,suffix):
    '''
    Return a string like: ./outputDir/mircx00550_suffix
    '''
    
    # Build diretory if needed
    ensure_dir (outputDir);

    # Get filename
    name = ('block%s'%(format(bnum, '04d')))

    # Clean as lower and continuous
    suffix = suffix.lower().replace('_',''); 

    # Return
    output = outputDir + '/' + name + '_' + suffix;
    return output;

def write (hdulist,filename,dpi=100):
    '''
    Write file. The input shall be a hdulist or
    a matplotlib figure handler.
    '''

    # Use this function to save figure as well
    if type(hdulist) is matplotlib.figure.Figure:
        log.info ('Write %s'%filename);
        hdulist.savefig (filename,dpi=dpi);
        os.chmod (filename,0o666);
        return;
    
    # Get header
    hdr = hdulist[0].header;
    
    fileinfo = filename + ' ('+hdr['FILETYPE']+')';
    log.info ('Write %s'%fileinfo);

    # Add the pipeline version
    hdr[HMP+'REV'] = (revision,'Version of mircx_pipeline');
    hdr[HMP+'GIT HASH'] = (git_hash);
    hdr[HMP+'GIT DATE'] = (git_date,'Git date of last commit');
    hdr[HMP+'GIT BRANCH'] = (git_branch,'Git branch');

    # Remove if existing
    if os.path.exists (filename):
        os.remove (filename);

    # Write and make it writable to all
    hdulist.writeto (filename);
    os.chmod (filename,0o666);
    # JDM: Can i delete and garbage collect to avoid problem with too many open files?


def linearize (data, gain):
    # subtract the first value from the array
    data -= (data[:,0,:,:])[:,None,:,:]
    # apply nonlinearity connection
    data[:] = (-1. + np.sqrt(1. + 4. * coef_flat(gain) * data))/(2. * coef_flat(gain))



def load_raw (hdrs, differentiate=True,
              removeBias=True, background=None, coaddRamp=False,
              badpix=None, flat=None, output='output',
              saturationThreshold=60000,
              continuityThreshold=10000,
              linear=False  ): 
    '''
    Load data and append into gigantic cube. The output cube is
    of shape: [nfile*nramp, nframes, ny, ny].

    If saturationThreshold is not None, then the non-differentiated
    data are compared to this threshold. Frames with some pixels
    exceeding this value are flagged and filled with zero.

    If continuityThreshold is not None, then the differentiated
    data are tested for continuity. When the difference between
    two consecutive frames is larger than this value, frames
    are flagged. Useful to detect cosmic rays.

    If differentiate==True, the consecutive frames of a ramp are
    subtracted together.

    If removeBias==True, the detector bias interference is removed
    by detecting the sinusoidal interference signal and subtracting it.

    If background is not None, thus background cube is subtracted
    from the data.

    If coaddRamp==True, the ramps inside each file are averaged together.
    Thus the resulting cube is of shape [nfile, nframes, ny, ny]

    Return (hdr, cubenp, cubemp) where hdr is the header of file, cubenp
    is the data as shape [nfile*nramp, nframes, ny, ny], and cubemp is
    the MJD of each frame as shape [nfile*nramp, nframes]
    '''
    log.info ('Load RAW files in mode coaddRamp=%s'%str(coaddRamp));

    # Build output header as the copy
    # of the first passed header
    hdr = hdrs[0].copy();
    hdr[HMQ+'NFILE'] = 0 #,'total number of files loaded');
    hdr[HMQ+'NRAMP'] = 0 #,'total number of ramp loaded');
    hdr[HMQ+'NSAT']  = 0 #,'total number of saturated frames');
    hdr['BZERO'] = 0;

    cube  = [];
    cubem = [];

    # Loop on files
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
            detector_gain = hdulist[1].header['GAIN'];
        # Read normal data. 
        else:
            data = hdulist[0].data;
            detector_gain = hdulist[0].header['GAIN'];

        # Convert to float
        data = data.astype ('float32');

        # Close file
        hdulist.close(); # JDM Memory leak worries...

        # Integrity check
        if np.min (data) == np.max (data):
            log.error ('All values are equal');
            raise ValueError ('RAW data are corrupted')

        # Dimensions
        nr,nf,ny,nx = data.shape;

        #  MJD of each frame
        mjd = headers.frame_mjd (h);
        mjd = mjd.reshape (nr,nf);

        # flag for invalid frames
        flag = np.zeros ((nr,nf), dtype=bool);

        # Guessed fringe window, FIXME: this may be wrong since we change the
        # orientation of images in saved data to match the header.
        # ys = ny - hdr['FR_ROW2'];
        # ye = ny - hdr['FR_ROW1'];
        # xc = int(nx - (hdr['FR_COL2'] + hdr['FR_COL1'])/2);
        # xs = xc - 10;
        # xe = xc + 10;

        # Frame is declared saturated if more than 10 pixels in
        # the center of the fringes are near saturation. flag is 0
        # if no saturation, or the id of the first saturated frame
        # if saturationThreshold is not None:
        #     flag = np.sum (data[:,:,ys:ye,xs:xe]>saturationThreshold, axis=(2,3));
        #     flag = np.argmax (flag > 10, axis=1);
        #     nsat = np.sum ( (flag.flatten() > 0) * (nf - flag.flatten()));
        #     hdr[HMQ+'NSAT'] += nsat;

        # Check if some frames are saturated. Check individual pixels, therefore
        # we make use of the badpixel mask if it was provided. flag array is 0
        # if no saturation, or the id of the first saturated frame. Note that we
        # don't check the edges of the images because badpixels are not properly
        # detected here
        
        if saturationThreshold is not None:
            # Discard badpixels from saturation check
            if badpix is None:
                tmp = data[:,:,2:-2,2:-2];
            else:
                tmp = (data * (badpix==False)[None,None,:,:])[:,:,2:-2,2:-2];
            # Discard bad lines and columns from saturation checks
            oks = (tmp.ptp (axis=(0,1,3)) != 0);
            tmp = tmp[:,:,oks,:];
            oks = (tmp.ptp (axis=(0,1,2)) != 0);
            tmp = tmp[:,:,:,oks];
            # Look for saturation
            flag += tmp.max (axis=(2,3)) > saturationThreshold;

        # deal with non-linearity
        if linear:
            linearize(data, detector_gain)

        # Take difference of consecutive frames
        if differentiate is True:
            data = np.diff (data,axis=1)[:,0:-1,:,:];
            mjd = 0.5 * (mjd[:,0:-1] + mjd[:,1:])[:,0:-1];
        
        # Remove bias. Note that the median should be taken
        # with an odd number of samples, to be unbiased.
        # WARNING: because of median, our best estimate of the bias
        # is to +/-1nph (convolved by the effect of gaussian_filter):
        if removeBias is True:
            # Subtract mean for
            breakpoint(); # don't do it this way.  interference will be removed after badpixels are known...
            ids = np.append (np.arange(15), data.shape[-1] - np.arange(1,15));
            bias = np.median (data[:,:,:,ids],axis=3);
            bias = gaussian_filter (bias,[0,0,1]);
            data = data - bias[:,:,:,None];

        if background is not None:
            data -= background;

        # Check continuity in flux, to detect cosmics. We consider that the
        # frames after a discontinuity are invalid (e.g saturated).
        if continuityThreshold is not None:
            if badpix is None:
                tmp = np.diff (data[:,:,2:-2,2:-2], axis=1);
            else:
                tmp = np.diff ((data * (badpix==False)[None,None,:,:])[:,:,2:-2,2:-2], axis=1);
            flag[:,1:-2] = tmp.max (axis=(2,3)) > continuityThreshold;
            
        # Loop on ramps
        nsat = 0;
        for r in range(data.shape[0]):
            # This ramp has flagged frames
            if flag[r,:].any():
                # Detect first flagged frame in ramp.
                mark = np.argmax (flag[r,:]);
                # Take margin, to start discarding few frames before mark.
                mark = np.maximum (mark-3,0);
                # Zero frames starting at mark
                data[r,mark:,:,:] = 0.0;
                # Count total number of zeroed frames
                nsat += nf - mark;

        #JDM. Apparantly the hdr object from fits is fancy and the code uses some features....

        # Increase nsat in header
        hdr[HMQ+'NSAT'] += nsat # hdr[HMQ+'NSAT'][0]+nsat,hdr[HMQ+'NSAT'][1] ## remove the tuple?

        # Add this RAW file in hdr
        keylist =list(hdr.keys())
        rawlist = list(filter(lambda x: 'MIRC PRO RAW' in x, keylist))
        nraw = len(rawlist)
        hdr['MIRC PRO RAW%i'%(nraw+1,)] = os.path.basename (h['ORIGNAME'])[-50:];
        hdr['MIRC QC NFILE'] += 1;
        hdr['MIRC QC NRAMP'] += data.shape[0];

        # Co-add ramp if required
        if coaddRamp == 'mean':
            data = np.mean (data,axis=0,keepdims=True);
            mjd  = np.mean (mjd, axis=0,keepdims=True);
        elif coaddRamp == 'sum':
            data = np.sum (data,axis=0,keepdims=True);
            mjd  = np.sum (mjd, axis=0,keepdims=True);

        # Append the data in the final cube
        cube.append  (data);
        cubem.append (mjd);

    # Allocate memory
    log.info ('Allocate memory');
    shape = cube[0].shape;
    nramp = sum ([c.shape[0] for c in cube]);
    cubenp = np.zeros ((nramp,shape[1],shape[2],shape[3]),dtype='float32');
    cubemp = np.zeros ((nramp,shape[1]));

    # Set data in cube, and free initial memory in its way
    log.info ('Set data in cube');
    ramp = 0;
    for c in range (len(cube)):
        cubenp[ramp:ramp+cube[c].shape[0],:,:,:] = cube[c];
        cubemp[ramp:ramp+cube[c].shape[0],:]     = cubem[c];
        ramp += cube[c].shape[0];
        cube[c]  = None;
        cubem[c] = None;

    # Apply flat
    if flat is not None:
        log.info ('Apply flat');
        cubenp /= flat[None,None,:,:];
    else:
        log.info ('No flat applied');

    # Recompute badpixels
    if badpix is None:
        log.info ('No badpixel map');
    else:
        log.info ('Recompute %i bad pixels (interpolate in spectral direction only)'%np.sum (badpix));
        ref = np.mean (cubenp, axis=(0,1));
        idx = np.argwhere (badpix);
        # cubenp[:,:,idx[:,0],idx[:,1]] = 0.25 * cubenp[:,:,idx[:,0]-1,idx[:,1]-1] + \
        #                                 0.25 * cubenp[:,:,idx[:,0]+1,idx[:,1]-1] + \
        #                                 0.25 * cubenp[:,:,idx[:,0]-1,idx[:,1]+1] + \
        #                                 0.25 * cubenp[:,:,idx[:,0]+1,idx[:,1]+1];
        cubenp[:,:,idx[:,0],idx[:,1]] = 0.5 * cubenp[:,:,idx[:,0]-1,idx[:,1]] + \
                                        0.5 * cubenp[:,:,idx[:,0]+1,idx[:,1]];
        # Figure
        fig,ax = plt.subplots (3,1);
        fig.suptitle (headers.summary (hdrs[0]));
        ax[0].imshow (badpix);
        ax[1].imshow (ref);
        ax[2].imshow (np.mean (cubenp, axis=(0,1)));
        write (fig, output+'_rmbad.png');

    # Some verbose
    nr,nf,ny,nx = cubenp.shape;
    log.info ('Number of files loaded = %i'%hdr[HMQ+'NFILE']);
    log.info ('Number of ramp loaded = %i'%hdr[HMQ+'NRAMP']);
    log.info ('Number of frames loaded = %i'%(hdr[HMQ+'NRAMP']*nf));
    log.info ('Number of saturated frames = %i'%hdr[HMQ+'NSAT']);

    # Fraction of saturation
    fsat = 1.0 * hdr[HMQ+'NSAT'] / (hdr[HMQ+'NRAMP']*nf);
    log.check (fsat,'Fraction of saturated frames = %.3f'%fsat);
    hdr[HMQ+'FSAT']  = (fsat,'fraction of saturated frames');

    return hdr,cubenp,cubemp;


def load_raw_only (hdrs):   
    '''
    Load data group and append into gigantic cube. The output cube is
    of shape: [nfile*nramp, nframes, ny, ny].

    Don't do any processing, just load the data.

    Return (hdr, cubenp, cubemp) where hdr is the header of file, cubenp
    is the data as shape [nfile*nramp, nframes, ny, ny], and cubemp is
    the MJD of each frame as shape [nfile*nramp, nframes]

    not if same_header is True then use the passed header info not the header in the file
    which may have been updated or corrected by the header csv file
    '''
    #log.info ('Load RAW files in mode coaddRamp=%s'%str(coaddRamp));

    # Build output header as the copy
    # of the first passed header
    continuityThreshold =10000
    hdr = hdrs[0].copy();
    hdr[HMQ+'NFILE'] = 0 #,'total number of files loaded');
    hdr[HMQ+'NRAMP'] = 0 #,'total number of ramp loaded');
    #hdr[HMQ+'NSAT']  = 0 #,'total number of saturated frames');
    hdr['BZERO'] = 0; #JDM. Is this needed?

    cube  = [];
    cubem = [];

    # Loop on files
    for h in hdrs:
        fileinfo = h['ORIGNAME'] + ' (' +h['FILETYPE']+')';
        log.debug ('Load %s'%fileinfo);
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
            detector_gain = hdulist[1].header['GAIN'];
        # Read normal data. 
        else:
            data = hdulist[0].data;
            detector_gain = hdulist[0].header['GAIN'];

        # Convert to float
        data = data.astype ('float32');

        # Close file
        hdulist.close(); # JDM Memory leak worries...

        # Integrity check
        if np.min (data) == np.max (data):
            log.error ('All values are equal');
            raise ValueError ('RAW data are corrupted')

        # Dimensions
        nr,nf,ny,nx = data.shape;
        flag = np.zeros ((nr,nf), dtype=bool);

        
        # if entire row is same value, then mark as nan.
        row_rms = np.repeat(np.nanstd(data,axis=3,keepdims=True,dtype=np.float64),nx,axis=3);

        #row_rms1d= np.nanstd(data[:,:,:,:-1],axis=(0,1,3),dtype=np.float64)
        row_rms1d=np.median(np.nanstd(data,axis=(3),dtype=np.float64),axis=(0,1) )

        goodrows = np.squeeze(np.argwhere(row_rms1d > 0.1))
        # purge bad rows rather than carry them around as NANs. subtly changes timing....
        data=data[:,:,goodrows,:]
        nr,nf,ny,nx = data.shape;
        #breakpoint()

        # if pixel doesn't change for an entire ramp then mark bad
        # ramp_rms = np.repeat(np.nanstd(data,axis=1,keepdims=True),nf,axis=1);
        # data = np.where(ramp_rms < 0.1, np.nan,data)

        #if hdr['INSTRUME'] == 'MIRC-X' or hdr['INSTRUME'] == 'MYSTIC':
        satlevel = 65534 # for MIRC-X and MYSTIC 
                             # use different load_raw_only for other instruments

        
        data = np.where(data > satlevel,np.nan,data) # to allow use of masked arrays
        data[:,:,0,0:7] = np.nan # First 7 pixels contain frame # (in header)
        # In some data, the last frame was corrupted at end of a sequence. Fixed around 2022.
        # will catch that since all values were the same.

        # Check continuity in flux, to detect cosmics. We consider that the
        # frames after a discontinuity are invalid (e.g saturated).
        if continuityThreshold is not None:
            #tmp = np.diff (data[:,:,2:-2,2:-2], axis=1); 
            tmp = np.diff (data, axis=1); 
            flag[:,1:] = np.nanmax(tmp,axis=(2,3)) > continuityThreshold;
            
            # Loop on ramps
            ncrays = 0;
            for r in range(nr):
                # This ramp has flagged frames
                if flag[r,:].any():
                    # Detect first flagged frame in ramp.
                    mark = np.argmax (flag[r,:]);
                    # Take margin, to start discarding few frames before mark.
                    mark = np.maximum (mark-3,0);
                    # Zero frames starting at mark
                    data[r,mark:,:,:] = np.nan;
                    # Count total number of zeroed frames
                    ncrays += nf - mark;
            if ncrays > 0: log.info("Number of frames removed due to cosmic rays: %i"%ncrays);
         
        log.debug ('Data size: '+str(data.shape));

        #  MJD of each frame
        mjd = headers.frame_mjd (h);
        mjd = mjd.reshape (nr,nf);


        # Loop on ramps
        

        # Add this RAW file in hdr
        keylist =list(hdr.keys())
        rawlist = list(filter(lambda x: 'MIRC PRO RAW' in x, keylist))
        nraw = len(rawlist)
        hdr['MIRC PRO RAW%i'%(nraw+1,)] = os.path.basename (h['ORIGNAME'])[-50:];
        hdr['MIRC QC NFILE'] += 1;
        hdr['MIRC QC NRAMP'] += data.shape[0];

        # Append the data in the final cube
        cube.append  (data);
        cubem.append (mjd);

    # Allocate memory
    #log.info ('Allocate memory');
    shape = cube[0].shape;
    nramp = sum ([c.shape[0] for c in cube]);
    cubenp = np.zeros ((nramp,shape[1],shape[2],shape[3]),dtype='float32');
    cubemp = np.zeros ((nramp,shape[1]));

    # Set data in cube, and free initial memory in its way
    #log.info ('Set data in cube');
    ramp = 0;
    for c in range (len(cube)):
        cubenp[ramp:ramp+cube[c].shape[0],:,:,:] = cube[c];
        cubemp[ramp:ramp+cube[c].shape[0],:]     = cubem[c];
        ramp += cube[c].shape[0];
        cube[c]  = None;
        cubem[c] = None;

    # Some verbose
    nr,nf,ny,nx = cubenp.shape;
    log.debug ('Number of files loaded = %i'%hdr[HMQ+'NFILE']);
    log.debug ('Number of ramp loaded = %i'%hdr[HMQ+'NRAMP']);
    log.debug ('Number of frames loaded = %i'%(hdr[HMQ+'NRAMP']*nf));
    #log.info ('Number of saturated frames = %i'%hdr[HMQ+'NSAT']);


    return hdr,cubenp,cubemp;