from astropy.time import Time
from astropy.io import fits as pyfits
from astropy.stats import sigma_clipped_stats
import numpy as np
import os

from . import log

def output (outputDir,hdr,suffix):
    '''
    Return a string like: ./outputDir/mircx00550_suffix
    '''
    
    # Build diretory if needed
    if not os.path.exists (outputDir):
        os.makedirs (outputDir);
        
    # Get filename 
    name = hdr['ORIGNAME'];
    name = os.path.splitext (os.path.basename(name))[0];
    if name[-5:] == '.fits':
        name = name[0:-5];
    
    # Return
    output = outputDir + '/' + name + '_' + suffix;
    return output;

def write (hdulist,filename):
    '''
    Write file.
    '''
    fileinfo = filename + ' ('+hdulist[0].header['FILETYPE']+')';
    log.info ('Write file %s'%fileinfo);
    
    if os.path.exists (filename):
        os.remove (filename);
    
    hdulist.writeto (filename);

def load_raw (hdrs, coaddRamp=False):
    '''
    Load data and append into gigantic cube. The output cube is
    of shape: [nfile*nramp, nframes, ny, ny].
    
    If coaddRamp==True, the ramps inside each file are averaged together.
    Thus the resulting cube is of shape [nfile, nframes, ny, ny]    
    '''

    # Build header
    hdr = hdrs[0].copy();
    hdr.set('HIERARCH MIRC QC NRAMP',0,'Total number of ramp loaded');
    
    cube = [];
    for h in hdrs:
        fileinfo = h['ORIGNAME'] + ' (' +h['FILETYPE']+')';
        log.info ('Load file %s'%fileinfo);
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
        
        # Preproc data
        data = np.diff (data.astype('float'),axis=1);
        ids = np.append (np.arange(10), data.shape[-1] - np.arange(1,11));
        bias = np.median (data[:,:,:,ids],axis=3);
        data = data - bias[:,:,:,None];

        # Append ramps or co-add them
        hdr['HIERARCH MIRC QC NRAMP'] += data.shape[0]
        if coaddRamp is True:
            cube.append (np.mean (data,axis=0)[None,:,:,:]);
        else:
            cube.append (data);

    # Convert to array
    log.info ('Convert to cube');
    cube = np.array(cube);

    # Concatenate all files into a single sequence
    log.info ('Reshape cube');
    (a,b,c,d,e) = cube.shape;
    cube.shape = (a*b,c,d,e);
    
    return hdr,cube;
