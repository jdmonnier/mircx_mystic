from astropy.time import Time
from astropy.io import fits as pyfits
import numpy as np
import os

import log

def calib_output (hdr):
    '''
    Return a string like: ./calib/mircx00550
    '''
    directory = './calib/';
    if not os.path.exists(directory):
        os.makedirs(directory)    
    
    name = hdr['ORIGNAME'];
    name = os.path.splitext(os.path.basename(name))[0];
    if name[-5:] == '.fits':
        name = name[0:-5];
    return directory + name;

def reduced_output (hdr):
    '''
    Return a string like: ./reduced/mircx00550
    '''
    directory = './reduced/';
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    name = hdr['ORIGNAME'];
    name = os.path.splitext(os.path.basename(name))[0];
    if name[-5:] == '.fits':
        name = name[0:-5];
    return directory + name;

def write (hdulist,filename):
    '''
    Write file.
    '''
    fileinfo = filename + ' ('+hdulist[0].header['FILETYPE']+')';
    log.notice ('Write file %s'%fileinfo);
    
    if os.path.exists (filename):
        os.remove (filename);
    
    hdulist.writeto (filename);

def load_raw (hdrs):
    '''
    Load data and append into gigantic cube
    '''
    cube = [];
    for h in hdrs:
        fileinfo = h['ORIGNAME'] + ' (' +h['FILETYPE']+')';
        log.notice ('Load file %s'%fileinfo);
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
        bias = 0.5 * (np.median(data[:,:,:,0:10],axis=3) + np.median(data[:,:,:,-11:-1],axis=3));
        data = data - bias[:,:,:,None];

        # Append        
        cube.append(data);

    # Convert to array
    log.notice ('Convert to cube');
    cube = np.array(cube);
    
    # Concatenate all files into a single sequence
    log.notice ('Reshape cube');
    (a,b,c,d,e) = cube.shape;
    cube.shape = (a*b,c,d,e);

    # Build header
    hdr = hdrs[0].copy();
    hdr.set('HIERARCH MIRC QC NFRAME',cube.shape[0],'Total number of frames used');
    
    return hdr,cube;
