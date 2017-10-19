import numpy as np
from astropy.io import fits as pyfits
from astropy.time import Time
import os

import log
import setup

def load (files):
    '''
    Load the headers of all input files. The following keywords
    are added to each header: MJD-OBS and ORIGNAME.
    The output is a list of FITS headers.
    '''
    log.trace('load_headers');
    
    hdrs = []    
    for f in files:
        try:
            # Open file
            hdulist = pyfits.open (f);
            
            # Load file header
            if f[-7:] == 'fits.fz':
                hdr = hdulist[1].header;
            else:
                hdr = hdulist[0].header;

            # Add file name
            hdr['ORIGNAME'] = f;

            # Add MJD-OBS
            t = Time(hdr['DATE-OBS'] + 'T'+ hdr['UTC-OBS'], format='isot', scale='utc');
            hdr['MJD-OBS'] = t.mjd;

            # Close
            hdrs.append(hdr);
            hdulist.close();
            log.notice('Read header for %s'%f);
        except:
            log.warning('Cannot get header of %s'%f);
            
    return hdrs;

def match (h1,h2,keys,delta):
    '''
    Return True fs all keys are the same in header h1
    and header h2, and if the time difference is less
    than delta (s). The keys shall be a list of string.
    '''
    # Check all keywords are the same
    answer = True;
    for k in keys:
        answer *= (h1[k] == h2[k]);
            
    # Check time is close-by
    answer *= (np.abs(h1['MJD-OBS'] - h2['MJD-OBS'])*24.*3600 < delta);

    # Ensure binary output
    return True if answer else False;

def group (hdrs, mtype, delta=300.0):
    '''
    Group the input headers into list of compatible files.
    A new group is started if:
    - a file of different type is interleaved,
    - the detector or instrument setup is different,
    - the time distance is larger than delta.
    The output is a list of list.
    '''
    log.trace('group_headers');
    
    groups = [[]];
    mjd = -10e9;

    # Key used to define setup
    keys = ['FILETYPE'] + setup.detector + setup.instrument;
    
    # Assume hdrs is sorted
    for h in hdrs:
        fileinfo = h['ORIGNAME'] + ' (' +h['FILETYPE']+')';
        
        # if different type, continue
        # and start new group
        if h['FILETYPE'] != mtype:
            # log.notice('Skip file %s'%fileinfo);
            if groups[-1] != []:
                groups.append([]);
            continue;

        # If no previous
        if groups[-1] == []:
            log.notice('New group %s'%fileinfo);
            groups[-1].append(h);
            continue;

        # If no match, we start new group
        if match (h,groups[-1][-1],keys,delta) is False:
            log.notice('New group %s'%fileinfo);
            groups.append([h]);
            continue;

        # Else, add to current group
        log.notice('Add file %s'%fileinfo);
        groups[-1].append(h);

    # Clean from void groups
    groups = [g for g in groups if g != []];
    return groups;

def assoc (h, allh, tag, keys, which='closest', required=0):
    '''
    Search for headers with tag and matching criteria
    '''

    # Keep only the requested tag matching the criteria
    atag = [a for a in allh if a['FILETYPE']==tag]
    out = []
    for a in atag:
        tmp = True
        for k in keys:
            tmp *= (h[k] == a[k])
        if tmp:
            out.append(a)

    # Check required
    if len (out) < required:
        log.warning ('Cannot find %i %s for %s'%(required,tag,h['ORIGNAME']))

    # Check closest
    if len (out) > required and which=='closest':
        # Case need closest and more than 1 not supported yet
        if required < 2:
            time_diffs = np.array([o['MJD-OBS'] - h['MJD-OBS'] for o in out])
            out = [out[np.abs(time_diffs).argmin()]]
        
    log.notice ('Find %i %s for %s'%(len(out),tag,h['ORIGNAME']));
    return out
