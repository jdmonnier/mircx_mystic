import numpy as np
from astropy.io import fits as pyfits
from astropy.time import Time
import os
import glob
import pickle

from . import log

# Global shortcut
HM  = 'HIERARCH MIRC ';
HMQ = 'HIERARCH MIRC QC ';
HMP = 'HIERARCH MIRC PRO ';
HMW = 'HIERARCH MIRC QC WIN ';

def str2bool (s):
    if s == True or s == 'TRUE': return True;
    if s == False or s == 'FALSE': return False;
    raise ValueError('Invalid boolean string');

def loaddir (dirs):
    '''
    Load the headers of all files mircx*.fit* from
    the input list of directory
    '''
    elog = log.trace ('loaddir');

    # Ensure this is a list
    if type(dirs) == str:
        dirs = [dirs];

    # Load all dirs
    hdrs = [];
    for dir in dirs:
        log.info ('Load directory: '+dir);
        files = glob.glob (dir+'/mircx*.fit*');
        files = sorted (files);

        # Load log
        hlog = [];
        fpkl = dir+'/mircx_hdrs.pkl';
        if os.path.isfile (fpkl):
            try:
                log.info ('Load binary log %s'%fpkl);
                hlog = pickle.load (open(fpkl, 'rb'));
            except:
                log.info ('Failed to load...');

        # Load header
        hdrs_here = load (files, hlog=hlog);
                
        # Dump log
        log.info ('Write binary log %s'%fpkl);
        if os.path.isfile (fpkl): os.remove (fpkl);
        pickle.dump (hdrs_here, open(fpkl, 'wb'), -1);
        
        # Append headers
        hdrs.extend (hdrs_here);

    return hdrs;

def load (files, hlog=[]):
    '''
    Load the headers of all input files. The following keywords
    are added to each header: MJD-OBS and ORIGNAME.
    The output is a list of FITS headers.
    '''
    elog = log.trace ('load');
    hdrs = []

    # Files available in log
    filesin = [os.path.split (h['ORIGNAME'])[1] for h in hlog];

    # Loop on files
    for fn,f in enumerate (files):
        try:
            
            # Recover or read header
            try:
                hdr = hlog[filesin.index (os.path.split (f)[1])];
                log.info('Recover header %i over %i (%s)'%(fn,len(files),f));
            except:
                if f[-7:] == 'fits.fz':
                    hdr = pyfits.getheader(f, 1);
                else:
                    hdr = pyfits.getheader(f, 0);
                log.info('Read header %i over %i (%s)'%(fn,len(files),f));

            # Add file name
            hdr['ORIGNAME'] = f;

            # Add MJD-OBS
            t = Time(hdr['DATE-OBS'] + 'T'+ hdr['UTC-OBS'], format='isot', scale='utc');
            hdr['MJD-OBS'] = t.mjd;

            # Append
            hdrs.append (hdr);
            
        except (KeyboardInterrupt, SystemExit):
            raise;
        except Exception as exc:
            log.warning ('Cannot get header of '+f+' ('+str(exc)+')');

    log.info ('%i headers loaded'%len(hdrs));
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

def group (hdrs, mtype, delta=300.0, Delta=300.0, continuous=True, keys=[]):
    '''
    Group the input headers into list of compatible files.
    A new group is started if:
    - a file of different type is interleaved,
    - the detector or instrument setup is different,
    - the time distance is larger than delta.
    The output is a list of list.
    '''
    elog = log.trace('group_headers');
    
    groups = [[]];
    mjd = -10e9;

    # Key used to define setup
    keys = ['FILETYPE'] + keys;

    # Sort by time
    hdrs = sorted (hdrs,key=lambda h: h['MJD-OBS']);
    
    # Assume hdrs is sorted
    for h in hdrs:
        fileinfo = h['ORIGNAME'] + ' (' +h['FILETYPE']+')';
        
        # if different type, start new group and continue
        if mtype not in h['FILETYPE']:
            if groups[-1] != [] and str2bool(continuous):
                groups.append([]);
            continue;

        # If no previous
        if groups[-1] == []:
            log.info('New group %s'%fileinfo);
            groups[-1].append(h);
            continue;

        # If no match with last, we start new group
        if match (h,groups[-1][-1],keys,delta) is False:
            log.info('New group (gap) %s'%fileinfo);
            groups.append([h]);
            continue;

        # If no match with first, we start new group
        if match (h,groups[-1][0],keys,Delta) is False:
            log.info('New group (integration) %s'%fileinfo);
            groups.append([h]);
            continue;
        
        # Else, add to current group
        log.info('Add file %s'%fileinfo);
        groups[-1].append(h);

    # Clean from void groups
    groups = [g for g in groups if g != []];
    return groups;

def assoc (h, allh, tag, keys=[], which='closest', required=0):
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

    # Check closest
    if len (out) > required and which=='closest':
        if required < 2:
            time_diffs = np.array([o['MJD-OBS'] - h['MJD-OBS'] for o in out])
            out = [out[np.abs(time_diffs).argmin()]]
        else:
            raise NotImplementedError('Not supported yet');
            
    # Check best quality
    if len (out) > required and which=='best':
        if required < 2:
            quality = np.array([o['HIERARCH MIRC QC QUALITY'] for o in out]);
            out = [out[np.argmax (quality)]];
        else:
            raise NotImplementedError('Not supported yet');
            
    # Check required
    if len (out) < required:
        log.warning ('Cannot find %i %s'%(required,tag))
    else:
        log.info ('Find %i %s (%s ...)'%(len(out),tag,out[0]['ORIGNAME']));
        
    return out

def check_input (hdrs, required=1, maximum=100000):
    '''
    Check the input when provided as hdrs
    '''

    # Ensure a list
    if type (hdrs) is not list:
        hdrs = [hdrs];

    # Check inputs are headers
    hdrs = [h for h in hdrs if type(h) is pyfits.header.Header or \
            type(h) is pyfits.hdu.compressed.CompImageHeader];

    if len(hdrs) < required:
        raise ValueError ('Missing mandatory input');

    if len(hdrs) > maximum:
        raise ValueError ('Too many input');

def rep_nan (val,*rep):
    if not rep: rep = 0.0;
    return val if np.isfinite (val) else rep;
