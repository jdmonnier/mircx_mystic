import pdb
from pydoc import pathdirs
from syslog import LOG_WARNING
import numpy as np;
import pandas as pd;
import sys

from astropy.io import fits as pyfits;
from astropy.time import Time;
from astropy.io import ascii;
from astropy.table import Table;


import os, glob, pickle, datetime, re, csv, gc;

from . import log
counters={'gpstime':0, 'etalon':0, 'sts':0}

# Global shortcut
# removing HIERARCH for 2.0
HM  = 'MIRC ';
HMQ = 'MIRC QC ';
HMP = 'MIRC PRO ';
HMW = 'MIRC QC WIN ';
HC = 'CHARA ';


def str2bool (s):
    if s == True or s == 'TRUE': return True;
    if s == False or s == 'FALSE': return False;
    raise ValueError('Invalid boolean string');

def getval (hdrs, key, default=np.nan):
    '''
    Return a numpy array with the values in header
    '''
    return np.array ([h.get(key,default) for h in hdrs]);

def summary (hdr):
    '''
    Return a short string to
    summarize the header
    '''
    value = 'G%i-L%i-R%i  %.4f %s'%(hdr.get('GAIN',0),hdr.get('NLOOPS',0),hdr.get('NREADS',0),
                                 hdr.get('MJD-OBS',0.0),hdr.get('OBJECT','unknown'));

    if 'MIRC PRO NCOHER' in hdr:
        value += ' NCOHER=%i'%(hdr.get('MIRC PRO NCOHER',0));

    return value;

def setup (hdr, params):
    '''
    Return the setup as string
    '''
    value = ' / '.join([str(hdr.get(p,'--')) for p in params]);
    return value;

def get_beam (hdr):
    '''
    Return the i of BEAMi
    '''
    n = hdr if type(hdr) is str else hdr['FILETYPE'];
    for i in range(1,7):
        if 'beam%i'%i in n: return i;
        if 'BEAM%i'%i in n: return i;
    return None;

def clean_date_obs (hdr):
    '''
    Clean DATE-OBS keyword to always match
    ISO format YYYY-MM-DD
    '''
    if 'DATE-OBS' not in hdr:
        return;
    
    if hdr['DATE-OBS'][4] == '/':
        # Reformat DATE-OBS YYYY/MM/DD -> YYYY-MM-DD
        hdr['DATE-OBS'] = hdr['DATE-OBS'][0:4] + '-' + \
          hdr['DATE-OBS'][5:7] + '-' + \
          hdr['DATE-OBS'][8:10];
    elif hdr['DATE-OBS'][2] == '/':
        # Reformat DATE-OBS MM/DD/YYYY -> YYYY-MM-DD
        hdr['DATE-OBS'] = hdr['DATE-OBS'][6:10] + '-' + \
          hdr['DATE-OBS'][0:2] + '-' + \
          hdr['DATE-OBS'][3:5];

def get_mjd (hdr, origin=['linux','gps','mjd'], check=2.0,Warning=True):
    '''
    Return the MJD-OBS as computed either by Linux time
    TIME_S + 1e-9 * TIME_US  (note than TIME_US is actually
    nanosec) or by GPS time DATE-OBS + UTC-OBS, or by an
    existing keyword 'MJD-OBS'.
    '''

    # Check input
    if type(origin) is not list: origin = [origin];
        
    # Read header silently
    try:    
        mjdu = Time (hdr['DATE-OBS'] + 'T'+ hdr['UTC-OBS'], format='isot', scale='utc').mjd;
    except:
        mjdu = 0.0;
    try:    
        mjdl = Time (hdr['TIME_S']+hdr['TIME_US']*1e-9,format='unix').mjd;
    except:
        mjdl = 0.0;
    try:    
        mjd  = hdr['MJD-OBS'];
    except:
        mjd  = 0.0;

    # Check the difference in [s]
    delta = np.abs (mjdu-mjdl) * 24 * 3600;
    if (delta > check) & (Warning == True):
        log.warning ('IN %s :\n   UTC-OBS and TIME are different by %.1f s!! '%(hdr['ORIGNAME'],delta));

    # Return the requested one
    for o in origin:  # if origin in array, then returns result in priority order.
        if o == 'linux' and mjdl != 0.0:
            return mjdl, (delta > check);
        if o == 'gps' and mjdu != 0.0:
            return mjdu, (delta > check);
        if o == 'mjd' and mjd != 0.0:
            return mjd, (delta > check);
    
    return 0.0, None;
    
def loaddir (dirs, uselog=True):
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
        if os.path.isdir (dir) is False:
            log.info ('Skip directory (does not exist): '+dir);
            continue;
        
        log.info ('Load directory: '+dir);
        files  = glob.glob (dir+'/mircx*.fits');
        files += glob.glob (dir+'/mystic*.fits');
        files += glob.glob (dir+'/mircx*.fits.fz');
        files += glob.glob (dir+'/mystic*.fits.fz');

        files = [ x for x in files if "fibexpmap" not in x ] # remove non-data files.

        # Check if any
        if len(files) == 0:
            log.warning ('No mircx or mystic data files in this directory');
            continue;

        # Sort them alphabetically
        files = sorted (files);

        # Load headers
        hdrs_here = load (files);
        
        # Append headers in case of multiple directories -- not used...
        hdrs.extend (hdrs_here);



    return hdrs;

def load (files):
    '''
    Load the headers of all input files. The following keywords
    are added to each header: MJD-OBS, MJD-LOAD and ORIGNAME.
    The output is a list of FITS headers.
    '''
    hdrs = []

    # Loop on files
    log.info("Number of files to read: %i "%len(files))
    log.info("First File: %s "%files[0])
    log.info("Last File : %s"%files[-1])
    for fn,f in enumerate (files):
        try:
            
            # Read compressed file
            if f[-7:] == 'fits.fz':
                #hdr = pyfits.getheader(f, 1);
                hdulist=pyfits.open(f,memmap=False)
                hdr=hdulist[1].header.copy()
                del hdulist[1].header
                hdulist.close()
                del hdulist
                fnum=int(f[-13:-8])  # might not always be true.

            # Read normal file
            else:
                #hdr = pyfits.getheader(f, 0);
                hdulist=pyfits.open(f,memmap=False)
                hdr=hdulist[0].header.copy()
                del hdulist[0].header # save a little memory along the way.
                hdulist.close()
                del hdulist
                fnum=int(f[-10:-5])

            # Add file name
            hdr['ORIGNAME'] = f;
            hdr['FILENUM'] = fnum;

            # Test if FRAME_RATE is in header
            if 'MIRC FRAME_RATE' not in hdr and 'EXPOSURE' in hdr:
                log.warning ('Assume FRAME_RATE is 1/EXPOSURE');
                hdr['MIRC FRAME_RATE'] = 1e3/hdr['EXPOSURE'];

            # Test if NBIN is in header
            if 'NBIN' not in hdr:
                hdr['NBIN'] = 1; 

            # Check change of card
            if 'ENDFR' in hdr:
                log.warning ('Old data with ENDFR');
                hdr.rename_keyword ('ENDFR','LASTFR');

            # Check NBIN
            if 'NBIN' not in hdr and hdr['FILETYPE'] != 'FLAT_MAP':
                log.warning ('Old data with no NBIN (set to one)');
                hdr['NBIN'] = 1;

            # Reformat DATE-OBS
            clean_date_obs (hdr);

            if 'DPOL_ROW' in hdr:
                    if hdr['DPOL_ROW'] !=0:
                    # check if config ends with _WOLL and hdr['CONF_NA']
                        conf_na = hdr['CONF_NA'].strip()
                        if conf_na[-5:] != '_WOLL': hdr['CONF_NA']=conf_na+'_WOLL'


            # Compute MJD from information in header
            mjd, temp_flag = get_mjd (hdr, Warning = (counters["gpstime"] == 0));
            #mjd, temp_flag = get_mjd (hdr, Warning = True);
            if (counters["gpstime"] == 0) & temp_flag:
                    log.warning("Additional time discrepancy warnings suppressed.")
            if (temp_flag): 
                counters["gpstime"]+=1            
            # Set in header
            hdr['MJD-OBS'] = (mjd, '[mjd] Observing time');

            # Add the loading time
            hdr['MJD-LOAD'] =  (Time.now().mjd, '[mjd] Last loading time (UTC)');

            # Check if STS data
            if hdr.get ('MIRC STS_IR_FOLD','OUT') == 'IN':
                #log.info ('Set OBJECT = STS because STS_IR_FOLD is IN');
                hdr['OBJECT'] = 'STS';
                counters["sts"] +=1

            
            # Check if ETALON
            if hdr.get ('MIRC ARMADA','OUT') == 'IN':
                counters["etalon"] +=1
                #if hdr['OBJECT'][-1]=='E':
                #    #log.info ('ETALON is IN for OBJECT');  
            #else:    
                #    #log.info ('Set OBJECT = OBJECT_E because ETALON is IN');
                if hdr['OBJECT'][-1] != 'E': 
                    hdr['OBJECT'] += '_E';  # JDM slightly preferes ETALON_OBJECT... but we will keep

            # Append
            hdrs.append (hdr);
    
        except (KeyboardInterrupt, SystemExit):
            raise;
        except Exception as exc:
            log.warning ('Cannot get header of '+f+' ('+str(exc)+')');
        
        #progress bar
        if fn == len(files)//4:   
            log.info("PROGRESS 25% Done")
        if fn == len(files)//2:   
            log.info("PROGRESS 50% Done")
        if fn == len(files)*3//4: 
            log.info("PROGRESS 75% Done")

    gc.collect()
    log.info('Number of files with time discrepancy: %i '%counters['gpstime'])
    log.info('Number of files with STS: %i '%counters['sts'])
    log.info('Number of files with Etalon: %i '%counters['etalon'])
    log.info ('%i headers loaded'%len(hdrs));
    return hdrs;

def frame_mjd (hdr):
    '''
    Compute MJD time for each frame from STARTFR to LASTFR.
    Assumig STARTFR has the MJD-OBS and the time between
    frame is given by HIERARCH MIRC FRAME_RATE.
    '''

    # Check consistency
    if hdr['LASTFR'] < hdr['STARTFR']:
        raise ValueError ('LASTFR is smaller than STARTFR');

    # Number of frame since start
    nframe = hdr['LASTFR'] - hdr['STARTFR'] + 1;

    # If binning
    nbin = hdr.get ('NBIN',1);
    if  nbin > 1:
        log.info ('Data are binned by %i'%nbin);

    # Build counter
    counter = np.arange (0, nframe, nbin);

    # Time step between frames in [d]
    # with new headers, the HIERRACH is removed from dictionary.

    delta = 1./hdr['MIRC FRAME_RATE'] / 24/3600; 

    
    # Compute assuming MJD-OBS is time of first frame
    mjd = hdr['MJD-OBS'] + delta * counter;

    return mjd;

def match (h1,h2,keys,delta):
    '''
    Return True fs all keys are the same in header h1
    and header h2, and if the time difference is less
    than delta (s). The keys shall be a list of string.
    '''
    # Check all keywords are the same
    answer = True;
    for k in keys:
        answer *= (h1.get(k,None) == h2.get(k,None));

    # Check time is close-by
    answer *= (np.abs(h1.get('MJD-OBS',0.0) - h2.get('MJD-OBS',0.0))*24.*3600 < delta);

    # Ensure binary output
    return True if answer else False;

def group (hdrs, mtype, delta=300.0, Delta=300.0, continuous=True, keys=[], logLevel=1):
    '''
    Group the input headers into list of compatible files.
    A new group is started if:
    - a file of different type is interleaved,
    - the detector or instrument setup is different,
    - the time distance between consecutive is larger than delta.
    - the total integration is larger than Delta
    The output is a list of list.
    '''
    elog = log.trace ('group_headers');
    
    groups = [[]];
    mjd = -10e9;

    # Key used to define setup
    keys = ['FILETYPE'] + keys;

    # Define the regular expression to match file type
    regex = re.compile ('^'+mtype+'$');

    # Sort by time
    hdrs = sorted (hdrs,key=lambda h: h['MJD-OBS']);
    
    # Assume hdrs is sorted
    for h in hdrs:
        fileinfo = h['ORIGNAME'] + ' (' +h['FILETYPE']+')';
        
        # if different type, start new group and continue
        if bool (re.match (regex, h['FILETYPE'])) is False:
            if groups[-1] != [] and str2bool (continuous):
                groups.append([]);
            continue;

        # If no previous
        if groups[-1] == []:
            if logLevel > 4: log.info('New group %s'%fileinfo);
            groups[-1].append(h);
            continue;

        # If no match with last, we start new group
        if match (h,groups[-1][-1],keys,delta) is False:
            if logLevel > 4: log.info('New group (gap) %s'%fileinfo);
            groups.append([h]);
            continue;

        # If no match with first, we start new group
        if match (h,groups[-1][0],keys,Delta) is False:
            if logLevel > 4: log.info('New group (integration) %s'%fileinfo);
            groups.append([h]);
            continue;
        
        # Else, add to current group
        if logLevel > 9: log.info('Add file %s'%fileinfo);
        groups[-1].append(h);

    # Clean from void groups
    groups = [g for g in groups if g != []];
    
    # For the BACKGROUND, remove the first file if there is more than 3 files
    # because it is often contaminated with light (slow shutter)
    # This needs to be more robust for all kinds of shutters. will be done later.
    #if mtype == 'BACKGROUND':
    #    for i in range(np.shape(groups)[0]):
    #        if np.shape(groups[i])[0] > 3:
    #            groups[i] = groups[i][1:];
    #            if logLevel > 4: log.info ('Ignore the first BACKGROUND files (more than 3)');
    
    return groups;


def assoc (h, allh, tag, keys=[], which='closest', required=0, quality=None):
    '''
    Search for headers with tag and matching criteria
    '''

    # Keep only the requested tag
    atag = [a for a in allh if a['FILETYPE']==tag]
    
    # Keep only the requested criteria
    out = [];
    for a in atag:
        tmp = True;
        for k in keys:
            tmp *= (h.get(k,None) == a.get(k,None));
        if tmp:
            out.append(a);

    # Keep only the requested quality
    l1 = len (out);
    if quality is not None:
        out = [o for o in out if o.get (HMQ+'QUALITY', 0.0) > quality];

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
            quality = np.array([o[HMQ+'QUALITY'] for o in out]);
            out = [out[np.argmax (quality)]];
        else:
            raise NotImplementedError('Not supported yet');
            
    # Check required
    if len (out) < required:
        log.warning ('Cannot find %i %s (%i rejected for quality)'%(required,tag,l1-len(out)))
    elif required > 0:
        log.info ('Find %i %s (%s ...)'%(len(out),tag,out[0]['ORIGNAME']));
        
    return out

def assoc_flat (h, allh):
    '''
    Return the best FLAT for a given file. Note that the flat header is return
    as a list of one to match the output of 'assoc' function.
    '''
    
    # Associate best FLAT based in gain
    flats = [a for a in allh if a['FILETYPE']=='FLAT_MAP'];

    # Check
    if len (flats) < 1:
        log.warning ('Cannot find FLAT');
        return [];

    # Get closest gain
    m = np.argmin ([np.abs (h['GAIN'] - f['GAIN']) for f in flats]);
    flat = flats[m];

    # Return
    log.info ('Find 1 FLAT (%s)'%os.path.basename(flat['ORIGNAME']));
    return [flat];

def clean_option (opt):
    '''
    Check options
    '''
    if opt == 'FALSE': return False;
    if opt == 'TRUE':  return True;

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
    ''' Replace nan by value'''
    rep = 0.0 if not rep else rep[0];
    return val if np.isfinite (val) else rep;

def parse_argopt_catalog (input):
    '''
    Parse the syntax 'NAME1,d1,e1,NAME2,d2,e2,...'
    and return an astropy Table with column NAME,
    ISCAL, MODEL_NAME, PARAM1 and PARAM2.
    '''
    if input == 'name1,diam,err,name2,diam,err':
        raise (ValueError('No calibrators specified'));

    # Catalog is a list
    if input[-5:] == '.list':
        log.info ('Calibrators given as list');
        catalog = ascii.read (input);
        return catalog;

    # Check it is a multiple of 3
    values = input.split(',');
    if float(len (values) / 3).is_integer() is False:
        raise (ValueError('Wrong syntax for calibrators'));
    
    # Parse each star
    names = np.array (values[0::3]);
    diam  = np.array (values[1::3]).astype(float);
    ediam = np.array (values[2::3]).astype(float);

    # Create catalog
    catalog = Table ();
    catalog['NAME'] = names;
    catalog['ISCAL'] = 'CAL';
    catalog['MODEL_NAME'] = 'UD_H';
    catalog['PARAM1'] =  diam;
    catalog['PARAM2'] = ediam;
        
    return catalog;

def update_diam_from_jmmc (catalog):
    '''
    For all stars with diam=0 and err=0 in the catalog, we try
    to get the information from the JMMC SearchCal.

    FIXME: this is not working anymore, need to deal with the new
    format for catalog based on astropy Table.
    '''
    
    # Init
    searchCal = 'http://apps.jmmc.fr/~sclws/getstar/sclwsGetStarProxy.php';
    voTableToTsv = os.path.dirname (log.__file__) + '/sclguiVOTableToTSV.xsl';

    # Loop on stars in catalog, query for the one
    # with err = 0 and diam = 0
    for c in catalog:
        if c[1] == 0 and c[2] == 0:

            try:
                # Call online SearchCal
                log.info ('Query JMMC SearchCal for star '+c[0]);
                os.system ('wget '+searchCal+'?star='+c[0]+' -O mircx_searchcal.vot -o mircx_searchcal.log');

                # Not found
                if 'has not been found' in open('mircx_searchcal.vot').read():
                    log.warning (c[0]+' has not been found');
                    continue;

                # Convert and parse
                os.system ('xsltproc '+voTableToTsv+' mircx_searchcal.vot > mircx_searchcal.tsv');
                answer = [l for l in csv.reader(open('mircx_searchcal.tsv'),delimiter='\t') if l[0][0] != '#'];
                c[1] = float (answer[1][answer[0].index('UD_H')]);
                c[2]  = float (answer[1][answer[0].index('e_LDD')]);
                
                log.info ('%s found %.4f +- %.4f mas'%(c[0],c[1],c[2]));
                
            except:
                log.error ('Cannot reach JMMC SearchCal or parse answer');
                
                
def get_sci_cal (hdrs, catalog):
    '''
    Spread the headers from SCI and CAL according to the 
    entries defined in catalog. Catalog should be an astropy
    Table with the columns NAME, ISCAL, PARAM1 and PARAM2.
    '''

    # Check format of catalog
    try:
        t = catalog['NAME'];
        t = catalog['ISCAL'];
        t = catalog['PARAM1'];
        t = catalog['PARAM2'];
    except:
        log.error ('Calibrators not specified correclty');
        raise (ValueError);

    # Check if enought
    if len (catalog) == 0:
        log.error ('No valid calibrators');
        raise (ValueError);

    # Get values
    name,iscal = catalog['NAME'], catalog['ISCAL'];

    # Loop on input headers
    scis, cals = [], [];
    for h in hdrs:
        if h['FILETYPE'] != 'OIFITS':
            continue;

        # Find where in catalog
        idx = np.where (name == h['OBJECT'])[0];
        
        if len(idx) > 0 and iscal[idx[0]] == 'CAL':
            idx = idx[0];
            log.info ('%s (%s) -> OIFITS_CAL (%s, %f,%f)'%(h['ORIGNAME'],h['OBJECT'], \
                      catalog[idx]['MODEL_NAME'],catalog[idx]['PARAM1'],catalog[idx]['PARAM2']));
            h['FILETYPE'] += '_CAL';
            h[HMP+'CALIB MODEL_NAME'] = (catalog[idx]['MODEL_NAME']);
            h[HMP+'CALIB PARAM1'] = (catalog[idx]['PARAM1']);
            h[HMP+'CALIB PARAM2'] = (catalog[idx]['PARAM2']);
            cals.append (h);
        else:
            log.info ('%s (%s) -> OIFITS_SCI'%(h['ORIGNAME'],h['OBJECT']));
            h['FILETYPE'] += '_SCI';
            scis.append (h);

    return scis,cals;

def p2h (phdrs): # convert panda frame to our standard header list of dictionaries
    hdr0=[]
    allh=phdrs.transpose().to_dict()
    keylist=list(allh.keys())
    for key in keylist:
        temp=allh[key]
        hdr0.append(temp)
    return hdr0;