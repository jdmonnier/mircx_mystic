import os, sys, re
import pandas as pd

from . import headers, files, mircx_mystic_log

try:
    from astroquery.simbad import Simbad
except ImportError:
    mircx_mystic_log.error('astroquery.simbad not found!')
    mircx_mystic_log.info('Assigning sci and cal types to targets requires access to SIMBAD')
    mircx_mystic_log.info('Try "sudo pip install astroquery"')
    raise ImportError
    sys.exit()

from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
from astropy import units as u
from requests.exceptions import ConnectionError


def targList(d,rawBase,redDir):
    """
    Write target list for the specified observing date and
    save in the reduction directory for that night.
        - d is a date string: YYYYMmmDD e.g. 2018Oct28;
        - rawBase is the path to base of the raw data 
        directory tree (the final character should not be 
        '/');
        - redDir is the path to the reduced data
        directory (the final character should not be
        '/');
    """
    dotargList = 'no'
    # Check to see whether summary files already exist (do nothing if true):
    if os.path.isfile(redDir+'/'+d+'_targets.list') != True:
        dotargList = 'yes'
    if dotargList == 'yes':
        # Load all the headers from observing date:
        mircx_mystic_log.info('Read headers from raw data directory')
        hdrs = headers.loaddir(rawBase+'/'+d)
        # create python list of object names:
        mircx_mystic_log.info('Retrieve object names from headers')
        objs = []
        for h in hdrs:
            try:
                if h['OBJECT'] != '' and h['OBJECT'] != 'NOSTAR' and h['OBJECT'] != 'STS':
                    objs.append(h['OBJECT'])
            except KeyError:
                mircx_mystic_log.warning('Not all headers contain OBJECT key word.')
                mircx_mystic_log.info('Continuing.')
        
        mircx_mystic_log.info('Cleanup memory')
        del hdrs
        
        objs = list(set(objs))
        # Check to see whether summary file already exists (do nothing if true):
        if os.path.isfile(redDir+'/'+d+'_targets.list') != True:
            files.ensure_dir(redDir);
            # write target list summary file:
            mircx_mystic_log.info('Write '+redDir+'/'+d+'_targets.list')
            with open(redDir+'/'+d+'_targets.list', 'w') as output:
                for obj in objs:
                    if type(obj) != str:
                        objs.remove(obj)
                    output.write(obj+'\n')
            if len(objs) == 0:
                mircx_mystic_log.error('No target names retrieved from headers.')
                mircx_mystic_log.info('Exiting.')
                sys.exit()
            else:
                mircx_mystic_log.info('File written successfully')
    else:
        mircx_mystic_log.info('Target lists already exist.')
        mircx_mystic_log.info('Reading target names from '+redDir+'/'+d+'_targets.list')
        objs = []
        with open(redDir+'/'+d+'_targets.list', 'r') as input:
            for line in input:
                objs.append(line.strip().replace('_', ' '))
    return objs

def queryJSDC(targ,m):
    connected = False
    mirrs = ['vizier.u-strasbg.fr','vizier.nao.ac.jp','vizier.hia.nrc.ca', 
             'vizier.ast.cam.ac.uk','vizier.cfa.harvard.edu','vizier.china-vo.org', 
             'www.ukirt.jach.hawaii.edu','vizier.iucaa.ernet.in']
    Vizier.VIZIER_SERVER = mirrs[m]
    try:
        result = Vizier.query_object(targ, catalog=['II/346'])
        connected = True
    except ConnectionError:
        connected = False
        mircx_mystic_log.warning(mirrs[m]+' VizieR server down')
        while connected == False:
            try:
                Vizier.VIZIER_SERVER=mirrs[m+1]
            except IndexError:
                mircx_mystic_log.error('Failed to connect to VizieR mirrors')
                mircx_mystic_log.error('Check internet connection and retry')
                sys.exit()
            try:
                result = Vizier.query_object(targ, catalog=['II/346'])
                connected = True
                mircx_mystic_log.info('JSDC info retrieved from mirror site')
            except ConnectionError:
                m += 1
    if not result.keys():
        # If nothing is returned from JSDC, assume the target is SCI:
        mircx_mystic_log.info('Nothing returned from JSDC for '+targ)
        mircx_mystic_log.info(targ+' will be treated as SCI')
        return 'sci'
    
    ind = -999
    alt_ids = Simbad.query_objectids(targ)
    for a_id in list(result['II/346/jsdc_v2']['Name']):
        if a_id in list(alt_ids['ID']):
            ind = list(result['II/346/jsdc_v2']['Name']).index(a_id)
        elif a_id in list([a.replace(' ', '') for a in alt_ids['ID']]):
            ind = list(result['II/346/jsdc_v2']['Name']).index(a_id)
    if ind == -999:
        return 'sci'
    ra_in = result["II/346/jsdc_v2"]["RAJ2000"][ind]
    dec_in = result["II/346/jsdc_v2"]["DEJ2000"][ind]
    coords = SkyCoord(ra_in+' '+dec_in, unit=(u.hourangle, u.deg))
    ra = str(coords.ra.deg)
    dec = str(coords.dec.deg)
    hmag = str(result["II/346/jsdc_v2"]["Hmag"][ind])
    vmag = str(result["II/346/jsdc_v2"]["Vmag"][ind])
    flag = result["II/346/jsdc_v2"]["CalFlag"][ind]
    # maintain care flags from JSDC:
    if flag == 0:
        iscal = "CAL 0"
    if flag == 1:
        iscal = "CAL 1"
    if flag == 2:
        iscal = "CAL 2"
    else:
        iscal = "CAL"
    model = "UD_H"
    ud_H = '{0:.6f}'.format(float(result["II/346/jsdc_v2"]["UDDH"][ind]))
    eud_H = '{0:.6f}'.format(float(result["II/346/jsdc_v2"]["e_LDD"][ind]))
    return ''.join(str([ra, dec, hmag, vmag, iscal, model, ud_H, eud_H])[1:-1]).replace("'", "")

def queryLocal(targs,db):
    """
    Query local database to identify science and calibrator targets.
    Calls queryJSDC if target match not found locally and writes new
    target file in this case.
        - targs is a python list of targets from MIRCX
        fits headers;
        - db is either the default distributed MIRCX
        targets database or it is user defined
    
    Produces:
        - 'calInf' which is the string containing calibrator names,
        uniform disk diameters and their errors. This will be 
        parsed to mircx_calibrate.py.
        - 'scical' which is a python list containing 'SCI', 'CAL',
        '(CAL)', 'NEW:SCI', or 'NEW:CAL' for the targets.
    """
    mirrs = ['vizier.u-strasbg.fr','vizier.nao.ac.jp','vizier.hia.nrc.ca', 
             'vizier.ast.cam.ac.uk','vizier.cfa.harvard.edu','vizier.china-vo.org', 
             'www.ukirt.jach.hawaii.edu','vizier.iucaa.ernet.in']
    localDB  = pd.read_csv(db)
    m_targs  = pd.Series.tolist(localDB['#NAME'])
    m_scical = pd.Series.tolist(localDB['ISCAL'])
    m_modTyp = pd.Series.tolist(localDB['MODEL_NAME'])
    m = 0
    calInf, scical = '', []
    for targ in targs:
        connected = False
        # First, retrieve alternative IDs for target from SIMBAD:
        try:
            alt_ids = Simbad.query_objectids(targ)
            mircx_mystic_log.info('Alternative IDs for '+targ+' retrieved from SIMBAD.')
            connected = True
        except ConnectionError:
            connected = False
            if m == 0:
                mircx_mystic_log.warning('Main SIMBAD server down')
            else:
                mircx_mystic_log.warning(mirrs[m]+' SIMBAD server down')
            while connected == False:
                try:
                    Simbad.SIMBAD_SERVER = mirrs[m+1]
                except IndexError:
                    mircx_mystic_log.error('Failed to connect to SIMBAD mirrors')
                    mircx_mystic_log.error('Check internet connection and try again')
                    sys.exit()
                try:
                    alt_ids = Simbad.query_objectids(targ)
                    connected = True
                    mircx_mystic_log.info('Alternative IDs for '+targ+' retrieved from SIMBAD mirror:')
                    mircx_mystic_log.info(mirrs[m])
                except ConnectionError:
                    m += 1
        # Then query all alternative IDs for target against MIRCX database
        id_count = 0
        targNew = None
        for id in alt_ids:
            id_count += m_targs.count(re.sub(' +',' ',id[0]))
            if id_count == 1 and targNew == None:
                # Remember the name for the target which matches with the database
                # (this may be the same as the original target name).
                targNew = re.sub(' +',' ',id[0])
        # If nothing is found in the local database, query JSDC:
        if id_count == 0:
            mircx_mystic_log.warning('Target '+targ+' not found in local database')
            mircx_mystic_log.info('Querying JSDC catalog at VizieR...')
            calsci = queryJSDC(targ,m)
            if len(calsci.split(',')) == 1:
                outline = targ.replace('_', ' ')+', , , , , SCI, , , \n'
                scical.append('NEW:SCI')
            else:
                outline = targ.replace('_',' ')+','+calsci+'\n'
                scical.append('NEW:CAL')
                calInf = calInf+targ.replace(' ','_')+','+','.join(calsci.split(',')[6:8])+','
            if os.environ['MIRCX_PIPELINE'][-1] != '/':
                outfile = os.environ['MIRCX_PIPELINE']+'/mircx_pipeline/mircx_newTargs.list'
            else:
                outfile = os.environ['MIRCX_PIPELINE']+'mircx_pipeline/mircx_newTargs.list'
            if not os.path.exists(outfile):
                with open(outfile, 'w') as output:
                    output.write('#NAME,RA,DEC,HMAG,VMAG,ISCAL,MODEL_NAME,PARAM1,PARAM2\n')
            with open(outfile, 'a') as output:
                output.write(outline)
        # If one match is found, read in the information from the local database
        elif id_count == 1:
            if targNew == targ:
                mircx_mystic_log.info('Target '+targ+' located in '+db)
            else:
                mircx_mystic_log.info('Target '+targ+' located in '+db+' as '+targNew)
            if 'SCI' in m_scical[m_targs.index(targNew)]:
                mircx_mystic_log.info(targ+' recognised as SCI')
                scical.append('SCI')
            else:
                mircx_mystic_log.info(targ+' recognised as CAL')
                if 'UD_H' in m_modTyp[m_targs.index(targNew)]:
                    ud_H = float(pd.Series.tolist(localDB['PARAM1'])[m_targs.index(targNew)])
                    eud_H = float(pd.Series.tolist(localDB['PARAM2'])[m_targs.index(targNew)])
                    calInf = calInf+targ.replace(' ','_')+','+'{0:.6f}'.format(ud_H)+','+'{0:.6f}'.format(eud_H)+','
                    scical.append('CAL')
                else:
                    mircx_mystic_log.error('Model type '+m_modTyp[m_targs.index(targNew)]+' not supported')
                    mircx_mystic_log.info('This CAL will not be used in the calibration')
                    scical.append('(CAL)')
        # If multiple entries are found for the same target, raise an error:
        elif id_count > 1:
            mircx_mystic_log.error('Multiple entries found for '+targ+' in '+db)
            mircx_mystic_log.error('Please rectify this before continuing.')
            sys.exit()
    return calInf.replace(' ', ''), scical

