import os, sys, re
import pandas as pd
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
from astropy import units as u
from requests.exceptions import ConnectionError

from . import headers, log, files

def targList(d,rawBase,redBase,opt):
    """
    Write target list for the specified observing date and
    save in the reduction directory for that night.
        - d is a date string: YYYYMmmDD e.g. 2018Oct28;
        - rawBase is the path to base of the raw data 
        directory tree (the final character should not be 
        '/');
        - redBase is the path to base of the reduced data
        directory tree (the final character should not be
        '/');
        - opt is a python list describing the input 
        options to be parsed to mircx_reduce.py.
    """
    dotargList = 'no'
    for i in range(0, len(opt)):
        # create file suffix from parsed options:
        suf = '_ncoh'+opt[i][0]+'ncs'+opt[i][1]+'nbs'+opt[i][2]+'snr'+opt[i][3]
        # Check to see whether summary files already exist (do nothing if true):
        if os.path.isfile(redBase+'/'+d+suf+'/'+d+'_targets.list') != True:
            dotargList = 'yes'
    if dotargList == 'yes':
        # Load all the headers from observing date:
        log.info('Read headers from raw data directory')
        hdrs = headers.loaddir(rawBase+'/'+d[0:7]+'/'+d)
        # create python list of object names:
        log.info('Retrieve object names from headers')
        objs = []
        for h in hdrs:
            try:
                if h['OBJECT'] != '' and h['OBJECT'] != 'NOSTAR':
                    objs.append(h['OBJECT'])
            except KeyError:
                log.warning('Not all headers contain OBJECT key word.')
                log.info('Continuing.')
        
        log.info('Cleanup memory')
        del hdrs
        
        objs = list(set(objs))
        for i in range(0, len(opt)):
            suf = '_ncoh'+opt[i][0]+'ncs'+opt[i][1]+'nbs'+opt[i][2]+'snr'+opt[i][3]
            # Check to see whether summary file already exists (do nothing if true):
            if os.path.isfile(redBase+'/'+d+suf+'/'+d+'_targets.list') != True:
                # create directory for d+suf/, if required
                files.ensure_dir(redBase+'/'+d+suf);
                # write target list summary file:
                log.info('Write '+redBase+'/'+d+suf+'/'+d+'_targets.list')
                with open(redBase+'/'+d+suf+'/'+d+'_targets.list', 'w') as output:
                    for obj in objs:
                        if type(obj) != str:
                            objs.remove(obj)
                        output.write(obj+'\n')
                if len(objs) == 0:
                    log.error('No target names retrieved from headers.')
                    log.info('Exiting.')
                    sys.exit()
                else:
                    log.info('File written successfully')
    else:
        log.info('Target lists already exist.')
        log.info('Reading target names from '+redBase+'/'+d+suf+'/'+d+'_targets.list')
        objs = []
        with open(redBase+'/'+d+suf+'/'+d+'_targets.list', 'r') as input:
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
        log.warning(mirrs[m]+' VizieR server down')
        while connected == False:
            try:
                Vizier.VIZIER_SERVER=mirrs[m+1]
            except IndexError:
                log.error('Failed to connect to VizieR mirrors')
                log.error('Check internet connection and retry')
                sys.exit()
            try:
                result = Vizier.query_object(targ, catalog=['II/346'])
                connected = True
                log.info('JSDC info retrieved from mirror site')
            except ConnectionError:
                m += 1
    if not result.keys():
        # If nothing is returned from JSDC, assume the target is SCI:
        log.info('Nothing returned from JSDC for '+targ)
        log.info(targ+' will be treated as SCI')
        return 'sci'
    else:
        ra_in = result["II/346/jsdc_v2"]["RAJ2000"][0]
        dec_in = result["II/346/jsdc_v2"]["DEJ2000"][0]
        coords = SkyCoord(ra_in+' '+dec_in, unit=(u.hourangle, u.deg))
        ra = str(coords.ra.deg)
        dec = str(coords.dec.deg)
        hmag = str(result["II/346/jsdc_v2"]["Hmag"][0])
        vmag = str(result["II/346/jsdc_v2"]["Vmag"][0])
        flag = result["II/346/jsdc_v2"]["CalFlag"][0]
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
        ud_H = '{0:.6f}'.format(float(result["II/346/jsdc_v2"]["UDDH"][0]))
        eud_H = '{0:.6f}'.format(float(result["II/346/jsdc_v2"]["e_LDD"][0]))
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
            log.info('Alternative IDs for '+targ+' retrieved from SIMBAD.')
            connected = True
        except ConnectionError:
            connected = False
            if m == 0:
                log.warning('Main SIMBAD server down')
            else:
                log.warning(mirrs[m]+' SIMBAD server down')
            while connected == False:
                try:
                    Simbad.SIMBAD_SERVER = mirr[m+1]
                except IndexError:
                    log.error('Failed to connect to SIMBAD mirrors')
                    log.error('Check internet connection and try again')
                    sys.exit()
                try:
                    alt_ids = Simbad.query_objectids(targ)
                    connected = True
                    log.info('Alternative IDs for '+targ+' retrieved from SIMBAD mirror:')
                    log.info(mirrs[m])
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
            log.warning('Target '+targ+' not found in local database')
            log.info('Querying JSDC catalog at VizieR...')
            calsci = queryJSDC(targ,m)
            if len(calsci.split(',')) == 1:
                outline = targ.replace('_', ' ')+', , , , , SCI, , , \n'
                scical.append('NEW:SCI')
            else:
                outline = targ.replace('_',' ')+','+calsci+'\n'
                scical.append('NEW:CAL')
                calInf = calInf+targ.replace(' ','_')+','+','.join(calsci.split(',')[6:8])+','
            outfile = os.environ['MIRCX_PIPELINE']+'mircx_pipeline/mircx_newTargs.list'
            if not os.path.exists(outfile):
                with open(outfile, 'w') as output:
                    output.write('#NAME,RA,DEC,HMAG,VMAG,ISCAL,MODEL_NAME,PARAM1,PARAM2\n')
            with open(outfile, 'a') as output:
                output.write(outline)
        # If one match is found, read in the information from the local database
        elif id_count == 1:
            if targNew == targ:
                log.info('Target '+targ+' located in '+db)
            else:
                log.info('Target '+targ+' located in '+db+' as '+targNew)
            if 'SCI' in m_scical[m_targs.index(targNew)]:
                log.info(targ+' recognised as SCI')
                scical.append('SCI')
            else:
                log.info(targ+' recognised as CAL')
                if m_modTyp[m_targs.index(targNew)] == 'UD_H':
                    ud_H = float(pd.Series.tolist(localDB['PARAM1'])[m_targs.index(targNew)])
                    eud_H = float(pd.Series.tolist(localDB['PARAM2'])[m_targs.index(targNew)])
                    calInf = calInf+targ.replace(' ','_')+','+'{0:.6f}'.format(ud_H)+','+'{0:.6f}'.format(eud_H)+','
                    scical.append('CAL')
                else:
                    log.error('Model type '+m_modTyp[m_targs.index(targNew)]+' not supported')
                    log.info('This CAL will not be used in the calibration')
                    scical.append('(CAL)')
        # If multiple entries are found for the same target, raise an error:
        elif id_count > 1:
            log.error('Multiple entries found for '+targ+' in '+db)
            log.error('Please rectify this before continuing.')
            sys.exit()
    return calInf.replace(' ', ''), scical

