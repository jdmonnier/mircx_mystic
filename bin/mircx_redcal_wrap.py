#! /usr/bin/env python                                                          
# -*- coding: iso-8859-15 -*-                                                   

import mircx_pipeline as mrx;
import argparse, glob, os, subprocess, sys, re, socket;
import numpy as np;
from astropy.io import fits as pyfits;
import matplotlib.pyplot as plt;

from mircx_pipeline import log, setup, plot, files, signal, headers;
from mircx_pipeline.headers import HM, HMQ, HMP;
# for emailing:
import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
from email.MIMEBase import MIMEBase
from email import encoders

#####################################################
# Describe the script
description = \
"""
description:
 Wrapper for mircx_reduce.py, mircx_calibrate.py
 and mircx_report.py. 

"""
epilog = \
"""
examples:
 mircx_redcal_wrapper.py --dates=2018Oct29,2018Oct28 
  --ncoherent= --ncs= --nbs=

"""
parser = argparse.ArgumentParser (description=description, epilog=epilog,
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 add_help=True);
parser.add_argument ("--raw-dir",dest="raw_dir",default='/.', #os.environ['MIRCX_RAW'],
                     type=str,help="base for the raw data paths [%(default)s]");
parser.add_argument ("--red-dir",dest="red_dir",default='/.', #os.environ['MIRCX_REDUCED'],
                     type=str,help="base for the reduction directory paths [%(default)s]");
parser.add_argument ("--dates", dest="dates", type=str,
                     help="dates of observations to be reduced [%(default)s]");
parser.add_argument ("--ncoherent", dest="ncoherent", type=str,default=5, 
                     help="list of number of frames for coherent integration [%(default)s]");
parser.add_argument ("--ncs", dest="ncs", type=str,default=1, 
                     help="list of number of frame-offset for cross-spectrum [%(default)s]");
parser.add_argument ("--nbs", dest="nbs", type=str,default=4, 
                     help="list of number of frame-offset for bi-spectrum [%(default)s]");
parser.add_argument ("--snr-threshold", dest="snr_threshold", type=str,default='2.0', 
                     help="list of SNR threshold for fringe selection [%(default)s]");
parser.add_argument ("--email", dest="email", type=str, 
                     default='mircx-reports@exeter.ac.uk', 
                     help='email address to send summary report file to [%(default)s]');

#####################################################
# Initialisation

# 1. Remove warning for invalid
np.seterr (divide='ignore',invalid='ignore');

# 2. Parse arguments
argopt = parser.parse_args ();

# 3. Verbose
elog = log.trace ('mircx_redcal_wrapper');

# 4. Check on the input criteria:
nco = str(argopt.ncoherent).split(',')
ncs = str(argopt.ncs).split(',')
nbs = str(argopt.nbs).split(',')
snr = str(argopt.snr_threshold).split(',')
if len(nco) == len(ncs) == len(nbs) == len(snr):
    log.info('Length of reduction options checked: ok')
else:
    log.error('Error in setup: length of ncoherent, ncs, nbs, snr_threshold not equal!')
    sys.exit()

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)
    
    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)
    
    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

#####################################################
# Set up the automatic email:
fromaddr = 'mircx.mystic@gmail.com'
toaddr   = argopt.email

msg = MIMEMultipart()

msg['From'] = fromaddr
msg['To']   = toaddr
# msg['Subject'] is defined below

def send_email(msg): 
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    try:
        server.login("mircx.mystic@gmail.com", os.environ['MAILLOGIN'])
    except KeyError:
        log.error('password for '+fromaddr+' not found')
        log.error('Please ensure environment variable MAILLOGIN is set')
        if fromaddr == 'mircx.mystic@gmail.com':
            log.error('Contact Narsi Anugu for password for '+fromaddr+' if necessary')
        server.quit()
        sys.exit()
    server.sendmail(fromaddr, toaddr, msg.as_string())
    server.quit()

#####################################################
# Query CDS to build a catalog of object information
# (This is protected since it may fail)

# 1. Load astroquery
try:
    from astroquery.vizier import Vizier;
    log.info ('Load astroquery.vizier');
    from astroquery.simbad import Simbad;
    log.info ('Load astroquery.simbad');
except:
    log.warning ('Cannot load astroquery, try:');
    log.warning ('sudo conda install -c astropy astroquery');

#####################################################
# Create a list of targets observed on each date parsed to --dates 
for date in argopt.dates.split(','):
    # Load all the headers from "date"
    if argopt.raw_dir[-1] =='/':
        hdrs = mrx.headers.loaddir (argopt.raw_dir+date[0:7]+'/'+date);
    else:
        hdrs = mrx.headers.loaddir (argopt.raw_dir+'/'+date[0:7]+'/'+date);
    # create object list as python list
    objlist = list(set([h['OBJECT'] for h in hdrs]));
    for i in range(0, len(nco)):
        # create file suffix from parsed options
        suffix = '_ncoh'+str(nco[i])+'ncs'+str(ncs[i])+'nbs'+str(nbs[i])+'snr'+str(snr[i]).replace('.','p');
        # create directory for date+suffix, if required
        mrx.files.ensure_dir (argopt.red_dir+'/'+date+suffix);
        # Check if summary file already exists:
        targlog = argopt.red_dir+'/'+date+suffix+'/target.log';
        if os.path.isfile(targlog) == True:
            log.info(date+' target list exists');
        else:
            with open(targlog, 'w') as output:
                for obj in objlist:
                    output.write(obj+'\n');
            log.info('Save '+date+' target list to '+targlog);
    hdrs = None;
    objlist = None;
    suffix = None;

#####################################################
# Using existing target history, identify sci and cal targets on "date" and execute
# reduction and calibration scripts.
with open(os.environ['MIRCX_PIPELINE']+'/mircx_pipeline/mircx_targets.list') as input:
    m_targs = [line.split(',')[0] for line in input]
with open(os.environ['MIRCX_PIPELINE']+'/mircx_pipeline/mircx_targets.list') as input:
    m_scical = [line.split(',')[5] for line in input]

rfiles = [['report_decoher.png', 'report_snr.png'], ['report_tf2.png', 'report_vis2.png'],
          ['report_trans.png']]
for date in argopt.dates.split(','):
    callist, scical = '', []
    hdrs = mrx.headers.loaddir(argopt.raw_dir+'/'+date[0:7]+'/'+date)
    targlist = list(set([h['OBJECT'] for h in hdrs]))
    # Define SIMBAD and VizieR mirror site addresses
    mirrs = ['vizier.u-strasbg.fr','vizier.nao.ac.jp','vizier.hia.nrc.ca', 
             'vizier.ast.cam.ac.uk','vizier.cfa.harvard.edu','vizier.china-vo.org', 
             'www.ukirt.jach.hawaii.edu','vizier.iucaa.ernet.in']
    
    for targ in targlist:
        log.info('Query SIMBAD for alternative IDs for target '+targ)
        try:
            alt_ids = Simbad.query_objectids(targ)
            log.info('Alternative IDs retrieved')
        except ConnectionError:
            connected == False
            log.warning('Main SIMBAD server down')
            m = 0
            while connected == False:
                try:
                    Simbad.SIMBAD_SERVER = mirr[m]
                except IndexError:
                    log.error('Failed to connect to SIMBAD mirror sites')
                    log.error('Check internet connection and retry')
                    sys.exit()
                try:
                    alt_ids = Simbad.query_objectids(targ)
                    connected == True
                    log.info('Alternative IDs retreived from mirror site')
                except ConnectionError:
                    m += 1
        
        # Query target names against mircx_targets.list
        id_count = 0
        targ_n = None
        for id in alt_ids:
            # As a check, count instances of target and its alternative IDs in the MIRC-X target list
            id_count += m_targs.count(re.sub(' +',' ',id[0]))
            if id_count == 1 and targ_n == None:
                targ_n = re.sub(' +',' ',id[0])
        if id_count == 0:
            log.warning('Target '+targ+' not found in mircx_target.list')
            log.info('Querying JSDC catalog at VizieR')
            # If target is not in MIRC-X target list, treat it as a new calibrator and query the JSDC catalog
            try:
                result = Vizier.query_object(obj, catalog=["II/346"])
            except ConnectionError:
                connected = False
                log.warning('Main VizieR server down')
                m = 0
                while connected == False:
                    try:
                        Vizier.VIZIER_SERVER = mirr[m]
                    except IndexError:
                        log.error('Failed to connect to VizieR mirror sites')
                        log.error('Check internet connection and retry')
                        sys.exit()
                    try:
                        result = Vizier.query_object(obj, catalog=["II/346"])
                        connected = True
                        log.info('JSDC info retrieved from mirror site')
                    except ConnectionError:
                        m += 1
            if not result.keys():
                # If nothing is returned from the JSDC, assume new target is a SCI
                log.info('Nothing returned from JSDC for '+targ)
                log.info('Marking '+targ+' as new SCI target')
                scical.append('NEW:SCI')
                # Check to see whether the new target file exists and create or append as needed
                if os.path.exists(os.environ['MIRCX_PIPELINE']+'mircx_pipeline/mircx_newTargs.list'):
                    with open(os.environ['MIRCX_PIPELINE']+'mircx_pipeline/mircx_newTargs.list', 'a') as output:
                        output.write(targ+', , , , , SCI, , , \n')
                else:
                    with open(os.environ['MIRCX_PIPELINE']+'mircx_pipeline/mircx_newTargs.list', 'w') as output:
                        output.write('#NAME,RA,DEC,HMAG,VMAG,ISCAL,MODEL_NAME,PARAM1,PARAM2,PARAM3,PARAM4\n')
                        output.write(targ+', , , , , SCI, , , \n')
            else:
                # If details are returned from JSDC, save these details to MIRC-X "new target" file
                ra = result["II/346/jsdc_v2"]["_RAJ2000"][0]
                dec = result["II/346/jsdc_v2"]["_DEJ2000"][0]
                hmag = result["II/346/jsdc_v2"]["Hmag"][0]
                vmag = result["II/346/jsdc_v2"]["Vmag"][0]
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
                ud_H = result["II/346/jsdc_v2"]["UDDH"][0]
                eud_H = result["II/346/jsdc_v2"]["e_LDD"][0]
                # Check to see whether the new target file exists and create or append as needed
                if os.path.exists(os.environ['MIRCX_PIPELINE']+'mircx_pipeline/mircx_newTargs.list'):
                    with open(os.environ['MIRCX_PIPELINE']+'mircx_pipeline/mircx_newTargs.list', 'a') as output:
                        output.write(targ+','+str(ra)+','+str(dec)+','+str(hmag)+','+str(vmag)+','+str(iscal)+','+str(model)+','+str(ud_H)+','+str(eud_H)+'\n')
                else:
                    with open(os.environ['MIRCX_PIPELINE']+'mircx_pipeline/mircx_newTargs.list', 'w') as output:
                        output.write('#NAME,RA,DEC,HMAG,VMAG,ISCAL,MODEL_NAME,PARAM1,PARAM2,PARAM3,PARAM4\n')
                        output.write(targ+','+str(ra)+','+str(dec)+','+str(hmag)+','+str(vmag)+','+str(iscal)+','+str(model)+','+str(ud_H)+','+str(eud_H)+'\n')
                # and mark this target as a new cal:
                callist = callist + targ.replace(' ','_')+','+ud_H+','+eud_H+','
                scical.append('NEW:CAL')
        
        elif id_count == 1:
            if targ_n == targ:
                log.info('Target '+targ+' located in mircx_target.list')
            else:
                log.info('Target '+targ+' located in mircx_target.list as '+targ_n)
            #Retrieve info for target from mircx_targets.list:
            print m_scical[m_targs.index(targ_n)]
            if 'SCI' in m_scical[m_targs.index(targ_n)]:
                scical.append('SCI')
            else:
                with open(os.environ['MIRCX_PIPELINE']+'mircx_pipeline/mircx_targets.list') as input:
                    for line in input:
                        if targ_n in line:
                            ud_H = line.split(',')[7]
                            eud_H = line.strip().split(',')[8]
                callist = callist + targ.replace(' ','_')+','+ud_H+','+eud_H+','
                scical.append('CAL')
        elif id_count > 1:
            log.error('Multiple entries found for '+targ+' in mircx_targets.list!')
            log.error('Please rectify this before continuing.')
            sys.exit()
    
    for i in range(0, len(nco)):
        # execute reduction
        suffix = '_ncoh'+str(nco[i])+'ncs'+str(ncs[i])+'nbs'+str(nbs[i])+'snr'+str(snr[i]).replace('.','p')
        optionals = '--ncoherent='+str(nco[i])+' --ncs='+str(ncs[i])+' --nbs='+str(nbs[i])+' --snr-threshold='+str(snr[i])
        if argopt.red_dir[-1] == '/' and argopt.raw_dir[-1] == '/':
            rawDir = argopt.raw_dir+date[0:7]+'/'+date
            redDir = argopt.red_dir+date+suffix
        elif argopt.red_dir[-1] == '/' and argopt.raw_dir[-1] != '/':
            rawDir = argopt.raw_dir+'/'+date[0:7]+'/'+date
            redDir = argopt.red_dir+date+suffix
        elif argopt.red_dir[-1] != '/' and argopt.raw_dir[-1] == '/':
            rawDir = argopt.raw_dir+date[0:7]+'/'+date
            redDir = argopt.red_dir+'/'+date+suffix
        elif argopt.red_dir[-1] != '/' and argopt.raw_dir[-1] != '/':
            rawDir = argopt.raw_dir+'/'+date[0:7]+'/'+date
            redDir = argopt.red_dir+'/'+date+suffix
        log.info('Execute mircx_reduce.py in directory '+redDir)
        with cd(redDir):
            subprocess.call("mircx_reduce.py "+optionals+" --raw-dir="+rawDir+" --oifits-dir="+redDir+'/oifits', shell=True)
        
        # Check whether reduction was successful:
        if os.path.isdir(redDir+'/oifits'):
            log.info('Execute mircx_report.py in directory '+redDir+'/oifits')
            # execute calibration and quality assessment report
            with cd(redDir+"/oifits"):
                subprocess.call("mircx_report.py", shell=True)
                log.info('Execute mircx_calibrate.py in directory '+redDir+'/oifits')
                subprocess.call("mircx_calibrate.py --calibrators="+callist[:-1], shell=True)
            
            calibfits = glob.glob(redDir+'/oifits/calibrated/*.fits')
            calibhdrs = mrx.headers.loaddir(redDir+'/oifits/calibrated') # headers of the calibrated files
            redhdrs = mrx.headers.loaddir(redDir+'/oifits') # headers of the reduced files
            calibtargs = list(set([h['OBJECT'] for h in calibhdrs]))
            #####################################################
            # Make uv coverage plot:
            for t in range(0, len(calibtargs)):
                usf, vsf = [], []
                for f in range(0, len(calibfits)):
                    with pyfits.open(calibfits[f]) as input:
                        if input[0].header['OBJECT'] == calibtargs[t]:
                            for u in range(0, len(input['OI_VIS2'].data['UCOORD'])):
                                for w in range(0, len(input['OI_WAVELENGTH'].data['EFF_WAVE'])):
                                    usf.append(0.0-(input['OI_VIS2'].data['UCOORD'][u]/input['OI_WAVELENGTH'].data['EFF_WAVE'][w]/1e6))
                                    vsf.append(input['OI_VIS2'].data['VCOORD'][u]/input['OI_WAVELENGTH'].data['EFF_WAVE'][w]/1e6)
                plt.plot(usf, vsf, ls='none', marker='o', color='k')
                plt.plot(0.0-np.array(usf), 0.0-np.array(vsf), ls='none', marker='o', color='k')
                ax = plt.gca()
                ax.set_ylabel('v [M$\lambda$]')
                ax.set_xlabel('u [M$\lambda$]')
                ax.set_ylim([-330./1.49, 330./1.49])
                ax.set_xticks([-200, -100, 0, 100, 200])
                ax.set_xticklabels(['200', '100', '0', '-100', '-200'])
                ax.set_xlim([-330./1.49, 330./1.49])
                plt.axes().set_aspect(1)
                plt.title(calibtargs[t])
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(redDir+'/oifits/calibrated/'+calibtargs[t]+'_uv_coverage.png')
                log.info('Write '+redDir+'/oifits/calibrated/'+calibtargs[t]+'_uv_coverage.png')
                reductionfailed = 'False'
        else:
            log.error('Reduction failed!')
            log.info('Consequently, cannot run mircx_report.py')
            log.info('and cannot run mircx_calibrate.py')
            reductionfailed = 'True'
        
        #####################################################
        # Produce summary file in each directory created containing the following:
        # 1. List of targets and a label of "SCI", "CAL", "NEW: SCI", and "NEW: CAL"
        # 2. uv coverage for the science stars
        # 3. Outputs of the mircx_report.py script 
        # 4. Calibrated visibilities: oifits/calibrated/*viscal_vis2.png
        # 5. Calibrated closure phases (not presently plotted)
        author = 'ncohrent='+str(nco[i])+'; ncs='+str(ncs[i])+'; nbs='+str(nbs[i])+'; snr\\_threshold='+str(snr[i])
        mfiles_v = glob.glob(redDir+'/oifits/calibrated/*viscal_vis2.png') # length zero if reduction failed
        mfiles_c = glob.glob(redDir+'/oifits/calibrated/*_t3phi.png')      # length zero if reduction failed
        uvplot = glob.glob(redDir+'/oifits/calibrated/*_uv_coverage.png')  # length zero if reduction failed
        rtsPlots = sorted(glob.glob(redDir+'/rts/*rts_psd.png'))           # may be length zero if reduction failed
        snrPlots = sorted(glob.glob(redDir+'/oifits/*oifits_snr.png'))     # length zero if reduction failed
        basePlots = sorted(glob.glob(redDir+'/oifits/*base_trend.png'))    # length zero if reduction failed
        with open(redDir+'/summary.tex', 'w') as outtex:
            # Set up latex file:
            outtex.write('\\documentclass[a4paper]{article}\n\n')
            outtex.write('\\usepackage{fullpage}\n\\usepackage{amsmath}\n')
            outtex.write('\\usepackage{hyperref}\n\\usepackage{graphicx}\n')
            outtex.write('\\usepackage{longtable}\n')
            outtex.write('\\usepackage[left=1.5cm,right=1.5cm,top=2cm,bottom=2cm]')
            outtex.write('{geometry}\n\n')
            # Write title and setup-specific sub-headings
            outtex.write('\\title{Summary report from mircx\\_redcal\\_wrap.py}\n')
            outtex.write('\\author{'+author+'}\n\\date{'+date+'}\n\n')
            outtex.write('\\begin{document}\n\n\\maketitle\n\n')
            if reductionfailed == 'False':
                outtex.write('\\subsubsection*{Reduced files located in : '+redDir+' on ')
                outtex.write(socket.gethostname()+'}\\n')
            else:
                outtex.write('\\subsubsection*{Reduction failed}\\n')
            # Output the PI(s)'s and observer(s)'s names & the program ID:
            outtex.write('\\subsubsection*{PI(s): ')
            outline = None
            for item in list(set([h['PI_NAME'] for h in hdrs])):
                if item != 'UNKNOWN':
                    try:
                        outline = outline+'; '+str(item)
                    except TypeError:
                        outline = str(item)
            
            outtex.write(outline+'}\n')
            outtex.write('\\subsubsection*{Observer(s): ')
            outline = None
            for item in list(set([h['OBSERVER'] for h in hdrs])):
                if item != 'Slimfringe':
                    try:
                        outlien = outline+'; '+str(item)
                    except TypeError:
                        outline = str(item)
            try:
                outtex.write(outline+'}\n')
            except TypeError:
                outtex.write('}\n')
            outtex.write('\\subsubsection*{Program ID(s): (info not yet retained in headers)}\n')
            # Print table containing target info (cal, sci or new)
            outtex.write('\\subsection*{Target summary}\n')
            outtex.write('\\begin{longtable}{p{.25\\textwidth} | p{.08\\textwidth} | ')
            outtex.write('p{.25\\textwidth}}\n    \\hline\n')
            outtex.write('    Target ID & used as & UD diam. for CALs (mas) \\\\ \n')
            outtex.write('    \\hline\n')
            log.info('callist = '+callist[:-1])
            for targ in targlist:
                log.info('targ = '+targ)
                log.info('scical = '+scical[targlist.index(targ)])
                try:
                    ud_H = callist.split(',')[callist.split(',').index(targ)+1]
                    eud_H = callist.split(',')[callist.split(',').index(targ)+2]
                    outtex.write('    '+targ.replace('_', ' ')+' & CAL')
                    outtex.write(' & $'+ud_H+'\\pm'+eud_H+'\\,$ \\\\ \n')
                except ValueError:
                    log.info(targ+' not in callist')
                    outtex.write('    '+targ.replace('_', ' ')+' & SCI')
                    outtex.write(' &  \\\\ \n')
            outtex.write('    \\hline\n\\end{longtable}\n')
            outtex.write('\n')
            # Print table containing data summary:
            outtex.write('\\subsection*{Reduced data summary}\n')
            outtex.write('\\begin{longtable}{p{.01\\textwidth} | p{.08\\textwidth} | ')
            outtex.write('p{.06\\textwidth} | p{.25\\textwidth} | p{.05\\textwidth} | ')
            outtex.write('p{.07\\textwidth} | p{.04\\textwidth} | p{.07\\textwidth} | ')
            outtex.write('p{.08\\textwidth} | p{.05\\textwidth}} \n    \\hline\n')
            outtex.write('    & Start & File & Target & Gain & Ncoher & Nps & Frames & ')
            outtex.write('Filter & seeing \\\\ \n')
            outtex.write('    & (UTC) & num. & & & & & $/$reset & & \\\\ \n    \\hline\n')
            if reductionfailed == 'False':
                tabRows = [[h['DATE'].split('T')[1],
                    h['HIERARCH MIRC PRO RTS'].split('/')[-1].split('mircx')[1].split('_')[0],
                    h['OBJECT'],h['GAIN'],h['NCOHER'],h['PSCOADD'],h['FRMPRST'],
                    h['FILTER1'],h['R0']] for h in redhdrs]
            else:
                tabRows = [[h['DATE'].split('T')[1],
                    h['COMMENT1'].split()[0],
                    h['OBJECT'],h['GAIN'],h['NCOHER'],h['PSCOADD'],h['FRMPRST'],
                    h['FILTER1'],h['R0']] for h in hdrs]
            r = 0.
            for row in tabRows:
                outstr = None
                for item in row:
                    try:
                        outstr = outstr+' & '+str(item).replace('_', ' ')
                    except TypeError:
                        outstr = str(r).split('.')[0]+' & '+str(item)
                outtex.write('        '+outstr+'\\\\ \n')
                r += 1.
            outtex.write('    \\hline\n\\end{longtable}\n\n')
            if reductionfailed == 'False':
                # Print figure showing uv-coverage of science target(s)
                outtex.write('\\newpage\n\\begin{figure}[h]\n    \\raggedright\n')
                outtex.write('    \\textbf{Full night $uv$-coverage for SCI target(s)}\\\\ \n')
                outtex.write('    \\centering\n')
                for uvp in uvplot[0:12]:
                    outtex.write('    \\includegraphics[trim=2.0cm 0.0cm 3.3cm 0.0cm, ')
                    outtex.write('clip=true, width=0.32\\textwidth]{'+uvp+'}\n')
                if len(uvplot) > 12:
                    for n in range(1, int(np.floor(len(uvplot)))):
                        outtex.write('\\end{figure}\n\n\\begin{figure}[h]\n')
                        outtex.write('    \\raggedright\n')
                        outtex.write('    \\textbf{Cont.}\\\\ \n    \\centering\n')
                        for uvp in uvplot[12*n:12*(n+1)]:
                            outtex.write('    \\includegraphics[trim=2.0cm 0.0cm 3.3cm 0.0cm, ')
                            outtex.write('clip=true, width=0.32\\textwidth]{'+uvp+'}\n')
                outtex.write('\\end{figure}\n\n\\newpage\n')
                # Print figures output by mircx_report.py
                for rfile in rfiles:
                    outtex.write('\\begin{figure}[h]\n    \\raggedright\n')
                    outtex.write('    \\textbf{Results from mircx\\_report.py for ')
                    outtex.write(date+'}\\\\ \n    \\centering\n')
                    for k in range(0, len(rfile)):
                        outtex.write('    \\includegraphics[trim=0.0cm 0.8cm 0.0cm 0.2cm, ')
                        outtex.write('clip=true, width=0.8\\textwidth]{'+redDir+'/oifits/')
                        outtex.write(rfile[k]+'}\n')
                    outtex.write('\\end{figure}\n\n')
                # Print Reduction QA plots: PSD
                outtex.write('\\newpage\n\\begin{figure}[h]\n    \\raggedright\n')
                outtex.write('    \\textbf{Reduction quality assessment: PSD}\\\\ \n')
                outtex.write('    \\centering\n')
                for rts in rtsPlots[0:15]:
                    outtex.write('    \\includegraphics[trim=0.7cm 0.9cm 1.5cm 0.0cm, ')
                    outtex.write('clip=true, width=0.32\\textwidth]{'+rts+'}\n')
                if len(rtsPlots) > 15:
                    for n in range(1, int(np.ceil(len(rtsPlots)/15.))):
                        outtex.write('\\end{figure}\n\n\\begin{figure}[h]\n')
                        outtex.write('    \\raggedright\n    \\textbf{Cont.}\\\\ \n')
                        outtex.write('    \\centering\n')
                        for rts in rtsPlots[15*n:15*(n+1)]:
                            outtex.write('    \\includegraphics[trim=0.7cm 0.9cm 1.5cm 0.0cm')
                            outtex.write(', clip=true, width=0.32\\textwidth]{'+rts+'}\n')
                outtex.write('\\end{figure}\n\n')
                # Print Reduction QA Plots: OIFITS SNR
                outtex.write('\\newpage\n\\begin{figure}[h]\n    \\raggedright\n')
                outtex.write('    \\textbf{Reduction quality assessment: SNR}\\\\ \n')
                outtex.write('    \\centering\n')
                for snr in snrPlots[0:6]:
                    outtex.write('    \\includegraphics[trim=2cm 0.9cm 1.5cm 0cm, clip=true, ')
                    outtex.write('width=0.49\\textwidth]{'+snr+'}\n')
                if len(snrPlots) > 6:
                    for n in range(1, int(np.ceil(len(snrPlots)/6.))):
                        outtex.write('\\end{figure}\n\n\\begin{figure}[h]\n')
                        outtex.write('    \\raggedright\n    \\textbf{Cont.}\\\\ \n')
                        outtex.write('    \\centering\n')
                        for snr in snrPlots[6*n:6*(n+1)]:
                            outtex.write('    \\includegraphics[trim=2cm 0.9cm 1.5cm 0cm, ')
                            outtex.write('clip=true, width=0.49\\textwidth]{'+snr+'}\n')
                outtex.write('\\end{figure}\n\n')
                # Print Reduction QA plots: OIFITS base trend
                outtex.write('\\newpage\n\\begin{figure}[h]\n    \\raggedright\n')
                outtex.write('    \\textbf{Reduction quality assessment: base trend}\\\\ \n')
                outtex.write('    \\centering\n')
                for ba in basePlots[0:6]:
                    outtex.write('    \\includegraphics[trim=2.2cm 0.9cm 1.5cm 0cm, clip=')
                    outtex.write('true, width=0.49\\textwidth]{'+ba+'}\n')
                if len(basePlots) > 6:
                    for n in range(1, int(np.ceil(len(basePlots)/6.))):
                        outtex.write('\\end{figure}\n\n\\begin{figure}[h]\n')
                        outtex.write('    \\raggedright\n    \\textbf{Cont.}\\\\ \n')
                        outtex.write('    \\centering\n')
                        for ba in basePlots[6*n:6*(n+1)]:
                            outtex.write('    \\includegraphics[trim=2.2cm 0.9cm 1.5cm 0cm, ')
                            outtex.write('clip=true, width=0.49\\textwidth]{'+ba+'}\n')
                outtex.write('\\end{figure}\n\n')
                # Print calibrated visibilities output by mircx_calibrate.py
                outtex.write('\\newpage\n\\begin{figure}\n    \\raggedright\n')
                outtex.write('    \\textbf{Calibrated visibility}\\\\ \n    \\centering\n')
                for mfile in sorted(mfiles_v)[0:15]:
                    outtex.write('    \\includegraphics[trim=1.1cm 0.2cm 1.5cm 0.0cm, ')
                    outtex.write('clip=true, width=0.32\\textwidth]{'+mfile+'}\n')
                if len(mfiles_v) > 15:
                    for n in range(1, int(np.ceil(len(mfiles_v)/15.))):
                        outtex.write('\\end{figure}\n\n\\begin{figure}[h]\n')
                        outtex.write('    \\raggedright\n    \\textbf{Cont.}\\\\ \n')
                        outtex.write('    \\centering\n')
                        for mfile in sorted(mfiles_v)[15*n:15*(n+1)]:
                            outtex.write('    \\includegraphics[trim=1.1cm 0.2cm 1.5cm 0.0c')
                            outtex.write('m, clip=true, width=0.32\\textwidth]{'+mfile+'}\n')
                outtex.write('\\end{figure}\n\n\\newpage\n')
                # Print calibrated closure phases
                outtex.write('\\begin{figure}\n    \\raggedright\n')
                outtex.write('    \\textbf{Calibrated closure phase}\\\\ \n    \\centering\n')
                for mf in sorted(mfiles_c)[0:15]:
                    outtex.write('    \\includegraphics[trim=1.1cm 0.2cm 1.5cm 0.0cm,')
                    outtex.write(' clip=true, width=0.32\\textwidth]{'+mf+'}\n')
                if len(mfiles_c) > 15:
                    for n in range(1, int(np.ceil(len(mfiles_c)/15.))):
                        outtex.write('\\end{figure}\n\n\\begin{figure}[h]\n')
                        outtex.write('    \\raggedright\n    \\textbf{Cont.}\\\\ \n')
                        outtex.write('    \\centering\n')
                        for mf in sorted(mfiles_c)[15*n:15*(n+1)]:
                            outtex.write('    \\includegraphics[trim=1.1cm 0.2cm 1.5cm 0.0c')
                            outtex.write('m, clip=true, width=0.32\\textwidth]{'+mf+'}\n')
            outtex.write('\\end{figure}\n\n\\end{document}\n')
        with cd(redDir):
            subprocess.call('pdflatex summary.tex', shell=True)
        log.info('Write and compile summary report')
        
        #####################################################
        # Email the summary file
        msg['Subject'] = 'MIRC-X redcal summary for '+date
        body = 'MIRC-X redcal summary for '+date+'\n'
        msg.attach(MIMEText(body, 'plain'))
        filename = 'summary.pdf'
        attachment = open(redDir+'/'+filename, 'rb')
        part = MIMEBase('application', 'octet-stream')
        part.set_payload((attachment).read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition',"attachment; filename= %s" % filename)
        msg.attach(part)
        try:
            send_email(msg)
            log.info('Emailed summary.pdf from '+date+' to:')
            log.info(argopt.email)
        except smtplib.SMTPAuthenticationError:
            log.error('Failed to send summary.pdf file to '+argopt.email)
            log.error('Check with Narsi Anugu for permissions')
            sys.exit()
        # Emailing works!
