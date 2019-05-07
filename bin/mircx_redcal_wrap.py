#! /usr/bin/env python                                                          
# -*- coding: iso-8859-15 -*-

import argparse, subprocess, os, glob, sys, socket, datetime
from mircx_pipeline import lookup_edit, summarise_edit, mailfile, headers, log, files
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits as pyfits
import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
from email.MIMEBase import MIMEBase
from email import encoders

class cd:
    """
    Context manager for changing the current working directory
    """
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)
    
    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)
    
    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

#####################################################
# Description of script and parsable options
description = \
"""
description use #1:
 Wrapper for mircx_reduce.py, mircx_calibrate.py,
 mircx_report.py and mircx_transmission.py. 
 
description use #2:
 Wrapper for mircx_reduce.py to explore different
 values of ncoherent and their effect on vis SNR 
 and T3PHI error.
"""

epilog = \
"""
examples use #1:
 mircx_redcal_wrap.py --dates=2018Oct29,2018Oct28 
  --ncoherent=5,10 --ncs=1,1 --nbs=4,4 --snr-threshold=2.0,2.0

NB: length of ncoherent, ncs, nbs, snr-threshold must be
 equal.
 
examples use #2:
 mircx_redcal_wrap.py --dates=2018Oct25 --ncoh-plots=TRUE 
 --email=observer@chara.observatory
"""

parser = argparse.ArgumentParser(description=description,epilog=epilog,
                       formatter_class=argparse.RawDescriptionHelpFormatter,add_help=True)

TrueFalseDefault = ['TRUE','FALSE','TRUEd']
TrueFalse = ['TRUE','FALSE']
TrueFalseOverwrite = ['TRUE','FALSE','OVERWRITE']

parser.add_argument("--raw-dir",dest="raw_dir",default='/data/CHARADATA/MIRCX',type=str,
            help="directory base for the raw data paths [%(default)s]")

parser.add_argument("--red-dir",dest="red_dir",default='/data/MIRCX/reduced',type=str,
            help="directory base for the reduced data paths [%(default)s]")

parser.add_argument("--dates",dest="dates",type=str,
            help="comma-separated list of observation dates to be reduced [%(default)s]")

preproc = parser.add_argument_group ('(1) preproc',
         '\nSet of options used to control the book-keeping'
         ' as well as the preproc and rts reduction steps.')

preproc.add_argument("--reduce",dest="reduce",default='TRUE',
            choices=TrueFalseOverwrite,
            help="(re)do the reduction process [%(default)s]")

preproc.add_argument("--ncs",dest="ncs",type=str,default='1d', 
            help="list of number of frame-offset for cross-spectrum [%(default)s]")

preproc.add_argument("--nbs",dest="nbs",type=str,default='4d', 
            help="list of number of frame-offset for bi-spectrum [%(default)s]")

preproc.add_argument ("--bbias", dest="bbias",default='TRUEd',choices=TrueFalseDefault,
            help="list of bools (compute the BBIAS_COEFF product [%(default)s]?)")

preproc.add_argument("--max-integration-time-preproc", dest="max_integration_time_preproc",
            default='30.d',type=str,
            help='maximum integration into a single file, in (s).\n'
            'This apply to PREPROC, and RTS steps [%(default)s]')

oifits = parser.add_argument_group ('(1) oifits',
         '\nSet of options used to control the oifits\n'
         ' reduction steps.')

oifits.add_argument("--ncoherent",dest="ncoherent",type=str,default='5d', 
            help="list of number of frames for coherent integration [%(default)s]")

oifits.add_argument("--snr-threshold",dest="snr_threshold",type=str,default='2.0d', 
            help="list of SNR threshold for fringe selection [%(default)s]")

oifits.add_argument("--max-integration-time-oifits", dest="max_integration_time_oifits",
            default='150.d',type=str,
            help='maximum integration into a single file, in (s).\n'
            'This apply to OIFITS steps [%(default)s]')

calib = parser.add_argument_group ('(1) calibrate',
        '\nSet of options used to control the calibration steps.')

calib.add_argument("--calibrate",dest="calibrate",default='TRUE',
            choices=TrueFalseOverwrite,
            help="(re)do the calibration process [%(default)s]")

calib.add_argument("--targ-list",dest="targ_list",default='mircx_targets.list',type=str,
            help="local database to query to identify SCI and CAL targets [%(default)s]")

summary = parser.add_argument_group ('(1) summary',
        '\nSet of options used to control the summary report\n'
        'file production and email alerts.')

summary.add_argument("--email",dest="email",type=str,default='', 
            help='email address to send summary report file TO [%(default)s]')

summary.add_argument("--sender",dest="sender",type=str,default='mircx.mystic@gmail.com',
            help='email address to send summary report file FROM [%(default)s]')

compare = parser.add_argument_group ('(1) compare',
        '\nOptions used to control the exploration of the impact'
        'of varying ncoherent on the vis SNR and T3ERR.')

compare.add_argument("--ncoh-plots", dest="ncoh_plots",default='FALSE',
            choices=TrueFalse,
            help="use the wrapper to produce plots of ncoherent vs\n"
            "vis SNR and T3ERR [%(default)s].")

# Parse arguments:
argopt = parser.parse_args ()

# Verbose:
elog = log.trace('mircx_redcal_wrapper')


# Check length of ncs,nbs,mitp,bbias,snr,mito and dates are equal
dates = argopt.dates.split(',')
ncs   = str(argopt.ncs).split(',')
nbs   = str(argopt.nbs).split(',')
mitp  = str(argopt.max_integration_time_preproc).split(',')
bbias = str(argopt.bbias).split(',')
snr   = str(argopt.snr_threshold).split(',')
mito  = str(argopt.max_integration_time_oifits).split(',')

for item in [ncs,nbs,mitp,bbias,snr,mito]:
    if isinstance(item, str):
        item = [item]

if len(ncs) == 1 and 'd' in ncs[0]:
    # Account for some being default settings:
    ncs = [ncs[0].replace('d','')]*len(dates)

if len(nbs) == 1 and 'd' in nbs[0]:
    # Account for some being default settings:
    nbs = [nbs[0].replace('d','')]*len(dates)

if len(mitp) == 1 and 'd' in mitp[0]:
    # Account for some being default settings:
    mitp = [mitp[0].replace('.d','')]*len(dates)

if len(bbias) == 1 and 'd' in bbias[0]:
    # Account for some being default settings:
    bbias = [bbias[0].replace('d','')]*len(dates)

if len(snr) == 1 and 'd' in snr[0]:
    # Account for some being default settings:
    snr = [snr[0].replace('d','')]*len(dates)

if len(mito) == 1 and 'd' in mito[0]:
    # Account for some being default settings:
    mito = [mito[0].replace('.d','')]*len(dates)
            
if len(ncs) == len(nbs) == len(mitp) == len(bbias) == len(snr) == len(mito) == len(dates):
    log.info('Length of reduction options checked: ok')
else:
    log.error('Error in setup: length of options is not equal!')
    sys.exit()


# Force choices of nbs and ncs when bbias=TRUE:
for bb in range(0, len(bbias)):
    if bb != 'FALSE':
        log.info('bbias instance set to true so setting corresponding ncs=1 and nbs=0')
        ncs[bb] = 1
        nbs[bb] = 0


# check argopt.ncoherent:
ncoh = str(argopt.ncoherent).split(',')
if argopt.ncoh_plots == 'FALSE':
    if len(ncoh) == 1 and 'd' in ncoh[0]:
        ncoh = [ncoh[0].replace('d','')]*len(dates)
    elif len(ncoh) != len(dates):
        log.error("Error: length of --ncoherent doesn't match length of --dates!")
        sys.exit()
else:
    if len(ncoh) == 1 and 'd' in ncoh[0]:
        ncoh = range(2,16)


# remove '/' from end of the reduction and raw base directories
if argopt.raw_dir[-1] == '/':
    rawBase = argopt.raw_dir[:-1]
else:
    rawBase = argopt.raw_dir
if argopt.red_dir[-1] == '/':
    redBase = argopt.red_dir[:-1]
else:
    redBase = argopt.red_dir

# Ensure emailing will work:
try:
    pw = os.environ['MAILLOGIN']
except KeyError:
    log.error('Password for '+argopt.sender+' not found!')
    log.info('The password for the email account parsed to --sender')
    log.info(' needs to be saved to environment variable $MAILLOGIN.')
    sys.exit()

# Ensure that the pipeline can be found
try:
    ext = os.environ['MIRCX_PIPELINE']
except KeyError:
    log.error('Environment variable $MIRCX_PIPELINE not found')
    log.info('Please rectify this before continuing')
    sys.exit()

if not os.path.isfile(os.environ['MIRCX_PIPELINE']+'/mircx_pipeline/'+argopt.targ_list):
    log.error(os.environ['MIRCX_PIPELINE']+'/mircx_pipeline/'+argopt.targ_list+' not found!')
    log.info('Please rectify this before continuing')
    sys.exit()
else:
    localDB = os.environ['MIRCX_PIPELINE']+'/mircx_pipeline/'+argopt.targ_list
    # ^-- this is the local target history database






for d in range(0, len(dates)):
    # special setting for execution on protostar @ exeter:
    if socket.gethostname() == 'protostar':
        rawBase_p = rawBase+'/'+dates[d][0:7]
        rawBase = rawBase_p
    
    
    # 1. Make directory dates_nbsncsbbiasmitp in argopt.red-dir
    if bbias[d] == 'TRUE':
        bbstr = 'T'
    else:
        bbstr = 'F'
    suf1 = '_nbs'+str(nbs[d])+'ncs'+str(ncs[d])+'bbias'+bbstr+'mitp'+mitp[d]
    redDir = redBase+'/'+dates[d]+suf1
    files.ensure_dir(redDir)
    
    # 2. run reduce.py with --oifits=FALSE
    opt1   = '--ncs='+str(ncs[d])+' --nbs='+str(nbs[d])+' --bbias='+str(bbias[d])
    opt2   = ' --max-integration-time-preproc='+str(mitp[d])
    opts = opt1+opt2
    rawDir = rawBase+'/'+dates[d]
    with cd(redDir):
        com = "mircx_reduce.py "+opts+" --raw-dir="+rawDir
        ma  = " --preproc-dir="+redDir+"/preproc --rts-dir="+redDir+"/rts"
        nd  = " --oifits=FALSE --reduce="+argopt.reduce
        pipe = "> nohup_preproc_rts.out"
        with open("nohup_preproc_rts.out", 'w') as output:
            output.write('\n')
        log.info('Execute nohup '+com+ma+nd+' '+pipe)
        subprocess.call('nohup '+com+ma+nd+' '+pipe+' &', shell=True)
        nf = open('nohup_preproc_rts.out', 'r')
        ll = 0
        while True:
            nf.seek(ll,0)
            last_line = nf.read()
            ll = nf.tell()
            if last_line:
                print last_line.strip()
                if 'Git last commit:' in last_line:
                    break
    
    # 3. Make directory snrmito in argopt.red-dir/dates_nbsncsbbiasmitp
    suf2 = 'snr'+str(snr[d]).replace('.','p')+'mito'+str(mito[d])
    files.ensure_dir(redDir+'/'+suf2)
    oiDir = redDir+'/'+suf2+"/oifits_nc"+str(ncoh[d])
    
    # 4: identify calibrators
    targs = lookup_edit.targList(dates[d],rawBase,redDir) # produces target summary file if directory is new
    calInfo, scical = lookup_edit.queryLocal(targs, localDB)
                
    if argopt.ncoh_plots == 'FALSE':
        # --------------------------------------------------------------
        # 5. Run reduce.py with --rts=FALSE and --preproc=FALSE
        #    assuming different ncoherent are for different argopt.dates
        # --------------------------------------------------------------
        opt3 = ' --max-integration-time-oifits='+str(mito[d])+' --snr-threshold='+str(snr[d])
        opts2 = opt1+' --ncoherent='+str(ncoh[d])+opt3
        with cd(redDir+'/'+suf2):
            com = "mircx_reduce.py "+opts2+" --raw-dir="+rawDir+" --preproc=FALSE"
            ma  = " --preproc-dir="+redDir+"/preproc --rts=FALSE --rts-dir="+redDir+"/rts"
            nd  = " --oifits-dir="+oiDir+" --reduce="+argopt.reduce
            pipe = "> nohup_oifits.out"
            with open("nohup_oifits.out", 'w') as output:
                output.write('\n')
            log.info('Execute nohup '+com+ma+nd+' '+pipe)
            subprocess.call('nohup '+com+ma+nd+' '+pipe+' &', shell=True)
            nf = open('nohup_oifits.out', 'r')
            ll = 0
            while True:
                nf.seek(ll,0)
                last_line = nf.read()
                ll = nf.tell()
                if last_line:
                    print last_line.strip()
                    if 'Git last commit:' in last_line:
                        break
            
        # 6. Check that the oifits step successfully created .fits files in oiDir:
        if os.path.isdir(oiDir):
            if len(glob.glob(oiDir+'/*.fits')) > 0:
                redF = False # reduction did not fail
                
                # a: run report.py script
                with cd(oiDir):
                    command = "mircx_report.py --oifits-dir="+oiDir
                    pipe = "> nohup_report.out"
                    with open('nohup_report.out', 'w') as output:
                        output.write('\n')
                    log.info('Execute nohup '+command+' '+pipe)
                    subprocess.call("nohup "+command+' '+pipe+' &', shell=True)
                    nf = open('nohup_report.out', 'r')
                    ll = 0
                    while True:
                        nf.seek(ll,0)
                        last_line = nf.read()
                        ll = nf.tell()
                        if last_line:
                            print last_line.strip()
                            if 'Git last commit:' in last_line:
                                break
                
                # b: run mircx_transmission.py
                today = datetime.datetime.strptime(dates[d], '%Y%b%d')
                nextDay = today + datetime.timedelta(days=1)
                nD = nextDay.strftime('%Y%b%d')
                with cd(redDir):
                    com = "mircx_transmission.py --dir="+redBase+" --date-from="+dates[d]
                    ma  = " --date-to="+nD+" --targ-list="+argopt.targ_list
                    nd  = " --oifits-dir="+suf2+"/oifits_nc"+str(ncoh[d])
                    pipe = "> nohup_transmission.out"
                    with open('nohup_transmission.out', 'w') as output:
                        output.write('\n')
                    log.info('Execute nohup '+com+ma+nd+' '+pipe)
                    subprocess.call("nohup "+com+ma+nd+' '+pipe+' &', shell=True)
                    nf = open('nohup_transmission.out', 'r')
                    ll = 0
                    while True:
                        nf.seek(ll,0)
                        last_line = nf.read()
                        ll = nf.tell()
                        if last_line:
                            print last_line.strip()
                            if 'Git last commit:' in last_line:
                                break
                
                # d: run calibrate.py
                if argopt.calibrate != 'FALSE':
                    with cd(oiDir):
                        com  = "mircx_calibrate.py --oifits-calibrated="+argopt.calibrate
                        ma   = " --calibrators="+calInfo[:-1]+" --oifits-dir="+oiDir
                        nd   = " --oifits-calibrated-dir="+oiDir+'/calibrated'
                        pipe = "> nohup_calibrate.out"
                        with open('nohup_calibrate.out', 'w') as output:
                            output.write('\n')
                        log.info('Execute nohup '+com+ma+nd+' '+pipe)
                        subprocess.call("nohup "+com+ma+nd+" "+pipe+" &", shell=True)
                        nf = open('nohup_calibrate.out', 'r')
                        ll = 0
                        while True:
                            nf.seek(ll,0)
                            last_line = nf.read()
                            ll = nf.tell()
                            if last_line:
                                print last_line.strip()
                                if 'Git last commit:' in last_line:
                                    break
            
            else:
                redF = True
        else:
            redF = True
        
        # 7. Check that the calibration step was successful:
        if os.path.isdir(oiDir+'/calibrated'):
            if len(glob.glob(oiDir+'/calibrated/*.fits')) > 0:
                calF = False
                # make summary uv coverage plots for the calibrated files:
                summarise_edit.plotUV(oiDir+'/calibrated')
            else:
                calF = True
        else:
            calF = True
        
        # 8. Write summary and report files
        log.info('Read headers from raw data directory')
        rawhdrs = headers.loaddir(rawBase+'/'+dates[d][0:7]+'/'+dates[d])
        log.info('Create report summary files')
        outfiles = summarise_edit.texSumTitle(oiDir, rawhdrs, redF, calF)
        summarise_edit.texSumTables(oiDir,targs,calInfo,scical,redF,rawhdrs,outfiles)
        log.info('Cleanup memory')
        del rawhdrs
        summarise_edit.texReportPlts(oiDir,outfiles,dates[d])
        summarise_edit.texSumUV(oiDir,calF,outfiles)
        summarise_edit.texSumPlots(oiDir,redF,calF,outfiles)
        with cd(redDir):
            subprocess.call('pdflatex '+outfiles[1], shell=True)
            subprocess.call('pdflatex '+outfiles[0] , shell=True)
            log.info('Write and compile summary report')
        
        # 9. Email summary file to argopt.email
        if '@' in argopt.email:
            mailfile.sendSummary(argopt.email,argopt.sender,outfiles[1].replace('.tex','.pdf'))
    
    else:
        log.info('Exploring impact of ncoherent on SNR and T3PHI')
        log.info('Values parsed to --ncoherent to be used for all --dates')
        # -------------------------------------------------------------------------------
        # 5. Run reduce.py with --rts=FALSE and --preproc=FALSE for each argopt.ncoherent
        # -------------------------------------------------------------------------------
        opt3 = ' --max-integration-time-oifits='+str(mito[d])+' --snr-threshold='+str(snr[d])
        for nc in ncoh:
            oiDir = redDir+'/'+suf2+"/oifits_nc"+str(nc)
            if not os.path.isdir(oiDir):
                opts2 = opt1+' --ncoherent='+str(nc)+opt3
                log.info('Run oifits step for ncoherent='+str(nc))
                with cd(redDir):
                    com = "mircx_reduce.py "+opts2+" --raw-dir="+rawDir+" --preproc=FALSE"
                    ma  = " --preproc-dir="+redDir+"/preproc --rts=FALSE --rts-dir="+redDir+"/rts"
                    nd  = " --oifits-dir="+oiDir+" --reduce="+argopt.reduce
                    pipe = "> nohup_oifits.out"
                    with open("nohup_oifits.out", 'w') as output:
                        output.write('\n')
                    log.info('Execute nohup '+com+ma+nd+' '+pipe)
                    subprocess.call('nohup '+com+ma+nd+' '+pipe+' &', shell=True)
                    nf = open('nohup_oifits.out', 'r')
                    ll = 0
                    while True:
                        nf.seek(ll,0)
                        last_line = nf.read()
                        ll = nf.tell()
                        if last_line:
                            print last_line.strip()
                            if 'Git last commit:' in last_line:
                                break
            else:
                log.info(oiDir+' already exists')
                log.info('Skipped ncoherent='+str(nc))
        
        # 6. Produce the plot of ncoherent vs SNR and ncoherent vs T3PHI:
        snr_keys = ['SNR01 MEAN', 'SNR02 MEAN', 'SNR03 MEAN', 'SNR04 MEAN', 'SNR05 MEAN', 
                    'SNR12 MEAN', 'SNR13 MEAN', 'SNR14 MEAN', 'SNR15 MEAN','SNR23 MEAN', 
                    'SNR24 MEAN', 'SNR25 MEAN', 'SNR34 MEAN', 'SNR35 MEAN', 'SNR45 MEAN']
        T3err_keys = ['T3PHI012 ERR', 'T3PHI013 ERR', 'T3PHI014 ERR', 'T3PHI015 ERR', 
                      'T3PHI023 ERR', 'T3PHI024 ERR', 'T3PHI025 ERR', 'T3PHI034 ERR', 
                      'T3PHI035 ERR','T3PHI045 ERR', 'T3PHI123 ERR', 'T3PHI124 ERR', 
                      'T3PHI125 ERR', 'T3PHI134 ERR', 'T3PHI135 ERR', 'T3PHI145 ERR', 
                      'T3PHI234 ERR', 'T3PHI235 ERR', 'T3PHI245 ERR', 'T3PHI345 ERR']
        nc_values = [float(n) for n in ncoh]
        snr_data = []
        T3err_data = []
        for nc in ncoh:
            fs = glob.glob(redDir+'/'+suf2+'/oifits_nc'+str(nc)+'/*.fits')[::2]
            log.info(redDir+'/'+suf2+'/oifits_nc'+str(nc)+" # files = "+str(len(fs)))
            
            hdrs = [];
            for f in files:
                hdulist = pyfits.open(f);
                hdrs.append(hdulist[0].header);
                hdulist.close();
            
            snr_data.append ( np.array([[ h['HIERARCH MIRC QC '+k] for k in snr_keys ] for h in hdrs]) )
            T3err_data.append ( np.array([[ h['HIERARCH MIRC QC '+k] for k in T3err_keys ] for h in hdrs]) )
            
        snr_data = np.asarray(snr_data)
        T3err_data = np.asarray(T3err_data)
        
        files.ensure_dir(redDir+'/'+suf2+'/PNG/')
        # SNR vs Ncoherent:
        for nf in range(0, snr_data.shape[1]): # number of files
            fig,ax = plt.subplots(5,3,figsize=(10,12)) # 15 SNR for each file
            ax = ax.flatten()
            
            for i in range(0, snr_data.shape[2]):
                ax[i].plot(nc_values, snr_data[:,nf,i], '-o')
                ax[i].set_ylabel('SNR')
                ax[i].set_xlabel('Ncoherent')
            fig.savefig(redDir+'/'+suf2+'/PNG/snr_vs_ncoh'+str(nf)+'.png', dpi=300,bbox_inches='tight')
            log.info('Created file: '+redDir+'/'+suf2+'/PNG/snr_vs_ncoh'+str(nf)+'.png')
            plt.close()
        
        
        # T3err vs Ncoherent:
        for nf in range(0, snr_data.shape[1]):
            fig,ax = plt.subplots(5,4,figsize=(10,12)) # 20 T3 for each file
            ax = ax.flatten()
            
            for i in range(0, T3err_data.shape[2]):
                ax[i].plot(nc_values, T3err_data[:,nf,i], '-o')
                ax[i].set_ylabel('T3 Err')
                ax[i].set_xlabel('Ncoherent')
            fig.savefig(redDir+'/'+suf2+'/PNG/t3err_vs_ncoh'+str(nf)+'.png', dpi=300,bbox_inches='tight')
            log.info('Created file: '+redDir+'/'+suf2+'/PNG/t3err_vs_ncoh'+str(nf)+'.png')
            plt.close()
        
        # 7. email user when this procedure finishes and prompt them to run the calibrate
        #    section of the script with the best value of ncoherent.
        line1 = 'ncoherent vs SNR and T3PHI plots for '+argopt.dates+' located in '+redDir+'/'+suf2+'/PNG/ \n\n'
        line2 = 'To calibrate the data with the best ncoherent value (X), use:\n\n'
        line3 = 'mircx_redcal_wrap.py --reduce=FALSE --dates='+dates[d]+' '+opt1+opt3+' --ncoherent=X\n\n'
        if '@' in argopt.email:
            msg = MIMEMultipart()
            msg['From'] = argopt.sender
            msg['To']   = argopt.email
            msg['Subject'] = 'Finished: MIRC-X redcal ncoherent vs SNR and T3PHI plots for '+argopt.dates
            body = line1+line2+line3
            msg.attach(MIMEText(body, 'plain'))
            try:
                mailfile.send_email(msg, argopt.sender, argopt.email)
                log.info('Emailed note to:')
                log.info(argopt.email)
            except smtplib.SMTPAuthenticationError:
                log.error('Failed to send note to '+argopt.email)
                log.error('Check with Narsi Anugu for permissions')
                sys.exit()
        else:
            log.info(line1)
            log.info(line2)
            log.info(line3)
