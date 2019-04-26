#! /usr/bin/env python                                                          
# -*- coding: iso-8859-15 -*-                                                   

# Author = Claire L. Davies 
# Uploaded to mircx_pipeline: 2018 Nov 27
# Summary of changes:
#
# 2019-04-26: added option to use wrapper to explore different ncoherent 
#
# 2019-03-14: added argument --bbias for consistency with
# updates to mircx_reduce.py script
#
# 2019-03-15: ensured bbias input is appended to directory name

import argparse, subprocess, os, glob, sys
from mircx_pipeline import lookup, summarise, mailfile, headers, log, files
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
description:
 Wrapper for mircx_reduce.py, mircx_calibrate.py
 and mircx_report.py. 
"""

epilog = \
"""
examples:
 mircx_redcal_wrapper.py --dates=2018Oct29,2018Oct28 
  --ncoherent=5,10 --ncs=1,1 --nbs=4,4 --snr-threshold=2.0,2.0

NB: length of ncoherent, ncs, nbs, snr-threshold must be
 equal.
"""

parser = argparse.ArgumentParser(description=description,epilog=epilog,
                       formatter_class=argparse.RawDescriptionHelpFormatter,add_help=True)

TrueFalseOverwrite = ['TRUE','FALSE','OVERWRITE']

parser.add_argument("--raw-dir",dest="raw_dir",default='/data/CHARADATA/MIRCX',type=str,
            help="directory base for the raw data paths [%(default)s]")
parser.add_argument("--red-dir",dest="red_dir",default='/data/MIRCX/reduced',type=str,
            help="directory base for the reduced data paths [%(default)s]")
parser.add_argument("--dates",dest="dates",type=str,
            help="comma-separated list of observation dates to be reduced [%(default)s]")
parser.add_argument("--ncoherent",dest="ncoherent",type=str,default='5', 
            help="list of number of frames for coherent integration [%(default)s]")
parser.add_argument("--ncs",dest="ncs",type=str,default='1', 
            help="list of number of frame-offset for cross-spectrum [%(default)s]")
parser.add_argument("--nbs",dest="nbs",type=str,default='4', 
            help="list of number of frame-offset for bi-spectrum [%(default)s]")
parser.add_argument("--snr-threshold",dest="snr_threshold",type=str,default='2.0', 
            help="list of SNR threshold for fringe selection [%(default)s]")
parser.add_argument("--reduce",dest="reduce",default='TRUE',
            choices=TrueFalseOverwrite,
            help="(re)do the reduction process [%(default)s]")
parser.add_argument("--calibrate",dest="calibrate",default='TRUE',
            choices=TrueFalseOverwrite,
            help="(re)do the calibration process [%(default)s]")
parser.add_argument("--targ-list",dest="targ_list",default='mircx_targets.list',type=str,
            help="local database to query to identify SCI and CAL targets [%(default)s]")
parser.add_argument("--email",dest="email",type=str,default='', 
            help='email address to send summary report file TO [%(default)s]')
parser.add_argument("--sender",dest="sender",type=str,default='mircx.mystic@gmail.com',
            help='email address to send summary report file FROM [%(default)s]')
parser.add_argument("--max-integration-time",dest="max_int_time",type=str,default='300',
            help='maximum integration into a single file, in (s).\n'
           'This applies to PREPROC, RTS and OIFITS steps [%(default)s]')
parser.add_argument ("--bbias", dest="bbias",default='FALSE',choices=TrueFalseOverwrite,
            help="list of bools (compute the BBIAS_COEFF product [%(default)s]?)")

parser.add_argument("--rts", dest="rts",default='TRUE',choices=TrueFalseOverwrite,
            help="compute the RTS products [%(default)s]")

parser.add_argument("--rts-dir", dest="rts_dir",default='./rts/',type=str,
            help="directory of RTS products [%(default)s]")

parser.add_argument("--preproc", dest="preproc",default='TRUE',choices=TrueFalseOverwrite,
            help="compute the PREPROC products [%(default)s]");

parser.add_argument("--preproc-dir", dest="preproc_dir",default='./preproc/',type=str,
            help="directory of PREPROC products [%(default)s]");

#####################################################
# Set-up script:
argopt = parser.parse_args ()

elog = log.trace('mircx_redcal_wrapper')

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
    if '@' in argopt.email:
        log.error('Password for '+argopt.sender+' not found!')
        log.info('Please add environment variable MAILLOGIN to your bash or')
        log.info(' csh profile before continuing.')
        log.info('If you do not own the password for '+argopt.sender+',')
        log.info(' please contact Narsi Anugu or use option --sender to change')
        log.info(' the email address from which the summary report will be sent.')
        log.info('[HINT: the password for the email parsed to --sender will need')
        log.info(' to be saved as environment varaible $MAILLOGIN.]')
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



# -------------------------------------
# Establish how the script is to be used:
# -------------------------------------
if argopt.rts=='FALSE' and argopt.preproc=='FALSE':
    log.info('Exploring impact of ncoherent on SNR and T3PHI')
    # 1a. require that one date be parsed...
    if len(argopt.dates.split(',')) != 1:
        log.error('Too many dates parsed')
        log.info('When using the code for this purpose, only one date should be parsed')
        sys.exit()
    
    # 1b. ...and that ncoherent is the only thing that varies:
    nco = str(argopt.ncoherent).split(',')
    if len(nco) != len(set(nco)):
        log.error('Values of ncoherent are not all different')
        sys.exit()
    
    # book-keeping:
    if argopt.preproc_dir[-1] == '/':
        dirPP = argopt.preproc_dir[:-1]
    else:
        dirPP = argopt.preproc_dir
    
    if argopt.rts_dir[-1] == '/':
        dirRTS = argopt.rts_dir[:-1]
    else:
        dirRTS = argopt.rts_dir
    
    # ----------------------------------
    # Set option variable values from directory path
    # ----------------------------------
    suf = dirPP.split('_')[-1].split('/')[0]
    ncs = suf.split('ncs')[1].split('nbs')[0]
    nbs = suf.split('nbs')[1].split('snr')[0]
    snr = suf.split('snr')[1].split('bbias')[0].replace('p', '.')
    if suf[-1] == 'T':
        bias = 'TRUE'
    elif suf[-1] == 'F':
        bias = 'FALSE'
    log.info('Options for ncs, nbs, snr-threshold and bbias retrieved from preproc directory path')
    
    redDir = redBase+'/'+argopt.dates+'_'+suf
    rawDir = rawBase+'/'+argopt.dates[0:7]+'/'+argopt.dates
    # ----------------------------------
    # Check if the preproc and rts directories already exist:
    # ----------------------------------
    if os.path.isdir(dirPP) and os.path.isdir(dirRTS):
        log.info('Saving oifits directories in existing folder')
    else:
        log.info('Preproc directory does not already exist')
        log.info('Preproc and rts steps to be executed first')
        redOpt = '--ncoherent='+nco[0]+' --ncs='+ncs+' --nbs='+nbs+' --snr-threshold='+snr+" --bbias="+bias
        with cd(redDir):
            com  = "mircx_reduce.py "+redOpt+" --raw-dir="+rawDir
            ma = " --preproc-dir="+dirPP+" --rts-dir="+dirRTS
            nd = " --oifits=FALSE"
            pipe = "> nohup_reduce.out"
            with open('nohup_reduce.out', 'w') as output:
                output.write('\n')
            log.info('Execute nohup '+com+ma+nd+' '+pipe)
            subprocess.call('nohup '+com+ma+nd+' '+pipe+' &', shell=True)
            nf = open('nohup_reduce.out', 'r')
            ll = 0
            while True:
                nf.seek(ll,0)
                last_line = nf.read()
                ll = nf.tell()
                if last_line:
                    print last_line.strip()
                    if 'Git last commit:' in last_line:
                        break
    
    #----------------------------------
    # Execute each of the oifits steps
    # ---------------------------------
    with cd(redDir):
        for i in range(0, len(nco)):
            log.info('Run oifits step for ncoherent='+nco[i])
            redOpt = '--ncoherent='+nco[i]+' --ncs='+ncs+' --nbs='+nbs+' --snr-threshold='+snr+" --bbias="+bias
            com = "mircx_reduce.py "+redOpt+" --raw-dir="+rawDir
            ma = " --preproc=FALSE --preproc-dir="+dirPP+" --rts=FALSE --rts-dir="+dirRTS
            nd = " --oifits-dir="+dirPP.replace('/preproc','/oifits_nc'+nco[i])
            pipe = "> nohup_reduce.out"
            with open('nohup_reduce.out', 'w') as output:
                output.write('\n')
            log.info('Execute nohup '+com+ma+nd+' '+pipe)
            subprocess.call('nohup '+com+ma+nd+' '+pipe+' &', shell=True)
            nf = open('nohup_reduce.out', 'r')
            ll = 0
            while True:
                nf.seek(ll,0)
                last_line = nf.read()
                ll = nf.tell()
                if last_line:
                    print last_line.strip()
                    if 'Git last commit:' in last_line:
                        break
    
    # --------------------------------
    # Produce the plot of ncoherent vs SNR and ncoherent vs T3PHI
    # --------------------------------
    snr_keys = ['SNR01 MEAN', 'SNR02 MEAN', 'SNR03 MEAN', 'SNR04 MEAN', 'SNR05 MEAN', 
                'SNR12 MEAN', 'SNR13 MEAN', 'SNR14 MEAN', 'SNR15 MEAN','SNR23 MEAN', 
                'SNR24 MEAN', 'SNR25 MEAN', 'SNR34 MEAN', 'SNR35 MEAN', 'SNR45 MEAN']
    T3err_keys = ['T3PHI012 ERR', 'T3PHI013 ERR', 'T3PHI014 ERR', 'T3PHI015 ERR', 
                  'T3PHI023 ERR', 'T3PHI024 ERR', 'T3PHI025 ERR', 'T3PHI034 ERR', 
                  'T3PHI035 ERR','T3PHI045 ERR', 'T3PHI123 ERR', 'T3PHI124 ERR', 
                  'T3PHI125 ERR', 'T3PHI134 ERR', 'T3PHI135 ERR', 'T3PHI145 ERR', 
                  'T3PHI234 ERR', 'T3PHI235 ERR', 'T3PHI245 ERR', 'T3PHI345 ERR']
    nc_values = [float(n) for n in nco]
    start_num_nc = min(nc_values)
    end_num_nc = max(nc_values)
    snr_data = []
    T3err_data = []
    for d in nc_values:
        dir = dirPP.replace('/preproc','/oifits_nc'+str(d).split('.')[0])
        fs = glob.glob(dir+'/mircx*.fits')[::2]
        
        log.info(dir+" # files = "+str(len(fs)))
        
        hdrs = []
        for f in fs:
            hdulist = pyfits.open(f)
            hdrs.append(hdulist[0].header)
            hdulist.close()
        
        snr_data.append(np.array([[ h['HIERARCH MIRC QC '+k] for k in snr_keys ] for h in hdrs]) )
        T3err_data.append(np.array([[ h['HIERARCH MIRC QC '+k] for k in T3err_keys ] for h in hdrs]) )
    
    snr_data = np.asarray(snr_data)
    T3err_data = np.asarray(T3err_data)
    
    files.ensure_dir(redDir+'/PNG/')
    # SNR vs Ncoherent:
    for nf in range(0, snr_data.shape[1]): # number of files
        fig,ax = plt.subplots(5,3,figsize=(10,12)) # 15 SNR for each file
        ax = ax.flatten()
        
        for i in range(0, snr_data.shape[2]):
            ax[i].plot(nc_values, snr_data[:,nf,i], '-o')
            ax[i].set_ylabel('SNR')
            ax[i].set_xlabel('Ncoherent')
        fig.savefig(redDir+'/PNG/snr_vs_ncoh'+str(nf)+'.png', dpi=300,bbox_inches='tight')
        log.info('Created file: '+redDir+'/PNG/snr_vs_ncoh'+str(nf)+'.png')
        plt.close()

    
    # T3err vs Ncoherent:
    for nf in range(0, snr_data.shape[1]):
        fig,ax = plt.subplots(5,4,figsize=(10,12)) # 20 T3 for each file
        ax = ax.flatten()
        
        for i in range(0, T3err_data.shape[2]):
            ax[i].plot(nc_values, T3err_data[:,nf,i], '-o')
            ax[i].set_ylabel('T3 Err')
            ax[i].set_xlabel('Ncoherent')
        fig.savefig(redDir+'/PNG/t3err_vs_ncoh'+str(nf)+'.png', dpi=300,bbox_inches='tight')
        log.info('Created file: '+redDir+'/PNG/t3err_vs_ncoh'+str(nf)+'.png')
        plt.close()
    
    # email the user to point to where these plots have been saved:
    if '@' in argopt.email:
        msg = MIMEMultipart()
        msg['From'] = argopt.sender
        msg['To']   = argopt.email
        msg['Subject'] = 'Finished: MIRC-X redcal ncoherent vs SNR and T3PHI plots for '+argopt.dates
        body = 'Finished: MIRC-X redcal ncoherent vs SNR and T3PHI plots for '+argopt.dates+'\n\n'+'Files are located in '+redDir+'/PNG/ \n'
        msg.attach(MIMEText(body, 'plain'))
        try:
            mailfile.send_email(msg, argopt.sender, argopt.email)
            log.info('Emailed note to:')
            log.info(argopt.email)
        except smtplib.SMTPAuthenticationError:
            log.error('Failed to send note to '+argopt.email)
            log.error('Check with Narsi Anugu for permissions')
            sys.exit()

elif not (argopt.rts == 'TRUE' and argopt.preproc == 'TRUE'):
    log.error('Script is not designed for this combination of --rts and --preproc')
    sys.exit()
else:
    # Check the input format:
    nco = str(argopt.ncoherent).split(',')
    ncs = str(argopt.ncs).split(',')
    nbs = str(argopt.nbs).split(',')
    snr = str(argopt.snr_threshold).split(',')
    bias = str(argopt.bbias).split(',')
    if len(nco) == len(ncs) == len(nbs) == len(snr) == len(bias):
        log.info('Length of reduction options checked: ok')
    else:
        log.error('Error in setup: length of ncoherent, ncs, nbs, snr_threshold and bbias not equal!')
        sys.exit()
    
    redoRed, redoCal = [], []
    # execute 
    for d in argopt.dates.split(','):
        opt = []
        for i in range(0, len(nco)):
            opt.append([str(nco[i]),str(ncs[i]),str(nbs[i]),str(snr[i]).replace('.','p'),str(bias[i])])
        
        # retrieve a list of target names from the file headers:
        targs = lookup.targList(d,rawBase,redBase,opt) # this also produces a target summary file if the directory is new
        
        # Check whether reduction and calibration need to be (re)run:
        r_overwrite = (argopt.reduce == 'OVERWRITE')
        c_overwrite = (argopt.calibrate == 'OVERWRITE')
        redoRed, redoCal = [], []
        for i in range(0, len(nco)):
            # Build the suffix for the directory name:
            suff = '_ncoh'+opt[i][0]+'ncs'+opt[i][1]+'nbs'+opt[i][2]+'snr'+opt[i][3]+'bbias'+opt[i][4][0]
            # Check whether user wished for reduction to be (re)done:
            if argopt.reduce != 'FALSE':
                # Check whether the directory exists and whether the files 
                #  should be overwritten:
                if os.path.isdir(redBase+'/'+d+suff+'/oifits') and r_overwrite is False:
                    redoRed.append(False)
                else:
                    redoRed.append(True)
                # This is nested as one requires the reduction outputs for calibration
                if argopt.calibrate != 'FALSE':
                    if os.path.isdir(redBase+'/'+d+suff+'/oifits/calibrated') and c_overwrite is False:
                        redoCal.append(False)
                    else:
                        redoCal.append(True)
                else:
                    redoCal.append(False)
            else:
                # Someone has requested not to redo the reduction process:
                redoRed.append(False)
                if argopt.calibrate != 'FALSE':
                    if os.path.isdir(redBase+'/'+d+suff+'/oifits/calibrated') and c_overwrite is False:
                        # The calibration has not been run or previously failed but
                        # should not be run at this time.
                        redoCal.append(False)
                    else:
                        # The calibration should be re-run
                        redoCal.append(True)
                else:
                    # Someone has requested not to redo the calibration process
                    redoCal.append(False)
        
        # Query database (local and JSDC) to retrieve whether targets are sci; cal; new:
        calInfo, scical = lookup.queryLocal(targs, localDB)
        
        for i in range(0, len(nco)):
            redF = True # Did the reduction fail? ('Yes' is default)
            calF = True # Did the calibration fail? ('Yes' is default)
            suf    = '_ncoh'+opt[i][0]+'ncs'+opt[i][1]+'nbs'+opt[i][2]+'snr'+opt[i][3]+'bbias'+opt[i][4][0]
            redOpt = '--ncoherent='+opt[i][0]+' --ncs='+opt[i][1]+' --nbs='+opt[i][2]+' --snr-threshold='+opt[i][3].replace('p','.')+" --bbias="+str(opt[i][4]) 
            rawDir = rawBase+'/'+d[0:7]+'/'+d
            redDir = redBase+'/'+d+suf
            if redoRed[i] == True:
                with cd(redDir):
                    com  = "mircx_reduce.py "+redOpt+" --raw-dir="+rawDir
                    ma = " --preproc-dir="+redDir+"/preproc --rts-dir="+redDir+"/rts"
                    nd = " --oifits-dir="+redDir+"/oifits"
                    pipe = "> nohup_reduce.out"
                    with open('nohup_reduce.out', 'w') as output:
                        output.write('\n')
                    log.info('Execute nohup '+com+ma+nd+' '+pipe)
                    subprocess.call('nohup '+com+ma+nd+' '+pipe+' &', shell=True)
                    nf = open('nohup_reduce.out', 'r')
                    ll = 0
                    while True:
                        nf.seek(ll,0)
                        last_line = nf.read()
                        ll = nf.tell()
                        if last_line:
                            print last_line.strip()
                            if 'Git last commit:' in last_line:
                                break
            
            # Check that at least some reduced fits files have been produced:
            if os.path.isdir(redDir+'/oifits'):
                if len(glob.glob(redDir+'/oifits/*.fits')) > 0:
                    redF = False
                    with cd(redDir+'/oifits'):
                        command = "mircx_report.py --oifits-dir="+redDir+"/oifits"
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
                    
                    if redoCal[i] == True:
                        with cd(redDir+'/oifits'):
                            com = "mircx_calibrate.py --calibrators="+calInfo[:-1]
                            ma = " --oifits-dir="+redDir+"/oifits"
                            nd = " --oifits-calibrated-dir="+redDir+"/oifits/calibrated"
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
            
            # Check that at least some calibrated fits files have been produced:
            if os.path.isdir(redDir+'/oifits/calibrated'): 
                if len(glob.glob(redDir+'/oifits/calibrated/*.fits')) > 0:
                    calF = False
                    # make summary uv coverage plots:
                    summarise.plotUV(redDir+'/oifits/calibrated')
            
            # make summary PDF files:
            log.info('Read headers from raw data directory')
            rawhdrs = headers.loaddir(rawBase+'/'+d[0:7]+'/'+d)
            log.info('Create report summary files')
            summarise.texSumTitle(redDir, rawhdrs, opt[i], redF, calF)
            summarise.texSumTables(redDir,targs,calInfo,scical,redF,rawhdrs)
            log.info('Cleanup memory')
            del rawhdrs
            
            summarise.texReportPlts(redDir)
            summarise.texSumUV(redDir,calF)
            summarise.texSumPlots(redDir,redF,calF)
            with cd(redDir):
                subprocess.call('pdflatex '+redDir+'/summary_'+d+suf+'.tex', shell=True)
                subprocess.call('pdflatex '+redDir+'/report_'+d+suf+'.tex' , shell=True)
                log.info('Write and compile summary report')
            if '@' in argopt.email:
                mailfile.sendSummary(redDir, argopt.email, argopt.sender)
