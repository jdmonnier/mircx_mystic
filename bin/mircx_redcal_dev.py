#! /usr/bin/env python                                                          
# -*- coding: iso-8859-15 -*-                                                   

# Author = Claire L. Davies 
# Uploaded to mircx_pipeline: 2018 Nov 27
# Summary of changes:
#

import argparse, subprocess, os, glob, sys
from mircx_pipeline import lookup, summarise, mailfile, headers, log

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

parser.add_argument("--raw-dir",dest="raw_dir",default='/.',type=str,
            help="directory base for the raw data paths [%(default)s]")
parser.add_argument("--red-dir",dest="red_dir",default='/.',type=str,
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
parser.add_argument("--targ-list",dest="targ_list",default='mircx_targets.list',
            type=str,
            help="local database to query to identify SCI and CAL targets [%(default)s]")
parser.add_argument("--email",dest="email",type=str,default='mircx-reports@exeter.ac.uk', 
            help='email address to send summary report file TO [%(default)s]')
parser.add_argument("--sender",dest="sender",type=str,default='mircx.mystic@gmail.com',
            help='email address to send summary report file FROM [%(default)s]')

#####################################################
# Set-up script:
argopt = parser.parse_args ()

elog = log.trace('mircx_redcal_wrapper')

# Check the inputs:
nco = str(argopt.ncoherent).split(',')
ncs = str(argopt.ncs).split(',')
nbs = str(argopt.nbs).split(',')
snr = str(argopt.snr_threshold).split(',')
if len(nco) == len(ncs) == len(nbs) == len(snr):
    log.info('Length of reduction options checked: ok')
else:
    log.error('Error in setup: length of ncoherent, ncs, nbs, snr_threshold not equal!')
    sys.exit()

try:
    pw = os.environ['MAILLOGIN']
except KeyError:
    log.error('Password for '+argopt.sender+' not found!')
    log.info('Please add environment variable MAILLOGIN to your bash or')
    log.info(' csh profile before continuing.')
    log.info('If you do not own the password for '+argopt.sender+',')
    log.info(' please contact Narsi Anugu or use option --sender to change')
    log.info(' the email address from which the summary report will be sent.')
    log.info('[HINT: the password for the email parsed to --sender will need')
    log.info(' to be saved as environment varaible $MAILLOGIN.]')
    sys.exit()

if argopt.raw_dir[-1] == '/':
    rawBase = argopt.raw_dir[:-1]
else:
    rawBase = argopt.raw_dir

if argopt.red_dir[-1] == '/':
    redBase = argopt.red_dir[:-1]
else:
    redBase = argopt.red_dir

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

#####################################################
for d in argopt.dates.split(','):
    opt = []
    for i in range(0, len(nco)):
        opt.append([str(nco[i]),str(ncs[i]),str(nbs[i]),str(snr[i]).replace('.','p')])
    # Save a target summary for the observation date to file:
    targs = lookup.targList(d,rawBase,redBase,opt)
    
    # Check whether reduction and calibration need to be (re)run:
    r_overwrite = (argopt.reduce == 'OVERWRITE')
    c_overwrite = (argopt.calibrate == 'OVERWRITE')
    redoRed, redoCal = [], []
    for i in range(0, len(nco)):
        # Build the suffix for the directory name:
        suff = '_ncoh'+opt[i][0]+'ncs'+opt[i][1]+'nbs'+opt[i][2]+'snr'+opt[i][3]
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
        redF = True
        calF = True
        suf    = '_ncoh'+opt[i][0]+'ncs'+opt[i][1]+'nbs'+opt[i][2]+'snr'+opt[i][3]
        redOpt = '--ncoherent='+opt[i][0]+' --ncs='+opt[i][1]+' --nbs='+opt[i][2]+' --snr-threshold='+opt[i][3].replace('p','.')
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
        mailfile.sendSummary(redDir, argopt.email, argopt.sender)
