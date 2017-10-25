from timeit import default_timer as timer
import time
import sys
import os
import logging

# Load colors
try:
    import colorama as col
except:
    RED     = '';
    MAGENTA = '';
    RESET   = '';
    BLUE    = '';
    GREEN   = '';
else:
    RED     = col.Fore.RED;
    MAGENTA = col.Fore.MAGENTA;
    RESET   = col.Fore.RESET;
    BLUE    = col.Fore.BLUE;
    GREEN   = col.Fore.GREEN;

# Get the logger
logger = logging.getLogger ('mircx_pipeline');

# Setup the configuration to log in the consol
logging.basicConfig (
     level=logging.DEBUG,
     format="[%(color)s%(levelname)-7.7s"+RESET+"] %(asctime)s: %(message)s",
     datefmt='%Y-%m-%dT%H:%M:%S');


# Set a logfile
def setFile (filename):
    info ('Set logFile: '+filename);
    for h in logger.handlers:
        logger.removeHandler (h);
    logfile = logging.FileHandler (filename, mode='w');
    logfile.setLevel (logging.DEBUG);
    formatter = logging.Formatter ("[%(levelname)-7.7s] "
                                   "%(asctime)s: %(message)s",
                                    datefmt='%Y-%m-%dT%H:%M:%S');
    logfile.setFormatter (formatter);
    logger.addHandler (logfile);

# Stop logging in files
def closeFile ():
    for h in logger.handlers:
        logger.removeHandler (h);

# Logging function
def info(msg):
    logger.info (msg, extra={'color':BLUE});

def warning(msg):
    logger.warning (msg, extra={'color':MAGENTA});

def error(msg):
    logger.error (msg, extra={'color':RED});

# Trace function (measure time until killed)
class trace:
    def __init__(self, funcname):
        logger.info('Start '+funcname,extra={'color':GREEN});
        self.funcname = funcname;
        self.stime = timer();

    def __del__(self):
        if self.stime is not None and self.funcname is not None:
            msg = 'End '+self.funcname+' in %.2fs'%(timer()-self.stime);
            logger.info (msg,extra={'color':GREEN});
        
