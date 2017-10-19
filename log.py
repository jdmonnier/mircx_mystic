from __future__ import print_function
#from builtins import object
import time
import sys

ERROR   = 1
WARNING = 2
NOTICE  = 4
TRACE   = 8

verbose_type = ERROR+WARNING+NOTICE+TRACE

def set_verbose_type(val):
    """
    Set the verbose type, e.i.:
      set_verbose_type(ERROR+WARNING) only warning and error will be verbosed
      set_verbose_type(ERROR+WARNING+NOTICE) all verbose
      etc ...

    """
    global verbose_type
    verbose_type = val

try:
    import colorama as col
except:
    str_mtype = {ERROR:"ERROR", WARNING:"WARNING", NOTICE:"NOTICE", TRACE:"TRACE"}
else:
    str_mtype = {
                 ERROR:col.Fore.RED+"ERROR"+col.Fore.RESET,
                 WARNING:col.Fore.MAGENTA+"WARNING"+col.Fore.RESET,
                 NOTICE:col.Fore.BLUE+"NOTICE"+col.Fore.RESET,
                 TRACE:col.Fore.GREEN+"TRACE"+col.Fore.RESET
                 }
        
def log(msg, mtype=WARNING):
    global verbose_type
    _message_format = """[MRX: {mtype}] {date}: {msg}"""

    print 
    if verbose_type & mtype:
        print(_message_format.format(mtype=str_mtype[mtype],
                                    date=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                                    msg =msg
                                   ))
        sys.stdout.flush()
                                                                                    
def error(msg):
    log(msg, mtype=ERROR)

def warning(msg, level=1):
    log(msg, mtype=WARNING)
    
def notice(msg, level=1):
    log(msg, mtype=NOTICE)

def trace(msg, level=1):
    log(msg, mtype=TRACE)
        
