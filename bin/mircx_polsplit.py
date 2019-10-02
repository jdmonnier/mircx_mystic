#! /usr/bin/env python

"""
This is a "quick and dirty" solution to getting polarization data through the pipeline.
This script creates new fits files with independent polarization states.
Make sure you have plenty of diskspace.
"""

from __future__ import print_function
import argparse
import os
from time import sleep
from astropy.io import fits

parser = argparse.ArgumentParser(description='Process MIRC-X raw data files')
parser.add_argument("--no-warn", action="store_true")
parser.add_argument("files", nargs="+", help="File(s) to process")
args = parser.parse_args()

if not args.no_warn:
    print("Warning: Make sure you have plenty of disk space; this is going to hurt.")
    print("(Hint: ^C while you still can!  Sleeping 10 seconds for your benefit.)")
    sleep(10)

for dir in ["pol1", "pol2"]:
    try:
        os.mkdir(dir)
    except FileExistsError:
        if os.path.isdir(dir):
            print("Warning: directory `" + dir + "` already exists")
        else:
            raise FileExistsError("Looks like you have a file named `" + dir + "`; please remove it.")

def polstate(file, state):
    f = fits.open(file)
    f[0].header["POLSTATE"] = state
    rows = f[0].header["CROPROWS"].split(",")
    if len(rows) != 2:
        raise ValueError("There must be exactly 2 detector regions. Is this a polarization data file?")
    span = 1 - eval(rows[0]) # 50-50 chance it should be rows[1]
    if state == 1:
        f[0].data = f[0].data[:,:,:span,:]
    elif state == 2:
        f[0].data = f[0].data[:,:,span:,:]
    else:
        raise ValueError("`state` (2nd arg of fcn `polstate`) must have the value either 1 or 2")
    path = "pol" + str(state) + "/" + file
    f.writeto(path)
    f.close()
    os.system("fpack " + path)
    os.remove(path)

for file in args.files:
    fz = file[-3:] == ".fz"
    if fz:
        os.system("funpack " + file)
        file = file[:-3]
    polstate(file, 1)
    polstate(file, 2)
    if fz:
        os.remove(file)
