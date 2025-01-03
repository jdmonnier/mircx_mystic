#! /usr/bin/env python
# -*- coding: iso-8859-15 -*-

# will archive the headers of a directory of data!
# choose a diretory of data or a directory of directories of data.
#will run the mircx_mystic_nightcat.py on all directories and send data to ~/ARCHIVE;.
# will NOT do the tar files since that is best doesn on great lakes directly using my existing scripts.

# TODO. fix permissions on filest olllow read/write by all, group,owner...

import mircx_mystic as mrx
import argparse
import glob
import os
import sys
import pickle
import json

from mircx_mystic import log, setup, files, headers
import datetime as datetime
import tkinter as tk
from tkinter import filedialog
import pandas as pd


def select_directories():
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Default directories
    default_input_dir = "/turbo/"
    default_output_dir = "/Users/monnier/ARCHIVE/"

    # Ask user to select input directory
    print("\033[1m\033[91mSelect the input directory with raw data, or a directory of raw data directories\033[0m")
    input_dir = filedialog.askdirectory(initialdir=default_input_dir, title="Select Input Directory")
    if not input_dir:
        input_dir = default_input_dir

    # Ask user to select output directory
    print("\033[1m\033[91mSelect the output directory for the summary files (default: is ~/ARCHIVE)\033[0m")
    output_dir = filedialog.askdirectory(initialdir=default_output_dir, title="Select Output Directory")
    if not output_dir:
        output_dir = default_output_dir

    return input_dir, output_dir

if __name__ == "__main__":
    input_dir, output_dir = select_directories()
    print(f"Input Directory: {input_dir}")
    print(f"Output Directory: {output_dir}")

    #check if thare are fits or fits.fz files in the input directory.
    fits_files = glob.glob(f"{input_dir}/*fits*", recursive=False)
    if not fits_files:
        print("No FITS files found in the input directory. So assume this is a directory of directories.")
        input_dir_list= [f.path for f in os.scandir(input_dir) if f.is_dir()]
    else:
        print("FITS files found in the input directory. So assume this is a directory of files.")
        input_dir_list = [input_dir]

    
    nightcat_script = os.path.join(os.path.dirname(__file__), 'mircx_mystic_nightcat.py')
    id_option = 'ARCHIVE'
    input_dir_list.sort()
    for directory in input_dir_list:
        raw_dir = directory
        mrx_dir = output_dir
        print("\033[1m\033[91mProcessing directory: "+raw_dir+"\033[0m")
        command = f"python {nightcat_script} --raw-dir={raw_dir} --mrx-dir={mrx_dir} --id={id_option}"
        os.system(command)
