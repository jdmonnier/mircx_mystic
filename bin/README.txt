README.txt

This file just helps to keep track of how to run pipeline v2.

STEP 1. must specify the raw_dir or a previously run SUMMARY dir.
mircx_mystic_nightcat.py

mircx_mystic_nightcat.py --raw-dir=/path/to/raw/data/ --mrx_dir=/path/to/reduced/data/ -id=JDM2022Jan04

Note that the _blocks,_headers, and json files are regenerated if missing.


STEP 2. must specify only previously run SUMMARY dir.

# validate
Check for bad files, frames
Check for wrong shutters
Check for fringes in foregrounds (or later?)
Regenerate blocks and headers csv and add on to log

# cleanup
Calculate the non-linearity corrections
Identify bad pixels from ensemble of backgrounds, skies. One bad pixel map for whole night...
Remove interference
Fix non-linear corrections
Save preproc
Estimate flat field from foregrounds (not its)



mircx_mystic_preproc.py


mircx_mystic_checkshutters.py


