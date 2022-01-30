[![Documentation](https://img.shields.io/badge/User%20Manual-Google%20Drive-blue)](https://docs.google.com/document/d/1ATfNWk1vxQ2w8VuO9MtwtYRyxBanWkicTAwuxZjMcxQ/edit?usp=sharing)
[![Issues](https://img.shields.io/badge/Issues-Google%20Drive-yellow)](https://docs.google.com/spreadsheets/d/1slbbHa1sOJtk-tYtwehQXtcHqG3bGiHn_u7lhgTBkxA/edit?usp=sharing)

## Description

This is data-quality pipeline of MIRCX-MYSTIC, based on an initial version written mainly by Jean-Baptiste le Bouquin and hosted by gitlab.chara.gsu.edu.


## Requirements

python3 in anaconda3
some packages (all from anaconda3)

    conda install matplotlib
    conda install astropy
    conda install -c astropy astroquery

## Usage

Connect on computer with pipeline downloaded.
    ssh <computer-name>

Make sure python can look for the pipeline package

    export MIRCX_MYSTIC_PIPELINE=/Users/Shared/
    export PYTHONPATH=$MIRCX_MYSTIC_PIPELINE:$PYTHONPATH
    export PATH=$MIRCX_MYSTIC_PIPELINE/mircx_pipeline/bin:$PATH

Go in the directory where you want to run your reduction.
    cd /Volumes/DRIVE1/MIRCX_MYSTIC_REDUCTION/

Copy a version of mircx_mystic_reduce.py script and customize

Run the reduction with default

    nohup mircx_mystic_reduce.py &
    tail -f nohup.out

Re-run the last step with more coherent integration
and different SNR threshold (for instance):

    nohup mircx_mystic_reduce.py --ncoherent=10 --snr-threshold=3 --oifits-dir=oifits_10 &
    tail -f nohup.out
  
Calibration script:

    cd oifits_nc10/
    mircx_calibrate.py --calibrators=HD_14055,0.51,0.03,HD_24398,0.70,0.03

Look at the results in 
  
    cd calibrated/
