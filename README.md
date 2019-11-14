[![Documentation](https://img.shields.io/badge/User%20Manual-Google%20Drive-blue)](https://docs.google.com/document/d/1zenNelkhVGTlm1v1tFRvtnb8i0EghIAUeX9F3t5asYU/edit)
[![Issues](https://img.shields.io/badge/Issues-Google%20Drive-yellow)](https://docs.google.com/spreadsheets/d/1u_0kam15HsIwaykv2pTbjCr-tc5QUNMNZxyS4CEOa1M/edit#gid=0)

## Description

This is data-quality pipeline of MIRCX.
We use it to explore the performances of the detectors.

## Requirements

python3 in anaconda3
some packages (all from anaconda3)

    conda install matplotlib
    conda install astropy
    conda install -c astropy astroquery

## Usage

Connect on orthanc.
    ssh orthanc.astro.lsa.umich.edu

Make sure python can look for the pipeline package

    export MIRCX_PIPELINE=/Users/Shared/
    export PYTHONPATH=$MIRCX_PIPELINE:$PYTHONPATH
    export PATH=$MIRCX_PIPELINE/mircx_pipeline/bin:$PATH

Go in the directory that you want to reduce:
 
    cd /nfs/Monnier2/MIRCX_DATA/MIRCX_2017Oct/

Run the reduction with default

    nohup mircx_reduce.py &
    tail -f nohup.out

Re-run the last step with more coherent integration
and different SNR threshold (for instance):

    nohup mircx_reduce.py --ncoherent=10 --snr-threshold=3 --oifits-dir=oifits_10 &
    tail -f nohup.out
  
Calibration script:

    cd oifits_nc10/
    mircx_calibrate.py --calibrators=HD_14055,0.51,0.03,HD_24398,0.70,0.03

Look at the results in 
  
    cd calibrated/
