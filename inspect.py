
def calTest(files, UDD, obj, outDir, uset3amp=False, fixUDD=True, detLim=True):
    """
     - files is a python list (can be of length one) of calibrated
       MIRC-X data files
     - uset3amp is a flag use to determine whether to use v2 and cp (False)
       or v2, cp and t3amp (True) in the fitting procedure.
     - fixUDD is a flag to determine whether the uniform disk diameter
       of the target is known and should be used. If False, the script
       compares the fitted UDD to the given value as an initial
       assessment of the goodness-of-fit of a no-point-source model.
     - UDD is the uniform disk diameter of the target (in mas)
     - detLim is a flag used to determine whether the detection limit
       step of CANDID is to be done.
    """
    import numpy as np
    try:
        import candid
    except ImportError as exception:
        print('ERROR: ', exception)
        if 'No module named' in str(exception):
            print('Download CANDID from https://github.com/amerand/CANDID')
            print('and install using:')
            print(' python setup.py install --user')
            print(' in CANDID directory')
        elif "cannot import name 'factorial'" in str(exception):
            print('CANDID is not python3 compatible (yet)')
            print(' ')
            print('To use CANDID within mircx_pipeline, you need to edit candid.py')
            print('Please replace the line which reads:')
            print('     from scipy.misc import factorial')
            print('with the following lines:')
            print('     try:')
            print('         from scipy.misc import factorial')
            print('     except ImportError:')
            print('         from scipy.special import factorial')
            print(' ')
            print('Then re-install CANDID by running')
            print('    python setup.py install --user')
            print('in your CANDID directory.')
            print(' ')
        return 'failed'
    
    try:
        import cyvis
    except ImportError:
        print('CANDID not installed. Use:')
        print(' python setup.py install --user')
        print(' in CANDID directory')
        return 'failed'
    
    import matplotlib.pyplot as plt
    
    o = candid.Open(files)
    if uset3amp == False:
        o.observables = ['v2', 'cp']
    
    if fixUDD == True:
        o.fitMap(fig=0, addParam={'diam*':float(UDD)}, doNotFit=['diam*'])
        plt.figure(0)
        plt.savefig(outDir+'/'+obj+'_fitMap_fixUDD.pdf')
        plt.close()
        plt.figure(1)
        plt.savefig(outDir+'/'+obj+'_Residuals_fitUDD.pdf')
        plt.close()
        ret = 'fixed'
    else:
        o.fitMap(fig=0)
        plt.figure(0)
        plt.savefig(outDir+'/'+obj+'_fitMap_fitUDD.pdf')
        plt.close()
        plt.figure(1)
        plt.savefig(outDir+'/'+obj+'_Residuals_fitUDD.pdf')
        plt.close()
        # comment on how this differs from mircx_target.list value
        fitUDD, eUDD = o.bestFit['best']['diam*'], o.bestFit['uncer']['diam*']
        if UDD < (fitUDD + eUDD) and UDD > (fitUDD - eUDD):
            ret = 'within range'
        else:
            ret = 'warning'
    
    if detLim == True:
        p = o.bestFit['best']
        candid.CONFIG['long exec warning'] = 900
        o.detectionLimit(fig=2, removeCompanion=p, methods=['injection'])
        plt.figure(2)
        plt.plot([o.rmin, o.rmax], [-2.5*np.log10(p['f']/100.)]*2, ls='--', color='k')
        plt.plot([np.sqrt(p['x']**2+p['y']**2)], [-2.5*np.log10(p['f']/100.)], ls=None, marker='*', ms=8)
        plt.savefig(outDir+'/'+obj+'_detLim.pdf')
        plt.close()
    
    return ret
