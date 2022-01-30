import glob, socket, os, sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits as pyfits
from matplotlib import cm

from . import headers, log, viscalib

def quickLook(oiDir, keys, setup, pltFile, st=0):
    """
    If you wish to plot all data for a specific target
    (irrespective of camera and reduction settings), use e.g
    quickLook(/full/path/to/data/directory, ['OBJECT'], 'HD_144', pltFile='all')
    -
    - oiDir is the directory containing oifits files;
    - keys is a python list of header keys;
    - setup is a python list of file header values
    which must match the 'keys' to be plotted;
    - pltFile is one of 'individual' (for use by the
    wrapper) or 'all' (to summarise all files matching
    setup);
    - st is required if pltFile=='individual' (acts as a
    sequential counter for file number).
    -
    """
    if 'calibrated' not in oiDir:
        fitsfiles = sorted(glob.glob(oiDir+'/*_oifits.fits'))
        suff = 'reduced'
    else:
        fitsfiles = sorted(glob.glob(oiDir+'/*_viscal.fits'))
        suff = 'calib'
    
    if st >= len(fitsfiles):
        return 'end of files reached'
    
    # select the fits files which match the setup
    if pltFile == 'individual':
        files, st = sortFits(fitsfiles, keys, setup, st)
    elif pltFile == 'all':
        files = []
        while st < len(fitsfiles):
            f, st = sortFits(fitsfiles, keys, setup, st)
            files = files + f
    
    if st > len(fitsfiles):
        return 'end of files reached'
    
    if files != []:
        plotV2CP(files, oiDir, suff, setup)
        if suff == 'reduced':
            fs_cal = ['/'.join(f.split('/')[:-1])+'/calibrated/'+f.split('/')[-1].replace('_oifits.fits','_oifits_viscal.fits') for f in files]
            files_cal = []
            for f in fs_cal:
                if os.path.isfile(f):
                    files_cal = files_cal + [f]
            if files_cal != []:
                plotV2CP(files_cal, oiDir+'/calibrated', 'calib', setup)
    else:
        #can end up here in batch mode if no calibrated data is
        #available for a target.
        return 'end of files reached'
    
    return st

def writeTable(oiDir, keys):
    """
    - oiDir if the full path to the directory containing 
    the oifits files.
    - keys is a python list of header keywords to be put
    in the table.
    """
    if 'calibrated' not in oiDir:
        fitsfiles = sorted(glob.glob(oiDir+'/*_oifits.fits'))
    else:
        fitsfiles = sorted(glob.glob(oiDir+'/*_viscal.fits'))
    
    hdrs = headers.loaddir(fitsfiles)
    with open(oiDir+'/quickLook_table.tex', 'w') as outtex:
        outtex.write('\\begin{longtable}\n    \\hline\n')
        outtex.write(' & '.join([str(k).replace('_','\\_') for k in keys])+' \\\\ \n')
        outtex.write('\\hline\n')
        # add entries to the table
        try:
            tabRows = [[str(h.get(k,'--')) for k in keys] for h in hdrs]
            for r in range(0, len(tabRows)-1):
                if r == 0:
                    outtex.write(' & '.join(str(s).replace('_','\\_') for s in tabRows[r])+'\\\\ \n')
                else:
                    lastrow = ' & '.join(str(s).replace('_','\\_') for s in tabRows[r-1])
                    thisrow = ' & '.join(str(s).replace('_','\\_') for s in tabRows[r])
                    if thisrow != lastrow:
                        outtex.write(' & '+thisrow+'\\\\ \n')
        except:
            something = 'went wrong'
            #print('Error: ',exception)
        outtex.write('    \\hline\n\\end{longtable}\n')
        outtex.write('\n')
    del hdrs

######
# Plotting functions:
######
def plotUV(direc):
    """
    Given a directory of calibrated files, locates all
    fits files for each science target and plots a
    full night's uv coverage plot for each of them.
        - dir is the directory of calibrated files
    """
    fitsfiles = sorted(glob.glob(direc+'/*_viscal.fits'))
    hdrs = headers.load(sorted(glob.glob(direc+'/*_viscal.fits')))
    objs = list(set([h['OBJECT'] for h in hdrs]))
    for item in ['NOSTAR', "", 'STS']:
        if item in objs:
            objs.remove(item)
    del hdrs
    for t in range(0, len(objs)):
        if not os.path.exists(direc+'/'+objs[t]+'_uv_coverage.png'):
            for f in range(0, len(fitsfiles)):
                with pyfits.open(fitsfiles[f]) as input:
                    if input[0].header['OBJECT'] == objs[t]:
                        lbd = input['OI_WAVELENGTH'].data['EFF_WAVE']*1e6
                        usf = 0.0-input['OI_VIS2'].data['UCOORD'][:,None]/lbd[None,:]
                        vsf = input['OI_VIS2'].data['VCOORD'][:,None]/lbd[None,:]
                        # trim u and v coordinates with nan vis2 data:
                        vis2 = input['OI_VIS2'].data['VIS2DATA']
                        flag = input['OI_VIS2'].data['FLAG']
                        usf[np.isfinite(vis2)==False] = np.nan
                        usf[flag] = np.nan
                        vsf[np.isfinite(vis2)==False] = np.nan
                        vsf[flag] = np.nan
                        # plot remaining data:
                        for u in range(15):
                            plt.plot(usf[u,:],vsf[u,:],marker='o',color='k',ls='None')
                            plt.plot(0.0-usf[u,:],0.0-vsf[u,:],marker='o',color='k',ls='None')
            ax = plt.gca()
            ax.set_ylabel('v [M$\lambda$]')
            ax.set_xlabel('u [M$\lambda$]')
            ax.set_ylim([-330./min(lbd), 330./min(lbd)])
            ax.set_xticks([-300, -200, -100, 0, 100, 200, 300])
            ax.set_xticklabels(['300', '200', '100', '0', '-100', '-200', '-300'])
            ax.set_xlim([-330./min(lbd), 330./min(lbd)])
            plt.axes().set_aspect(1)
            plt.title(objs[t])
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(direc+'/'+objs[t]+'_uv_coverage.png')
            plt.close()
            log.info('Save '+direc+'/'+objs[t]+'_uv_coverage.png')
            del lbd, usf, vsf
        else:
            log.info('File '+direc+'/'+objs[t]+'_uv_coverage.png already exists.')
    log.info('Cleanup memory')
    del objs
    return

def plotV2CP(files, outDir, suff, setup, withUV=False, cmap='winter'):
    # Load first file
    hdu    = pyfits.open(files[0])
    object = hdu[0].header['OBJECT']
    wave   = hdu['OI_WAVELENGTH'].data['EFF_WAVE']*1e6
    flag   = hdu['OI_VIS2'].data['FLAG']
    vis2   = hdu['OI_VIS2'].data['VIS2DATA']
    evis2  = hdu['OI_VIS2'].data['VIS2ERR']
    ucoord, vcoord = reshape_uv(hdu)
    saveAsStr = str(hdu[0].header['HIERARCH MIRC PRO RTS']).split('_')[0]
    
    flag2 = hdu['OI_T3'].data['FLAG']
    cp    = hdu['OI_T3'].data['T3PHI']
    ecp   = hdu['OI_T3'].data['T3PHIERR']
    
    spf, usf, vsf = get_spfreqs(hdu,'OI_VIS2')
    max_sf = np.max(get_spfreqs(hdu,'OI_T3'),axis=0)
    
    # Load other files
    for file in files[1:]:
        hdu    = pyfits.open(file);
        flag   = np.append(flag, hdu['OI_VIS2'].data['FLAG'], axis=0)
        vis2   = np.append(vis2, hdu['OI_VIS2'].data['VIS2DATA'], axis=0)
        evis2  = np.append(evis2, hdu['OI_VIS2'].data['VIS2ERR'], axis=0)
        ucoord = np.append(ucoord, reshape_uv(hdu)[0], axis=0)
        vcoord = np.append(vcoord, reshape_uv(hdu)[1], axis=0)
        spf    = np.append(spf, get_spfreqs(hdu,'OI_VIS2')[0], axis=0)
        usf    = np.append(usf, get_spfreqs(hdu,'OI_VIS2')[1], axis=0)
        vsf    = np.append(vsf, get_spfreqs(hdu,'OI_VIS2')[2], axis=0)
        
        flag2  = np.append(flag2, hdu['OI_T3'].data['FLAG'], axis=0)
        cp     = np.append(cp, hdu['OI_T3'].data['T3PHI'], axis=0)
        ecp    = np.append(ecp, hdu['OI_T3'].data['T3PHIERR'], axis=0)
        max_sf = np.append(max_sf, np.max(get_spfreqs(hdu,'OI_T3'),axis=0), axis=0)
    
    # Interpolate flagged values
    for item in [vis2, evis2, vsf, usf, ucoord, vcoord]:
        item[flag] = np.nan
    
    for item in [cp, ecp, max_sf]:
        item[flag2] = np.nan
    
    mask = evis2/vis2 > 0.28
    vis2[mask] = np.nan
    evis2[mask] = np.nan
    
    mask2 = ecp > 30.
    cp[mask2] = np.nan
    ecp[mask2] = np.nan
    
    mask3 = vis2 < 0.
    vis2[mask3] = np.nan
    evis2[mask3] = np.nan
    
    copcol = cm.get_cmap(cmap, len(wave))
    
    # Plot squared vis:
    figv2 = plt.figure(1, figsize=(6,4))
    axv2  = plt.subplot2grid((1, 1), (0, 0))
    for w in range(0, len(wave)):
        axv2.errorbar(spf[:,w],vis2[:,w],yerr=evis2[:,w],fmt='o',ms=1,color=copcol(w))

    axv2.set_xlim(0,230)
    axv2.set_ylim(0.0,1.0)
    axv2.set_xlabel('Baseline (M$\lambda$)')
    axv2.set_ylabel('Vis2')
    figv2.suptitle(', '.join(str(s) for s in setup))
    figv2.savefig(outDir+'/'+saveAsStr+'_'+suff+'_vis2.png')
    log.info('Written '+outDir+'/'+saveAsStr+'_'+suff+'_vis2.png')
    
    #
    # Plot vis
    #
    figV = plt.figure(2, figsize=(6,4))
    axV  = plt.subplot2grid((1, 1), (0, 0))
    for w in range(0, len(wave)):
        evis = 0.5*evis2[:,w]*np.power(vis2[:,w], -0.5)
        axV.errorbar(spf[:,w],np.sqrt(vis2[:,w]),yerr=evis,fmt='o',ms=1,color=copcol(w))
    
    axV.set_xlim(0,230)
    axV.set_ylim(0.0,1.0)
    axV.set_xlabel('Baseline (M$\lambda$)')
    axV.set_ylabel('Vis')
    figV.suptitle(', '.join(str(s) for s in setup))
    figV.savefig(outDir+'/'+saveAsStr+'_'+suff+'_vis.png')
    
    #
    # Plot CP vs Bmax
    #
    figCP = plt.figure(3, figsize=(6,4))
    axCP  = plt.subplot2grid((1, 1), (0, 0))
    axCP.hlines(0, 0,230, ls='--', color='grey')
    axCP.hlines(-200, 0,230, ls='--', color='w')
    axCP.hlines(200, 0,230, ls='--', color='w')
    for w in range(0, len(wave)):
        axCP.errorbar(max_sf[:,w],cp[:,w],yerr=ecp[:,w],fmt='o',ms=1,color=copcol(w))
    
    axCP.set_xlim=(0.,230.)
    axCP.set_xlabel('Max baseline (M$\lambda$)')
    axCP.set_ylabel('$\phi_{CP}$')
    figCP.suptitle(', '.join(str(s) for s in setup))
    figCP.savefig(outDir+'/'+saveAsStr+'_'+suff+'_t3phi.png')
    log.info('Written '+outDir+'/'+saveAsStr+'_'+suff+'_t3phi.png')
    
    if withUV != False:
        #
        # Plot UV plane (Recall: +u = East; +v = North)
        #
        figUVm = plt.figure(4, figsize=(4,4))
        axUVm  = plt.subplot2grid((1, 1), (0, 0))
        axUVm.plot(ucoord,vcoord,'o',color='b')
        axUVm.plot(-ucoord,-vcoord,'o',color='b')
        axUVm.set_xlim(-330,330)
        axUVm.set_ylim(-330,330)
        axUVm.set_xlabel('u (m)') # $\longleftarrow$ East
        axUVm.set_ylabel('v (m)') # North $\longrightarrow$
        axUVm.set_aspect(1)
        axUVm.grid(True)
        figUVm.tight_layout()
        figUVm.suptitle(', '.join(str(s) for s in setup))
        figUVm.savefig(outDir+'/'+saveAsStr+'_uv_m.png')
    return

def sortFits(fitsfiles, keys, setup, st=0):
    """
    Return a subset of fits files matching set header keywords
     - fitfiles is a python list of file paths
     - keys is a list of header keywords
     - setup is the required values of 'keys'
     - st is the list index of fitsfiles to start sorting from
    """
    first = True
    files = []
    for f in range(st, len(fitsfiles)):
        with pyfits.open(fitsfiles[f]) as input:
            if [str(input[0].header.get(k,'--')) for k in keys] == setup and first == True:
                if fitsfiles[f] == fitsfiles[-1]:
                    st = f+1
                else:
                    files.append(fitsfiles[f])
                    first = False
            elif [str(input[0].header.get(k,'--')) for k in keys] == setup and first != True:
                files.append(fitsfiles[f])
                if fitsfiles[f] == fitsfiles[-1]:
                    st = f
            elif [str(input[0].header.get(k,'--')) for k in keys] != setup and first != True:
                if fitsfiles[f] == fitsfiles[-1]:
                    st = f+1
                else:
                    st = f
                break
    if files == []:
        return files, len(fitsfiles)+1
    return files, st

def reshape_uv(hdu):
    '''
    Return the u and v coordinates in the same shape as vis2
    '''
    
    dumY = np.ones(np.shape(hdu['OI_WAVELENGTH'].data['EFF_WAVE']))
    u = 0.0-hdu['OI_VIS2'].data['UCOORD']
    v = hdu['OI_VIS2'].data['VCOORD']
    return [u[:,None]/dumY[None,:],v[:,None]/dumY[None,:]]

def get_spfreqs(hdu,name):
    '''
    Return the spatial frequency B/lbd in M$\lambda
    '''
    lbd = hdu['OI_WAVELENGTH'].data['EFF_WAVE']*1e6
    
    if name == 'OI_VIS2':
        u = hdu['OI_VIS2'].data['UCOORD']
        # the sign doesn't matter here because things are squared
        v = hdu['OI_VIS2'].data['VCOORD']
        return [np.sqrt(u**2+v**2)[:,None]/lbd[None,:],-u[:,None]/lbd[None,:],v[:,None]/lbd[None,:]]
    
    if name == 'OI_T3':
        u1 = hdu['OI_T3'].data['U1COORD']
        v1 = hdu['OI_T3'].data['V1COORD']
        u2 = hdu['OI_T3'].data['U2COORD']
        v2 = hdu['OI_T3'].data['V2COORD']
        u = np.array([u1,u2,u1+u2])
        v = np.array([v1,v2,v1+v2])
        return np.sqrt (u**2 + v**2)[:,:,None] / lbd[None,None,:]


######
# LaTex writing functions:
######
def texSumTitle(oiDir,hdrs,redF,calF):
    """
    Produce header section of tex files containing reduction
    and calibration summary.
        - oiDir is the directory containing the products of the
        oifits reduction step;
        - hdrs is the raw data headers;
        - redF and calF are flags for if the reduction
        and/or calibration process failed;
    """
    # oiDir has the form redBase+/date_nbsXncsXbbiasXmitp/snrXfthXmitoX/oifits_ncX
    ncoh = oiDir.split('oifits_nc')[-1]
    ncs  = oiDir.split('/')[-3].split('ncs')[-1].split('bbias')[0]
    nbs  = oiDir.split('/')[-3].split('nbs')[-1].split('ncs')[0]
    bb   = oiDir.split('/')[-3].split('bbias')[-1].split('mitp')[0]
    snr  = oiDir.split('/')[-2].split('snr')[-1].split('fth')[0]
    fth  = oiDir.split('/')[-2].split('fth')[-1].split('mito')[0]
    dates = oiDir.split('/')[-3].split('_')[0]
    
    for item in snr, fth:
        item = item.replace('p','.')
    auth = 'ncohrent='+ncoh+'; ncs='+ncs+'; nbs='+nbs+'; snr\\_thresh='+snr.replace('p','.')+'; flux\\_thresh='+fth.replace('p','.')+'; bbias='+bb
    suf1 = oiDir.split('/')[-3]
    suf2 = oiDir.split('/')[-2]+oiDir.split('_')[-1].replace('nc','ncoh')
    direc = '/'.join(oiDir.split('/')[:-2])
    outFiles = [direc+'/report_'+suf1+suf2+'.tex',direc+'/summary_'+suf1+suf2+'.tex']
    # ^-- outFiles[0] is not to be emailed. It exceeds the 10MB gmail attachment limit.
    # outFiles[1] is a smaller file containing a night log summary as well as uv 
    # coverage plots for sci targets and vis2 vs sf and CP vs max_sf plots for reduced
    # and calibrated (where applicable) data.
    for outFile in outFiles:
        with open(outFile, 'w') as outtex:
            outtex.write('\\documentclass[a4paper]{article}\n\n')
            outtex.write('\\usepackage{fullpage}\n\\usepackage{amsmath}\n')
            outtex.write('\\usepackage{hyperref}\n\\usepackage{graphicx}\n')
            outtex.write('\\usepackage{longtable}\n')
            outtex.write('\\usepackage[left=1.5cm,right=1.5cm,top=2cm,bottom=2cm]')
            outtex.write('{geometry}\n\n')
    with open(outFiles[0], 'a') as outtex:
        outtex.write('\\title{Summary report from mircx\\_redcal\\_wrap.py}\n')
    with open(outFiles[1], 'a') as outtex:
        outtex.write('\\title{Brief summary report from mircx\\_redcal\\_wrap.py}\n')
    for outFile in outFiles:
        with open(outFile, 'a') as outtex:
            outtex.write('\\author{'+auth+'}\n\\date{'+dates+'}\n\n')
            outtex.write('\\begin{document}\n\n\\maketitle\n\n')
            if redF == False:
                outtex.write('\\subsubsection*{Reduced files located in: ')
                outtex.write(direc.replace('_','\\_')+' on ')
                outtex.write(socket.gethostname()+'}\n')
                if calF == True:
                    outtex.write('\\subsubsection*{Calibration failed}\n')
            else:
                outtex.write('\\subsubsection*{Reduction failed}\n')
            try:
                princInv = list(set([h['PI_NAME'] for h in hdrs]))
            except:
                princInv = ['UNKNOWN']
            try:
                princInv.remove('UNKNOWN')
            except:
                statement = 'no PI_NAMES match UNKNOWN'
            log.info('Recovered PI NAMES '+'; '.join(princInv)+' from headers')
            outtex.write('\n\\subsubsection*{PI(s): '+'; '.join(princInv).replace('_',' ')+'}\n')
            outtex.write('\\subsubsection*{Observer(s): ')
            try:
                obsPerson = list(set([h['OBSERVER'] for h in hdrs]))
                outtex.write('; '.join(obsPerson))
                log.info('Recovered OBSERVERS '+'; '.join(obsPerson).replace('_',' ')+' from headers')
            except:
                outtex.write('(info not recovered from header)')
                log.error('No OBSERVER keyword in fits headers')
            outtex.write('}\n')
            outtex.write('\\subsubsection*{Program ID(s): ')
            try:
                progID = list(set([h['PROGRAM'] for h in hdrs]))
            except:
                progID = ['UNKNOWN']
            try:
                progID.remove('UNKNOWN')
            except:
                statement = 'no PROGRAM match UNKNOWN'
            log.info('Recovered PROGRAM '+'; '.join(progID)+' from headers')
            outtex.write('; '.join(progID))
            outtex.write('}\n')
    return outFiles

def texTargTable(targs,calInf,redF,outFiles):
    """
    Append target summary table to existing summary tex files.
        - targs is a python list of target names;
        - calInf is a string of information which was
        parsed to mircx_reduce.py;
        - redF is a flag highlighting whether the reduction
        was successful;
    """
    try:
        from astroquery.vizier import Vizier;
        log.info('Load astroquery.vizier');
        from astroquery.simbad import Simbad;
        log.info('Load astroquery.simbad');
    except:
        log.warning('Cannot load astroquery')
        log.warning('H-magnitude will not be able to be tabulated')
    
    for outFile in outFiles:
        with open(outFile, 'a') as outtex:
            outtex.write('\\subsection*{Target summary}\n')
            outtex.write('\\begin{longtable}{p{.25\\textwidth} | p{.08\\textwidth} | ')
            outtex.write('p{.25\\textwidth} | p{.08\\textwidth}}\n    \\hline\n')
            outtex.write('    Target ID & used as & UD diam. for CALs (mas) & H-mag \\\\ \n')
            outtex.write('    \\hline\n')
            for targ in targs:
                try:
                    result = Vizier.query_object(targ, catalog=['II/346'])
                    ind = -999
                    alt_ids = Simbad.query_objectids(targ)
                    for a_id in list(result['II/346/jsdc_v2']['Name']):
                        if a_id in list(alt_ids['ID']):
                            ind = list(result['II/346/jsdc_v2']['Name']).index(a_id)
                        elif a_id in list([a.replace(' ', '') for a in alt_ids['ID']]):
                            ind = list(result['II/346/jsdc_v2']['Name']).index(a_id)
                    if ind == -999:
                        hmag = '--'
                    else:
                        hmag = str(result["II/346/jsdc_v2"]["Hmag"][ind])
                except:
                    hmag = '--'
                    log.warning('H band magnitude not retrieved from JSDC')
                try:
                    ud_H = calInf.split(',')[calInf.split(',').index(targ.replace(' ','_'))+1]
                    eud_H = calInf.split(',')[calInf.split(',').index(targ.replace(' ','_'))+2]
                    outtex.write('    '+targ.replace('_', ' ')+' & CAL')
                    outtex.write(' & $'+ud_H+'\\pm'+eud_H+'\\,$ & '+hmag+' \\\\ \n')
                except ValueError:
                    outtex.write('    '+targ.replace('_', ' ')+' & SCI')
                    outtex.write(' &  & '+hmag+' \\\\ \n')
            outtex.write('    \\hline\n\\end{longtable}\n')
            outtex.write('\n')

def texReducTable(oiDir,redF,outFiles):
    """
    Append reduced data summary table to existing summary tex files.
        - oiDir is the directory containing the products of the
        oifits reduction step;
        - redF is a flag highlighting whether the reduction
        was successful;
    """
    if redF == False:
        # specifying the inclusion only of *_oifits.fits files ensures that BG and FG
        # files are not summarised.
        redhdrs = headers.load(sorted(glob.glob(oiDir+'/*_oifits.fits')))
    for outFile in outFiles:
        with open(outFile, 'a') as outtex:
            outtex.write('\\subsection*{Reduced data summary}\n')
            outtex.write('{\\fontsize{7pt}{7pt}\n \\selectfont\n')
            outtex.write(' \\begin{longtable}{p{.03\\textwidth} | p{.06\\textwidth} | ')
            outtex.write('p{.04\\textwidth} | p{.20\\textwidth} | p{.03\\textwidth} | ')
            outtex.write('p{.03\\textwidth} | p{.03\\textwidth} | p{.03\\textwidth} | ')
            outtex.write('p{.06\\textwidth} | p{.06\\textwidth} | p{.10\\textwidth}} \n')
            outtex.write('    \\hline\n')
            outtex.write('    & Start & File & Target & Gain & Nco & Nps & Frm & ')
            outtex.write('Filter1 & Filter2 & Config \\\\ \n')
            outtex.write('    & (UTC) & num. & & & & & $/$rst & & & \\\\ \n    \\hline\n')
            keys = ['DATE','HIERARCH MIRC PRO RTS','OBJECT','GAIN','NCOHER','PSCOADD','FRMPRST','FILTER1','FILTER2','CONF_NA']
            try:
                tabRows = [[str(h.get(k,'--')) for k in keys] for h in redhdrs]
                for row in tabRows:
                    row[0] = row[0].split('T')[1]
                    row[1] = row[1].split('/')[-1].split('mircx')[1].split('_')[0]
                skipd = 17
                for r in range(0, len(tabRows)-1):
                    if r == 0:
                        outtex.write('        '+str(r)+' & '+' & '.join(str(s).replace('_',' ') for s in tabRows[r])+'\\\\ \n')
                    else:
                        nextrow = ' & '.join(str(s).replace('_',' ') for s in tabRows[r+1])
                        thisrow = ' & '.join(str(s).replace('_',' ') for s in tabRows[r])
                        if nextrow[skipd:] != thisrow[skipd:]:
                            outtex.write('        '+str(r)+' & '+nextrow+'\\\\ \n')
                        del nextrow, thisrow
                del tabRows
                log.info('Cleanup memory')
            except:
                skipd = 11
            outtex.write('    \\hline\n\\end{longtable}\n}\n')
    if redF == False:
        del redhdrs
    return

def texReportPlts(oiDir,outFiles,d):
    """
    Locates the report files output by mircx_report.py
    which check the camera performance and includes
    them into the tex summary and report PDF files.
        - oiDir is the directory containing the products of the
        oifits reduction step;
    """
    reportFiles = glob.glob(oiDir+'/report*.png')
    transPlots   = glob.glob('/'.join(oiDir.split('/')[:-3])+'/*transmission_*'+d+'.png')
    # NB: transmission plots will not appear if 'd' is not the most recent night that has been reduced
    for outFile in outFiles:
        with open(outFile, 'a') as outtex:
            if len(reportFiles) == 0 and len(transPlots) == 0:
                outtex.write('\\subsubsection*{No outputs from mircx\\_report.py available')
                outtex.write(' to show} \n')
            else:
                outtex.write('\\newpage \n')
            r = 0
            while r < len(reportFiles):
                outtex.write('\\begin{figure}[h]\n    \\raggedright\n')
                outtex.write('    \\textbf{Results from mircx\\_report.py for '+d)
                if r == 0:
                    outtex.write('}\\\\ \n')
                else:
                    outtex.write(' (cont.) }\\\\ \n')
                outtex.write('    \\centering\n')
                # r is the top plot, r+1 is the bottom plot
                outtex.write('    \\includegraphics[trim=0.0cm 0.8cm 0.0cm 0.2cm, ')
                outtex.write('clip=true, width=0.8\\textwidth]{'+reportFiles[r]+'}\n')
                log.info('Added '+reportFiles[r]+' to summary report PDF')
                try:
                    x = reportFiles[r+1]
                    outtex.write('    \\includegraphics[trim=0.0cm 0.8cm 0.0cm 0.2cm, ')
                    outtex.write('clip=true, width=0.8\\textwidth]{'+reportFiles[r+1]+'}')
                    outtex.write(' \n')
                    log.info('Added '+reportFiles[r+1]+' to summary report PDF')
                except IndexError:
                    log.info('End of report files reached')
                r += 2
                outtex.write('\\end{figure}\n\n')
            r = 0
            try:
                x = transPlots[0]
                outtex.write('\\begin{figure}[h]\n    \\raggedright\n')
                outtex.write('    \\textbf{Results from mircx\\_report.py for '+d)
                outtex.write(' (cont.) }\\\\ \n')
                outtex.write('    \\centering\n')
                outtex.write('    \\includegraphics[trim=0.0cm 0.2cm 0.0cm 0.2cm, ')
                outtex.write('clip=true, width=0.8\\textwidth]{'+x+'}')
                outtex.write(' \n')
                log.info('Added transmission plot to summary report PDF')
                outtex.write('\\end{figure}\n\n')
            except:
                log.error('No transmission plot found for '+d)
    return

def texSumUV(oiDir,calF,outFiles):
    """
    If the calibration process was succesful, the 
    uv coverage plots for the science targets will
    be appended to the summary PDF files.
        - oiDir is the directory containing the products of the
        oifits reduction step;
        - calF is a flag highlighting whether the calibration
        failed (True means that the calibration did fail - sorry
        for confusing logic);
    """
    if calF == False:
        # locate the calibrated files that have been produced:
        uvPlt = glob.glob(oiDir+'/calibrated/*_uv_coverage.png') 
        for outFile in outFiles:
            with open(outFile, 'a') as outtex:
                outtex.write('\\newpage\n\n\\begin{figure}[h]\n    \\raggedright\n')
                outtex.write('    \\textbf{Full night $uv$-coverage for SCI target(s)}')
                outtex.write('\\\\ \n    \\centering\n')
                for uvp in uvPlt[0:12]:
                    outtex.write('    \\includegraphics[trim=2.0cm 0.0cm 2.0cm 0.0cm, ')
                    outtex.write('clip=true, width=0.32\\textwidth]{'+uvp+'}\n')
                if len(uvPlt) > 12:
                    # if more than 12 science targets have been observed, the page
                    # will not be big enough to host them all. This part of the script
                    # handles the required page break.  
                    for n in range(1, int(np.ceil(len(uvPlt)/12.))):
                        outtex.write('\\end{figure}\n\n\\clearpage\n')
                        outtex.write('\\begin{figure}[h]\n')
                        outtex.write('    \\raggedright\n')
                        outtex.write('    \\textbf{Cont.}\\\\ \n    \\centering\n')
                        for uvp in uvPlt[12*n:12*(n+1)]:
                            outtex.write('    \\includegraphics[trim=2.0cm 0.0cm 2.0cm')
                            outtex.write(' 0.0cm, clip=true, width=0.32\\textwidth]{')
                            outtex.write(uvp+'}\n')
                outtex.write('\\end{figure}\n\n\\newpage\n')
    return

def getFileNum(file):
    return int(file.split('/')[-1].split('_')[0].split('x')[1])

def texSumPlots(oiDir,redF,calF,outFiles,calIDs):
    """
    Appends vis2 and CP plots to the summary files.
    Whether any files are appended depends on if
    the reduction process succeeded. Whether the
    calibrated vis2 and CP plots are also appended
    depends on whether the calibration succeeded.
        - dir is the directory containing this night's 
        reduced data;
        - redF and calF are flags highlighting whether
        the reduction and calibration were successful,
        respectively;
        - calIDs is a python list of calibrator names.
    """
    if redF == True:
        # catches instances where the reduction failed so there are no outputs to display
        for outFile in outFiles:
            with open(outFile, 'a') as outtex:
                outtex.write('\\end{document}\n')
        return
    # sort the reduced files by camera settings and target, ensuring that FG and BG files
    # are ignored:
    redhdrs = headers.load(sorted(glob.glob(oiDir+'/*_oifits.fits')))
    log.info('Retrieve targets and camera settings from successfully reduced files')
    keys = ['OBJECT','GAIN','NCOHER','PSCOADD','FRMPRST','FILTER1','FILTER2','CONF_NA']
    setupL = [[str(h.get(k,'--')) for k in keys] for h in redhdrs]
    del redhdrs
    setups, setupsP = [],[]
    setups.append(setupL[0])
    setupsP.append(setupL[0])
    log.info('Targets and camera settings:')
    for m in range(0, len(setupL)-1):
        if setupL[m+1] not in setupsP:
            setupsP.append(setupL[m+1])
        if setupL[m+1] != setupL[m]:
            setups.append(setupL[m+1])
    # make reduced and calibrated vis2 and CP plots
    log.info('Plotting data in '+oiDir+' split according to '+','.join(keys))
    for sup in range(0, len(setupsP)):
        log.info(' '+', '.join(str(s) for s in setupsP[sup]))
        # reduced:
        st = 0
        while isinstance(st, (float, int)):
            st = quickLook(oiDir, keys, setupsP[sup], pltFile='individual', st=st)
    # Read in mircx numbers of vis2 plots created in reduced and calibrated directories:
    redPlts = sorted(glob.glob(oiDir+'/*reduced_vis2.png'))
    redNum = [int(i.split('/')[-1].split('_')[0].split('x')[1]) for i in redPlts]
    RTS_p  = sorted(glob.glob('/'.join(oiDir.split('/')[:-2])+'/rts/*datarts_psd.png')) # e.g. mircx00000_datarts_psd.png or mircx00000_foregroundrts_psd.png or mircx00000_backgroundrts_psd.png
    SNR_p  = sorted(glob.glob(oiDir+'/*_oifits_snr.png'))
    for num in range(0, len(redNum)):
        # ensure correct number of leading zeros are added to redNum for file name:
        strnum = '0'*(5-len(str(redNum[num])))+str(redNum[num])
        log.info('Gather plots for summary report for file mircx'+strnum)
        redV2plt = oiDir+'/mircx'+strnum+'_reduced_vis2.png'
        redCPplt = oiDir+'/mircx'+strnum+'_reduced_t3phi.png'
        for outFile in outFiles:
            with open(outFile, 'a') as outtex:
                outtex.write('\n\\begin{figure}[h]\n    \\raggedright\n')
                outtex.write("    \\textbf{"+', '.join(str(n).replace('_',' ') for n in setups[num])+"}")
                outtex.write('\\\\ \n    \\centering\n')
                outtex.write('    \\textbf{Reduced vis2 and CP:}\\\\ \n')
                outtex.write('    \\includegraphics[trim=0.0cm 0.2cm 0.0cm 0.0cm, ')
                outtex.write('clip=true, width=0.32\\textwidth]{'+redV2plt+'}\n')
                outtex.write('    \\includegraphics[trim=0.0cm 0.2cm 0.0cm 0.0cm,')
                outtex.write(' clip=true, width=0.32\\textwidth]{'+redCPplt+'}\\\\ \n')
                if os.path.isfile(oiDir+'/calibrated/mircx'+strnum+'_calib_vis2.png'):
                    outtex.write('    \\textbf{Calibrated vis2 and CP:}\\\\\n')
                    outtex.write('    \\includegraphics[trim=0.0cm 0.2cm 0.0cm 0.0cm, ')
                    outtex.write('clip=true, width=0.32\\textwidth]{')
                    outtex.write(oiDir+'/calibrated/mircx'+strnum+'_calib_vis2.png')
                    outtex.write('}\n')
                    outtex.write('    \\includegraphics[trim=0.0cm 0.2cm 0.0cm 0.0cm,')
                    outtex.write(' clip=true, width=0.32\\textwidth]{')
                    outtex.write(oiDir+'/calibrated/mircx'+strnum+'_calib_t3phi.png')
                    outtex.write('}\\\\ \n')
                outtex.write('\\end{figure}\n\n')
        if int(float(num/30.)) == float(num/30.) and int(float(num)) != 1:
            with open(outFile, 'a') as outtex:
                outtex.write('\\clearpage\n') # avoids latex 'too many unprocessed floats' error
    
    # Then append reduction QA plots and CANDID plots to the file that won't be emailed:
    with open(outFiles[0], 'a') as outtex:
         outtex.write('\n\\clearpage\n\n\\newpage\n\\begin{figure}[h]\n    \\raggedright\n')
         outtex.write('    \\textbf{Reduction quality assessment: PSD}\\\\ \n')
         outtex.write('    \\centering\n')
         for rts in RTS_p[0:15]:
             outtex.write('    \\includegraphics[trim=0.7cm 0.9cm 1.5cm 0.0cm, ')
             outtex.write('clip=true, width=0.32\\textwidth]{'+rts+'}\n')
         if len(RTS_p) > 15:
             for n in range(1, int(np.ceil(len(RTS_p)/15.))):
                 outtex.write('\\end{figure}\n\n\\clearpage\n')
                 outtex.write('\\begin{figure}[h]\n')
                 outtex.write('    \\raggedright\n    \\textbf{Cont.}\\\\ \n')
                 outtex.write('    \\centering\n')
                 for rts in RTS_p[15*n:15*(n+1)]:
                     outtex.write('    \\includegraphics[trim=0.7cm 0.9cm 1.5cm 0.0cm')
                     outtex.write(', clip=true, width=0.32\\textwidth]{'+rts+'}\n')
         outtex.write('\\end{figure}\n\n')
         outtex.write('\\newpage\n\\begin{figure}[h]\n    \\raggedright\n')
         outtex.write('    \\textbf{Reduction quality assessment: SNR}\\\\ \n')
         outtex.write('    \\centering\n')
         for snr in SNR_p[0:6]:
             # include the first 6 snr plots in the report file if they exist
             outtex.write('    \\includegraphics[width=0.49\\textwidth]{'+snr+'}\n')
         if len(SNR_p) > 6:
             for n in range(1, int(np.ceil(len(SNR_p)/6.))):
                 # include the remainder of the snr plots in the report file (if they exist)
                 outtex.write('\\end{figure}\n\n\\clearpage\n')
                 outtex.write('\\begin{figure}[h]\n')
                 outtex.write('    \\raggedright\n    \\textbf{Cont.}\\\\ \n')
                 outtex.write('    \\centering\n')
                 for snr in SNR_p[6*n:6*(n+1)]:
                     outtex.write('    \\includegraphics[width=0.49\\textwidth]{'+snr+'}\n')
         outtex.write('\\end{figure}\n\n')
         outtex.write('\\newpage\n\\begin{figure}[h]\n    \\raggedright\n')
         outtex.write('    \\textbf{Reduction quality assessment: base trend}\\\\ \n')
         outtex.write('    \\centering\n')
         for ba in SNR_p[0:6]:
             # include the first 6 base_trend plots if they exist
             if os.path.isfile(ba.replace('_oifits_snr', '_oifits_base_trend')):
                 outtex.write('    \\includegraphics[trim=2.2cm 0.9cm 1.5cm 0cm, clip=')
                 outtex.write('true, width=0.49\\textwidth]{')
                 outtex.write(ba.replace('_oifits_snr', '_oifits_base_trend'))
                 outtex.write('}\n')
         if len(SNR_p) > 6:
             for n in range(1, int(np.ceil(len(SNR_p)/6.))):
                 outtex.write('\\end{figure}\n\n\\clearpage\n')
                 outtex.write('\\begin{figure}[h]\n')
                 outtex.write('    \\raggedright\n    \\textbf{Cont.}\\\\ \n')
                 outtex.write('    \\centering\n')
                 for ba in SNR_p[6*n:6*(n+1)]:
                     # include the remaining base_trend plots if they exist
                     if os.path.isfile(ba.replace('_oifits_snr', '_oifits_base_trend')):
                         outtex.write('    \\includegraphics[trim=2.2cm 0.9cm 1.5cm 0cm, ')
                         outtex.write('clip=true, width=0.49\\textwidth]{')
                         outtex.write(ba.replace('_oifits_snr', '_oifits_base_trend'))
                         outtex.write('}\n')
         outtex.write('\\end{figure}\n\n')
         #
         # Find CANDID outputs and include them:
         #
         for calID in calIDs:
             fitMap_plt = glob.glob(oiDir+'/'+calID+'_fitMap_fitUDD.pdf')
             detLim_plt = glob.glob(oiDir+'/'+calID+'_detLim.pdf')
             resid_plt  = glob.glob(oiDir+'/'+calID+'_Residuals_fitUDD.pdf')
             outtex.write('\\clearpage\\newpage\n\\begin{figure*}[h]\n    \\raggedright\n')
             outtex.write('    \\textbf{CANDID output: fitMap with free UDD for ')
             outtex.write(calID.replace('_',' ')+'}\\\\ \n    \\centering\n')
             try:
                 outtex.write('    \\includegraphics[width=0.9\\textwidth]{'+fitMap_plt[0]+'}\n')
             except IndexError:
                 log.info('No fitMap plot found for '+calID)
             try:
                 outtex.write('    \\includegraphics[width=0.9\\textwidth]{'+resid_plt[0]+'}\n')
             except IndexError:
                 log.info('No residuals plot found for '+calID)
             outtex.write('\\end{figure*}\n\n\\clearpage\n')
             outtex.write('\\newpage\n\\begin{figure*}[h]\n    \\raggedright\n')
             outtex.write('    \\textbf{CANDID output: detectionLimit for ')
             outtex.write(calID.replace('_',' ')+'}\\\\ \n    \\centering\n')
             try:
                 outtex.write('    \\includegraphics[width=0.9\\textwidth]{'+detLim_plt[0]+'}\n')
             except IndexError:
                 log.info('No detLim plot found for '+calID)
             outtex.write('\\end{figure*}\n\n\\clearpage\n')
    
    for outFile in outFiles:
        with open(outFile, 'a') as outtex:
            outtex.write('\\end{document}\n')
    return

