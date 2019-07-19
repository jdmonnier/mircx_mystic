import glob, socket, os, sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits as pyfits

from . import headers, log, viscalib

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
    if 'NOSTAR' in objs:
        objs.remove('NOSTAR')
    if '' in objs:
        objs.remove('')
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
                        usf[np.isfinite(vis2)==False] = np.nan
                        vsf[np.isfinite(vis2)==False] = np.nan
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
    del hdrs, objs
    return

def addV2CP(input, viscp, fig, axes):
    """
    Reads in interferometric data from 'input' and adds vis2
    or cp to figure 'fig' with axes 'axes'
    """
    # Then read in the data from the fits file:
    sf = viscalib.get_spfreq(input,'OI_VIS2')
    vis2 = input['OI_VIS2'].data['VIS2DATA']
    evis = input['OI_VIS2'].data['VIS2ERR']
    max_sf = np.max(viscalib.get_spfreq(input,'OI_T3'),axis=0)
    cp = input['OI_T3'].data['T3PHI']
    ecp = input['OI_T3'].data['T3PHIERR']
    if viscp == 'vis':
        for b in range(15):
            # Plot data if the errors on the non-extreme (in wavelength) values
            # is below 28%.
            checkVals = evis[b,1:-1]/vis2[b,1:-1]
            if all(checkVals < 0.28):
                axes.errorbar(1e-6*sf[b,:],vis2[b,:],yerr=evis[b,:],marker='o',ms=1)
    elif viscp == 'cp':
        for b in range(20):
            # Plot data if the t3phi errors on the non-extreme (in wavelength) values
            # are below 30 degrees
            if not any(ecp[b,1:-1] > 30.):
                axes.errorbar(1e-6*max_sf[b,:],cp[b,:],yerr=ecp[b,:],marker='o',ms=1)
    return

def calibPlots(calibfiles,viscp,saveAsStr,setup):
    """
    Plots the calibrated files corresponding to each 
    setup (if they exist).
    """
    first = True
    for file in calibfiles:
        f = '/'.join(file.split('/')[:-1])+'/calibrated/'+file.split('/')[-1].replace('.fits','_viscal.fits')
        if os.path.isfile(f):
            # calibrated file exists
            if first == True:
                fig,axes = plt.subplots()
                fig.suptitle(', '.join(str(s) for s in setup))
                with pyfits.open(f) as input:
                    addV2CP(input, viscp, fig, axes)
                first = False
            else:
                with pyfits.open(f) as input:
                    addV2CP(input, viscp, fig, axes)
    if first == False:
        axes.set_xlim(0.,225.)
        if viscp == 'vis':
            axes.set_ylim(-0.1,1.2)
            axes.set_xlabel('sp. freq. (M$\lambda$)')
            axes.set_ylabel('vis2')
            plt.savefig('/'.join(f.split('/')[:-1])+'/'+saveAsStr+'_calib_vis2.png')
            plt.close()
            log.info('    - Write '+'/'.join(f.split('/')[:-1])+'/'+saveAsStr+'_calib_vis2.png')
        elif viscp == 'cp':
            axes.set_xlabel('max sp. freq. (M$\lambda$)');
            axes.set_ylabel('$\phi_{CP}$')
            axes.set_ylim(-200,200)
            plt.savefig('/'.join(f.split('/')[:-1])+'/'+saveAsStr+'_calib_t3phi.png')
            plt.close()
            log.info('    - Write '+'/'.join(f.split('/')[:-1])+'/'+saveAsStr+'_calib_t3phi.png')
        plt.close("all")
    return

def plotV2CP(oiDir,setups,viscp):
    """
    Searches a directory for fits files and plots vis vs sf
    and CP vs max_sf for all the files found. The files are
    grouped together by the target name and the camera 
    settings as defined in 'setup'.
        - oiDir is the directory containing the products of the
        oifits reduction step;
        - setups defines the object name and the camera 
        settings;
        - viscp is either 'vis' or 'cp' to decide what is
        plotted.
    """
    if 'calibrated' not in oiDir:
        fitsfiles = sorted(glob.glob(oiDir+'/*_oifits.fits'))
    else:
        fitsfiles = sorted(glob.glob(oiDir+'/*_viscal.fits'))
    if  'oifits' in oiDir.split('/')[-1]:
        suff = 'reduced'
    p, first = 0, True
    calibfiles = []
    for file in fitsfiles:
        # keywords from file headers read in
        with pyfits.open(file) as input:
            keys = ['OBJECT','GAIN','NCOHER','PSCOADD','FRMPRST','FILTER1']
            teststr = [str(input[0].header.get(k,'--')) for k in keys]
            if teststr == setups[p] and first == True:
                # option i) file matches current setup and is first file to match it
                log.info('    - '+file+' matches setup '+', '.join(str(s) for s in teststr))
                fig,axes = plt.subplots()
                fig.suptitle(', '.join(str(s) for s in teststr))
                saveAsStr = str(input[0].header['HIERARCH MIRC PRO RTS']).split('_')[0]
                addV2CP(input, viscp, fig, axes)
                calibfiles.append(file)
                first = False
            elif teststr == setups[p] and first != True:
                # option ii) file matches current setup but is not first file to match it
                log.info('    - '+file+' also matches setup '+', '.join(str(s) for s in teststr))
                addV2CP(input, viscp, fig, axes)
                calibfiles.append(file)
            elif teststr != setups[p]:
                # option iii) file doesn't match current setup at all:
                if first != True:
                    # if there is data plotted already, close the plot
                    axes.set_xlim(0., 225.)
                    if viscp == 'vis':
                        axes.set_ylim(-0.1,1.2)
                        axes.set_xlabel('sp. freq. (M$\lambda$)')
                        axes.set_ylabel('vis2')
                        plt.savefig(oiDir+'/'+saveAsStr+'_'+suff+'_vis2.png')
                        plt.close()
                        log.info('    - Write '+oiDir+'/'+saveAsStr+'_'+suff+'_vis2.png')
                    elif viscp == 'cp':
                        axes.set_xlabel('max sp. freq. (M$\lambda$)');
                        axes.set_ylabel('$\phi_{CP}$')
                        axes.set_ylim(-200,200)
                        plt.savefig(oiDir+'/'+saveAsStr+'_'+suff+'_t3phi.png')
                        plt.close()
                        log.info('    - Write '+oiDir+'/'+saveAsStr+'_'+suff+'_t3phi.png')
                    plt.close("all")
                    del fig,axes
                    first = True
                    # If there is corresponding calibrated data, plot it:
                    calibPlots(calibfiles, viscp, saveAsStr, setups[p])
                    calibfiles = []
                # increase the value of p until a match is found for the current file:
                p += 1
                while first == True:
                    try:
                        if teststr == setups[p]:
                            log.info('    -- '+file+' matches setup '+', '.join(str(s) for s in teststr))
                            fig,axes = plt.subplots()
                            fig.suptitle(', '.join(str(s) for s in teststr))
                            saveAsStr = str(input[0].header['HIERARCH MIRC PRO RTS']).split('_')[0]
                            addV2CP(input, viscp, fig, axes)
                            calibfiles.append(file)
                            first = False
                        else:
                            p += 1
                    except IndexError:
                        log.info('End of setups list reached')
                        return
        del teststr
        log.info('   - Close '+file)
    try:
       axes.set_xlim(0.,225.)
       if viscp == 'vis':
           axes.set_ylim(-0.1,1.2)
           axes.set_xlabel('sp. freq. (M$\lambda$)')
           axes.set_ylabel('vis2')
           plt.savefig(oiDir+'/'+saveAsStr+'_'+suff+'_vis2.png')
           plt.close()
           log.info('    - Write '+oiDir+'/'+saveAsStr+'_'+suff+'_vis2.png')
       elif viscp == 'cp':
           axes.set_xlabel('max sp. freq. (M$\lambda$)');
           axes.set_ylabel('$\phi_{CP}$')
           axes.set_ylim(-200,200)
           plt.savefig(oiDir+'/'+saveAsStr+'_'+suff+'_t3phi.png')
           plt.close()
           log.info('    - Write '+oiDir+'/'+saveAsStr+'_'+suff+'_t3phi.png')
       plt.close("all")
       calibPlots(calibfiles, viscp, saveAsStr, teststr)
    except:
        return
    return

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
    # oiDir has the form redBase+/date_nbsXncsXbbiasXmitp/snrXmitoX/oifits_ncX
    ncoh = oiDir.split('oifits_nc')[-1]
    ncs  = oiDir.split('/')[-3].split('ncs')[-1].split('bbias')[0]
    nbs  = oiDir.split('/')[-3].split('nbs')[-1].split('ncs')[0]
    bb   = oiDir.split('/')[-3].split('bbias')[-1].split('mitp')[0]
    snr  = oiDir.split('/')[-2].split('snr')[-1].split('mito')[0]
    dates = oiDir.split('/')[-3].split('_')[0]
    
    auth = 'ncohrent='+ncoh+'; ncs='+ncs+'; nbs='+nbs+'; snr\\_threshold='+snr.replace('p','.')+'; bbias='+bb
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
                outtex.write('\\subsubsection*{Reduced files located in : ')
                outtex.write(direc.replace('_','\\_')+' on ')
                outtex.write(socket.gethostname()+'}\n')
                if calF == True:
                    outtex.write('\\subsubsection*{Calibration failed}\n')
            else:
                outtex.write('\\subsubsection*{Reduction failed}\n')
            outtex.write('\n\\subsubsection*{PI(s): ')
            princInv = list(set([h['PI_NAME'] for h in hdrs]))
            try:
                princInv.remove('UNKNOWN')
            except ValueError:
                statement = 'unknown not in princInv list'
            if len(princInv) > 1:
                outtex.write('; '.join(princInv)+'}\n')
            elif len(princInv) == 1:
                outtex.write(princInv[0]+'}\n')
            else:
                outtex.write('}\n')
            outtex.write('\\subsubsection*{Observer(s): ')
            outline = []
            for h in hdrs:
                try:
                    obsPerson = h['OBSERVER']
                except KeyError:
                    obsPerson = 'Slimfringe'
                if obsPerson != 'Slimfringe':
                    outline.append(obsPerson)
            out = list(set(outline))
            outtex.write('; '.join(out)+'}\n')
            outtex.write('\\subsubsection*{Program ID(s): (info not yet retained in headers)}\n')
    return outFiles

def texSumTables(oiDir,targs,calInf,scical,redF,rawhdrs,outFiles):
    """
    Append tables section of summary file to existing
    summary tex files. 
        - oiDir is the directory containing the products of the
        oifits reduction step;
        - targs is a python list of target names;
        - calInf is a string of information which was
        parsed to mircx_reduce.py;
        - scical is a python list identifying each target
        as sci or cal;
        - redF is a flag highlighting whether the reduction
        was successful;
        - rawhdrs is the fits headers from the raw data.
    """
    if redF == False:
        # specifying the inclusion only of *_oifits.fits files ensures that BG and FG
        # files are not summarised.
        redhdrs = headers.load(sorted(glob.glob(oiDir+'/*_oifits.fits')))
    for outFile in outFiles:
        with open(outFile, 'a') as outtex:
            outtex.write('\\subsection*{Target summary}\n')
            outtex.write('\\begin{longtable}{p{.25\\textwidth} | p{.08\\textwidth} | ')
            outtex.write('p{.25\\textwidth}}\n    \\hline\n')
            outtex.write('    Target ID & used as & UD diam. for CALs (mas) \\\\ \n')
            outtex.write('    \\hline\n')
            for targ in targs:
                try:
                    ud_H = calInf.split(',')[calInf.split(',').index(targ.replace(' ','_'))+1]
                    eud_H = calInf.split(',')[calInf.split(',').index(targ.replace(' ','_'))+2]
                    outtex.write('    '+targ.replace('_', ' ')+' & CAL')
                    outtex.write(' & $'+ud_H+'\\pm'+eud_H+'\\,$ \\\\ \n')
                except ValueError:
                    outtex.write('    '+targ.replace('_', ' ')+' & SCI')
                    outtex.write(' &  \\\\ \n')
            outtex.write('    \\hline\n\\end{longtable}\n')
            outtex.write('\n')
            outtex.write('\\subsection*{Reduced data summary}\n')
            outtex.write('\\begin{longtable}{p{.04\\textwidth} | p{.08\\textwidth} | ')
            outtex.write('p{.06\\textwidth} | p{.25\\textwidth} | p{.05\\textwidth} | ')
            outtex.write('p{.07\\textwidth} | p{.04\\textwidth} | p{.07\\textwidth} | ')
            outtex.write('p{.08\\textwidth}} \n    \\hline\n')
            outtex.write('    & Start & File & Target & Gain & Ncoher & Nps & Frames & ')
            outtex.write('Filter \\\\ \n')
            outtex.write('    & (UTC) & num. & & & & & $/$reset & \\\\ \n    \\hline\n')
            keys = ['DATE','HIERARCH MIRC PRO RTS','OBJECT','GAIN','NCOHER','PSCOADD','FRMPRST','FILTER1']
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
            outtex.write('    \\hline\n\\end{longtable}\n\n')
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
                outtex.write('    \\includegraphics[angle=90,origin=c,trim=0.0cm 0.2cm 0.0cm 0.2cm, ')
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
                    for n in range(1, int(np.floor(len(uvPlt)))):
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

def texSumPlots(oiDir,redF,calF,outFiles):
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
    keys = ['OBJECT','GAIN','NCOHER','PSCOADD','FRMPRST','FILTER1']
    setupL = [[str(h.get(k,'--')) for k in keys] for h in redhdrs]
    del redhdrs
    setups = []
    setups.append(setupL[0])
    log.info('Targets and camera settings:')
    for m in range(0, len(setupL)-1):
        if setupL[m+1] != setupL[m]:
            log.info('    '+', '.join(str(s) for s in setupL[m+1]))
            setups.append(setupL[m+1])
    # make reduced and calibrated vis2 and CP plots
    plotV2CP(oiDir, setups, 'vis')
    plotV2CP(oiDir, setups, 'cp')
    # Read in mircx numbers of vis2 plots created in reduced and calibrated directories:
    redPlts = sorted(glob.glob(oiDir+'/*reduced_vis2.png'))
    redNum = [int(i.split('/')[-1].split('_')[0].split('x')[1]) for i in redPlts]
    RTS_p  = sorted(glob.glob('/'.join(oiDir[:-2])+'/rts/*datarts_psd.png')) # e.g. mircx00000_datarts_psd.png or mircx00000_foregroundrts_psd.png or mircx00000_backgroundrts_psd.png
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
    
    # Then append reduction QA plots to the file that won't be emailed:
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
         for snr in RTS_p[0:6]:
             # include the first 6 snr plots in the report file if they exist
             if os.path.isfile(snr.replace('/rts', '/oifits').replace('datarts_psd', 'oifits_snr')):
                 outtex.write('    \\includegraphics[trim=2cm 0.9cm 1.5cm 0cm, clip=true, ')
                 outtex.write('width=0.49\\textwidth]{')
                 outtex.write(snr.replace('/rts', '/oifits').replace('datarts_psd', 'oifits_snr'))
                 outtex.write('}\n')
         if len(RTS_p) > 6:
             for n in range(1, int(np.ceil(len(RTS_p)/6.))):
                 # include the remainder of the snr plots in the report file (if they exist)
                 outtex.write('\\end{figure}\n\n\\clearpage\n')
                 outtex.write('\\begin{figure}[h]\n')
                 outtex.write('    \\raggedright\n    \\textbf{Cont.}\\\\ \n')
                 outtex.write('    \\centering\n')
                 for snr in RTS_p[6*n:6*(n+1)]:
                     if os.path.isfile(snr.replace('/rts', '/oifits').replace('datarts_psd', 'oifits_snr')):
                         outtex.write('    \\includegraphics[trim=2cm 0.9cm 1.5cm 0cm, ')
                         outtex.write('clip=true, width=0.49\\textwidth]{')
                         outtex.write(snr.replace('/rts', '/oifits').replace('datarts_psd', 'oifits_snr'))
                         outtex.write('}\n')
         outtex.write('\\end{figure}\n\n')
         outtex.write('\\newpage\n\\begin{figure}[h]\n    \\raggedright\n')
         outtex.write('    \\textbf{Reduction quality assessment: base trend}\\\\ \n')
         outtex.write('    \\centering\n')
         for ba in RTS_p[0:6]:
             # include the first 6 base_trend plots if they exist
             if os.path.isfile(ba.replace('/rts', '/oifits').replace('datarts_psd', 'oifits_base_trend')):
                 outtex.write('    \\includegraphics[trim=2.2cm 0.9cm 1.5cm 0cm, clip=')
                 outtex.write('true, width=0.49\\textwidth]{')
                 outtex.write(ba.replace('/rts', '/oifits').replace('datarts_psd', 'oifits_base_trend'))
                 outtex.write('}\n')
         if len(RTS_p) > 6:
             for n in range(1, int(np.ceil(len(RTS_p)/6.))):
                 outtex.write('\\end{figure}\n\n\\clearpage\n')
                 outtex.write('\\begin{figure}[h]\n')
                 outtex.write('    \\raggedright\n    \\textbf{Cont.}\\\\ \n')
                 outtex.write('    \\centering\n')
                 for ba in RTS_p[6*n:6*(n+1)]:
                     # include the remaining base_trend plots if they exist
                     if os.path.isfile(ba.replace('/rts', '/oifits').replace('datarts_psd', 'oifits_base_trend')):
                         outtex.write('    \\includegraphics[trim=2.2cm 0.9cm 1.5cm 0cm, ')
                         outtex.write('clip=true, width=0.49\\textwidth]{')
                         outtex.write(ba.replace('/rts', '/oifits').replace('datarts_psd', 'oifits_base_trend'))
                         outtex.write('}\n')
         outtex.write('\\end{figure}\n\n')
    for outFile in outFiles:
        with open(outFile, 'a') as outtex:                    
            outtex.write('\\end{document}\n')
    return

