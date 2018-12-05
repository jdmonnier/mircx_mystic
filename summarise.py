import glob, socket, os
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
    fitsfiles = glob.glob(direc+'/*.fits')
    hdrs  = headers.loaddir(direc)
    objs  = list(set([h['OBJECT'] for h in hdrs]))
    for t in range(0, len(objs)):
        if not os.path.exists(direc+'/'+objs[t]+'_uv_coverage.png'):
            for f in range(0, len(fitsfiles)):
                with pyfits.open(fitsfiles[f]) as input:
                    if input[0].header['OBJECT'] == objs[t]:
                        lbd = input['OI_WAVELENGTH'].data['EFF_WAVE']*1e6
                        usf = 0.0-input['OI_VIS2'].data['UCOORD'][:,None]/lbd[None,:]
                        vsf = input['OI_VIS2'].data['VCOORD'][:,None]/lbd[None,:]
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
            axes.errorbar(1e-6*sf[b,:],vis2[b,:],yerr=evis[b,:],marker='o',ms=1)
    elif viscp == 'cp':
        for b in range(20):
            axes.errorbar(1e-6*max_sf[b,:],cp[b,:],yerr=ecp[b,:],fmt='o',ms=1)
    return

def plotV2CP(direc,setups,viscp):
    """
    Searches a directory for fits files and plots vis vs sf
    and CP vs max_sf for all the files found. The files are
    grouped together by the target name and the camera 
    settings as defined in 'setup'.
        - dir is the full path to the directory containing
        fits files;
        - setups defines the object name and the camera 
        settings;
        - viscp is either 'vis' or 'cp' to decide what is
        plotted.
    """
    fitsfiles = sorted(glob.glob(direc+'/*.fits'))
    if direc.split('/')[-1] == 'oifits':
        suff = 'reduced'
    elif direc.split('/')[-1] == 'calibrated':
        suff = 'calib'
    p, first = 0, True
    for file in fitsfiles:
        # keywords from file headers read in
        with pyfits.open(file) as input:
            keys = ['OBJECT','GAIN','NCOHER','PSCOADD','FRMPRST','FILTER1','R0']
            teststr = [str(input[0].header.get(k,'--')) for k in keys]
            fname = str(input[0].header['HIERARCH MIRC PRO RTS']).split('_')[0]
        if teststr == setups[p] and first == True:
            # option i) file matches current setup and is first file to match it
            fig,axes = plt.subplots()
            fig.suptitle(', '.join(str(s) for s in teststr))
            saveAsStr = fname
            addV2CP(input, viscp, fig, axes)
            first = False
        elif teststr == setups[p] and first != True:
            # option ii) file matches current setup but is not first file to match it
            addV2CP(input, viscp, fig, axes)
        elif teststr != setups[p]:
            # option iii) file doesn't match current setup at all:
            if first != True:
                # if there is data plotted already, close the plot
                axes.set_xlim(0)
                if viscp == 'vis':
                    axes.set_ylim(-0.1,1.2)
                    axes.set_xlabel('sp. freq. (M$\lambda$)')
                    axes.set_ylabel('vis2')
                    plt.savefig(direc+'/'+saveAsStr+'_'+suff+'_vis2.png')
                elif viscp == 'cp':
                    axes.set_xlabel('max sp. freq. (M$\lambda$)');
                    axes.set_ylabel('$\phi_{CP}$')
                    plt.savefig(direc+'/'+saveAsStr+'_'+suff+'_t3phi.png')
                plt.close("all")
            # increase the value of p until a match is found for the current file:
            p += 1
            while first == True:
                try:
                    if teststr == setups[p]:
                        fig,axes = plt.subplots()
                        fig.suptitle(', '.join(str(s) for s in teststr))
                        saveAsStr = str(h['HIERARCH MIRC PRO RTS']).split('_')[0]
                        addV2CP(input, viscp, fig, axes)
                        first = False
                    else:
                        p += 1
                except IndexError:
                    return
        del teststr
    return

######
# LaTex writing functions:
######
def texSumTitle(direc,hdrs,opt,redF,calF):
    """
    Produce header section of tex files containing reduction  
    and calibration summary. 
        - dir is the directory containing this night's 
        reduced data;
        - hdrs is the raw data headers;
        - opt is a python list of options parsed to 
        mircx_reduce.py;
        - redF and calF are flags for if the reduction
        and/or calibration process failed;
    """
    auth = 'ncohrent='+opt[0]+'; ncs='+opt[1]+'; nbs='+opt[2]+'; snr\\_threshold='+opt[3].replace('p','.')
    suf = direc.split('/')[-1]
    outFiles = [direc+'/report_'+suf+'.tex',direc+'/summary_'+suf+'.tex']
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
            outtex.write('\\author{'+auth+'}\n\\date{'+suf.split('_')[0]+'}\n\n')
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
            outline = []
            for h in hdrs:
                try:
                    princInv = h['PI_NAME']
                except KeyError:
                    princInv = 'UNKNOWN'
                if princInv != 'UNKNOWN':
                    outline.append(princInv)
            outtex.write('; '.join(str(out) for out in list(set(outline)))+'}\n')
            outtex.write('\\subsubsection*{Observer(s): ')
            outline = []
            for h in hdrs:
                try:
                    obsPerson = h['OBSERVER']
                except KeyError:
                    obsPerson = 'Slimfringe'
                if obsPerson != 'Slimfringe':
                    outline.append(obsPerson)
            outtex.write('; '.join(str(out) for out in list(set(outline)))+'}\n')
            outtex.write('\\subsubsection*{Program ID(s): (info not yet retained in headers)}\n')
    return

def texSumTables(direc,targs,calInf,scical,redF,rawhdrs):
    """
    Append tables section of summary file to existing
    summary tex files. 
        - dir is the directory containing this night's 
        reduced data;
        - targs is a python list of target names;
        - calInf is a string of information which was
        parsed to mircx_reduce.py;
        - scical is a python list identifying each target
        as sci or cal;
        - redF is a flag highlighting whether the reduction
        was successful;
        - rawhdrs is the fits headers from the raw data.
    """
    suf = direc.split('/')[-1]
    outFiles = [direc+'/report_'+suf+'.tex',direc+'/summary_'+suf+'.tex']
    if redF == False:
        redhdrs = headers.loaddir(direc+'/oifits')
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
            outtex.write('p{.08\\textwidth} | p{.05\\textwidth}} \n    \\hline\n')
            outtex.write('    & Start & File & Target & Gain & Ncoher & Nps & Frames & ')
            outtex.write('Filter & seeing \\\\ \n')
            outtex.write('    & (UTC) & num. & & & & & $/$reset & & \\\\ \n    \\hline\n')
            keys = ['DATE','HIERARCH MIRC PRO RTS','OBJECT','GAIN','NCOHER','PSCOADD','FRMPRST','FILTER1','R0']
            tabRows = [[str(h.get(k,'--')) for k in keys] for h in redhdrs]
            for row in tabRows:
                row[0] = row[0].split('T')[1]
                row[1] = row[1].split('/')[-1].split('mircx')[1].split('_')[0]
            if redF == False:
                skipd = 17
            else:
                skipd = 11
            for r in range(0, len(tabRows)-1):
                if r == 0:
                    outtex.write('        '+str(r)+' & '+' & '.join(str(s).replace('_',' ') for s in tabRows[r])+'\\\\ \n')
                else:
                    nextrow = ' & '.join(str(s).replace('_',' ') for s in tabRows[r+1])
                    thisrow = ' & '.join(str(s).replace('_',' ') for s in tabRows[r])
                    if nextrow[skipd:] != thisrow[skipd:]:
                        outtex.write('        '+str(r)+' & '+nextrow+'\\\\ \n')
                    del nextrow, thisrow
            outtex.write('    \\hline\n\\end{longtable}\n\n')
            del tabRows
    log.info('Cleanup memory')
    if redF == False:
        del redhdrs
    return

def texReportPlts(direc):
    """
    Locates the report files output by mircx_report.py
    which check the camera performance and includes
    them into the tex summary and report PDF files.
        - dir is the reduced files directory
    """
    reportFiles = glob.glob(direc+'/oifits/report*.png')
    suf = direc.split('/')[-1]
    d = suf.split('_')[0]
    outFiles = [direc+'/report_'+suf+'.tex',direc+'/summary_'+suf+'.tex']
    for outFile in outFiles:
        with open(outFile, 'a') as outtex:
            if len(reportFiles) == 0:
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
                outtex.write('\\end{figure}\n\n')
                r += 2
    return

def texSumUV(direc,calF):
    """
    If the calibration process was succesful, the 
    uv coverage plots for the science targets will
    be appended to the summary PDF files.
        - dir is the directory containing this night's 
        reduced data;
        - calF is a flag highlighting whether the calibration
        was successful;
    """
    if calF == False:
        uvPlt = glob.glob(direc+'/oifits/calibrated/*_uv_coverage.png')
        suf = direc.split('/')[-1]
        outFiles = [direc+'/report_'+suf+'.tex',direc+'/summary_'+suf+'.tex']
        for outFile in outFiles:
            with open(outFile, 'a') as outtex:
                outtex.write('\\newpage\n\\begin{figure}[h]\n    \\raggedright\n')
                outtex.write('    \\textbf{Full night $uv$-coverage for SCI target(s)}')
                outtex.write('\\\\ \n    \\centering\n')
                for uvp in uvPlt[0:12]:
                    outtex.write('    \\includegraphics[trim=2.0cm 0.0cm 2.0cm 0.0cm, ')
                    outtex.write('clip=true, width=0.32\\textwidth]{'+uvp+'}\n')
                if len(uvPlt) > 12:
                    for n in range(1, int(np.floor(len(uvPlt)))):
                        outtex.write('\\end{figure}\n\n\\begin{figure}[h]\n')
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

def texSumPlots(direc,redF,calF):
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
    suf = direc.split('/')[-1]
    outFiles = [direc+'/report_'+suf+'.tex',direc+'/summary_'+suf+'.tex']
    if redF == True:
        for outFile in outFiles:
            with open(outFile, 'a') as outtex:                    
                outtex.write('\\end{document}\n')
        return
    # sort the reduced files by camera settings and target:
    redFiles = sorted(glob.glob(direc+'/oifits/*.fits'))
    redhdrs = headers.loaddir(direc+'/oifits')
    keys = ['OBJECT','GAIN','NCOHER','PSCOADD','FRMPRST','FILTER1','R0']
    setupL = [[str(h.get(k,'--')) for k in keys] for h in redhdrs]
    del redhdrs
    setups = []
    setups.append(setupL[0])
    for m in range(0, len(setupL)-1):
        if setupL[m+1] != setupL[m]:
            setups.append(setupL[m+1])
    # make reduced vis2 and CP plots
    plotV2CP(direc+'/oifits', setups, 'vis')
    plotV2CP(direc+'/oifits', setups, 'cp')
    # Then do the same for calibrated files if calibration process was successful:
    if calF == False:
        calFiles = sorted(glob.glob(direc+'/oifits/calibrated/*.fits'))
        plotV2CP(direc+'/oifits/calibrated', setups, 'vis')
        plotV2CP(direc+'/oifits/calibrated', setups, 'cp')
    # Read in mircx numbers of vis2 plots created in reduced and calibrated directories:
    redPlts = sorted(glob.glob(direc+'/oifits/*reduced_vis2.png'))
    redNum = [int(i.split('/')[-1].split('_')[0].split('x')[1]) for i in redPlts]
    RTS_p  = sorted(glob.glob(direc+'/rts/*rts_psd.png'))
    for num in range(0, len(redNum)):
        # ensure correct number of leading zeros are added to redNum for file name:
        strnum = '0'*(5-len(str(redNum[num])))+str(redNum[num])
        log.info('Gather plots for summary report for file mircx'+strnum)
        redV2plt = direc+'/oifits/mircx'+strnum+'_reduced_vis2.png'
        redCPplt = direc+'/oifits/mircx'+strnum+'_reduced_t3phi.png'
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
                if os.path.isfile(direc+'/oifits/calibrated/mircx'+strnum+'_calib_vis2.png'):
                    outtex.write('    \\textbf{Calibrated vis2 and CP:}\\\\\n')
                    outtex.write('    \\includegraphics[trim=0.0cm 0.2cm 0.0cm 0.0cm, ')
                    outtex.write('clip=true, width=0.32\\textwidth]{')
                    outtex.write(direc+'/oifits/calibrated/mircx'+strnum+'_calib_vis2.png')
                    outtex.write('}\n')
                    outtex.write('    \\includegraphics[trim=0.0cm 0.2cm 0.0cm 0.0cm,')
                    outtex.write(' clip=true, width=0.32\\textwidth]{')
                    outtex.write(direc+'/oifits/calibrated/mircx'+strnum+'_calib_t3phi.png')
                    outtex.write('}\\\\ \n')
                outtex.write('\\end{figure}\n\n')
    # Then append reduction QA plots to the file that won't be emailed:
    with open(outFiles[0], 'a') as outtex:
         outtex.write('\\newpage\n\\begin{figure}[h]\n    \\raggedright\n')
         outtex.write('    \\textbf{Reduction quality assessment: PSD}\\\\ \n')
         outtex.write('    \\centering\n')
         for rts in RTS_p[0:15]:
             outtex.write('    \\includegraphics[trim=0.7cm 0.9cm 1.5cm 0.0cm, ')
             outtex.write('clip=true, width=0.32\\textwidth]{'+rts+'}\n')
         if len(RTS_p) > 15:
             for n in range(1, int(np.ceil(len(RTS_p)/15.))):
                 outtex.write('\\end{figure}\n\n\\begin{figure}[h]\n')
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
             outtex.write('    \\includegraphics[trim=2cm 0.9cm 1.5cm 0cm, clip=true, ')
             outtex.write('width=0.49\\textwidth]{')
             outtex.write(snr.replace('/rts', '/oifits').replace('rts_psd', 'oifits_snr'))
             outtex.write('}\n')
         if len(RTS_p) > 6:
             for n in range(1, int(np.ceil(len(RTS_p)/6.))):
                 outtex.write('\\end{figure}\n\n\\begin{figure}[h]\n')
                 outtex.write('    \\raggedright\n    \\textbf{Cont.}\\\\ \n')
                 outtex.write('    \\centering\n')
                 for snr in RTS_p[6*n:6*(n+1)]:
                     outtex.write('    \\includegraphics[trim=2cm 0.9cm 1.5cm 0cm, ')
                     outtex.write('clip=true, width=0.49\\textwidth]{')
                     outtex.write(snr.replace('/rts', '/oifits').replace('rts_psd', 'oifits_snr'))
                     outtex.write('}\n')
         outtex.write('\\end{figure}\n\n')
         outtex.write('\\newpage\n\\begin{figure}[h]\n    \\raggedright\n')
         outtex.write('    \\textbf{Reduction quality assessment: base trend}\\\\ \n')
         outtex.write('    \\centering\n')
         for ba in RTS_p[0:6]:
             outtex.write('    \\includegraphics[trim=2.2cm 0.9cm 1.5cm 0cm, clip=')
             outtex.write('true, width=0.49\\textwidth]{')
             outtex.write(ba.replace('/rts', '/oifits').replace('rts_psd', 'oifits_base_trend'))
             outtex.write('}\n')
         if len(RTS_p) > 6:
             for n in range(1, int(np.ceil(len(RTS_p)/6.))):
                 outtex.write('\\end{figure}\n\n\\begin{figure}[h]\n')
                 outtex.write('    \\raggedright\n    \\textbf{Cont.}\\\\ \n')
                 outtex.write('    \\centering\n')
                 for ba in RTS_p[6*n:6*(n+1)]:
                     outtex.write('    \\includegraphics[trim=2.2cm 0.9cm 1.5cm 0cm, ')
                     outtex.write('clip=true, width=0.49\\textwidth]{')
                     outtex.write(ba.replace('/rts', '/oifits').replace('rts_psd', 'oifits_base_trend'))
                     outtex.write('}\n')
         outtex.write('\\end{figure}\n\n')
    for outFile in outFiles:
        with open(outFile, 'a') as outtex:                    
            outtex.write('\\end{document}\n')
    return

