import subprocess
import sys, os, socket
import smtplib
from . import log

try:
    from email.mime.multipart import MIMEMultipart
    log.info('Loaded python3 MIMEMultipart')
except ModuleNotFoundError:
    from email.MIMEMultipart import MIMEMultipart
    log.info('Loaded python MIMEMultipart')
try:
    from email.mime.text import MIMEText
except ModuleNotFoundError:
    from email.MIMEText import MIMEText
try:
    from email.mime.base import MIMEBase
except ModuleNotFoundError:
    from email.MIMEBase import MIMEBase

from email import encoders

def send_email(msg, fromaddr, toaddr): 
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    try:
        server.login("mircx.mystic@gmail.com", os.environ['MAILLOGIN'])
    except KeyError:
        log.error('password for '+fromaddr+' not found')
        log.error('Please ensure $MAILLOGIN is set')
        if fromaddr == 'mircx.mystic@gmail.com':
            log.error('Contact Narsi Anugu for password for '+fromaddr)
        server.quit()
        sys.exit()
    server.sendmail(fromaddr, toaddr, msg.as_string())
    server.quit()

def sendSummary(toaddr,fromaddr,outFile,inDir):
    """
    Emails the summary report PDF file for the reduced and calibrated
    night of observations to 'addr'
    """
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To']   = toaddr
    
    filename = outFile.split('/')[-1]
    msg['Subject'] = 'MIRC-X redcal summary '+filename
    if socket.gethostname() == 'mircx':
        # this is where we need to change to include text from archive log
        bod = []
        if os.path.isfile(inDir+'/mircx_archivedata.summary.log'):
            subprocess.call('cat '+inDir+'/mircx_archivedata.log | grep "ERROR" | grep -v "*** ERROR" | sed -e "y#:#_#" >> '+inDir+'/mircx_archivedata.summary.log',shell=True)
            print('cat '+inDir+'/mircx_archivedata.log | grep "ERROR" | grep -v "*** ERROR" | sed -e "y#:#_#" >> '+inDir+'/mircx_archivedata.summary.log')
            with open(inDir+'/mircx_archivedata.summary.log') as readin:
                for line in readin:
                    if line.strip() not in bod:
                        bod.append(line.strip())
            bod.append('\n\n')
        else:
            bod.append('File '+inDir+'/mircx_archivedata.summary.log not found')
            bod.append('MIRC-X redcal summary '+filename+'\n')
        body = '\n'.join(bod)
    else:
        body = 'MIRC-X redcal summary '+filename+'\n'
    msg.attach(MIMEText(body, 'plain'))
    attachment = open(outFile, 'rb')
    part = MIMEBase('application', 'octet-stream')
    part.set_payload((attachment).read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition',"attachment; filename= %s" % filename)
    msg.attach(part)
    try:
        send_email(msg, fromaddr, toaddr)
        log.info('Emailed summary report ('+filename+') to:')
        log.info(toaddr)
    except smtplib.SMTPAuthenticationError:
        log.error('Failed to send summary report ('+filename+') to '+toaddr)
        log.error('Check with Narsi Anugu for permissions')
        sys.exit()
