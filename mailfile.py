import sys, os
import smtplib
try:
    from email.mime.multipart import MIMEMultipart
except ModuleNotFoundError:
    from email.MIMEMultipart import MIMEMultipart
try:
    from email.mime.text import MIMEText
except ModuleNotFoundError:
    from email.MIMEText import MIMEText
try:
    from email.mime.base import MIMEBase
except ModuleNotFoundError:
    from email.MIMEBase import MIMEBase

from email import encoders

from . import log

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

def sendSummary(toaddr, fromaddr,outFile):
    """
    Emails the summary report PDF file for the reduced and calibrated 
    night of observations to 'addr'
    """
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To']   = toaddr
    
    filename = outFile.split('/')[-1]
    msg['Subject'] = 'MIRC-X redcal summary '+filename
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
