import sys, os
import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
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

def sendSummary(dir, toaddr, fromaddr):
    """
    Emails the report PDF file for the reduced and calibrated 
    night of observations to 'addr'
    """
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To']   = toaddr
    
    d = dir.split('/')[-1].split('_')[0]
    msg['Subject'] = 'MIRC-X redcal summary for '+d
    body = 'MIRC-X redcal summary for '+d+'\n'
    msg.attach(MIMEText(body, 'plain'))
    filename = d+'_report.pdf'
    attachment = open(dir+'/'+filename, 'rb')
    part = MIMEBase('application', 'octet-stream')
    part.set_payload((attachment).read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition',"attachment; filename= %s" % filename)
    msg.attach(part)
    try:
        send_email(msg, fromaddr, toaddr)
        log.info('Emailed summary report ('+filename+') to:')
        log.info(addr)
    except smtplib.SMTPAuthenticationError:
        log.error('Failed to send summary report ('+filename+') to '+addr)
        log.error('Check with Narsi Anugu for permissions')
        sys.exit()
