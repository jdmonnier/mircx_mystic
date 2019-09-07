# This file will eventually be in its own library.
from __future__ import print_function
import json, os, re

try:
    with open("/home/spooler/secret/slack.json") as json_data:
        slack_data = json.load(json_data)
        slack_key = slack_data["channel-id"]
        slack_user = slack_data["users"]
        slack_oauth = slack_data["OAUTH"]["mirc-bot"]
except:
    print("Slack integration not found.")
    slack_key = {}
    slack_user = {}
    slack_oauth = ""

def user(name):
    if name in ["channel", "everyone"]:
        return "<!" + name + ">"
    elif name in slack_user:
        return "<@" + slack_user[name] + ">"
    else:
        return "@" + name

def parse(msg):
    newmsg = msg
    users = re.findall("@[a-zA-Z0-9\-]+", msg)
    for usr in users:
        newmsg = newmsg.replace(usr, user(usr.lower()[1:]))
    return newmsg

def post(channel, message, attach=None):
    msg = parse(message)
    if channel in slack_key:
        if attach is not None:
            if os.path.isfile(attach):
                cmd = "curl -F file=@" + attach + ' -F "initial_comment=' + msg + '''" -F channels=''' + slack_key[channel] + ''' -H "Authorization: Bearer ''' + slack_oauth + '''" https://slack.com/api/files.upload'''
                os.system(cmd)
            else:
                print("Attached file does not exist.  No message posted to Slack.")
        else:
            cmd = '''curl -F text="''' + msg + '''" -F channel=''' + slack_key[channel] + ''' -H "Authorization: Bearer ''' + slack_oauth + '''" https://slack.com/api/chat.postMessage'''
            os.system(cmd)
    else:
        print("Warning, slack key to channel #" + channel + " not found.)
