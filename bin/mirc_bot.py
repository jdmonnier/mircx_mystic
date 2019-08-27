# This file will eventually be in its own library.
from __future__ import print_function
import json, os, re

try:
    with open("/home/spooler/secret/slack.json") as json_data:
        slack_data = json.load(json_data)
        slack_key = slack_data["channels"]
        slack_user = slack_data["users"]
except:
    slack_key = {}
    slack_user = {}

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

def post(channel, msg, notify=None):
    alert = ""
    if isinstance is not None:
        if isinstance(notify, (list, tuple)):
            for name in notify:
                alert = alert + user(name) + ", "
            alert = alert[:-2] + ":\n"
        elif isinstance(notify, str):
            alert = user(notify) + ":\n"
    msg = alert + parse(msg)
    if channel in slack_key:
        cmd = '''curl -X POST -H 'Content-type: application/json' --data '{"text":"''' + msg + '''"}' https://hooks.slack.com/services/''' + slack_key[channel]
        os.system(cmd)
    else:
        print("Warning, slack key to channel #" + channel + " not found.  Message that should have been posted:\n" + msg)
