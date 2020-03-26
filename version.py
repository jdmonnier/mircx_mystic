import os, psutil;
import matplotlib as mpl;

def get_from_dir (arg):
    '''
    Run command in the installation directory of this file.
    And return the output
    '''
    try:
        directory = os.path.dirname (os.path.abspath(__file__));
        cmd = 'cd '+directory+'; '+arg;
        return os.popen(cmd).read().split('\n')[0];
    except:
        return 'unknown';

def info ():
    print ('---------------------------------------------');
    print ('Module mircx_pipeline version %s'%revision);
    print ('Please report any bugs to lebouquj@umich.edu');
    print ('Git commit: %s (%s)'%(git_hash[:8],git_date));
    print ('Git branch: %s%s'%(git_branch,git_status));
    print ('Matplotlib backend: '+mpl.get_backend());
    print ('Total memory: %.1fG'%(psutil.virtual_memory().total/1e9));
    print ('---------------------------------------------');

# Revision hardcoded name
revision = '1.3.0';

# some information from the GIT repository, if available
git_date = get_from_dir ('git log -1 --format=%cd --date=format:%Y-%m-%dT%H:%M:%S');

git_hash = get_from_dir ('git log -1 --format=%H');

git_branch = get_from_dir ('git branch | grep \* | cut -d \' \' -f2');

git_status = get_from_dir ('git diff-index --name-only HEAD | tr \'\\n\' \' \'');

if git_status != '':
    git_status = ' + uncommited changes:\n  ' + git_status;
