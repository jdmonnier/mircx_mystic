import os;

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
    print ('Please report any bug to lebouquj@umich.edu');
    print ('Git branch: %s'%git_branch);
    print ('Git last commit: %s'%git_date);
    print ('---------------------------------------------');

revision = '1.0.1';

git_date = get_from_dir ('git log -1 --format=%cd --date=format:%Y-%M-%dT%H:%M:%S');

git_hash = get_from_dir ('git log -1 --format=%H');

git_branch = get_from_dir ('git branch | grep \* | cut -d \' \' -f2');
