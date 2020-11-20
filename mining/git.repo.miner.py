'''
Akond Rahman 
Nov 19, 2020 
Mine Git-based repos 
'''


import pandas as pd 
import csv 
import subprocess
import numpy as np
import shutil
from git import Repo
from git import exc 
from xml.dom import minidom
from xml.parsers.expat import ExpatError
import time 
import  datetime 
import os 

def deleteRepo(dirName, type_):
    print(':::' + type_ + ':::Deleting ', dirName)
    try:
        if os.path.exists(dirName):
            shutil.rmtree(dirName)
    except OSError:
        print('Failed deleting, will try manually')        


def makeChunks(the_list, size_):
    for i in range(0, len(the_list), size_):
        yield the_list[i:i+size_]

def cloneRepo(repo_name, target_dir):
    cmd_ = "git clone " + repo_name + " " + target_dir 
    try:
       subprocess.check_output(['bash','-c', cmd_])    
    except subprocess.CalledProcessError:
       print('Skipping this repo ... trouble cloning repo:', repo_name )

def dumpContentIntoFile(strP, fileP):
    fileToWrite = open( fileP, 'w')
    fileToWrite.write(strP )
    fileToWrite.close()
    return str(os.stat(fileP).st_size)

def getPythonCount(path2dir): 
    usageCount = 0
    for root_, dirnames, filenames in os.walk(path2dir):
        for file_ in filenames:
            full_path_file = os.path.join(root_, file_) 
            if (file_.endswith('py') ):
                usageCount +=  1 
    return usageCount                         


def cloneRepos(repo_list): 
    counter = 0     
    str_ = ''
    for repo_batch in repo_list:
        for repo_ in repo_batch:
            counter += 1 
            print('Cloning ', repo_ )
            dirName = '/Users/arahman/FSE2021_ML_REPOS/GITLAB_REPOS/' + repo_.split('/')[-2] + '@' + repo_.split('/')[-1] 
            cloneRepo(repo_, dirName )
            ### get file count 
            all_fil_cnt = sum([len(files) for r_, d_, files in os.walk(dirName)])
            if (all_fil_cnt <= 0):
               deleteRepo(dirName, 'NO_FILES')
            else: 
               py_file_count = getPythonCount( dirName  )
               prop_py = float(py_file_count) / float(all_fil_cnt)
               if(prop_py < 0.25):
                   deleteRepo(dirName, 'LOW_PYTHON_' + str( round(prop_py, 5) ) )
            print("So far we have processed {} repos".format(counter) )
            if((counter % 100) == 0):
                dumpContentIntoFile(str_, 'tracker_completed_repos.csv')
            elif((counter % 1000) == 0):
                print(str_)                
            print('#'*100)

if __name__=='__main__':
    repos_df = pd.read_csv('/Users/arahman/Documents/OneDriveWingUp/OneDrive-TennesseeTechUniversity/Research/VulnStrategyMining/ForensicsinML/Datasets/INITIAL_PYTHON_REPOS_GITLAB.csv')
    list_    = repos_df['URL'].tolist()
    list_    = np.unique(list_)
    print('Repos to download:', len(list_)) 
    ## need to create chunks as too many repos 
    chunked_list = list(makeChunks(list_, 1000))  ### list of lists, at each batch download 1000 repos 
    cloneRepos(chunked_list)

