'''
Akond Rahman 
Nov 16, 2020 
Mine Metrics for Paper Writing
'''
import pandas as pd 
import numpy as np 
from git import Repo
from git import exc 
import os 
from datetime import datetime
import subprocess
from collections import Counter 

def getBranch(path):
    dict_ = { 
             '/Users/arahman/FSE2021_ML_REPOS/MODELZOO/NATURAL_LANGUAGE_PROCESSING/magic282@MXNMT':'next' 
    } 
    if path in dict_:
        return dict_[path] 
    else:
        return 'master' 

def getFileLength(file_):
    return sum(1 for line in open(file_, encoding='latin-1'))

def getDevEmailForCommit(repo_path_param, hash_):
    author_emails = []

    cdCommand     = "cd " + repo_path_param + " ; "
    commitCountCmd= " git log --format='%ae'" + hash_ + "^!"
    command2Run   = cdCommand + commitCountCmd

    author_emails = str(subprocess.check_output(['bash','-c', command2Run]))
    author_emails = author_emails.split('\n')
    author_emails = [x_.replace(hash_, '') for x_ in author_emails if x_ != '\n' and '@' in x_ ] 
    author_emails = [x_.replace('^', '') for x_ in author_emails if x_ != '\n' and '@' in x_ ] 
    author_emails = [x_.replace('!', '') for x_ in author_emails if x_ != '\n' and '@' in x_ ] 
    author_emails = [x_.replace('\\n', ',') for x_ in author_emails if x_ != '\n' and '@' in x_ ] 
    try:
        author_emails = author_emails[0].split(',')
        author_emails = [x_ for x_ in author_emails if len(x_) > 3 ] 
        author_emails = list(np.unique(author_emails) )
    except IndexError as e_:
        pass
    return author_emails  

def getDevDayCommits(full_path_to_repo, branchName='master', explore=1000):
    repo_emails = []
    all_commits = []
    repo_emails = []
    all_time_list = []
    if os.path.exists(full_path_to_repo):
        repo_  = Repo(full_path_to_repo)
        try:
           all_commits = list(repo_.iter_commits(branchName))   
        except exc.GitCommandError:
           print('Skipping this repo ... due to branch name problem', full_path_to_repo )
        for commit_ in all_commits:
                commit_hash = commit_.hexsha

                emails = getDevEmailForCommit(full_path_to_repo, commit_hash)
                repo_emails = repo_emails + emails

                timestamp_commit = commit_.committed_datetime
                str_time_commit  = timestamp_commit.strftime('%Y-%m-%d') ## date with time 
                all_time_list.append( str_time_commit )

    else:
        repo_emails = [ str(x_) for x_ in range(10) ]
    all_day_list   = [datetime(int(x_.split('-')[0]), int(x_.split('-')[1]), int(x_.split('-')[2]), 12, 30) for x_ in all_time_list]

    repo_emails = np.unique( repo_emails ) 
    return len(repo_emails) , len(all_commits) , all_day_list

def days_between(d1_, d2_): ## pass in date time objects 
    return abs((d2_ - d1_).days)


def getAllCommits(all_repos):
    full_list = []
    total_commits  = 0 
    total_devs     = 0 
    all_days       = []
    tracker        = 0 
    for repo_ in all_repos:
        tracker += 1 
        branchName = getBranch(repo_) 
        dev_cnt, com_cnt, _days = getDevDayCommits(repo_, branchName)  
        per_repo_min_day        = min(_days) 
        per_repo_max_day        = max(_days)   
        day_diff                = days_between( per_repo_min_day, per_repo_max_day  )       
        the_tuple = (repo_, dev_cnt, com_cnt, per_repo_min_day, per_repo_max_day, day_diff) 
        print(tracker) 
        print(the_tuple) 
        full_list.append(  the_tuple  )
        total_commits = total_commits + com_cnt 
        total_devs    = total_devs + dev_cnt 
        all_days      = all_days + _days 
    
    min_day        = min(all_days) 
    max_day        = max(all_days) 

    temp_df  = pd.DataFrame( full_list )
    temp_df.to_csv( 'COMMIT.STATS.csv', header=['REPO', 'DEVS', 'COMMITS', 'START_DATE', 'END_DATE', 'DURATION_DAYS'], index=False, encoding='utf-8')     
    return min_day, max_day, total_commits, total_devs 

           

def getAllFileCount(df_):
    tot_fil_size = 0 
    file_names_ =  np.unique( df_['FILE_FULL_PATH'].tolist() )
    for file_ in file_names_:
        tot_fil_size = tot_fil_size + getFileLength( file_ )
    return tot_fil_size, len( file_names_ ) 


def getGeneralStats(all_dataset_list):
    for result_file in all_dataset_list:
        print('='*50)
        print(result_file)
        print('='*50)
        if 'ZOO' in result_file:
            all_repos = [] 
            res_df    = pd.read_csv( result_file ) 
            temp_dirs = np.unique( res_df['REPO_FULL_PATH'].tolist() ) 
            for temp_dir in temp_dirs:
                list_subfolders_with_paths = [f.path for f in os.scandir(temp_dir) if f.is_dir()]
                all_repos = all_repos + list_subfolders_with_paths 
        print('REPO_COUNT:', len(all_repos) ) 
        file_size, file_count   = getAllFileCount(res_df)
        print('ALL_FILE_COUNT:', file_count  ) 
        print('ALL_FILE_SIZE:', file_size  )   
        # start_date, end_date, coms, devs  = getAllCommits( all_repos ) 
        # print('COMMIT_COUNT:', coms )
        # print('DEVS_COUNT:', devs )
        # print('START_DATE:', start_date )
        # print('END_DATE:', end_date )
        print('='*50)

if __name__=='__main__':
    MODEL_ZOO_RESULTS_FILE = '/Users/arahman/Documents/OneDriveWingUp/OneDrive-TennesseeTechUniversity/Research/VulnStrategyMining/ForensicsinML/Output/V5_OUTPUT_MODELZOO.csv'
    all_datasets = [MODEL_ZOO_RESULTS_FILE]
    
    getGeneralStats(all_datasets)  