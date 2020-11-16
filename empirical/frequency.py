'''
Akond Rahman 
Nov 15, 2020
Frequency: RQ2
'''
import numpy as np 
import os 
import pandas as pd 
import time 
import datetime 

def giveTimeStamp():
  tsObj = time.time()
  strToret = datetime.datetime.fromtimestamp(tsObj).strftime( '%Y-%m-%d %H:%M:%S' ) 
  return strToret


def reportProportion( res_file ):
    res_df = pd.read_csv( res_file )
    all_py_files   = np.unique( res_df['FILE_FULL_PATH'].tolist() )
    fields2explore = ['DATA_LOAD_COUNT', 'MODEL_LOAD_COUNT', 'DATA_DOWNLOAD_COUNT',	'MODEL_LABEL_COUNT', 'MODEL_OUTPUT_COUNT',	
                      'DATA_PIPELINE_COUNT', 'ENVIRONMENT_COUNT', 'STATE_OBSERVE_COUNT', 'DNN_DECISION_COUNT', 'TOTAL_EVENT_COUNT'
                     ]
    for field in fields2explore:
        field_atleast_one_df = res_df[res_df[field] > 0 ]
        atleast_one_files    = np.unique( field_atleast_one_df['FILE_FULL_PATH'].tolist() )
        prop_metric          = round(float(len( atleast_one_files ) )/float(len(all_py_files)) , 5) * 100
        print('TOTAL_FILES:{}, CATEGORY:{}, ATLEASTONE:{}, PROP_VAL:{}'.format( len(all_py_files), field, len(atleast_one_files) , prop_metric  ))
        # print(atleast_one_files)
        print('-'*50) 

        # print(field_df.head())  



if __name__=='__main__': 
    print('*'*100 )
    t1 = time.time()
    print('Started at:', giveTimeStamp() )
    print('*'*100 )


    # DATASET_NAME = 'TEST'
    # RESULTS_FILE = '/Users/arahman/Documents/OneDriveWingUp/OneDrive-TennesseeTechUniversity/Research/VulnStrategyMining/ForensicsinML/Output/V3_OUTPUT_TEST.csv'

    DATASET_NAME = 'MODEL_ZOO'
    RESULTS_FILE = '/Users/arahman/Documents/OneDriveWingUp/OneDrive-TennesseeTechUniversity/Research/VulnStrategyMining/ForensicsinML/Output/V3_OUTPUT_MODELZOO.csv'    
    
    reportProportion( RESULTS_FILE )
    
    print('*'*100 )
    print('Ended at:', giveTimeStamp() )
    print('*'*100 )