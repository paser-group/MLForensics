'''
Farzana Ahamed Bhuiyan (Lead) 
Akond Rahman 
Oct 20, 2020 
Main executor 
'''
import lint_engine
import constants 
import time 
import datetime 
import os 
import pandas as pd


def giveTimeStamp():
  tsObj = time.time()
  strToret = datetime.datetime.fromtimestamp(tsObj).strftime(constants.TIME_FORMAT) 
  return strToret
  

def getCSVData(dic_, dir_repo):
	temp_list = []
	for TEST_ML_SCRIPT in dic_:
		# Section 1.1a
		data_load_counta = lint_engine.getDataLoadCount( TEST_ML_SCRIPT ) 

		# Section 1.1b
		data_load_countb = lint_engine.getDataLoadCountb( TEST_ML_SCRIPT ) 

		# Section 1.1c
		data_load_countc = lint_engine.getDataLoadCountc( TEST_ML_SCRIPT ) 

		# Section 1.2a
		model_load_counta = lint_engine.getModelLoadCounta( TEST_ML_SCRIPT ) 

		# Section 1.2b
		model_load_countb = lint_engine.getModelLoadCountb( TEST_ML_SCRIPT ) 

		# Section 1.2c
		model_load_countc = lint_engine.getModelLoadCountc( TEST_ML_SCRIPT ) 

		# Section 1.2d
		model_load_countd = lint_engine.getModelLoadCountd( TEST_ML_SCRIPT ) 

		# Section 2.1a
		data_download_counta = lint_engine.getDataDownLoadCount( TEST_ML_SCRIPT ) 

		# Section 2.1b
		data_download_countb = lint_engine.getDataDownLoadCountb( TEST_ML_SCRIPT )

		# Section 3.1
		model_feature_count = lint_engine.getModelFeatureCount( TEST_ML_SCRIPT ) 

		# Section 3.2a
		model_label_counta = lint_engine.getModelLabelCount( TEST_ML_SCRIPT ) 
	
		# Section 3.2b
		model_label_countb = lint_engine.getModelLabelCountb( TEST_ML_SCRIPT ) 

		# Section 3.3a
		model_output_counta = lint_engine.getModelOutputCount( TEST_ML_SCRIPT ) 
	
		# Section 3.3a
		model_output_countb = lint_engine.getModelOutputCountb( TEST_ML_SCRIPT ) 

		# Section 3.3a
		model_output_countc = lint_engine.getModelOutputCountc( TEST_ML_SCRIPT ) 

		# Section 4.1
		data_pipeline_counta = lint_engine.getDataPipelineCount( TEST_ML_SCRIPT ) 

		# Section 4.2
		data_pipeline_countb = lint_engine.getDataPipelineCountb( TEST_ML_SCRIPT ) 

		# Section 4.3
		data_pipeline_countc = lint_engine.getDataPipelineCountc( TEST_ML_SCRIPT ) 

		# Section 4.4
		data_pipeline_countd = lint_engine.getDataPipelineCountd( TEST_ML_SCRIPT ) 

		# Section 5.1a
		environment_counta = lint_engine.getEnvironmentCount( TEST_ML_SCRIPT ) 

		# Section 5.1b
		environment_countb = lint_engine.getEnvironmentCountb( TEST_ML_SCRIPT ) 

		# Section 5.2
		state_observe_count = lint_engine.getStateObserveCount( TEST_ML_SCRIPT ) 

		# Section 6.1
		dnn_decision_counta = lint_engine.getDNNDecisionCount( TEST_ML_SCRIPT ) 

		# Section 6.2
		dnn_decision_countb = lint_engine.getDNNDecisionCountb( TEST_ML_SCRIPT ) 
		# the following checks except related blocks 

		# Section 7
		except_flag = lint_engine.getExcepts( TEST_ML_SCRIPT ) 

		# Section 8
		incomplete_logging_count = lint_engine.getIncompleteLoggingCount( TEST_ML_SCRIPT ) 
		
		data_load_count = data_load_counta + data_load_countb + data_load_countc
		model_load_count = model_load_counta + model_load_countb + model_load_countc + model_load_countd
		data_download_count = data_download_counta + data_download_countb
		model_label_count = model_label_counta + model_label_countb
		model_output_count = model_output_counta + model_output_countb + model_output_countc
		data_pipeline_count = data_pipeline_counta + data_pipeline_countb + data_pipeline_countc + data_pipeline_countd
		environment_count = environment_counta + environment_countb
		dnn_decision_count = dnn_decision_counta + dnn_decision_countb
		
		the_tup = ( dir_repo, TEST_ML_SCRIPT, data_load_count, model_load_count, data_download_count, model_feature_count, \
  				  model_label_count, model_output_count, data_pipeline_count, environment_count, state_observe_count, \
  				  dnn_decision_count, incomplete_logging_count, except_flag)
  				  
		temp_list.append( the_tup )
	return temp_list
  
  
def getAllPythonFilesinRepo(path2dir):
    valid_list = []
    for root_, dirnames, filenames in os.walk(path2dir):
        for file_ in filenames:
            full_path_file = os.path.join(root_, file_) 
            if (file_.endswith( constants.PY_FILE_EXTENSION ) ):
                valid_list.append(full_path_file) 
    return valid_list


def runFameML(inp_dir, csv_fil):
    output_event_dict = {}
    df_list = [] 
    list_subfolders_with_paths = [f.path for f in os.scandir(inp_dir) if f.is_dir()]
    for subfolder in list_subfolders_with_paths: 
        events_with_dic =  getAllPythonFilesinRepo(subfolder)  
        if subfolder not in output_event_dict:
          output_event_dict[subfolder] = events_with_dic
        print(constants.ANALYZING_KW, subfolder)
        temp_list  = getCSVData(events_with_dic, subfolder)
        df_list    = df_list + temp_list 
    full_df = pd.DataFrame( df_list ) 
    # print(full_df.head())
    full_df.to_csv(csv_fil, header= constants.CSV_HEADER, index=False, encoding= constants.UTF_ENCODING)     
    return output_event_dict


if __name__=='__main__':
	
	t1 = time.time()
	print('Started at:', giveTimeStamp() )
	print('*'*100 )
	
	repo_dir   = '/Users/arahman/FSE2021_ML_REPOS/MODELZOO/'
	output_csv = '../../Output/v1_OUTPUT_MODELZOO.csv'
	full_dict = runFameML(repo_dir, output_csv)
	
	print('*'*100 )
	print('Ended at:', giveTimeStamp() )
	print('*'*100 )
	
	t2 = time.time()
	time_diff = round( (t2 - t1 ) / 60, 5) 
	print('Duration: {} minutes'.format(time_diff) )
	print('*'*100 )


