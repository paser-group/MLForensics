'''
Farzana Ahamed Bhuiyan (Lead) 
Akond Rahman 
Oct 20, 2020 
Executes the pattern matching and data flow analysis 
'''

import py_parser
import constants 

def getDataLoadCount( py_file ):
    data_load_count = 0 
    py_tree = py_parser.getPythonParseObject(py_file)
    func_def_list  = py_parser.getPythonAtrributeFuncs( py_tree ) 
    print('----------------------------------------------')
    print(func_def_list)
    print('----------------------------------------------')
    for def_ in func_def_list:
        class_name, func_name, func_line, arg_call_list = def_ 
        
        if(( class_name == constants.TORCH_KW ) and (func_name == constants.LOAD_KW ) ):
            data_load_count += 1 
            print(def_)
            
        elif(( class_name == constants.DATA_KW ) and (func_name == constants.LOAD_KW ) ):
            data_load_count += 1 
            print(def_)
            
        elif(( class_name == constants.PICKLE_KW ) and (func_name == constants.LOAD_KW ) ):
            data_load_count += 1 
            print(def_)
            
        elif(( class_name == constants.JSON_KW ) and (func_name == constants.LOAD_KW ) ):
            data_load_count += 1 
            print(def_)
            
        elif(( class_name == constants.NP_KW ) and (func_name == constants.LOAD_KW ) ):
            data_load_count += 1 
            print(def_)
            
        elif(( class_name == constants.LATEST_BLOB_KW ) and (func_name == constants.DOWNLOAD_TO_FILENAME_KW ) ):
            data_load_count += 1 
            print(def_)
            
        elif(( class_name == constants.BLOB_KW ) and (func_name == constants.UPLOAD_FROM_FILENAME_KW ) ):
            data_load_count += 1 
            print(def_)
            
        elif(( class_name == constants.VISDOM_LOGGER_KW ) and (func_name == constants.LOAD_PREVIOUS_VALUES_KW ) ):
            data_load_count += 1 
            print(def_)
            
        elif(( class_name == constants.COCO_GT_KW ) and (func_name == constants.LOADRES_KW ) ):
            data_load_count += 1 
            print(def_)
            
        elif(( class_name == constants.YAML_KW ) and (func_name == constants.LOAD_KW ) ):
            data_load_count += 1 
            print(def_) 
            
        elif(( class_name == constants.HUB_KW ) and (func_name == constants.LOAD_KW ) ):
            data_load_count += 1 
            print(def_)
            
        elif(( class_name == constants.DATA_LOADER_FACTORY_KW ) and (func_name == constants.GET_DATA_LOADER_KW ) ):
            data_load_count += 1 
            print(def_)
            
        elif(( class_name == constants.IO_KW ) and (func_name == constants.READ_FILE_KW ) ):
            data_load_count += 1 
            print(def_)
            
        elif(( class_name == constants.DATASET_KW ) and (func_name == constants.TENSOR_SLICE_KW ) ):
            data_load_count += 1 
            print(def_)
            
        elif(( class_name == constants.SP_MODEL_KW ) and (func_name == constants.LOAD_CAPITAL_KW) ):
            data_load_count += 1 
            print(def_)
            
        elif(( class_name == constants.TAGGING_DATA_LOADER_KW ) and (func_name == constants.LOAD_KW ) ):
            data_load_count += 1 
            print(def_)
            
        elif(( class_name == constants.PD_KW ) and (func_name == constants.READ_CSV_KW ) ):
            data_load_count += 1 
            print(def_)
            
        elif(( class_name == constants.FILES_KW ) and (func_name == constants.LOAD_FILES_LIST_KW ) ):
            data_load_count += 1 
            print(def_)
            
        elif(( class_name == constants.IBROSA_KW ) and (func_name == constants.LOAD_KW ) ):
            data_load_count += 1 
            print(def_)
            
        elif(( class_name == constants.DATA_UTILS_KW ) and (func_name == constants.LOAD_CELEBA_KW ) ):
            data_load_count += 1 
            print(def_)
            
        elif(( class_name == constants.DSET_KW ) and (func_name == constants.MNIST_KW ) ):
            data_load_count += 1 
            print(def_)
            
        elif(( class_name == constants.TARFILE_KW ) and (func_name == constants.OPEN_KW ) ):
            data_load_count += 1 
            print(def_)
            
        elif(( class_name == constants.AUDIO_KW ) and (func_name == constants.LOAD_WAV_KW ) ):
            data_load_count += 1 
            print(def_)
            
        elif(( class_name == constants.IMAGE_KW) and (func_name == constants.OPEN_KW ) ):
            data_load_count += 1 
            print(def_)    
                    
        elif(( class_name == constants.REPLAY_BUFFER_KW ) and (func_name == constants.LOAD_KW ) ):
            data_load_count += 1 
            print(def_)
            
        elif(( class_name == constants.H5PY_KW ) and (func_name == constants.FILE_KW ) ):
            data_load_count += 1 
            print(def_)

    # LOGGING_IS_ON_FLAG = py_parser.checkLogging( py_tree,  func_def_list, 'akond' )
    # this will be used to check if the file_name passed in as file to read, is logged  
    LOGGING_IS_ON_FLAG = py_parser.checkLoggingPerData( py_tree, constants.DUMMY_LOG_KW ) 
    print(LOGGING_IS_ON_FLAG, data_load_count) 
    return data_load_count 
    
    
def getDataLoadCountb( py_file ):
    data_load_countb = 0 
    py_tree = py_parser.getPythonParseObject(py_file)
    func_assign_list  = py_parser.getFunctionAssignments( py_tree ) 
    print('----------------------------------------------')
    print(func_assign_list)
    print('----------------------------------------------')
    for assign_ in func_assign_list:
        lhs, func_name, func_line, func_arg_list = assign_ 
        
        if( (func_name == constants.GET_LOADER_KW ) and (len(func_arg_list) > 0) ):
            data_load_countb += 1 
            print(assign_)
        
        elif( (func_name == constants.FROM_BUFFER_KW ) and (len(func_arg_list) > 0) ):
            data_load_countb += 1 
            print(assign_)
            
    LOGGING_IS_ON_FLAG = py_parser.checkLoggingPerData( py_tree, constants.DUMMY_LOG_KW ) 
    print(LOGGING_IS_ON_FLAG, data_load_countb) 
    return data_load_countb 


def getDataLoadCountc( py_file ):
    data_load_countc = 0 
    py_tree = py_parser.getPythonParseObject(py_file)
    func_assign_list  = py_parser.getFunctionDefinitions( py_tree ) 
    print('----------------------------------------------')
    print(func_assign_list)
    print('----------------------------------------------')
    for func_ in func_assign_list:
        func_name, func_line, func_arg_list = func_ 
        
        if( (func_name == constants.LOAD_RANDOMLY_AUGMENTED_AUDIO_KW ) and (len(func_arg_list) > 0) ):
            data_load_countc += 1 
            print(func_)
            
        elif( (func_name == constants._DOWNLOAD_KW ) and (len(func_arg_list) > 0) ):
            data_load_countc += 1 
            print(func_)
            
        elif( (func_name == constants.OPEN_KW ) and (len(func_arg_list) > 0) ):
            data_load_countc += 1 
            print(func_)
            
        elif( (func_name == constants.LOAD_KW ) and (len(func_arg_list) > 0) ):
            data_load_countc += 1 
            print(func_)
            
        elif( (func_name == constants.LOAD_GENERIC_AUDIO_KW ) and (len(func_arg_list) > 0) ):
            data_load_countc += 1 
            print(func_)
            
        elif( (func_name == constants.LOAD_AUDIO_KW ) and (len(func_arg_list) > 0) ):
            data_load_countc += 1 
            print(func_)
            
        elif( (func_name == constants.LOAD_IMAGE_DATASET_KW ) and (len(func_arg_list) > 0) ):
            data_load_countc += 1 
            print(func_)
            
        elif( (func_name == constants.DOWNLOAD_FROM_URL_KW ) and (len(func_arg_list) > 0) ):
            data_load_countc += 1 
            print(func_)
            
        elif( (func_name == constants.GET_RAW_FILES_KW ) and (len(func_arg_list) > 0) ):
            data_load_countc += 1 
            print(func_)
            
        elif( (func_name == constants.LOAD_VOCAB_FILE_KW ) and (len(func_arg_list) > 0) ):
            data_load_countc += 1 
            print(func_)
            
        elif( (func_name == constants.LOAD_ATTRIBUTE_DATASET_KW ) and (len(func_arg_list) > 0) ):
            data_load_countc += 1 
            print(func_)
            
        elif( (func_name == constants.READ_H5FILE_KW ) and (len(func_arg_list) > 0) ):
            data_load_countc += 1 
            print(func_)
            
        elif( (func_name == constants.LOAD_LUA_KW) and (len(func_arg_list) > 0) ):
            data_load_countc += 1 
            print(func_)
            
    LOGGING_IS_ON_FLAG = py_parser.checkLoggingPerData( py_tree, constants.DUMMY_LOG_KW ) 
    print(LOGGING_IS_ON_FLAG, data_load_countc) 
    return data_load_countc 


def getModelLoadCounta( py_file ):
    model_load_counta = 0 
    py_tree = py_parser.getPythonParseObject(py_file)
    func_def_list  = py_parser.getPythonAtrributeFuncs( py_tree ) 
    print('----------------------------------------------')
    print(func_def_list)
    print('----------------------------------------------')
    for def_ in func_def_list:
        class_name, func_name, func_line, arg_call_list = def_ 
        
        if(( class_name == constants.DEEP_SPEECH_KW ) and (func_name == constants.LOAD_MODEL_PACKAGE_KW) ):
            model_load_counta += 1 
            print(def_)
        
        elif(( class_name == constants.MODELS_KW ) and (func_name == constants.LOAD_MODEL_KW) ):
            model_load_counta += 1 
            print(def_)
            
        elif(( class_name == constants.MODEL_KW ) and (func_name == constants.LOAD_STATE_DICT_KW) ):
            model_load_counta += 1 
            print(def_)
            
        elif(( class_name == constants.NETWORK_KW ) and (func_name == constants.LOAD_NET_KW) ):
            model_load_counta += 1 
            print(def_)
            
        elif(( class_name == constants.VGG_KW ) and (func_name == constants.LOAD_FROM_NPY_FILE_KW) ):
            model_load_counta += 1 
            print(def_)
            
        elif(( class_name == constants.CAFFE_PARSER_KW ) and (func_name == constants.READ_CAFFE_MODEL_KW) ):
            model_load_counta += 1 
            print(def_)
            
        elif(( class_name == constants.TRAIN_KW ) and (func_name == constants.CHECK_POINT_KW) ):
            model_load_counta += 1 
            print(def_)
            
        elif(( class_name == constants.TF_HUB_KW ) and (func_name == constants.LOAD_KW) ):
            model_load_counta += 1 
            print(def_)
            
        elif(( class_name == constants.MISC_KW ) and (func_name == constants.IMRE_SIZE_KW) ):
            model_load_counta += 1 
            print(def_)
            
    LOGGING_IS_ON_FLAG = py_parser.checkLoggingPerData( py_tree, constants.DUMMY_LOG_KW )    
    print(LOGGING_IS_ON_FLAG, model_load_counta) 
    return model_load_counta 
    
    
def getModelLoadCountb( py_file ):
    model_load_countb = 0 
    py_tree = py_parser.getPythonParseObject(py_file)
    func_assign_list  = py_parser.getFunctionAssignments( py_tree ) 
    print('----------------------------------------------')
    print(func_assign_list)
    print('----------------------------------------------')
    for assign_ in func_assign_list:
        lhs, func_name, func_line, func_arg_list = assign_ 
        
        if( (func_name == constants.PATCH_PATH_KW  ) and (len(func_arg_list) > 0) ):
            model_load_countb += 1 
            print(assign_)
        
        elif( (func_name == constants.CAFFE_FUNCTION_KW ) and (len(func_arg_list) > 0) ):
            model_load_countb += 1 
            print(assign_)
            
    LOGGING_IS_ON_FLAG = py_parser.checkLoggingPerData( py_tree, constants.DUMMY_LOG_KW ) 
    print(LOGGING_IS_ON_FLAG, model_load_countb) 
    return model_load_countb 
    
    
def getModelLoadCountc( py_file ):
    model_load_countc = 0 
    py_tree = py_parser.getPythonParseObject(py_file)
    func_assign_list  = py_parser.getFunctionDefinitions( py_tree ) 
    print('----------------------------------------------')
    print(func_assign_list)
    print('----------------------------------------------')
    for func_ in func_assign_list:
        func_name, func_line, func_arg_list = func_ 
        
        if( (func_name == constants.LOAD_MODEL_KW ) and (len(func_arg_list) > 0) ):
            model_load_countc += 1 
            print(func_)
            
        elif( (func_name == constants.LOAD_DECODER_KW ) and (len(func_arg_list) > 0) ):
            model_load_countc += 1 
            print(func_)
            
        elif( (func_name == constants.LOAD_PREVIOUS_VALUES_KW ) and (len(func_arg_list) > 0) ):
            model_load_countc += 1 
            print(func_)
            
        elif( (func_name == constants.LOAD_PRETRAINED_KW ) and (len(func_arg_list) > 0) ):
            model_load_countc += 1 
            print(func_)
            
        elif( (func_name == constants.LOAD_PARAM_KW ) and (len(func_arg_list) > 0) ):
            model_load_countc += 1 
            print(func_)
            
    LOGGING_IS_ON_FLAG = py_parser.checkLoggingPerData( py_tree, constants.DUMMY_LOG_KW ) 
    print(LOGGING_IS_ON_FLAG, model_load_countc) 
    return model_load_countc 
    
    
def getModelLoadCountd( py_file ):
    model_load_countd = 0 
    py_tree = py_parser.getPythonParseObject(py_file)
    func_assign_list  = py_parser.getFunctionAssignmentsWithMultipleLHS( py_tree ) 
    print('----------------------------------------------')
    print(func_assign_list)
    print('----------------------------------------------')
    for assign_ in func_assign_list:
        lhs, func_name, func_line, func_arg_list = assign_ 
        
        if( (func_name == constants.SEQ_LABEL_KW  ) and (len(func_arg_list) > 0) ):
            model_load_countd += 1 
            print(assign_)
        
        elif( (func_name == constants.LOAD_CHECKPOINT_KW ) and (len(func_arg_list) > 0) ):
            model_load_countd += 1 
            print(assign_)
            
    LOGGING_IS_ON_FLAG = py_parser.checkLoggingPerData( py_tree, constants.DUMMY_LOG_KW ) 
    print(LOGGING_IS_ON_FLAG, model_load_countd) 
    return model_load_countd 
    
    
def getDataDownLoadCount( py_file ):
    data_download_count = 0 
    py_tree = py_parser.getPythonParseObject(py_file)
    func_def_list  = py_parser.getPythonAtrributeFuncs( py_tree ) 
    print('----------------------------------------------')
    print(func_def_list)
    print('----------------------------------------------')
    for def_ in func_def_list:
        class_name, func_name, func_line, arg_call_list = def_ 
        
        if(( class_name == constants.WGET_KW ) and (func_name == constants.DOWNLOAD_KW ) ):
            data_download_count += 1 
            print(def_)
            
        elif(( class_name == constants.REQUEST_KW ) and (func_name == constants.URL_OPEN_KW ) ):
            data_download_count += 1 
            print(def_)
            
        elif(( class_name == constants.MODEL_ZOO_KW ) and (func_name == constants.LOAD_URL_KW ) ):
            data_download_count += 1 
            print(def_)
            
        elif(( class_name == constants.URL_LIB_KW  ) and (func_name == constants.URL_RETRIEVE_KW ) ):
            data_download_count += 1 
            print(def_)
            
        elif(( class_name == constants.AGENT_KW ) and (func_name == constants.LOAD_KW ) ):
            data_download_count += 1 
            print(def_)
            
    LOGGING_IS_ON_FLAG = py_parser.checkLoggingPerData( py_tree, constants.DUMMY_LOG_KW ) 
    print(LOGGING_IS_ON_FLAG, data_download_count) 
    return data_download_count 
    
    
def getDataDownLoadCountb( py_file ):
    data_download_countb = 0 
    py_tree = py_parser.getPythonParseObject(py_file)
    func_assign_list  = py_parser.getFunctionDefinitions( py_tree ) 
    print('----------------------------------------------')
    print(func_assign_list)
    print('----------------------------------------------')
    for func_ in func_assign_list:
        func_name, func_line, func_arg_list = func_ 
        
        if( (func_name == constants.PREPARE_URL_IMAGE_KW ) and (len(func_arg_list) > 0) ):
            data_download_countb += 1 
            print(func_)
            
    LOGGING_IS_ON_FLAG = py_parser.checkLoggingPerData( py_tree, constants.DUMMY_LOG_KW ) 
    print(LOGGING_IS_ON_FLAG, data_download_countb) 
    return data_download_countb
            
            
def getModelFeatureCount( py_file ):
    model_feature_count = 0 
    py_tree = py_parser.getPythonParseObject(py_file)
    feature_list  = py_parser.getModelFeature( py_tree ) 
    print('----------------------------------------------')
    print(feature_list)
    print('----------------------------------------------')
    for feature_ in feature_list:
        lhs, class_name, feature_name, feature_line = feature_ 
        
        if( (class_name == constants.DATA_KW ) and (feature_name == constants.HP_BATCH_SIZE_KW ) ):
            model_feature_count += 1 
            print(feature_)
            
    LOGGING_IS_ON_FLAG = py_parser.checkLoggingPerData( py_tree, constants.DUMMY_LOG_KW ) 
    print(LOGGING_IS_ON_FLAG,  model_feature_count) 
    return model_feature_count
    

def getModelLabelCount( py_file ):
    model_label_count = 0 
    py_tree = py_parser.getPythonParseObject(py_file)
    func_assign_list  = py_parser.getFunctionAssignmentsWithMultipleLHS( py_tree ) 
    print('----------------------------------------------')
    print(func_assign_list)
    print('----------------------------------------------')
    for assign_ in func_assign_list:
        lhs, func_name, func_line, func_arg_list = assign_ 
        
        if( (func_name == constants.READ_H5FILE_KW  ) and (len(func_arg_list) > 0) ):
            model_label_count += 1 
            print(assign_)
            
        elif( (func_name == constants.ARRAY_KW ) and (len(func_arg_list) > 0) ):
            model_label_count += 1 
            print(assign_)
            
        elif( (func_name == constants.CONVERT_KW ) and (len(func_arg_list) > 0) ):
            model_label_count += 1 
            print(assign_)
            
        elif( (func_name == constants.AS_TYPE_KW ) and (len(func_arg_list) > 0) ):
            model_label_count += 1 
            print(assign_)
            
        elif( (func_name == constants.BASENAME_KW ) and (len(func_arg_list) > 0) ):
            model_label_count += 1 
            print(assign_)
            
        elif( (func_name == constants.LOAD_DATA_AND_LABELS_KW ) and (len(func_arg_list) > 0) ):
            model_label_count += 1 
            print(assign_)
            
        elif( (func_name == constants.CREATE_DATASET_KW ) and (len(func_arg_list) > 0) ):
            model_label_count += 1 
            print(assign_)
            
    LOGGING_IS_ON_FLAG = py_parser.checkLoggingPerData( py_tree, constants.DUMMY_LOG_KW ) 
    print(LOGGING_IS_ON_FLAG, model_label_count) 
    return model_label_count 
    

def getModelLabelCountb( py_file ):
    model_label_countb = 0 
    py_tree = py_parser.getPythonParseObject(py_file)
    func_assign_list  = py_parser.getTupAssiDetails( py_tree ) 
    print('----------------------------------------------')
    print(func_assign_list)
    print('----------------------------------------------')
    for assign_ in func_assign_list:
        lhs, var_s, var_d, rhs_var_iter, func_line = assign_ 
        
        if ( (var_s == constants.SENT_KW ) and (var_d == constants.SENT_KW )  and (rhs_var_iter == constants.INPUT_BATCH_LIST_KW ) ):
            model_label_countb += 1 
            print(assign_)
            
    LOGGING_IS_ON_FLAG = py_parser.checkLoggingPerData( py_tree, constants.DUMMY_LOG_KW ) 
    print(LOGGING_IS_ON_FLAG, model_label_countb) 
    return model_label_countb 
    
    
def getModelOutputCount( py_file ):
    model_output_count = 0 
    py_tree = py_parser.getPythonParseObject(py_file)
    func_def_list  = py_parser.getPythonAtrributeFuncs( py_tree ) 
    print('----------------------------------------------')
    print(func_def_list)
    print('----------------------------------------------')
    for def_ in func_def_list:
        class_name, func_name, func_line, arg_call_list = def_ 
        
        if(( class_name == constants.MODEL_KW ) and (func_name == constants.SUMMARY_KW ) ):
            model_output_count += 1 
            print(def_)
            
        elif(( class_name == constants.DATA_KW ) and (func_name == constants.SHOW_DATA_SUMMARY_KW ) ):
            model_output_count += 1 
            print(def_)
            
    LOGGING_IS_ON_FLAG = py_parser.checkLoggingPerData( py_tree, constants.DUMMY_LOG_KW ) 
    print(LOGGING_IS_ON_FLAG, model_output_count) 
    return model_output_count 
    

def getModelOutputCountb( py_file ):
    model_output_countb = 0 
    py_tree = py_parser.getPythonParseObject(py_file)
    func_assign_list  = py_parser.getFunctionAssignments( py_tree ) 
    print('----------------------------------------------')
    print(func_assign_list)
    print('----------------------------------------------')
    for assign_ in func_assign_list:
        lhs, func_name, func_line, func_arg_list = assign_ 
        
        if( (func_name == constants.GET_TENSOR_KW ) and (len(func_arg_list) > 0) ):
            model_output_countb += 1 
            print(assign_)
            
        elif( (func_name == constants.EVALUATE_KW ) and (len(func_arg_list) > 0) ):
            model_output_countb += 1 
            print(assign_)
                          
        elif(( func_name == constants.EVAL_KW ) ):
            model_output_countb += 1 
            print(assign_)
            
    LOGGING_IS_ON_FLAG = py_parser.checkLoggingPerData( py_tree, constants.DUMMY_LOG_KW ) 
    print(LOGGING_IS_ON_FLAG, model_output_countb) 
    return model_output_countb 
    
    
def getModelOutputCountc( py_file ):
    model_output_countc = 0 
    py_tree = py_parser.getPythonParseObject(py_file)
    func_assign_list  = py_parser.getFunctionAssignments( py_tree ) 
    print('----------------------------------------------')
    print(func_assign_list)
    print('----------------------------------------------')
    for func_ in func_assign_list:
        lhs, func_name, func_line, func_arg_list = func_ 
        
        if( (func_name == constants.CONFUSION_MATRIX_KW ) and (len(func_arg_list) > 0) ):
            model_output_countc += 1 
            print(func_)
            
        elif( (func_name == constants.F1_SCORE_KW ) and (len(func_arg_list) > 0) ):
            model_output_countc += 1 
            print(func_)
            
        elif( (func_name == constants.ACCURACY_SCORE_KW ) and (len(func_arg_list) > 0) ):
            model_output_countc += 1 
            print(func_)
            
        elif( (func_name == constants.CLASSIFICATION_LOSS_KW ) and (len(func_arg_list) > 0) ):
            model_output_countc += 1 
            print(func_)
            
    LOGGING_IS_ON_FLAG = py_parser.checkLoggingPerData( py_tree, constants.DUMMY_LOG_KW ) 
    print(LOGGING_IS_ON_FLAG, model_output_countc) 
    return model_output_countc 
    
    
def getDataPipelineCount( py_file ):
    data_pipeline_count = 0 
    py_tree = py_parser.getPythonParseObject(py_file)
    func_def_list  = py_parser.getPythonAtrributeFuncs( py_tree ) 
    print('----------------------------------------------')
    print(func_def_list)
    print('----------------------------------------------')
    for def_ in func_def_list:
        class_name, func_name, func_line, arg_call_list = def_ 
        
        if(( class_name == constants.ARG_PARSE_KW ) and (func_name == constants.ARGUMENT_PARSER_KW ) and (len(arg_call_list) > 0)):
            data_pipeline_count += 1 
            print(def_)
            
    LOGGING_IS_ON_FLAG = py_parser.checkLoggingPerData( py_tree, constants.DUMMY_LOG_KW ) 
    print(LOGGING_IS_ON_FLAG, data_pipeline_count) 
    return data_pipeline_count 
    
    
def getDataPipelineCountb( py_file ):
    data_pipeline_countb = 0 
    py_tree = py_parser.getPythonParseObject(py_file)
    func_assign_list  = py_parser.getFunctionAssignments( py_tree ) 
    print('----------------------------------------------')
    print(func_assign_list)
    print('----------------------------------------------')
    for assign_ in func_assign_list:
        lhs, func_name, func_line, func_arg_list = assign_ 
        
        if( (func_name == constants.TRAIN_EVAL_PIPELINE_CONFIG_KW ) ):
            data_pipeline_countb += 1 
            print(assign_)
            
    LOGGING_IS_ON_FLAG = py_parser.checkLoggingPerData( py_tree, constants.DUMMY_LOG_KW ) 
    print(LOGGING_IS_ON_FLAG, data_pipeline_countb) 
    return data_pipeline_countb 


def getDataPipelineCountc( py_file ):
    data_pipeline_countc = 0 
    py_tree = py_parser.getPythonParseObject(py_file)
    func_assign_list  = py_parser.getFunctionDefinitions( py_tree ) 
    print('----------------------------------------------')
    print(func_assign_list)
    print('----------------------------------------------')
    for func_ in func_assign_list:
        func_name, func_line, func_arg_list = func_ 
        
        if( (func_name == constants.GET_CONFIGS_FROM_PIPELINE_FILE_KW ) and (len(func_arg_list) > 0) ):
            data_pipeline_countc += 1 
            print(func_)
            
    LOGGING_IS_ON_FLAG = py_parser.checkLoggingPerData( py_tree, constants.DUMMY_LOG_KW ) 
    print(LOGGING_IS_ON_FLAG, data_pipeline_countc) 
    

def getDataPipelineCountd( py_file ):
	data_pipeline_countd = 0 
	py_tree = py_parser.getPythonParseObject(py_file)
	feature_list  = py_parser.getModelFeature( py_tree ) 
	print('----------------------------------------------')
	print(feature_list)
	print('----------------------------------------------')
	for feature_ in feature_list:
		lhs, class_name, feature_name, feature_line = feature_ 
		
		if( (class_name == constants.PIPELINE_CONFIG_KW  ) and (feature_name == constants.MODEL_KW ) ):
			data_pipeline_countd += 1 
			print(feature_)
			
	LOGGING_IS_ON_FLAG = py_parser.checkLoggingPerData( py_tree, constants.DUMMY_LOG_KW ) 
	print(LOGGING_IS_ON_FLAG,  data_pipeline_countd) 
	return data_pipeline_countd
    

def getExcepts( py_file ) :
    py_tree = py_parser.getPythonParseObject(py_file)
    except_list  = py_parser.getPythonExcepts( py_tree )  
    except_func_list = py_parser.checkAttribFuncsInExcept( except_list )    
    EXCEPT_LOGGING_IS_ON_FLAG = py_parser.checkExceptLogging( except_func_list )      
    print(EXCEPT_LOGGING_IS_ON_FLAG) 