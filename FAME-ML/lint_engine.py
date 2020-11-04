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
            
        elif(( class_name == constants.WGET_KW ) and (func_name == constants.DOWNLOAD_KW ) ):
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

    # LOGGING_IS_ON_FLAG = py_parser.checkLogging( py_tree,  func_def_list, 'akond' )
    # this will be used to check if the file_name passed in as file to read, is logged  
    LOGGING_IS_ON_FLAG = py_parser.checkLoggingPerData( py_tree, constants.DUMMY_LOG_KW ) 

    print(LOGGING_IS_ON_FLAG, data_load_count) 
    return data_load_count 



def getExcepts( py_file ) :
    py_tree = py_parser.getPythonParseObject(py_file)
    except_list  = py_parser.getPythonExcepts( py_tree )  
    except_func_list = py_parser.checkAttribFuncsInExcept( except_list )    
    EXCEPT_LOGGING_IS_ON_FLAG = py_parser.checkExceptLogging( except_func_list )      
    print(EXCEPT_LOGGING_IS_ON_FLAG) 