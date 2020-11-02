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
    # print(func_def_list)
    for def_ in func_def_list:
        class_name, func_name, func_line, arg_call_list = def_ 
        if(( class_name == constants.DATA_KW ) and (func_name == constants.LOAD_KW ) and (len(arg_call_list) > 0) ):
            data_load_count += 1 

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