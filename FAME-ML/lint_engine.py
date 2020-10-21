'''
Farzana Ahamed Bhuiyan (Lead) 
Akond Rahman 
Oct 20, 2020 
Executes the pattern matching and data flow analysis 
'''

import py_parser

def getDataLoadCount( py_file ):
    data_load_count = 0 
    py_tree = py_parser.getPythonParseObject(py_file)
    func_def_list = py_parser.getPythonAtrributeFuncs( py_tree ) 
    for def_ in func_def_list:
        class_name, func_name, func_line = def_ 
        if(( class_name == 'data' ) and (func_name == 'load' ) ):
            data_load_count += 1 
    return data_load_count 

