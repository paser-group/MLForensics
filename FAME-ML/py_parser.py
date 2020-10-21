'''
Farzana Ahamed Bhuiyan (Lead) 
Akond Rahman 
Oct 20, 2020 
Parser needed to implement FAME-ML 
'''

import ast 
import os 
import constants 


def getPythonParseObject( pyFile ): 
    full_tree = ast.parse( open( pyFile  ).read() )    
    return full_tree 

def getPythonAtrributeFuncs(pyTree):
    attrib_call_list = []
    for stmt_ in pyTree.body:
        for node_ in ast.walk(stmt_):
            if isinstance(node_, ast.Call):
                funcDict = node_.__dict__ 
                func_, funcArgs, funcLineNo =  funcDict[ constants.FUNC_KW ], funcDict[constants.ARGS_KW], funcDict[constants.LINE_NO_KW] 
                if( isinstance(func_, ast.Attribute ) ):
                    func_as_attrib_dict = func_.__dict__ 
                    # print(func_as_attrib_dict ) 
                    func_name    = func_as_attrib_dict[constants.ATTRIB_KW] 
                    func_parent  = func_as_attrib_dict[constants.VALUE_KW]
                    if( isinstance(func_parent, ast.Name )   ):                    
                        attrib_call_list.append( ( func_parent.id, func_name , funcLineNo  ) )
    return attrib_call_list 