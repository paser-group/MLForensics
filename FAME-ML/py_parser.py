'''
Farzana Ahamed Bhuiyan (Lead) 
Akond Rahman 
Oct 20, 2020 
Parser needed to implement FAME-ML 
'''

import ast 
import os 
import constants 


def checkLogging(tree_object, list_func_defs, name2track):
    LOGGING_EXISTS_FLAG = False 
    IMPORT_FLAG, FUNC_FLAG, ARG_FLAG  = False, False , False 
    for stmt_ in tree_object.body:
        for node_ in ast.walk(stmt_):
            if isinstance(node_, ast.Import) :
                funcDict = node_.__dict__     
                # print(funcDict) 
                import_name_objects = funcDict[constants.NAMES_KW]
                for obj in import_name_objects:
                    if ( constants.LOGGING_KW in  obj.__dict__[constants.NAME_KW]): 
                        IMPORT_FLAG = True 
    func_decl_list = getPythonAtrributeFuncs(tree_object)
    for func_decl_ in func_decl_list:
        func_parent_id, func_name , funcLineNo, call_arg_list = func_decl_ # the class in which the method belongs, func_name, line no, arg_list 
        
        if ( constants.LOGGING_KW in func_parent_id ) or ( constants.LOGGING_KW in func_name) : 
            # print(func_parent_id, func_name, call_arg_list)  
            FUNC_FLAG = True 
            for arg_ in call_arg_list:
                if name2track in arg_:
                    ARG_FLAG = True 
    if (IMPORT_FLAG) and (FUNC_FLAG) and (ARG_FLAG):
        LOGGING_EXISTS_FLAG = True 
    return LOGGING_EXISTS_FLAG 
    




  


def getPythonParseObject( pyFile ): 
    full_tree = ast.parse( open( pyFile  ).read() )    
    return full_tree 

def getPythonAtrributeFuncs(pyTree):
    attrib_call_list  = [] 
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
                        call_arg_list = []                
                        for x_ in range(len(funcArgs)):
                            funcArg = funcArgs[x_] 
                            if( isinstance(funcArg, ast.Name ) )  :
                                call_arg_list.append( (  funcArg.id, constants.INDEX_KW + str(x_ + 1) )  ) 
                            elif( isinstance(funcArg, ast.Attribute) ): 
                                arg_dic  = funcArg.__dict__
                                arg_name = arg_dic[constants.ATTRIB_KW] 
                                call_arg_list.append( (  arg_name, constants.INDEX_KW + str(x_ + 1) )  ) 
                            elif(isinstance( funcArg, ast.Str ) ):
                                call_arg_list.append( ( funcArg.s, constants.INDEX_KW + str( x_ + 1 )  ) )
                        attrib_call_list.append( ( func_parent.id, func_name , funcLineNo, call_arg_list  ) )                
    return attrib_call_list 