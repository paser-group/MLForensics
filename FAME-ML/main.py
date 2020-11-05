'''
Farzana Ahamed Bhuiyan (Lead) 
Akond Rahman 
Oct 20, 2020 
Main executor 
'''
import lint_engine

if __name__=='__main__':
    TEST_ML_SCRIPT = '../Data/sample.py' 
    # the followign checks all data loading related methods 
    
    print("*"*100)
    print("Section 1.1a")
    data_load_count = lint_engine.getDataLoadCount( TEST_ML_SCRIPT ) 
    print(data_load_count)
    
    print("*"*100)
    print("Section 1.1b")
    data_load_countb = lint_engine.getDataLoadCountb( TEST_ML_SCRIPT ) 
    print(data_load_countb) 
    
    print("*"*100)
    print("Section 1.1c")
    data_load_countc = lint_engine.getDataLoadCountc( TEST_ML_SCRIPT ) 
    print(data_load_countc) 
    
    # the following checks except related blocks 
    lint_engine.getExcepts( TEST_ML_SCRIPT ) 