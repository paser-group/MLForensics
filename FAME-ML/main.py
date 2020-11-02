'''
Farzana Ahamed Bhuiyan (Lead) 
Akond Rahman 
Oct 20, 2020 
Main executor 
'''
import lint_engine

if __name__=='__main__':
    TEST_ML_SCRIPT = '../../Datasets/sample.test.py' 
    # the followign checks all data loading related methods 
    data_load_count = lint_engine.getDataLoadCount( TEST_ML_SCRIPT ) 
    # the following checks except related blocks 
    lint_engine.getExcepts( TEST_ML_SCRIPT ) 
    

