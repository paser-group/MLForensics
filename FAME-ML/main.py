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
    
    print("*"*100)
    print("Section 1.2a")
    model_load_counta = lint_engine.getModelLoadCounta( TEST_ML_SCRIPT ) 
    print(model_load_counta)
    
    
    print("*"*100)
    print("Section 1.2b")
    model_load_countb = lint_engine.getModelLoadCountb( TEST_ML_SCRIPT ) 
    print(model_load_countb) 
    
    print("*"*100)
    print("Section 1.2c")
    model_load_countc = lint_engine.getModelLoadCountc( TEST_ML_SCRIPT ) 
    print(model_load_countc) 
    
    print("*"*100)
    print("Section 1.2d")
    model_load_countd = lint_engine.getModelLoadCountd( TEST_ML_SCRIPT ) 
    print(model_load_countd)
    
    print("*"*100)
    print("Section 2.1a")
    data_download_count = lint_engine.getDataDownLoadCount( TEST_ML_SCRIPT ) 
    print(data_download_count)

    print("*"*100)
    print("Section 2.2b")
    data_download_countb = lint_engine.getDataDownLoadCountb( TEST_ML_SCRIPT ) 
    print(data_download_countb) 
    
    # the following checks except related blocks 
    lint_engine.getExcepts( TEST_ML_SCRIPT ) 