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
    
    print("*"*100)
    print("Section 3.1")
    model_feature_count = lint_engine.getModelFeatureCount( TEST_ML_SCRIPT ) 
    print(model_feature_count) 
    
    print("*"*100)
    print("Section 3.2a")
    model_label_count = lint_engine.getModelLabelCount( TEST_ML_SCRIPT ) 
    print(model_label_count) 
    
    print("*"*100)
    print("Section 3.2b")
    model_label_countb = lint_engine.getModelLabelCountb( TEST_ML_SCRIPT ) 
    print(model_label_countb) 
    
    print("*"*100)
    print("Section 3.3a")
    model_output_count = lint_engine.getModelOutputCount( TEST_ML_SCRIPT ) 
    print(model_output_count) 
    
    print("*"*100)
    print("Section 3.3b")
    model_output_countb = lint_engine.getModelOutputCountb( TEST_ML_SCRIPT ) 
    print(model_output_countb) 
    
    print("*"*100)
    print("Section 3.3c")
    model_output_countc = lint_engine.getModelOutputCountc( TEST_ML_SCRIPT ) 
    print(model_output_countc) 
    
        
    print("*"*100)
    print("Section 4.1")
    data_pipeline_count = lint_engine.getDataPipelineCount( TEST_ML_SCRIPT ) 
    print(data_pipeline_count) 
    
    print("*"*100)
    print("Section 4.2")
    data_pipeline_countb = lint_engine.getDataPipelineCountb( TEST_ML_SCRIPT ) 
    print(data_pipeline_countb) 
    
    print("*"*100)
    print("Section 4.3")
    data_pipeline_countc = lint_engine.getDataPipelineCountc( TEST_ML_SCRIPT ) 
    print(data_pipeline_countc) 
    
    print("*"*100)
    print("Section 4.4")
    data_pipeline_countd = lint_engine.getDataPipelineCountd( TEST_ML_SCRIPT ) 
    print(data_pipeline_countd) 
    
    print("*"*100)
    print("Section 5.1")
    environment_count = lint_engine.getEnvironmentCount( TEST_ML_SCRIPT ) 
    print(environment_count) 
    
    print("*"*100)
    print("Section 5.1b")
    environment_countb = lint_engine.getEnvironmentCountb( TEST_ML_SCRIPT ) 
    print(environment_countb) 
    
    print("*"*100)
    print("Section 5.2")
    state_observe_count = lint_engine.getStateObserveCount( TEST_ML_SCRIPT ) 
    print(state_observe_count) 
    
    print("*"*100)
    print("Section 6.1")
    dnn_decision_count = lint_engine.getDNNDecisionCount( TEST_ML_SCRIPT ) 
    print(dnn_decision_count) 
    
    print("*"*100)
    print("Section 6.2")
    dnn_decision_countb = lint_engine.getDNNDecisionCountb( TEST_ML_SCRIPT ) 
    print(dnn_decision_countb) 
    
    # the following checks except related blocks 
    lint_engine.getExcepts( TEST_ML_SCRIPT ) 