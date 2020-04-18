
import numpy as np
import xgboost as xgb
from preprosses import preprocess
from sklearn.decomposition import PCA
from transform import Transform
def train_xgb(model_options, X_train, X_test, y_train, y_test):
    if model_options['use_pca'] :
        scaler = PCA(pca_var_hold)
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    if model_options['transform_targets']:
        TRANSF = Transform(y_train)
        y_train = TRANSF.fit(y_train)
        y_test = TRANSF.fit(y_test)
    
    xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree =model_options['colsample_bytree'], learning_rate = model_options['lr'],
                    max_depth =model_options['max_depth'], alpha = model_options['aplha'], n_estimators = model_options['n_estimators'],verbosity=1,gamma = model_options['gamma'],max_delta_step =model_options['max_delta_step'])
        
    xg_reg.fit(X_train,y_train)
    preds = xg_reg.predict(X_test)
    
    print(np.round(preds[:5],2),y_test[:5])
    if model_options['transform_targets'] :
        y_test = TRANSF.decode(y_test)
        preds = TRANSF.decode(preds)
    print(np.round(preds[:5],2),y_test[:5])
    diff = abs(preds- y_test )
    print(diff[:5])
    accuracy = (len(diff[diff<0.5]) )/preds.shape[0]
    return accuracy    
    # print(' accuracy, ',accuracy)