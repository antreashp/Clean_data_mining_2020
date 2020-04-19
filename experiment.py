import numpy as np
from preprosses import preprocess
from train_model import train_xgb,train_mlp

import torch

from sklearn.model_selection import train_test_split
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Experiment():
    def __init__(self,options = None):
        
        self.model_type = options['model_type']
        self.data =  options['data']
        self.win_size = options['win_size']
        self.transform_targets = options['transform_targets']
        self.model_options = options['model_options']
        if self.data is None:
            self.methods  = ['average', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max',
                   'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'average', 'average',
                   'average', 'average', 'average', 'average', 'average', 'average', 'average', 'average'] if options['methods'] is None else options['methods']
            filename = 'data/RAW_Data.pickle'
            self.preprocess_instance = preprocess(filename, window_size=self.win_size,transform_appcat='arctan',appcat_scale=1/60, methods=self.methods)
            self.preprocess_instance.normalize()

            self.preprocess_instance.bin(include_remainder=False)
            processed_df = self.preprocess_instance.create_dataframe_pros()
        else:
            processed_df = options['data']
        indexNames = processed_df[ processed_df['user_id'] =='AS14.7' ].index
        processed_df= processed_df.drop(indexNames )


        print('mean',processed_df['appCat.communication'].mean())

        # processed_df = self.preprocess_instance
        # if self.transform_targets:
            # self.preprocess_instance.transform_target()
        

        self.split = options['split']
        X,y = self.df_to_np(processed_df)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.split , random_state=42)
    def update_opt(self,options):
        self.model_options = options 
    def df_to_np(self,df):
        X = []
        y =[]
        for index, row in df.iterrows():
            # print(index)
            # print(list(row))
            rowl = list(row)
            X.append(rowl[3:])
            y.append(rowl[2])
        X = np.array(X)
        y = np.array(y)
        return X,y
    
    def train_and_test(self):
        if self.model_type == 'xgb':
            res,loss = train_xgb(self.model_options,self.X_train, self.X_test, self.y_train, self.y_test)
        elif self.model_type == 'mlp':
            res,loss = train_mlp(self.model_options,self.X_train, self.X_test, self.y_train, self.y_test)
        return res,loss

if __name__ == "__main__":

    model_type = 'mlp'
    trans_trg = False
    win_size = 1
    mod_opt_mlp ={'exp_name'      : None, #default if dont want to specify 
              'win_size'      : win_size,
              'batch_size'    : 128,
              'epochs'        : 100,
              'lr'            : 0.0338,
              'use_pca'       : True,
              'pca_var_hold'  : 0.995,
              'model_type'    : 'reg', #'cls'
              'transform_targets'  : trans_trg,
              'loss_fn'       : 'mse', #cross-entropy
              'optim'         : 'adam',#sgd
              'use_scheduler' : False, #true decreaseing  
              'debug_mode'    : False 
    }
    # mod_opt_xgb ={'max_depth'      : 15, 
    #           'aplha'              : 100,
    #           'colsample_bytree'   : 0.1,
    #           'n_estimators'       : 10000,
    #           'lr'                 : 0.0338,
    #           'use_pca'            : False,
    #           'pca_var_hold'       : 0.995,
    #           'transform_targets'  : trans_trg,
    #           'max_delta_step'     : 2000, 
    #           'gamma'              : 0.01 

    # }

    options ={'model_type'        : model_type, # xgb  mlp
              'win_size'          : 3,
              'batch_size'        : 128,
              'data'              : None,
              'transform_targets' : trans_trg,
              'split' : 0.2,
              'model_options' : mod_opt_xgb if 'xgb' in model_type else  mod_opt_mlp,
              'methods'       : None
    }
              
    exp = Experiment(options)
    res,loss = exp.train_and_test()
    print(res,loss)





