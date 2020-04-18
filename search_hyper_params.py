from experiment import Experiment
import skopt



model_type = 'mlp'
trans_trg = False
win_size = 1
mod_opt_mlp ={'exp_name'      : None, #default if dont want to specify 
            'win_size'      : win_size,
            'batch_size'    : 128,
            'epochs'        : 50,
            'lr'            : 0.0003,
            'use_pca'       : False,
            'pca_var_hold'  : 0.995,
            'model_type'    : 'reg', #'cls'
            'transform_targets'  : trans_trg,
            'loss_fn'       : 'mse', #cross-entropy
            'optim'         : 'adam',#sgd
            'use_scheduler' : False, #true decreaseing  
            'debug_mode'    : False 
}
mod_opt_xgb ={'max_depth'      : 15, 
            'aplha'              : 100,
            'colsample_bytree'   : 0.1,
            'n_estimators'       : 10000,
            'lr'                 : 0.1,
            'use_pca'            : False,
            'pca_var_hold'       : 0.995,
            'transform_targets'  : trans_trg,
            'max_delta_step'     : 2000, 
            'gamma'              : 0.01 

}

options ={'model_type'        : model_type, # xgb  mlp
            'win_size'          : win_size,
            'batch_size'        : 128,
            'data'              : None,
            'transform_targets' : trans_trg,
            'split' : 0.2,
            'model_options' : mod_opt_xgb if 'xgb' in model_type else  mod_opt_mlp,
            'methods'       : None
}
SPACE =[skopt.space.Real(0.001, 0.1, name='lr', prior='log-uniform'),
        skopt.space.Integer(5, 9, name='batch_size'),
        skopt.space.Integer(7, 11, name='hid_layer1'),
        skopt.space.Integer(7, 11, name='hid_layer2'),
        skopt.space.Integer(0, 2, name='dim_redu_params'),
        skopt.space.Integer(0, 3, name='name'),]
exp = Experiment(options)
res = exp.train_and_test()
print(res)





