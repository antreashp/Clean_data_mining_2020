from experiment import Experiment
import skopt

from preprosses import preprocess



HPO_PARAMS = {'n_calls':100,
              'n_random_starts':10,
              'base_estimator':'ET',
              'acq_func':'EI',
              'xi':0.02,
              'kappa':1.96,
              'n_points':10000,
             }
methods = ['average','max','max','max','max','max','max','max','max','max',
'max','max','max','max','max','max','max','max','average','average',
'average','average','average','average','average','average','average','average']

win_size = 3
batch_size =128

filename = 'data/RAW_Data.pickle'
preprocess_instance = preprocess(filename, window_size=win_size, methods=methods)
preprocess_instance.normalize()
data = preprocess_instance.bin(include_remainder=False)

SPACE =[skopt.space.Real(0.001, 0.1, name='lr', prior='log-uniform'),
        skopt.space.Integer(5, 10, name='max_depth'),
        skopt.space.Real(0.001, 0.1, name='gamma'),
        skopt.space.Integer(10, 200, name='alpha'),
        skopt.space.Real(0.01, 0.3, name='colsample_bytree'),
        skopt.space.Categorical([64, 128,512,1024,4096], name='n_estimators'),
        skopt.space.Categorical([True,False], name='use_pca'),
        skopt.space.Categorical([True,False], name='transform_targets'),]


# exp = Experiment(options)
# res = exp.train_and_test()
# print(res)

@skopt.utils.use_named_args(SPACE)
def search(**params):
        all_params = params
        global data 

        mod_opt_xgb = { 'max_depth'          : all_params['max_depth'], 
                        'aplha'              : all_params['alpha'],
                        'colsample_bytree'   : all_params['colsample_bytree'],
                        'n_estimators'       : all_params['n_estimators'],
                        'lr'                 : all_params['lr'],
                        'use_pca'            : all_params['use_pca'],
                        'pca_var_hold'       : 0.995,
                        'transform_targets'  : all_params['transform_targets'],
                        'max_delta_step'     : 2000, 
                        'gamma'              : all_params['gamma'] 

        }

        options = {     'model_type'        : 'xgb', # xgb  mlp
                        'win_size'          : win_size,
                        'batch_size'        : 128,
                        'data'              : data,
                        'transform_targets' : all_params['transform_targets'],
                        'split' : 0.2,
                        'model_options' : mod_opt_xgb ,
                        'methods'       : None
        }
        options.update(mod_opt_xgb)

        exp = Experiment(options)




results = skopt.forest_minimize(search, SPACE, **HPO_PARAMS)