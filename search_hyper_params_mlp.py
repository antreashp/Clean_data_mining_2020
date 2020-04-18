from experiment import Experiment
import skopt

from preprosses import preprocess

import matplotlib.pyplot as plt

HPO_PARAMS = {'n_calls':200,
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
preprocess_instance.bin(include_remainder=False)
data = preprocess_instance.create_dataframe_pros()
SPACE =[skopt.space.Real(0.001, 0.1, name='lr', prior='log-uniform'),#
        skopt.space.Real(0.994, 0.999, name='pca_var_hold'),#
        skopt.space.Categorical(['sgd', 'adam'], name='optim'),
        skopt.space.Categorical([64, 128], name='batch_size'),
        skopt.space.Categorical([True,False], name='use_pca'),]
        # skopt.space.Categorical([True,False], name='transform_targets'),]

options = {     'model_type'        : 'mlp', # xgb  mlp
                'win_size'          : win_size,
                'batch_size'        : 128,
                'data'              : data,
                'transform_targets' : True,
                'split'             : 0.2,
                'model_options' : None ,
                'methods'       : None
}
# options.update(mod_opt_xgb)
# exp = Experiment(options)
# res = exp.train_and_test()
# print(res)
exp = Experiment(options)
@skopt.utils.use_named_args(SPACE)
def search(**params):
        global exp
        all_params = params
        global data 
        mod_opt_mlp ={'exp_name'      : None, #default if dont want to specify 
              'win_size'      : win_size,
              'batch_size'    : all_params['batch_size'],
              'epochs'        : 50,
              'lr'            :all_params['lr'],
              'use_pca'       : all_params['use_pca'],
              'pca_var_hold'  : all_params['pca_var_hold'],
              'model_type'    : 'reg', #'cls'
              'transform_targets'  : True,
              'loss_fn'       : 'mse', #cross-entropy
              'optim'         : all_params['optim'],#sgd
              'use_scheduler' : False, #true decreaseing  
              'debug_mode'    : False 
    }
       
        print('Experiment with vars:  ',mod_opt_mlp)
        # print(mod_opt_xgb)
        options = {     'model_type'        : 'xgb', # xgb  mlp
                        'win_size'          : win_size,
                        'batch_size'        : 128,
                        'data'              : data,
                        'transform_targets' : True,
                        'split'             : 0.2,
                        'model_options' : mod_opt_mlp ,
                        'methods'       : None
        }
        options.update(mod_opt_mlp)

        exp.update_opt(options)
        res = exp.train_and_test()
        print('accuracy: ',res)
        return 1 - res



results = skopt.forest_minimize(search, SPACE, **HPO_PARAMS)
print(results)

from skopt.plots import plot_convergence

plot_convergence(results)
plt.savefig("plots/mlp_hyper_accinv.png")
plt.show()