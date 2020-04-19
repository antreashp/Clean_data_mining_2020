from experiment import Experiment
import skopt

from preprosses import preprocess

import matplotlib.pyplot as plt

HPO_PARAMS = {'n_calls':20,
              'n_random_starts':2,
              'base_estimator':'ET',
              'acq_func':'EI',
              'xi':0.02,
              'kappa':1.96,
              'n_points':10000,
              'n_jobs':2
             }
methods = ['average','max','max','max','max','max','max','max','max','max',
'max','max','max','max','max','max','max','max','average','average',
'average','average','average','average','average','average','average','average']

win_size = 3
batch_size =128

filename = 'data/RAW_Data.pickle'
preprocess_instance = preprocess(filename, window_size=win_size, methods=methods , appcat_scale=15 / 60)
preprocess_instance.normalize()
preprocess_instance.bin(include_remainder=False)
processed_df = preprocess_instance.create_dataframe_pros()
indexNames = processed_df[ processed_df['user_id'] =='AS14.7' ].index
data= processed_df.drop(indexNames )
SPACE =[skopt.space.Real(0.0001, 0.1, name='lr', prior='log-uniform'),
        skopt.space.Integer(4, 15, name='max_depth'),
        skopt.space.Real(0.0001, 0.1, name='gamma'),
        # skopt.space.Real(0.994, 0.999, name='pca_var_hold'),
        skopt.space.Integer(10, 256, name='alpha'),
        skopt.space.Real(0.01, 0.3, name='colsample_bytree'),
        skopt.space.Integer(5, 10000, name='n_estimators'),
        skopt.space.Categorical([True,False], name='use_pca'),]
        # skopt.space.Categorical([True,False], name='transform_targets'),]

options = {     'model_type'        : 'xgb', # xgb  mlp
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

        mod_opt_xgb = { 'max_depth'          : all_params['max_depth'], 
                        'aplha'              : all_params['alpha'],
                        'colsample_bytree'   : all_params['colsample_bytree'],
                        'n_estimators'       : all_params['n_estimators'],
                        'lr'                 : all_params['lr'],
                        'use_pca'            : all_params['use_pca'],
                        'pca_var_hold'       : .995,
                        'transform_targets'  : True,
                        'max_delta_step'     : 2000, 
                        'gamma'              : all_params['gamma'] 

        }
        print('Experiment with vars:  ',mod_opt_xgb)
        # print(mod_opt_xgb)
        options = {     'model_type'        : 'xgb', # xgb  mlp
                        'win_size'          : win_size,
                        'batch_size'        : 128,
                        'data'              : data,
                        'transform_targets' : True,
                        'split'             : 0.2,
                        'model_options' : mod_opt_xgb ,
                        'methods'       : None
        }
        options.update(mod_opt_xgb)

        exp.update_opt(options)
        res,loss = exp.train_and_test()
        print('accuracy: ',res,'loss: ',loss)
        return 1 - res



from skopt.plots import plot_convergence
dummy_results = skopt.dummy_minimize(search, SPACE,n_calls=20)
skopt.dump(dummy_results, 'results/dummy_xgb_hyper.pkl')
plt.figure(1)
plot_convergence(dummy_results)
plt.savefig("plots/dummy_xgb_hyper_accinv.png")

gp_results = skopt.gp_minimize(search, SPACE,n_calls=20)
skopt.dump(gp_results, 'results/gp_xgb_hyper.pkl')
plt.figure(2)
plot_convergence(gp_results)
plt.savefig("plots/gp_xgb_hyper_accinv.png")

gbrt_results = skopt.gbrt_minimize(search, SPACE, **HPO_PARAMS)
skopt.dump(gbrt_results, 'results/gbrt_xgb_hyper.pkl')
plt.figure(3)
plot_convergence(gbrt_results)
plt.savefig("plots/gbrt_xgb_hyper_accinv.png")
# print(results)


# plot_convergence(results)
# plt.savefig("plots/xgb_hyper_accinv.png")
results = [('random_results', dummy_results),
           ('gbrt_results', gbrt_results),
           ('gp_results', gp_results)]
plt.figure(4)
plot_convergence(*results)
plt.savefig("plots/all_xgb_hyper_accinv.png")
# plt.show()