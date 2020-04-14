import itertools, os, datetime, multiprocessing, time
import numpy as np

from analyz.workflow.saving import filename_with_datetime
from analyz.optimization.dual_annealing import run_dual_annealing
from analyz.IO.npz import save_dict, load_dict
import neural_network_dynamics.main as ntwk

from model import REC_POPS, AFF_POPS, Model
from Umodel import Umodel

import warnings

residual_inclusion_threshold = 1000 # above this value of residual, we do not store the config
INITIAL_LIMS = [[1,1500],
                [1,80],
                [1,320],
                [1,40],
                [1,40],
                [1,40]]


BOUNDS = []
for i, spop in enumerate(REC_POPS+AFF_POPS):
    for j, tpop in enumerate(REC_POPS):
        BOUNDS.append(INITIAL_LIMS[i])


class StochasticSearch:
    """
    relies on a system of subfolders in data/

    data/batch1/
    data/batch2/
    ...

    """
    def __init__(self, Model, REC_POPS, AFF_POPS,
                 DA_folder='DA_test',
                 run=False,
                 desired_Vm_key='pyrExc'):

        self.Model = Model
        self.REC_POPS, self.AFF_POPS = REC_POPS, AFF_POPS
        
        self.desired_Vm_key = desired_Vm_key

        self.BOUNDS = BOUNDS
        
        self.DA_folder = os.path.join('data', DA_folder)

        if not os.path.exists(self.DA_folder):
            os.makedirs(self.DA_folder)

        self.CONFIGS, self.RESIDUAL = None, None
        self.result, self.new_result = None, None

        self.Umodel = Umodel() # initialize U-model

        if run:
            # initialize mean-field
            self.mf = ntwk.FastMeanField(self.Model, REC_POPS, AFF_POPS, tstop=6.)
            self.X0 = [0*self.mf.t for i in range(len(REC_POPS))]
            self.mf.build_TF_func(100, with_Vm_functions=True, sampling='log')
            # initialize desired Vm trace from U-model
            self.desired_Vm = self.Umodel.predict_Vm(self.mf.t, self.mf.FAFF[0,:])+\
                Model['%s_El' % self.desired_Vm_key] # in mV
        else:
            self.prev_files = [f for f in os.listdir(self.DA_folder) if f.endswith('.npy') ]

        
    ##########################################################################
    #################    ANALYSIS    ##########################################
    ##########################################################################
    
    def show_residuals_over_trials(self, graph=None, ax=None, threshold=100):

        if graph is None:
            from datavyz import ges as graph
        if ax is None:
            # fig, ax = graph.figure(axes=(1,3), wspace=.2,
            #                        left=.7, bottom=.7, figsize=(.7,1.))
            fig, ax = graph.figure(left=.7, bottom=.7, figsize=(.7,1.))
        else:
            fig = None

        result = self.result
        if self.new_result is not None:
            RESULTS = [self.result, self.new_result]
        else:
            RESULTS = [self.result]
        for res in RESULTS:
            indices = np.argsort(res['residuals'])
            x = res['residuals'][indices]
            full_x = np.concatenate([np.arange(1, len(indices)+1)[x<=threshold],
                                     [len(indices)+1, res['Ntot']]])
            full_y = np.concatenate([x[x<=threshold], [threshold, threshold]])
            ax.plot(np.log10(full_x), np.log10(full_y), clip_on=False)

        graph.set_plot(ax,
                       # yticks=np.log10([1, 2, 5, 10, 20]),
                       # yticks_labels=['1', '2', '5', '10', '>20'],
                       # yminor_ticks=np.log10([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20]),
                       # ylim=[1, np.log10(threshold)],
                       xminor_ticks=np.log10(np.concatenate([[i*10**k for i in range(1,10)]\
                                                             for k in range(0,7)])),
                       xlim=[np.log10(result['Ntot']), -1.],
                       xticks=np.arange(7),
                       xticks_labels=['1','','','$10^3$','','','$10^6$'],
                       xlabel='config #',
                       ylabel=r'fit $\chi^2$ (norm.)       ', ylabelpad=-7)
        graph.annotate(ax, '4 pop. model', (.4, 1.2), ha='center')

        return fig, ax

    
    def get_x_best_configs(self, x):
        
        indices = np.argsort(self.result['residuals'])[::-1][:int(x)]
        return [self.result['configs'][i] for i in indices]

    
    def translate_SynapseMatrix_into_connectivity_proba(self, Matrix, Model, REC_POPS, AFF_POPS):

        for i, source_pop in enumerate(REC_POPS+AFF_POPS):
            for j, target_pop in enumerate(REC_POPS):
                Model['p_%s_%s' % (source_pop, target_pop)] = Matrix[i,j]/Model['N_%s' % source_pop]
                # print('p_%s_%s = %.1f%%' % (source_pop, target_pop,\
                #                             100*Matrix[i,j]/Model['N_%s' % source_pop]))


    def reshape_SynapseMatrix_into_ConnecMatrix(self, Matrix, Model, REC_POPS, AFF_POPS):
        """
        uses the population number from "Model" to translate into connectivity
        """
        pconnMatrix = 0*Matrix
        for i, pop in enumerate(REC_POPS+AFF_POPS):
            N = Model['N_%s' % pop]
            pconnMatrix[i,:] = Matrix[i,:]/N
        return pconnMatrix    


    
    ##########################################################################
    #################    SIMULATION    #######################################
    ##########################################################################

    def sort_random_config(self, seed=1):

        if seed is not None:
            np.random.seed(seed)

        x = np.zeros(len(self.BOUNDS))
        for i, bounds in enumerate(self.BOUNDS):
            x[i] = np.random.uniform(bounds[0], bounds[1])

        return x

    
    def save_config(self, result, i):

        fn = filename_with_datetime('', folder=self.DA_folder,
                                    with_microseconds=True,
                                    extension='npz')
        save_dict(fn, result)
        print('----------- result for scan %i saved as "%s"' % (i, fn))

        
    def compute_Vm(self, X):
        Vm = self.mf.convert_to_mean_Vm_trace(X, self.desired_Vm_key)
        return 1e3*Vm # in mV
    
        
    def compute_Vm_residual(self, X):
        Vm = self.compute_Vm(X)
        cond = (Vm>-100) & (Vm<=-30)
        residual = np.sum((Vm-self.desired_Vm)**2)/np.sum((self.desired_Vm)**2)
        if np.isfinite(residual):
            return residual
        else:
            return 1e12


    def flatten_Matrix(self, Matrix):
        return Matrix.flatten()

    def reshape_Matrix(self, flattened_Matrix):
        return flattened_Matrix.reshape(len(self.REC_POPS)+len(self.AFF_POPS), len(self.REC_POPS))
    
    def run_single_sim(self, flattened_Matrix):
        try:
            X = self.mf.run_single_connectivity_sim(self.reshape_Matrix(flattened_Matrix))
            res = self.compute_Vm_residual(X)
            return res
        except RuntimeWarning:
            return 1e3

    def run_simulation_set(self, i, n=100):

        seed = datetime.datetime.now().microsecond
        
        x0 = self.sort_random_config(seed)
        
        start_time=time.time()
        res = run_dual_annealing(self.run_single_sim,
                                 x0 = x0,
                                 bounds=self.BOUNDS,
                                 seed=seed,
                                 maxiter=n, maxfun=100*n,
                                 no_local_search=True,
                                 verbose=False)
        print('---------- Parameter Search took %.1f seconds ' % (time.time()-start_time))
        
        X = self.mf.run_single_connectivity_sim(self.reshape_Matrix(res.x))
        Vm = self.compute_Vm(X)
        result = {'Ntot':n, 'x':res.x, 'residual':res.fun, 'x0':x0, 'X':X, 'Vm':Vm}
                
        self.save_config(result, i)
        
    def launch_search(self, n=3):
        
        warnings.filterwarnings("error")
        
        i=1
        try:
            while True:
                print('\n---------- running scan %i [...]' % i)
                print('---------- (approx. duration:',n*0.27/100.,'hours)')
                self.run_simulation_set(i, n=n)
                i+=1
        except KeyboardInterrupt:
            print('\nScan stopped !')
        

if __name__=='__main__':
    
    from model import Model
    import sys
    
    if len(sys.argv)==2:
        
        _, protocol = sys.argv
        folder, N = 'test', 1
        
    elif len(sys.argv)==3:
        
        _, protocol, folder = sys.argv
        N = 1
        
    elif len(sys.argv)==4:

        _, protocol, folder, N = sys.argv
        
    else:
        protocol, folder, N  = '', '', 1

        
    if protocol=='-r': # run

        search = StochasticSearch(Model,REC_POPS,AFF_POPS,
                                  DA_folder=folder,
                                  run=True)
        search.launch_search(int(N))

        
    elif protocol=='-a': # analysis
        
        from datavyz import ges
        search = StochasticSearch(Model,REC_POPS,AFF_POPS,
                                  DA_folder=folder,
                                  run=False)
        if folder_prev!='batch_test':
            search.load_results(on_prev=True)
            search.load_results(on_prev=False)
        else:
            search.load_results()
        
        fig, ax = search.show_residuals_over_trials(ges)
        ges.show()

        
    elif protocol=='-t': # test

        from model import Model, REC_POPS, AFF_POPS
        from ntwk_sim import run_sim
        
        search = StochasticSearch(Model,REC_POPS,AFF_POPS,
                                  DA_folder=folder,
                                  run=False)

        fn = sorted(os.listdir(os.path.join('data', folder)))[::-1][int(N)]
        print('loading %s [...]' % fn)
        data = load_dict(os.path.join('data',folder,fn))

        M = search.reshape_Matrix(data['x'])
        search.translate_SynapseMatrix_into_connectivity_proba(M, Model, REC_POPS, AFF_POPS)
        run_sim(Model,
                filename=os.path.join('data',folder,fn.replace('.npz','.h5')),
                verbose=False)
            
    else:
        print("""
        use as :
        python iterative_random_search.py -r bt
        """)
        search = StochasticSearch(Model, REC_POPS, AFF_POPS, run=True)
        search.run_single_sim(search.mf.ecMatrix)
        
    # search.run_simulation_set_batch()
    # search.save_config()
    # save_config(CONFIGS, RESIDUAL)
    
