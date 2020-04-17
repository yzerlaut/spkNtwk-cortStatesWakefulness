import itertools, os, datetime, multiprocessing, time
import numpy as np

from analyz.workflow.saving import filename_with_datetime
from analyz.optimization.dual_annealing import run_dual_annealing
from analyz.IO.npz import save_dict, load_dict
import neural_network_dynamics.main as ntwk

from Umodel import Umodel

import warnings

from iterative_random_search import translate_SynapseMatrix_into_connectivity_proba
    

def build_initial_grid(REC_POPS,
                       AFF_POPS,
                       INITIAL_LIMS = [[1,1500],
                                       [1,80],
                                       [1,320],
                                       [1,40],
                                       [1,40],
                                       [1,40]]):
    BOUNDS = []
    for i, spop in enumerate(REC_POPS+AFF_POPS):
        for j, tpop in enumerate(REC_POPS):
            BOUNDS.append(INITIAL_LIMS[i])
    return BOUNDS


class StochasticSearch:
    """
    relies on a system of subfolders in data/

    data/batch1/
    data/batch2/
    ...

    """
    def __init__(self, Model, REC_POPS, AFF_POPS,
                 folder='data/DA_test',
                 run=False,
                 Ngrid_for_TF=100,
                 BOUNDS = None,
                 residual_inclusion_threshold=1000,# above this value, we do not store the config
                 desired_Vm_key='pyrExc'):

        self.Model = Model
        self.REC_POPS, self.AFF_POPS = REC_POPS, AFF_POPS
        self.folder = folder
        
        self.desired_Vm_key = desired_Vm_key
        self.idesired_Vm_key = np.argwhere(np.array(REC_POPS)==desired_Vm_key)[0][0]

        if BOUNDS is None:
            self.BOUNDS = build_initial_grid(REC_POPS,AFF_POPS)
        else:
            self.BOUNDS = BOUNDS
            
        if run:

            if not os.path.exists(folder):
                os.makedirs(folder)

            self.CONFIGS, self.RESIDUAL = None, None
            self.result, self.new_result = None, None

            self.Umodel = Umodel() # initialize U-model

            # initialize mean-field
            self.mf = ntwk.FastMeanField(self.Model, REC_POPS, AFF_POPS, tstop=6.)
            self.X0 = [0*self.mf.t for i in range(len(REC_POPS))]
            self.mf.build_TF_func(Ngrid_for_TF, with_Vm_functions=True, sampling='log')
            # initialize desired Vm trace from U-model
            self.desired_Vm = self.Umodel.predict_Vm(self.mf.t, self.mf.FAFF[0,:])+\
                Model['%s_El' % self.desired_Vm_key] # in mV
            
        
    ##########################################################################
    #################    ANALYSIS    ##########################################
    ##########################################################################
    

    
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

        fn = filename_with_datetime('', folder=self.folder,
                                    with_microseconds=True,
                                    extension='npz')
        save_dict(fn, result)
        print('----------- result for scan %i saved as "%s"' % (i, fn))

        
    def compute_Vm_residual(self, Vm):
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
            X, Vm = self.mf.run_single_connectivity_sim(self.reshape_Matrix(flattened_Matrix))
            res = self.compute_Vm_residual(1e3*Vm[self.idesired_Vm_key,:])
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
        
        X, Vm = self.mf.run_single_connectivity_sim(self.reshape_Matrix(res.x))
        result = {'Ntot':n, 'x':res.x, 'residual':res.fun,
                  'Model':self.Model, 'REC_POPS':self.REC_POPS,'AFF_POPS':self.AFF_POPS,
                  'x0':x0, 'X':X, 'Vm':Vm, 'desired_Vm':self.desired_Vm}
                
        self.save_config(result, i)
        
    def launch_search(self, n=3):
        
        warnings.filterwarnings("error")
        
        i=1
        try:
            while True:
                print('\n---------- running scan %i [...]' % i)
                print('--- EXPECTED ENDTIME:', time.strftime('%Y-%m-%d %H:%M:%S',\
                                                             time.localtime(time.time()+10*n)))
                self.run_simulation_set(i, n=n)
                i+=1
        except KeyboardInterrupt:
            print('\nScan stopped !')
        

if __name__=='__main__':
    

    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("protocol",\
                        help="""
                        Mandatory argument, either:
                        - run
                        - run-associated-spiking-network (or "rasn")
                        - analyze
                        - check
                        - 
                        """)
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument('-fn', "--filename", help="filename",type=str, default='')
    parser.add_argument('-fd', "--folder", help="folder",type=str, default='')
    parser.add_argument('-pfd', "--previous_folder", help="folder",type=str, default='')
    parser.add_argument('-n', "--Nmax", help="folder",type=int, default=1)
    
    args = parser.parse_args()
    
    
        
    if args.protocol=='run': # run

        if os.path.isdir(args.folder):
            search = StochasticSearch(Model,REC_POPS,AFF_POPS,
                                      folder=args.folder,
                                      run=True)
            print('launching analysis with N=%i for folder %s [...]' % (args.Nmax, args.folder))
            search.launch_search(args.Nmax)
        else:
            print('Need to create the folder, folder "%s" does not exists' % args.folder)
            
        
    elif args.protocol in ["run-associated-spiking-network", "rasn"]:

        from model import Model, REC_POPS, AFF_POPS # TO BE REMOVED
        from ntwk_sim import run_sim
        
        if args.folder and os.path.isdir(args.folder):
            lf = [os.path.join(args.folder, f) for f in os.listdir(args.folder) if f.endswith('.npz')]
        elif args.filename and os.path.isfile(args.filename):
            lf = [args.filename]
        else:
            lf = []
            print('provide a valid filename or a valid folder !')
            
        for fn in lf:

            print('loading %s [...]' % fn)
            data = load_dict(fn)

            # TO BE REPLACED by "data['Model'], ..."
            search = StochasticSearch(Model, REC_POPS, AFF_POPS, run=False) 
            
            M = search.reshape_Matrix(data['x'])
            translate_SynapseMatrix_into_connectivity_proba(M, Model, REC_POPS, AFF_POPS)
            run_sim(Model,REC_POPS,AFF_POPS,
                    filename=fn.replace('.npz','.h5'), verbose=False)

        
    elif args.protocol=='analysis': # analysis
        
        from datavyz import ges
        search = StochasticSearch(Model,REC_POPS,AFF_POPS,
                                  folder=args.folder,
                                  run=False)
        
        if args.prev_folder!='':
            search.load_results(on_prev=True)
            search.load_results(on_prev=False)
        else:
            search.load_results()
        
        fig, ax = search.show_residuals_over_trials(ges)
        ges.show()

        
    elif args.protocol=='check':
        # calibration protocol, we insure that the MF is accurately describing the NTWK
            
        from model import Model, REC_POPS, AFF_POPS
        
        search = StochasticSearch(Model, REC_POPS, AFF_POPS,
                                  folder='data/test',
                                  run=True)

        # a few random configs:
        RESIDUALS = []
        for i in range(10):
            X, Vm = search.mf.run_single_connectivity_sim(search.reshape_Matrix(search.sort_random_config(i)))
            RESIDUALS.append(search.compute_Vm_residual(1e3*Vm[search.idesired_Vm_key,:]))
            print('Residual:', RESIDUALS[-1])
        i0 = np.argmin(np.array(RESIDUALS))

        flattened_Matrix = search.sort_random_config(i0)
        Matrix = search.reshape_Matrix(flattened_Matrix)
        
        X, Vm = search.mf.run_single_connectivity_sim(Matrix)
        
        from ntwk_sim import run_sim, ge, COLORS
        # we need to translate the synapse matrix into a connectivity one
        translate_SynapseMatrix_into_connectivity_proba(Matrix, Model, REC_POPS, AFF_POPS)
        run_sim(Model, REC_POPS, AFF_POPS, filename='data/test/stochastic-calibration.h5')
        
        import neural_network_dynamics.main as ntwk
        
        data = ntwk.load_dict_from_hdf5('data/test/stochastic-calibration.h5')
        fig, AX = ntwk.activity_plots(data,
                                      COLORS=COLORS,
                                      smooth_population_activity=10,
                                      pop_act_log_scale=True)

        AX[2].plot(1e3*search.mf.t, search.desired_Vm, 'k--')
        
        mf2 = ntwk.FastMeanField(Model, REC_POPS, AFF_POPS, tstop=6.)
        for i, label in enumerate(REC_POPS):
            AX[-1].plot(1e3*search.mf.t, 1e-2+X[i,:], lw=1, color='k', alpha=.5)
            AX[i+2].plot(1e3*search.mf.t, 1e3*Vm[i,:], 'k-')
            AX[i+2].set_ylim([-72,-45])

        ge.show()
