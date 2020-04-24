import itertools, os, datetime, multiprocessing, time, copy
import numpy as np

from analyz.workflow.saving import filename_with_datetime
from analyz.workflow.shell import printProgressBar
from analyz.IO.npz import save_dict, load_dict
from analyz.IO.hdf5 import save_dict_to_hdf5, load_dict_from_hdf5
import neural_network_dynamics.main as ntwk

import warnings

from Umodel import Umodel

def build_initial_grid(REC_POPS, AFF_POPS,
                       INITIAL_LIMS = [[1,1500],
                                       [1,80],
                                       [1,320],
                                       [1,40],
                                       [1,40],
                                       [1,40]]):
    GRID0 = np.zeros((len(REC_POPS)+len(AFF_POPS), len(REC_POPS),2))
    for i, j in itertools.product(range(len(REC_POPS)+len(AFF_POPS)), range(len(REC_POPS))):
        GRID0[i,j,:] = np.array(INITIAL_LIMS[i])
    return GRID0

def translate_SynapseMatrix_into_connectivity_proba(Matrix, Model, verbose=False):
    """
    Sets the "Matrix" as the connectivity proba in "Model"
    """
    for i, source_pop in enumerate(list(Model['REC_POPS'])+list(Model['AFF_POPS'])):
        for j, target_pop in enumerate(list(Model['REC_POPS'])):
            Model['p_%s_%s' % (source_pop, target_pop)] = Matrix[i,j]/Model['N_%s' % source_pop]
            if verbose:
                print("Model['p_%s_%s'] = %.3f" % (source_pop, target_pop,
                                                   Matrix[i,j]/Model['N_%s' % source_pop]))

def reshape_SynapseMatrix_into_ConnecMatrix(Matrix, Model):
    """
    uses the population number from "Model" to translate into connectivity
    """
    pconnMatrix = 0*Matrix
    for i, pop in enumerate(Model['REC_POPS']+Model['AFF_POPS']):
        pconnMatrix[i,:] = Matrix[i,:]/Model['N_%s' % pop]
    return pconnMatrix    
            
    
class IterativeSearch:
    """
    relies on a system of subfolders in data/

    data/batch1/
    data/batch2/
    ...

    """
    def __init__(self, Model,
                 folder='data/test',
                 run=False,
                 Ngrid_for_TF=100,
                 grid_file = '',
                 residual_inclusion_threshold=1000,# above this value, we do not store the config
                 desired_Vm_key='pyrExc'):

        self.Model = Model
        self.REC_POPS, self.AFF_POPS = Model['REC_POPS'], Model['AFF_POPS']
        
        self.desired_Vm_key = desired_Vm_key
        self.idesired_Vm_key = np.argwhere(np.array(self.REC_POPS)==desired_Vm_key)[0][0]
        
        self.folder = folder
        self.CONFIGS, self.RESIDUAL = None, None
        self.result = None
        self.residual_inclusion_threshold=residual_inclusion_threshold

        if not grid_file:
            print('/!\ No grid file provided, running on default (large) grid !!')
            self.GRID = build_initial_grid(self.REC_POPS, self.AFF_POPS)
        else:
            self.GRID = np.load(grid_file)
            print('running on grid:\n', self.GRID)

        if run:
            
            if not os.path.exists(folder):
                os.makedirs(folder)
                
            self.Umodel = Umodel() # initialize U-model
            
            # initialize mean-field
            self.mf = ntwk.FastMeanField(self.Model, tstop=6.)
            self.X0 = [0*self.mf.t for i in range(len(self.REC_POPS))]
            # initialize FasT-MF TF
            self.mf.build_TF_func(tf_sim_file='neural_network_dynamics/theory/tf_sim_points.npz')
            # initialize desired Vm trace from U-model
            self.desired_Vm = self.Umodel.predict_Vm(self.mf.t, self.mf.FAFF[0,:])+\
                Model['%s_El' % self.desired_Vm_key] # in mV

        else:

            self.result = load_results(folder)
        
        
    ##########################################################################
    #################    ANALYSIS    ##########################################
    ##########################################################################
    
    def analyze_previous_batch(self):
        
        # filename_with_datetime(filename, folder='./', extension='')
        # PREV_CONFIGS, PREV_RESIDUALS = np.load()
        
        pass

    def show_new_grid(self, new_GRID, graph=None):

        if graph is None:
            from datavyz import ges as ge
            
        fig, ax = ge.parallel_plot(Y=[self.GRID[:,:,0].flatten(),
                                      self.GRID[:,:,1].flatten(),
                                      new_GRID[:,:,0].flatten(),
                                      new_GRID[:,:,1].flatten()],
                                   COLORS=['k', 'k', 'r', 'r'],
                                   lw=2, fig_args={'figsize':(4,1)})
        
        
    
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

    
    
    ##########################################################################
    #################    SIMULATION    #######################################
    ##########################################################################

    def compute_new_grid(self,
                         Nbest_criteria=100,
                         variance_factor=1.5):

        print('computing new grid from the data of %s [...]' % self.folder)
        GRID1 = np.zeros((len(self.REC_POPS)+len(self.AFF_POPS), len(self.REC_POPS), 2))

        CONFIGS = np.array(self.get_x_best_configs(Nbest_criteria))
        for i, spop in enumerate(self.REC_POPS+self.AFF_POPS):
            for j, tpop in enumerate(self.REC_POPS):
                mean, std = CONFIGS[:,i,j].mean(), CONFIGS[:,i,j].std()
                GRID1[i,j,:] = [np.max([self.GRID[i,j,0],mean-variance_factor*std]),
                                np.min([self.GRID[i,j,1],mean+variance_factor*std])]

        return GRID1
    
    
    def sort_random_configs(self,
                            seed=1, n=10000):

        if seed is not None:
            np.random.seed(seed)
            
        CONFIGS = np.zeros((n,len(self.REC_POPS)+len(self.AFF_POPS), len(self.REC_POPS)))
        for i, j in itertools.product(range(len(self.REC_POPS)+len(self.AFF_POPS)),\
                                      range(len(self.REC_POPS))):
            CONFIGS[:,i,j] = np.random.uniform(self.GRID[i,j,0], self.GRID[i,j,1], size=n)

        return CONFIGS

    

    def compute_Vm_residual(self, Vm):
        cond = (Vm>-100) & (Vm<=-30)
        residual = np.sqrt(np.sum((Vm-self.desired_Vm)**2)/np.sum((self.desired_Vm)**2))
        if np.isfinite(residual):
            return residual
        else:
            return 1e12

        
    def run_single_sim(self, Matrix):
        try:
            X, Vm = self.mf.run_single_connectivity_sim(Matrix)
            res = self.compute_Vm_residual(1e3*Vm[self.idesired_Vm_key,:])
            return X, Vm, res
        except RuntimeWarning:
            return None, None, 1e12
        
    
    def run_simulation_set(self, seed, n=10000):

        # initialize results
        result = {'Ntot':n, 'configs':[], 'residuals':[], 'seed':seed,
                  'Vm':[], 'X':[], 'associated_ntwk_sim_filename':[],
                  'Model':self.Model, 'REC_POPS':self.REC_POPS, 'AFF_POPS':self.AFF_POPS,
                  'desired_Vm':self.desired_Vm, 't':self.mf.t}
        
        CONFIGS = self.sort_random_configs(seed+datetime.datetime.now().microsecond, n=n)

        printProgressBar(0, n, prefix = 'Progress:', suffix = 'Complete', length = 50)
        for i in range(n):
            X, Vm, res = self.run_single_sim(CONFIGS[i,:,:])
            if res<self.residual_inclusion_threshold:
                result['configs'].append(CONFIGS[i,:,:])
                result['Vm'].append(Vm)
                result['X'].append(X)
                ntwk_fn = filename_with_datetime('', folder=self.folder,
                                                 with_microseconds=True,
                                                 extension='.ntwk.h5')
                # even if never generated, we force the future name of the ntwk sim
                result['associated_ntwk_sim_filename'].append(ntwk_fn)
                result['residuals'].append(res)
            if i%10==0:
                printProgressBar(i+1, n, prefix = 'Progress:', suffix = 'Complete', length = 50)
                
        save_dict_to_hdf5(result, filename_with_datetime('', folder=self.folder,
                                                         with_microseconds=True,
                                                         extension='.scan.h5'))
        
    def launch_search(self, batch_size=1000, n_batch=0):
        
        warnings.filterwarnings("error") # so that Runtime warnings are catched !
        if n_batch==0:
            n_batch = 1e10
        i=1
        try:
            while i<(n_batch+1):
                print('\n---------- running scan %i [...]' % i)
                self.run_simulation_set(i, n=batch_size)
                i+=1
        except KeyboardInterrupt:
            print('\nScan stopped !')


def load_results(folder,
                 i0=0, i1=100000,
                 residual_inclusion_threshold=0.1):

    lf = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.scan.h5')]
    
    for i, f in enumerate(lf[i0:i1]):
        result_set = load_dict_from_hdf5(f)
        if i==0:
            full_result = copy.deepcopy(result_set)
        else:
            full_result['Ntot'] += result_set['Ntot']
            # possibilityto re-sharpen the inclusion threshold here:
            cond = result_set['residuals']<residual_inclusion_threshold
            full_result['Vm'] = list(full_result['Vm']) +list(result_set['Vm'][cond])
            full_result['X'] = list(full_result['X']) +list(result_set['X'][cond])
            
            full_result['configs'] = list(full_result['configs']) +list(result_set['configs'][cond])
            full_result['residuals'] = np.concatenate([full_result['residuals'],
                                                       result_set['residuals'][cond]])
            full_result['associated_ntwk_sim_filename'] = np.concatenate(\
                                        [full_result['associated_ntwk_sim_filename'],
                                         result_set['associated_ntwk_sim_filename'][cond]])

    full_result['i_sorted_residuals'] = np.argsort(full_result['residuals'])
            
    return full_result
            

if __name__=='__main__':
    
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("protocol",\
                        help="""
                        Mandatory argument, either:
                        - run
                        - analyze-grid
                        - run-associated-spiking-network (or "rasn")
                        - analyze
                        - check
                        - 
                        """)
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("--force", help="flag to force un-natural behavior", action="store_true")
    parser.add_argument('-fn', "--filename", help="filename",type=str, default='')
    parser.add_argument('-gf', "--grid_file", help="filename for grid file",type=str,
                        default='data/Mscan/batch1/grid.npy')
    parser.add_argument('-df', "--data_folder", help="folder",type=str, default='')
    parser.add_argument('-pfd', "--previous_folder", help="folder",type=str, default='')
    parser.add_argument('-n', "--Nbest", help="N criteria",type=int, default=100)
    parser.add_argument('-vf', "--variance_factor", type=float, default=1.5)
    parser.add_argument('-rit', "--residual_inclusion_threshold", type=float, default=0.06)
    parser.add_argument('-bs', "--batch_size", help="batch size of random search",
                        type=int, default=1000)
    parser.add_argument('-nb', "--n_batch", help="number of batches of random search",
                        type=int, default=0)
    parser.add_argument("--i0", type=int, default=0)
    parser.add_argument("--i1", type=int, default=1000000)
    
    
    args = parser.parse_args()
    
    
    if args.protocol=='run':
        
        if args.data_folder and os.path.isdir(args.data_folder):
            from model import Model
            search = IterativeSearch(Model,
                                     grid_file=args.grid_file,
                                     folder=args.data_folder,
                                     residual_inclusion_threshold=args.residual_inclusion_threshold,
                                     run=True)
            search.launch_search(args.batch_size,
                                 n_batch=args.n_batch)
        else:
            print('provide a folder name, "%s" is not a valid folder')

            
    elif args.protocol in ['analyze-grid', 'ag']:

        from model import Model
        search = IterativeSearch(Model,
                                 grid_file=args.grid_file,
                                 folder=args.data_folder,
                                 run=False)
        
        new_GRID = search.compute_new_grid(Nbest_criteria=args.Nbest,
                                           variance_factor=args.variance_factor)

        search.show_new_grid(new_GRID)
        np.save(os.path.join(args.data_folder, 'grid.npy'), new_GRID)
        print('Grid array now saved as: %s ' % os.path.join(args.data_folder, 'grid.npy'))

        ntwk.show()

        
    elif args.protocol in ["run-associated-spiking-network", "rasn"]:

        if args.data_folder and os.path.isdir(args.data_folder):
            results = load_results(args.data_folder)
        else:
            raise NameError('provide a valid folder !')

        from ntwk_sim import run_sim, run_slow_mf

        for i in results['i_sorted_residuals'][:args.Nbest]:
            print(i, results['Ntot'], results['residuals'][i])

            fn, sim = results['associated_ntwk_sim_filename'][i], True
            if os.path.isfile(fn) and not args.force:
                print('"%s" already exists, leaving existing datafile, use "--force" to redo the simulation ' % fn)
                sim = False

            if sim:
                Matrix = results['configs'][i]
                Model = results['Model'].copy()
                translate_SynapseMatrix_into_connectivity_proba(Matrix, Model)
                run_sim(Model, filename=fn, verbose=False)
                run_slow_mf(fn)
            

    elif args.protocol in ["reduce-data", "rd"]:

        if args.data_folder and os.path.isdir(args.data_folder):
            results = load_results(args.data_folder,
                                   args.i0, args.i1,
                                   args.residual_inclusion_threshold)
            save_dict_to_hdf5(results, filename_with_datetime('', folder=args.data_folder,
                                                              with_microseconds=True,
                                                              extension='.Mscan.h5'))
            
            
        else:
            raise NameError('provide a valid folder !')

                    
    elif args.protocol in ["analyze-sim", "as"]:

        if args.data_folder and os.path.isdir(args.data_folder):
            results = load_results(args.data_folder)
        else:
            raise NameError('provide a valid folder !')

        from ntwk_sim import plot_sim
        from datavyz import ge
        
        Umodel_data = {'t':1e3*results['t'], 'desired_Vm':results['desired_Vm']}
        for i in results['i_sorted_residuals'][:args.Nbest]:
            print(i, results['residuals'][i])
            print(results['configs'][i])
            
            fn = results['associated_ntwk_sim_filename'][i]
            if os.path.isfile(fn):
                plot_sim(fn, ge, Umodel_data=Umodel_data)
            else:
                print('file "%s" of index "%i" is not available' % (fn, i))
                
    
