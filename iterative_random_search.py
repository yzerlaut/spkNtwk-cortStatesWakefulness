import itertools, os, datetime, multiprocessing, time
import numpy as np

from analyz.workflow.saving import filename_with_datetime
from analyz.workflow.shell import printProgressBar
from analyz.IO.npz import save_dict, load_dict
import neural_network_dynamics.main as ntwk

from model import REC_POPS, AFF_POPS, Model
from Umodel import Umodel

def build_initial_grid(REC_POPS,
                       AFF_POPS,
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

def translate_SynapseMatrix_into_connectivity_proba(Matrix, Model, REC_POPS, AFF_POPS):
    """
    Sets the "Matrix" as the connectivity proba in "Model"
    """
    for i, source_pop in enumerate(REC_POPS+AFF_POPS):
        for j, target_pop in enumerate(REC_POPS):
            Model['p_%s_%s' % (source_pop, target_pop)] = Matrix[i,j]/Model['N_%s' % source_pop]

def reshape_SynapseMatrix_into_ConnecMatrix(Matrix, Model, REC_POPS, AFF_POPS):
    """
    uses the population number from "Model" to translate into connectivity
    """
    pconnMatrix = 0*Matrix
    for i, pop in enumerate(REC_POPS+AFF_POPS):
        pconnMatrix[i,:] = Matrix[i,:]/Model['N_%s' % pop]
    return pconnMatrix    
            
    
class IterativeSearch:
    """
    relies on a system of subfolders in data/

    data/batch1/
    data/batch2/
    ...

    """
    def __init__(self, Model, REC_POPS, AFF_POPS,
                 folder='data/test',
                 run=False,
                 Ngrid_for_TF=100,
                 grid_file = '',
                 residual_inclusion_threshold=1000,# above this value, we do not store the config
                 desired_Vm_key='pyrExc'):

        self.Model = Model
        self.REC_POPS, self.AFF_POPS = REC_POPS, AFF_POPS
        
        self.desired_Vm_key = desired_Vm_key
        self.idesired_Vm_key = np.argwhere(np.array(REC_POPS)==desired_Vm_key)[0][0]
        
        self.folder = folder
        self.CONFIGS, self.RESIDUAL = None, None
        self.result, self.new_result = None, None

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
            self.mf = ntwk.FastMeanField(self.Model, REC_POPS, AFF_POPS, tstop=6.)
            self.X0 = [0*self.mf.t for i in range(len(REC_POPS))]
            self.mf.build_TF_func(100, with_Vm_functions=True, sampling='log')
            # initialize desired Vm trace from U-model
            self.desired_Vm = self.Umodel.predict_Vm(self.mf.t, self.mf.FAFF[0,:])+\
                Model['%s_El' % self.desired_Vm_key] # in mV

        else:

            self.load_results()
        
        
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
        GRID1 = np.zeros((len(REC_POPS)+len(AFF_POPS), len(REC_POPS), 2))

        CONFIGS = np.array(self.get_x_best_configs(Nbest_criteria))
        for i, spop in enumerate(REC_POPS+AFF_POPS):
            for j, tpop in enumerate(REC_POPS):
                mean, std = CONFIGS[:,i,j].mean(), CONFIGS[:,i,j].std()
                GRID1[i,j,:] = [np.max([self.GRID[i,j,0],mean-variance_factor*std]),
                                np.min([self.GRID[i,j,1],mean+variance_factor*std])]

        return GRID1
    
    
    def sort_random_configs(self,
                            seed=1, n=10000):

        if seed is not None:
            np.random.seed(seed)
            
        CONFIGS = np.zeros((n,len(REC_POPS)+len(AFF_POPS), len(REC_POPS)))
        for i, j in itertools.product(range(len(REC_POPS)+len(AFF_POPS)), range(len(REC_POPS))):
            CONFIGS[:,i,j] = np.random.uniform(self.GRID[i,j,0], self.GRID[i,j,1], size=n)

        return CONFIGS

    
    def save_config(self, result, i):

        save_dict(filename_with_datetime('', folder=self.new_batch_folder,
                                         with_microseconds=True,
                                         extension='npz'), result)
        print('--------------- result for scan %i: %i/%i' % (i, len(result['residuals']), result['Ntot']))

        
    def compute_Vm(self, X):
        Vm = self.mf.convert_to_mean_Vm_trace(X, self.desired_Vm_key)
        return 1e3*Vm # in mV
    
        
    def compute_Vm_residual(self, X):
        Vm = self.compute_Vm(X)
        cond = (Vm>-100) & (Vm<=-30)
        residual = np.sqrt(np.sum((Vm-self.desired_Vm)**2))
        if np.isfinite(residual):
            return residual/self.mf.t[-1]
        else:
            return 1e12

        
    def run_single_sim(self, Matrix):
        X = self.mf.run_single_connectivity_sim(Matrix)
        return self.compute_Vm_residual(X)

    
    def run_simulation_set(self, seed, n=10000):

        result = {'Ntot':n, 'configs':[], 'residuals':[]}
        
        CONFIGS = self.sort_random_configs(seed+datetime.datetime.now().microsecond, n=n)

        printProgressBar(0, n, prefix = 'Progress:', suffix = 'Complete', length = 50)
        for i in range(n):
            res = self.run_single_sim(CONFIGS[i,:,:])
            if res<residual_inclusion_threshold:
                result['configs'].append(CONFIGS[i,:,:])
                result['residuals'].append(res)
            if i%10==0:
                printProgressBar(i+1, n, prefix = 'Progress:', suffix = 'Complete', length = 50)
                
        self.save_config(result, i)
        
    def launch_search(self):
        i=1
        try:
            while True:
                print('\n---------- running scan %i [...]' % i)
                self.run_simulation_set(i, n=1000)
                i+=1
        except KeyboardInterrupt:
            print('\nScan stopped !')


    def load_results(self):

        result = {'Ntot':0, 'configs':[], 'residuals':[]}

        lf = [os.path.join(self.folder, f) for f in os.listdir(self.folder) if f.endswith('.npz')]
        for i, f in enumerate(lf):
            data = load_dict(f)
            result['Ntot'] += data['Ntot']
            if type(data['residuals']) is float:
                result['configs'] = result['configs'] +list([data['configs']])
                result['residuals'] = np.concatenate([result['residuals'],[data['residuals']]])
            else:
                result['configs'] = result['configs'] +list(data['configs'])
                result['residuals'] = np.concatenate([result['residuals'],data['residuals']])

        self.result = result
            
        

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
    parser.add_argument('-fn', "--filename", help="filename",type=str, default='')
    parser.add_argument('-gf', "--grid_file", help="filename for grid file",type=str,default='')
    parser.add_argument('-fd', "--folder", help="folder",type=str, default='')
    parser.add_argument('-pfd', "--previous_folder", help="folder",type=str, default='')
    parser.add_argument('-n', "--Nbest", help="N criteria",type=int, default=100)
    parser.add_argument('-vf', "--variance_factor", type=float, default=1.5)
    
    args = parser.parse_args()

    if args.protocol=='run':
        if args.folder and os.path.isdir(args.folder):
            from model import Model, REC_POPS, AFF_POPS
            search = IterativeSearch(Model,REC_POPS, AFF_POPS,
                                     grid_file=args.grid_file,
                                     folder=args.folder,
                                     run=True)
            search.launch_search()
        else:
            print('provide a folder name, "%s" is not a valid folder')
            
    elif args.protocol in ['analyze-grid', 'ag']:
        search = IterativeSearch(Model,REC_POPS, AFF_POPS,
                                 grid_file=args.grid_file,
                                 folder=args.folder,
                                 run=False)
        
        new_GRID = search.compute_new_grid(Nbest_criteria=args.Nbest,
                                           variance_factor=args.variance_factor)

        search.show_new_grid(new_GRID)
        np.save(os.path.join(args.folder, 'grid.npy'), new_GRID)
        print('Grid array now saved as: %s ' % os.path.join(args.folder, 'grid.npy'))

        ntwk.show()

    # elif protocol=='-a': # analysis
    #     from datavyz import ges
    #     search = IterativeSearch(Model,
    #                               previous_batch_folder=folder_prev,
    #                               new_batch_folder=folder,
    #                               run=False)
    #     if folder_prev!='batch_test':
    #         search.load_results(on_prev=True)
    #         search.load_results(on_prev=False)
    #     else:
    #         search.load_results()
        
    #     fig, ax = search.show_residuals_over_trials(ges)
    #     ges.show()

    # elif protocol=='-t': # test

    #     from model import Model, REC_POPS, AFF_POPS
    #     from ntwk_sim import run_sim
    #     search = IterativeSearch(Model,
    #                               previous_batch_folder=folder,
    #                               run=True)
    #     search.load_results()

    #     for i, M in enumerate(search.get_x_best_configs(int(N))):
    #         print("""


    #         --------------------------------------------------------------------
    #         """)
    #         X = search.mf.run_single_connectivity_sim(M)
    #         mf_data = {'X':X, 'Vm':search.compute_Vm(X), 't':1e3*search.mf.t,
    #                    'residual':search.compute_Vm_residual(X),
    #                    'desired_Vm':search.desired_Vm}
    #         save_dict(os.path.join('data',folder,'mf_%i.npz' % (i+1)), mf_data)
    #         search.translate_SynapseMatrix_into_connectivity_proba(M, Model, REC_POPS, AFF_POPS)
    #         run_sim(Model, filename=os.path.join('data',folder,'ntwk_%i.h5' % (i+1)), verbose=False)
            
    # else:
    #     print("""

    #     use as :

    #     python iterative_random_search.py -r bt


    #     """)
    #     search = IterativeSearch(Model, run=True)
    #     search.run_single_sim(search.mf.ecMatrix)
        
    # # search.run_simulation_set_batch()
    # # search.save_config()
    # # save_config(CONFIGS, RESIDUAL)
    
