import itertools, os, datetime, multiprocessing, time
import numpy as np

from analyz.workflow.saving import filename_with_datetime
from analyz.workflow.shell import printProgressBar
from analyz.IO.npz import save_dict
import neural_network_dynamics.main as ntwk

from model import REC_POPS, AFF_POPS, Model
from Umodel import Umodel

INITIAL_LIMS = [1,40]

GRID0 = np.zeros((len(REC_POPS)+len(AFF_POPS), len(REC_POPS),2))
for i, j in itertools.product(range(len(REC_POPS)+len(AFF_POPS)), range(len(REC_POPS))):
    GRID0[i,j,:] = np.array(INITIAL_LIMS)

                
class InterativeSearch:
    """
    relies on a system of subfolders in data/

    data/batch1/
    data/batch2/
    ...

    """
    def __init__(self, Model,
                 previous_batch_folder='batch_test',
                 new_batch_folder='batch_test',
                 run=False):

        self.Model = Model
        
        self.previous_batch_folder = os.path.join('data', 'Mscan', previous_batch_folder)
        self.new_batch_folder = os.path.join('data', 'Mscan', new_batch_folder)

        if not os.path.exists(self.new_batch_folder):
            os.makedirs(self.new_batch_folder)

        self.CONFIGS, self.RESIDUAL = None, None

        self.GRID = GRID0

        self.Umodel = Umodel() # initialize U-model

        if run:
            # initialize mean-field
            self.mf = ntwk.FastMeanField(self.Model, REC_POPS, AFF_POPS, tstop=6.)
            self.X0 = [0*self.mf.t for i in range(len(REC_POPS))]
            self.mf.build_TF_func(100, with_Vm_functions=True, sampling='log')
            # initialize desired Vm trace from U-model
            self.desired_Vm = self.Umodel.predict_Vm(self.mf.t, self.mf.FAFF[0,:])
        else:
            self.prev_files = [f for f in os.listdir(self.previous_batch_folder) if f.endswith('.npy') ]

        
    def analyze_previous_batch(self):
        
        # filename_with_datetime(filename, folder='./', extension='')
        # PREV_CONFIGS, PREV_RESIDUALS = np.load()
        
        pass

        
        
    def sort_random_configs(self, seed=1, n=10000):

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

    def compute_Vm_residual(self, X):
        Vm = self.mf.convert_to_mean_Vm_trace(X, 'pyrExc')
        cond = (Vm>-100e-3) & (Vm<=-30e-3)
        residual = np.sqrt(np.sum((1e3*Vm-self.Model['pyrExc_El']-self.desired_Vm)**2))
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
        CONFIGS[0,:,:] = self.mf.ecMatrix

        print(self.mf.ecMatrix)
        
        printProgressBar(0, n, prefix = 'Progress:', suffix = 'Complete', length = 50)
        for i in range(n):
            res = self.run_single_sim(CONFIGS[i,:,:])
            if res<1000:
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
            

if __name__=='__main__':
    
    from model import Model
    import sys
    
    if len(sys.argv)==3:
        
        _, protocol, folder = sys.argv
        
        if protocol=='-r': # run
            search = InterativeSearch(Model,
                                      new_batch_folder=folder,
                                      run=True)
            search.launch_search()
        if protocol=='-a': # analysis
            pass
    else:
        print("""

        use as :

        python iterative_random_search.py -r bt


        """)
        search = InterativeSearch(Model,
                                  run=True)
        search.run_single_sim(search.mf.ecMatrix)
    # search.run_simulation_set_batch()
    # search.save_config()
    # save_config(CONFIGS, RESIDUAL)
    
