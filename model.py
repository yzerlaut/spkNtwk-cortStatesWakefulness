import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import neural_network_dynamics.main as ntwk
from graphs.my_graph import graphs
mg = graphs() # initiate a custom plotting environment


################################################################
## ------ Construct populations with their equations -------- ##
## ------------- with recurrent connections ----------------- ##
################################################################

REC_POPS = ['pyrExc', 'oscillExc', 'recInh', 'dsInh']
AFF_POPS = ['AffExc', 'NoiseExc']

# adding the same LIF props to all recurrent pops
LIF_props = {'Gl':10., 'Cm':200.,'Trefrac':5.,
             'El':-70, 'Vthre':-50., 'Vreset':-70., 'deltaV':0.,
             'a':0., 'b': 0., 'tauw':1e9}

Model = {
    ## -----------------------------------------
    ### Initialisation by default parameters
    ## UNIT SYSTEM is : ms, mV, pF, nS, pA, Hz
    ## (arbitrary and unconsistent, so see code)
    ## ------------------------------------------
    # numbers of neurons in population
    'N_pyrExc':4000, 'N_recInh':1000, 'N_dsInh':500, 'N_AffExc':100, 'N_NoiseExc':200, 'N_oscillExc':100,
    # synaptic time constants
    'Tse':5., 'Tsi':5.,
    # synaptic reversal potentials
    'Ee':0., 'Ei': -80.,
    # afferent stimulation
    'F_AffExc':10., 'F_NoiseExc':1.,
    # simulation parameters
    'dt':0.1, 'tstop': 100., 'SEED':3, # low by default, see later
}

for pop in REC_POPS:
    for key, val in LIF_props.items():
        Model['%s_%s' % (pop, key)] = val
# adding the oscillatory feature to the oscillExc pop
Model['oscillExc_Ioscill_freq']=3.
Model['oscillExc_Ioscill_amp']= 10.*15.

# === adding synaptic weights ===
Qe, Qi = 2., 10 # nS
# loop oover the two population types
for aff in ['pyrExc', 'oscillExc', 'NoiseExc', 'AffExc']:
    for target in REC_POPS:
        Model['Q_%s_%s' % (aff, target)] = Qe
for aff in ['recInh', 'dsInh']:
    for target in REC_POPS:
        Model['Q_%s_%s' % (aff, target)] = Qi

# === initializing connectivity === #         
for aff in REC_POPS+AFF_POPS:
    for target in REC_POPS:
        Model['p_%s_%s' % (aff, target)] = 0.

# -------------------------------
# --- connectivity parameters ---
# -------------------------------
# ==> pyrExc
Model['p_pyrExc_pyrExc'] = 0.01
Model['p_pyrExc_oscillExc'] = 0.01
Model['p_pyrExc_recInh'] = 0.02
# ==> oscillExc
Model['p_oscillExc_oscillExc'] = 0.01
# Model['p_oscillExc_pyrExc'] = 0.02
# Model['p_oscillExc_recInh'] = 0.02
# ==> recInh
Model['p_recInh_recInh'] = 0.05
Model['p_recInh_pyrExc'] = 0.05
Model['p_recInh_oscillExc'] = 0.1
# ==> dsInh
Model['p_dsInh_recInh'] = 0.1
# ==> AffExc
Model['p_AffExc_dsInh'] = 0.2
Model['p_AffExc_recInh'] = 0.2
Model['p_AffExc_pyrExc'] = 0.05
# ==> NoiseExc
Model['p_NoiseExc_recInh'] = 0.1
Model['p_NoiseExc_pyrExc'] = 0.02
Model['p_NoiseExc_dsInh'] = 0.02
Model['p_NoiseExc_oscillExc'] = 0.2

if __name__=='__main__':

    print(Model)        
