import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import neural_network_dynamics.main as ntwk
from graphs.my_graph import graphs
mg = graphs() # initiate a custom plotting environment


################################################################
## ------ Construct populations with their equations -------- ##
## ------------- with recurrent connections ----------------- ##
################################################################

REC_POPS = ['Exc', 'oscillExc', 'PvInh', 'SstInh', 'VipInh']
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
    'N_Exc':4000, 'N_PvInh':500, 'N_VipInh':500, 'N_AffExc':100,
    'N_SstInh':500, 'N_NoiseExc':200, 'N_oscillExc':100,
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
Model['oscillExc_Ioscill_amp']= 10.*20.

# === adding synaptic weights ===
Qe, Qi = 2., 10 # nS
# loop oover the two population types
for aff in ['Exc', 'oscillExc', 'NoiseExc', 'AffExc']:
    for target in REC_POPS:
        Model['Q_%s_%s' % (aff, target)] = Qe
for aff in ['PvInh', 'SstInh', 'VipInh']:
    for target in REC_POPS:
        Model['Q_%s_%s' % (aff, target)] = Qi

# === initializing connectivity === #         
for aff in REC_POPS+AFF_POPS:
    for target in REC_POPS:
        Model['p_%s_%s' % (aff, target)] = 0.

# -------------------------------
# --- connectivity parameters ---
# -------------------------------
# ==> Exc
Model['p_Exc_Exc'] = 0.03
Model['p_Exc_PvInh'] = 0.03
Model['p_Exc_SstInh'] = 0.03
# ==> oscillExc
Model['p_oscillExc_oscillExc'] = 0.05
Model['p_oscillExc_Exc'] = 0.1
Model['p_oscillExc_PvInh'] = 0.1
# ==> PvInh
Model['p_PvInh_PvInh'] = 0.03
Model['p_PvInh_SseInh'] = 0.03
Model['p_PvInh_Exc'] = 0.03
# Model['p_PvInh_SstInh'] = 0.
# Model['p_PvInh_VipInh'] = 0.1
Model['p_PvInh_oscillExc'] = 0.3
# ==> SstInh
Model['p_SstInh_Exc'] = 0.03
# Model['p_SstInh_PvInh'] = 0.05
# Model['p_SstInh_oscillExc'] = 0.2
# ==> VipInh
Model['p_VipInh_SstInh'] = 0.05
# Model['p_VipInh_PvInh'] = 0.05
# Model['p_VipInh_oscillExc'] = 0.05
# ==> AffExc
Model['p_AffExc_VipInh'] = 0.2
Model['p_AffExc_PvInh'] = 0.2
Model['p_AffExc_Exc'] = 0.2
# ==> NoiseExc
Model['p_NoiseExc_PvInh'] = 0.1
# Model['p_NoiseExc_Exc'] = 0.05
Model['p_NoiseExc_SstInh'] = 0.2
# Model['p_NoiseExc_VipInh'] = 0.02
Model['p_NoiseExc_oscillExc'] = 0.1

if __name__=='__main__':

    print(Model)        
