"""

"""
import sys
sys.path.append('./neural_network_dynamics')

import numpy as np
import matplotlib.pylab as plt
import ntwk

################################################################
## ------ Construct populations with their equations -------- ##
## ------------- with recurrent connections ----------------- ##
################################################################

Model = {
    ## ---------------------------------------------------------------------------------
    ## UNIT SYSTEM is : ms, mV, pF, nS, pA, Hz (arbitrary and unconsistent, so see code)
    ## ---------------------------------------------------------------------------------
    # numbers of neurons in population
    'N_TcExc':100, 
    'N_ReInh':100, 
    'N_L4Exc':100, 
    'N_AffExc':50,
    # synaptic weights
    'Q_TcExc_L4Exc':10., 
    'Q_TcExc_ReInh':10., 
    'Q_ReInh_TcExc':30., 
    'Q_ReInh_ReInh':30., 
    'Q_L4Exc_TcExc':10., 
    'Q_AffExc_TcExc':10.,
    # synaptic time constants
    'Tsyn_Exc':5., 
    'Tsyn_Inh':5., 
    # synaptic reversal potentials
    'Erev_Exc':0., 
    'Erev_Inh':-80., 
    # connectivity parameters
    'p_TcExc_L4Exc':0.05, 
    'p_TcExc_ReInh':0.05, 
    'p_L4Exc_TcExc':0.05, 
    'p_ReInh_ReInh':0.08, 
    'p_ReInh_TcExc':0.08, 
    'p_AffExc_TcExc':0.1, 
    'p_AffExc_L4Exc':0.1, 
    # simulation parameters
    'dt':0.1, 'tstop': 2000., 'SEED':3, # low by default, see later
    ## ---------------------------------------------------------------------------------
    # === cellular properties (based on AdExp), population by population ===
    # --> Thalamo-Cortical population (Exc)
    'TcExc_Gl':10., 'TcExc_Cm':200.,'TcExc_Trefrac':3.,
    'TcExc_El':-60., 'TcExc_Vthre':-50., 'TcExc_Vreset':-60., 'TcExc_deltaV':2.5,
    'TcExc_a':60., 'TcExc_b': 50, 'TcExc_tauw':200.,
    # --> Layer 4 cortical population (Exc)
    'L4Exc_Gl':10., 'L4Exc_Cm':200.,'L4Exc_Trefrac':3.,
    'L4Exc_El':-60., 'L4Exc_Vthre':-50., 'L4Exc_Vreset':-60., 'L4Exc_deltaV':2.5,
    'L4Exc_a':0., 'L4Exc_b': 50, 'L4Exc_tauw':200.,
    # --> Reticular population (Inh)
    'ReInh_Gl':10., 'ReInh_Cm':200.,'ReInh_Trefrac':3.,
    'ReInh_El':-60., 'ReInh_Vthre':-50., 'ReInh_Vreset':-60., 'ReInh_deltaV':2.5,
    'ReInh_a':80., 'ReInh_b': 30, 'ReInh_tauw':600.,
    'F_AffExc':50,
}


NTWK = ntwk.build.populations(Model, ['TcExc', 'ReInh', 'L4Exc'],
                              AFFERENT_POPULATIONS=['AffExc'],
                              with_raster=True,
                              with_pop_act=True,
                              with_Vm=10,
                              # with_synaptic_currents=True,
                              # with_synaptic_conductances=True,
                              verbose=True)

ntwk.build.recurrent_connections(NTWK, SEED=5, verbose=True)

#######################################
########### AFFERENT INPUTS ###########
#######################################

t_array = ntwk.arange(int(Model['tstop']/Model['dt']))*Model['dt']
f_array = 0.*t_array+Model['F_AffExc']
# f_array[t_array<100] = 
# # # afferent excitation onto cortical excitation and inhibition
for i, tpop in enumerate(['TcExc']): # both on excitation and inhibition
    ntwk.stim.construct_feedforward_input(NTWK, tpop, 'AffExc',
                                          t_array, f_array,
                                          verbose=True,
                                          SEED=int(37+i)%37)

################################################################
## --------------- Initial Condition ------------------------ ##
################################################################
ntwk.build.initialize_to_rest(NTWK)

#####################
## ----- Run ----- ##
#####################
network_sim = ntwk.collect_and_run(NTWK, verbose=True)

ntwk.recording.write_as_hdf5(NTWK, filename='thal-oscill.h5')

## load file
data = ntwk.recording.load_dict_from_hdf5('thal-oscill.h5')

# ## plot
fig, _ = ntwk.plots.activity_plots(data, 
                                   Vm_args={'lw':0.3, 'subsampling':4},
                                   pop_act_args=dict(smoothing=4,
                                                     subsampling=4))
plt.show()
