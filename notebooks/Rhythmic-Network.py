# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import sys
sys.path.append('../neural_network_dynamics')

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
    'N_TcExc':100, 'N_ReInh':100, 'N_L4Exc':100, 'N_L23Exc':100, 
    # synaptic weights
    'Q_TcExc_L4Exc':10., 'Q_TcExc_ReInh':10., 'Q_ReInh_TcExc':30., 
    'Q_ReInh_ReInh':30., 'Q_L4Exc_TcExc':10., 'Q_L4Exc_L23Exc':2., 'Q_AffExc_TcExc':10.,
    # synaptic time constants
    'Tsyn_Exc':5.,  'Tsyn_Inh':5., 
    # synaptic reversal potentials
    'Erev_Exc':0., 'Erev_Inh':-80., 
    # connectivity parameters
    'p_TcExc_L4Exc':0.05, 'p_TcExc_ReInh':0.05, 'p_L4Exc_TcExc':0.05,  'p_L4Exc_L23Exc':0.1,
    'p_ReInh_ReInh':0.08,  'p_ReInh_TcExc':0.08, 'p_AffExc_TcExc':0.1, 
    # simulation parameters
    'dt':0.1, 'tstop': 3000., 'SEED':3, # low by default, see later
    ## ---------------------------------------------------------------------------------
    # === cellular properties (based on AdExp), population by population ===
    # --> Thalamo-Cortical population (Exc)
    'TcExc_Gl':10., 'TcExc_Cm':200.,'TcExc_Trefrac':3.,
    'TcExc_El':-60., 'TcExc_Vthre':-50., 'TcExc_Vreset':-60., 'TcExc_deltaV':2.5,
    'TcExc_a':60., 'TcExc_b': 50, 'TcExc_tauw':200.,
    # --> Reticular population (Inh)
    'ReInh_Gl':10., 'ReInh_Cm':200.,'ReInh_Trefrac':3.,
    'ReInh_El':-60., 'ReInh_Vthre':-50., 'ReInh_Vreset':-60., 'ReInh_deltaV':2.5,
    'ReInh_a':80., 'ReInh_b': 30, 'ReInh_tauw':600.,
    # --> Layer 4 cortical population (Exc)
    'L4Exc_Gl':10., 'L4Exc_Cm':200.,'L4Exc_Trefrac':3.,
    'L4Exc_El':-60., 'L4Exc_Vthre':-50., 'L4Exc_Vreset':-60., 'L4Exc_deltaV':2.5,
    'L4Exc_a':0., 'L4Exc_b': 50, 'L4Exc_tauw':200.,
    # --> Layer 23 cortical population (Exc)
    'L23Exc_Gl':10., 'L23Exc_Cm':200.,'L23Exc_Trefrac':3.,
    'L23Exc_El':-60.,'L23Exc_Vthre':-50.,'L23Exc_Vreset':-60.,'L23Exc_deltaV':2.5,
    'L23Exc_a':0., 'L23Exc_b': 50, 'L23Exc_tauw':200.,
    'F_AffExc':30,
}

POP_KEYS = ['TcExc', 'ReInh', 'L4Exc','L23Exc']

NTWK = ntwk.build.populations(Model, POP_KEYS,
                              AFFERENT_POPULATIONS=['AffExc'],
                              with_raster=True,
                              with_pop_act=True,
                              with_Vm=10,
                              # with_synaptic_currents=True,
                              # with_synaptic_conductances=True,
                              verbose=False)

ntwk.build.recurrent_connections(NTWK, SEED=5, verbose=False)

#######################################
########### AFFERENT INPUTS ###########
#######################################

t_array = ntwk.arange(int(Model['tstop']/Model['dt']))*Model['dt']
f_array = 0.*t_array+Model['F_AffExc']
# f_array[t_array<100] = 
# # # afferent excitation onto cortical excitation and inhibition
for i, tpop in enumerate(['TcExc']):
    ntwk.stim.construct_feedforward_input(NTWK, tpop, 'AffExc',
                                          t_array, f_array,
                                          verbose=False,
                                          SEED=int(37+i)%37)

################################################################
## --------------- Initial Condition ------------------------ ##
################################################################
ntwk.build.initialize_to_rest(NTWK)

#####################
## ----- Run ----- ##
#####################
network_sim = ntwk.collect_and_run(NTWK, verbose=False)

ntwk.recording.write_as_hdf5(NTWK, filename='../data/thal-oscill.h5')

# %%
## load file
data = ntwk.recording.load_dict_from_hdf5('../data/thal-oscill.h5')

# ## plot
fig, AX = ntwk.plots.activity_plots(data, 
                                   Vm_args=None,#{'lw':0.3, 'subsampling':4},
                                   pop_act_args=dict(smoothing=4,
                                                     subsampling=4))
for i in range(4):
    ntwk.plots.pt.annotate(AX[0], i*'\n'+POP_KEYS[i], (1,1), va='top', color=ntwk.plots.pt.tab10(i), fontsize=9)
fig, ax = ntwk.plots.pt.figure(figsize=(4,12), top=0, left=0, bottom=0)
ntwk.plots.few_Vm_plot(data, lw=1, vpeak=10, shift=100,
                       clip_spikes=True,
                       spike_style='-', ax=ax,
                       COLORS=[ntwk.plots.pt.tab10(i) for i in range(4)],
                       NVMS=[[0],[0],[0],[0,2,3]])

# %%
