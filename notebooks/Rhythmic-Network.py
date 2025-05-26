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
    'N_TcExc':100, 'N_ReInh':100, 'N_L5Exc':100, 'N_L5Inh':25, 'N_AffExc':100, 
    # synaptic weights
    'Q_TcExc_L5Exc':10., 'Q_TcExc_ReInh':10., # 'Q_TcExc_L5Inh':10., 
    'Q_ReInh_TcExc':30., 'Q_ReInh_ReInh':30., 
    'Q_L5Exc_TcExc':10., 
    #'Q_L5Exc_L5Exc':2., 'Q_L5Exc_L5Inh':2., 'Q_L5Inh_L5Exc':10., 'Q_L5Inh_L5Inh':0., 
    'Q_AffExc_TcExc':10.,
    # synaptic time constants
    'Tsyn_Exc':5.,  'Tsyn_Inh':5., 
    # synaptic reversal potentials
    'Erev_Exc':0., 'Erev_Inh':-80., 
    # connectivity parameters
    'p_TcExc_L5Exc':0.05, 'p_TcExc_L5Inh':0.05, 'p_TcExc_ReInh':0.05, 
    #'p_L5Exc_L5Exc':0.04, 'p_L5Exc_L5Inh':0.04, 'p_L5Inh_L5Exc':0.04, 'p_L5Inh_L5Inh':0.04, 
    'p_L5Exc_TcExc':0.05,  
    'p_ReInh_ReInh':0.08,  'p_ReInh_TcExc':0.08, 
    'p_AffExc_TcExc':0.1, 
    # simulation parameters
    'dt':0.1, 'tstop': 3000., 'SEED':3, # low by default, see later
    ## ---------------------------------------------------------------------------------
    # === cellular properties (based on AdExp), population by population ===
    # --> Thalamo-Cortical population (Exc)
    'TcExc_Gl':10., 'TcExc_Cm':200.,'TcExc_Trefrac':3.,
    'TcExc_El':-60., 'TcExc_Vthre':-50., 'TcExc_Vreset':-60., 'TcExc_deltaV':2.5,
    'TcExc_a':60., 'TcExc_b': 60., 'TcExc_tauw':250.,
    # --> Reticular population (Inh)
    'ReInh_Gl':10., 'ReInh_Cm':200.,'ReInh_Trefrac':3.,
    'ReInh_El':-60., 'ReInh_Vthre':-50., 'ReInh_Vreset':-60., 'ReInh_deltaV':2.5,
    'ReInh_a':80., 'ReInh_b': 30, 'ReInh_tauw':600.,
    # --> Layer 5 cortical population (Exc)
    'L5Exc_Gl':10., 'L5Exc_Cm':200.,'L5Exc_Trefrac':3.,
    'L5Exc_El':-60., 'L5Exc_Vthre':-50., 'L5Exc_Vreset':-60., 'L5Exc_deltaV':2.5,
    'L5Exc_a':0., 'L5Exc_b': 100, 'L5Exc_tauw':250.,
    # --> Layer 5 cortical population (Inh)
    'L5Inh_Gl':10., 'L5Inh_Cm':200.,'L5Inh_Trefrac':3.,
    'L5Inh_El':-60., 'L5Inh_Vthre':-53., 'L5Inh_Vreset':-60., 'L5Inh_deltaV':2.5,
    'L5Inh_a':0., 'L5Inh_b': 0., 'L5Inh_tauw':10000000.,
    # --> Layer 23 cortical population (Exc)
    'L23Exc_Gl':10., 'L23Exc_Cm':200.,'L23Exc_Trefrac':3.,
    'L23Exc_El':-60.,'L23Exc_Vthre':-50.,'L23Exc_Vreset':-60.,'L23Exc_deltaV':2.5,
    'L23Exc_a':0., 'L23Exc_b': 0, 'L23Exc_tauw':250.,
    # 
    'REC_POPS':['TcExc', 'ReInh', 'L5Exc', 'L4Exc', 'L23Exc'],
    'AFF_POPS':['BgExc'],
    'F_AffExc':30,
}

t= ntwk.arange(int(Model['tstop']/Model['dt']))*Model['dt']
Model['Farray_AffExc'] = np.array([50. if tt<100 else Model['F_AffExc'] for tt in t])

# %%
ntwk.quick_run.simulation(Model, with_Vm=3, verbose=False,
                          filename='../data/thal-oscill.ntwk.h5')

# %%
## load file
data = ntwk.recording.load_dict_from_hdf5('../data/thal-oscill.ntwk.h5')

# ## plot
fig, AX = ntwk.plots.activity_plots(data, 
                                   Vm_args=None,#{'lw':0.3, 'subsampling':4},
                                   pop_act_args=dict(smoothing=4,
                                                     subsampling=4))
for i, m in enumerate(Model['REC_POPS']):
    ntwk.plots.pt.annotate(AX[0], i*'\n'+m, (1,1), va='top', color=ntwk.plots.pt.tab10(i), fontsize=9)
fig, ax = ntwk.plots.pt.figure(figsize=(4,12), top=0, left=0, bottom=0)
ntwk.plots.few_Vm_plot(data, lw=1, vpeak=20, shift=100,
                       clip_spikes=True,
                       spike_style='-', ax=ax,
                       COLORS=[ntwk.plots.pt.tab10(i) for i in range(4)],
                       NVMS=[[0,1] for m in Model['REC_POPS']])

# %% [markdown]
# ## Playing with Neuromodulation

# %%
Model['tstop'] = 6000.
def custom_membrane_equation(neuron_params, synaptic_array):
    # Vm dynamics
    eqs = """
    dV/dt = (%(Gl)f*nS*(%(El)f*mV - V) + %(Gl)f*nS*%(deltaV)f*mV*exp(-(%(Vthre)f*mV-V)/(%(deltaV)f*mV)) + I - w_adapt)/(%(Cm)f*pF) : volt (unless refractory) """ % neuron_params
    # adaptation dynamics, we keep
    eqs += """
    dw_adapt/dt = ( - NM * %(a)f*nS *( %(El)f*mV - V) - w_adapt )/(%(tauw)f*ms) : amp  """ % neuron_params    
    # now synaptic currents:
    eqs += """
    I = I0
    """
    eqs = ntwk.cells.cell_construct.add_synaptic_currents(synaptic_array, eqs)
    eqs += ' : amp'
    eqs += """
        I0 : amp """
    eqs = ntwk.cells.cell_construct.add_synaptic_dynamics(synaptic_array, eqs)
    eqs += """
        NM : 1 """
    
    neurons = ntwk.NeuronGroup(neuron_params['N'], model=eqs,
                   method='euler', refractory=str(neuron_params['Trefrac'])+'*ms',
                   threshold='V>'+str(neuron_params['Vthre']+5.*neuron_params['deltaV'])+'*mV',
                   reset="""V=%(Vreset)f*mV; w_adapt+=%(b)f*pA""" % neuron_params)
    neurons.NM = 1.
    return neurons


NTWK = ntwk.build.populations(Model, Model['REC_POPS'],
                              AFFERENT_POPULATIONS=Model['AFF_POPS'],
                              custom_membrane_equation = custom_membrane_equation,
                              with_raster=True,
                              with_pop_act=True,
                              with_Vm=5,
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
def update_NM(NTWK):
    print('removing adaptation')
    for pop in NTWK['POPS']:
        pop.NM = 0.
    
network_sim = ntwk.collect_and_run(NTWK, verbose=False,
                                   INTERMEDIATE_INSTRUCTIONS=[{'time':3000, 'function':update_NM}])

ntwk.recording.write_as_hdf5(NTWK, filename='../data/thal-oscill.ntwk.h5')

# %%
## load file
data = ntwk.recording.load_dict_from_hdf5('../data/thal-oscill.ntwk.h5')

# ## plot
fig, AX = ntwk.plots.activity_plots(data, 
                                   Vm_args=None,#{'lw':0.3, 'subsampling':4},
                                   pop_act_args=dict(smoothing=4,
                                                     subsampling=4))
for i, m in enumerate(Model['REC_POPS']):
    ntwk.plots.pt.annotate(AX[0], i*'\n'+m, (1,1), va='top', color=ntwk.plots.pt.tab10(i), fontsize=9)
fig, ax = ntwk.plots.pt.figure(figsize=(4,12), top=0, left=0, bottom=0)
ntwk.plots.few_Vm_plot(data, lw=1, vpeak=20, shift=100,
                       clip_spikes=True,
                       spike_style='-', ax=ax,
                       COLORS=[ntwk.plots.pt.tab10(i) for i in range(4)],
                       NVMS=[[0,1] for m in Model['REC_POPS']])

# %%
