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
    'N_TcExc':200, 
    'N_ReInh':200, 
    'N_L5Exc':1000, 
    'N_L4Exc':1000, 
    'N_L23Exc':1000, 
    'N_L23Inh':200, 
    'N_BgExc':100,
    'N_ModExc':100,
    # synaptic weights
    'Q_TcExc_L5Exc':5., 
    'Q_TcExc_L4Exc':5., 
    'Q_TcExc_ReInh':5., 
    'Q_ReInh_TcExc':15., 
    'Q_ReInh_ReInh':15., 
    'Q_L5Exc_TcExc':5., 
    'Q_L4Exc_L23Exc':4., 
    'Q_BgExc_TcExc':10.,
    'Q_ModExc_L5Inh':5.,
    'Q_ModExc_ReInh':5.,
    # synaptic time constants
    'Tsyn_Exc':5., 
    'Tsyn_Inh':5., 
    # synaptic reversal potentials
    'Erev_Exc':0., 
    'Erev_Inh':-80., 
    # connectivity parameters
    'p_TcExc_L5Exc':0.05, 
    'p_TcExc_ReInh':0.05, 
    'p_L5Exc_TcExc':0.01, 
    # 'p_TcExc_L4Exc':0.05, 
    'p_L4Exc_L23Exc':0.05, 
    'p_ReInh_ReInh':0.08, 
    'p_ReInh_TcExc':0.08, 
    'p_BgExc_TcExc':0.1, 
    'p_ModExc_ReInh':0.2, 
    # simulation parameters
    'dt':0.1, 'tstop': 9000., 'SEED':3, # low by default, see later
    ## ---------------------------------------------------------------------------------
    # === cellular properties (based on AdExp), population by population ===
    # --> Thalamo-Cortical population (Exc)
    'TcExc_Gl':10., 'TcExc_Cm':200.,'TcExc_Trefrac':3.,
    'TcExc_El':-60., 'TcExc_Vthre':-50., 'TcExc_Vreset':-60., 'TcExc_deltaV':2.5,
    'TcExc_a':60., 'TcExc_b': 70, 'TcExc_tauw':200.,
    # --> Reticular population (Inh)
    'ReInh_Gl':10., 'ReInh_Cm':200.,'ReInh_Trefrac':3.,
    'ReInh_El':-60., 'ReInh_Vthre':-50., 'ReInh_Vreset':-60., 'ReInh_deltaV':2.5,
    'ReInh_a':50., 'ReInh_b': 30, 'ReInh_tauw':200.,
    # --> Layer 4 cortical population (Exc)
    'L5Exc_Gl':10., 'L5Exc_Cm':200.,'L5Exc_Trefrac':3.,
    'L5Exc_El':-60., 'L5Exc_Vthre':-50., 'L5Exc_Vreset':-60., 'L5Exc_deltaV':2.5,
    'L5Exc_a':0., 'L5Exc_b': 80, 'L5Exc_tauw':200.,
    # --> Layer 4 cortical population (Exc)
    'L4Exc_Gl':10., 'L4Exc_Cm':200.,'L4Exc_Trefrac':3.,
    'L4Exc_El':-60.,'L4Exc_Vthre':-50.,'L4Exc_Vreset':-60.,'L4Exc_deltaV':2.5,
    'L4Exc_a':0., 'L4Exc_b': 50, 'L4Exc_tauw':200.,
    # --> Layer 23 cortical population (Exc)
    'L23Exc_Gl':10., 'L23Exc_Cm':200.,'L23Exc_Trefrac':3.,
    'L23Exc_El':-60.,'L23Exc_Vthre':-50.,'L23Exc_Vreset':-60.,'L23Exc_deltaV':2.5,
    'L23Exc_a':0., 'L23Exc_b': 50, 'L23Exc_tauw':200.,
    #
    'REC_POPS':['TcExc', 'ReInh', 'L5Exc', 'L4Exc', 'L23Exc'],
    'AFF_POPS':['BgExc', 'ModExc'],
    # External drives
    'F_BgExc':8.,
}

t = ntwk.arange(int(Model['tstop']/Model['dt']))*Model['dt']
Model['Farray_BgExc'] = Model['F_BgExc']+0*t
Model['Farray_ModExc'] = ntwk.stim.waveform_library.varying_levels_function(\
                                                        t, [0, 10, 20], [0, 3e3, 6e3], 200)


# %% [markdown]
# ## Playing with Neuromodulation

# %%
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


# # # afferent excitation onto cortical excitation and inhibition
for i, tpop in enumerate(Model['REC_POPS']):
    for apop in Model['AFF_POPS']:
        ntwk.stim.construct_feedforward_input(NTWK, tpop, apop,
                                              t, Model['Farray_%s' % apop],
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
