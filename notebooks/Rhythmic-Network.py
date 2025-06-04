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
    'N_L5Inh':200, 
    'N_L4Exc':1000, 
    'N_L4Inh':200, 
    'N_L23Exc':4000, 
    'N_L23Inh':1000,
    'N_BgExc':100,
    'N_ModExc':200,
    # synaptic weights: recurrent
    'Q_TcExc_ReInh':5., 'Q_TcExc_L5Exc':5., 'Q_TcExc_L5Inh':5., 'Q_TcExc_L4Exc':5., 'Q_TcExc_L4Inh':5., 
    'Q_ReInh_TcExc':15., 'Q_ReInh_ReInh':15., 
    'Q_L5Exc_TcExc':5., 
    'Q_L4Exc_L23Exc':4., 'Q_L4Exc_L4Inh':4., 
    'Q_L4Inh_L4Exc':10., 
    "Q_L23Exc_L23Exc":2.0, "Q_L23Exc_L23Inh":2.0, 
    "Q_L23Inh_L23Exc":10.0, "Q_L23Inh_L23Inh":10.0, 
    # synaptic weights: afferent
    'Q_BgExc_TcExc':10.,
    'Q_ModExc_TcExc':5., 'Q_ModExc_L5Inh':5., 'Q_ModExc_L5Exc':5., 'Q_ModExc_ReInh':5.,
    # synaptic time constants
    'Tsyn_Exc':5., 
    'Tsyn_Inh':5., 
    # synaptic reversal potentials
    'Erev_Exc':0., 
    'Erev_Inh':-80., 
    # simulation parameters
    'dt':0.1, 'tstop': 4000., 'SEED':3, # low by default, see later
    ## ---------------------------------------------------------------------------------
    # === cellular properties (based on AdExp), population by population ===
    # --> Thalamo-Cortical population (Exc)
    'TcExc_Gl':10., 'TcExc_Cm':200.,'TcExc_Trefrac':3.,
    'TcExc_El':-60., 'TcExc_Vthre':-50., 'TcExc_Vreset':-60., 'TcExc_deltaV':2.5,
    'TcExc_a':60., 'TcExc_b': 80, 'TcExc_tauw':200.,
    # --> Reticular population (Inh)
    'ReInh_Gl':10., 'ReInh_Cm':200.,'ReInh_Trefrac':3.,
    'ReInh_El':-60., 'ReInh_Vthre':-50., 'ReInh_Vreset':-60., 'ReInh_deltaV':2.5,
    'ReInh_a':50., 'ReInh_b': 30, 'ReInh_tauw':200.,
    # --> Layer 4 cortical population (Exc)
    'L5Exc_Gl':10., 'L5Exc_Cm':200.,'L5Exc_Trefrac':3.,
    'L5Exc_El':-60., 'L5Exc_Vthre':-50., 'L5Exc_Vreset':-60., 'L5Exc_deltaV':2.5,
    'L5Exc_a':0., 'L5Exc_b': 100, 'L5Exc_tauw':200.,
    # --> Layer 5 cortical population (Inh)
    'L5Inh_Gl':10., 'L5Inh_Cm':200.,'L5Inh_Trefrac':3.,
    'L5Inh_El':-60., 'L5Inh_Vthre':-50., 'L5Inh_Vreset':-60., 'L5Inh_deltaV':2.5,
    'L5Inh_a':0., 'L5Inh_b': 80, 'L5Inh_tauw':1e5,
    # --> Layer 4 cortical population (Exc)
    'L4Exc_Gl':10., 'L4Exc_Cm':200.,'L4Exc_Trefrac':3.,
    'L4Exc_El':-60.,'L4Exc_Vthre':-50.,'L4Exc_Vreset':-60.,'L4Exc_deltaV':2.5,
    'L4Exc_a':0., 'L4Exc_b': 100, 'L4Exc_tauw':200.,
    # --> Layer 4 cortical population (Inh)
    'L4Inh_Gl':10., 'L4Inh_Cm':200.,'L4Inh_Trefrac':3.,
    'L4Inh_El':-60.,'L4Inh_Vthre':-50.,'L4Inh_Vreset':-60.,'L4Inh_deltaV':2.5,
    'L4Inh_a':0., 'L4Inh_b': 0, 'L4Inh_tauw':200.,
    # --> Layer 23 cortical population (Exc)
    'L23Exc_Gl':10., 'L23Exc_Cm':200.,'L23Exc_Trefrac':3.,
    'L23Exc_El':-60.,'L23Exc_Vthre':-50.,'L23Exc_Vreset':-60.,'L23Exc_deltaV':2.5,
    'L23Exc_a':0., 'L23Exc_b': 0, 'L23Exc_tauw':1e4,
    # --> Layer 23 cortical population (Inh)
    'L23Inh_Gl':10., 'L23Inh_Cm':200.,'L23Inh_Trefrac':3.,
    'L23Inh_El':-60.,'L23Inh_Vthre':-53.,'L23Inh_Vreset':-60.,'L23Inh_deltaV':2.5,
    'L23Inh_a':0., 'L23Inh_b': 0, 'L23Inh_tauw':200.,
    #
    'REC_POPS':['TcExc', 'ReInh', 'L5Exc', 'L5Inh', 'L4Exc', 'L4Inh', 'L23Exc', 'L23Inh'],
    'AFF_POPS':['BgExc', 'ModExc'],
    # External drives
    'F_BgExc':8.,
    # connectivity parameters: recurrent populations
    'p_TcExc_L5Exc':0.05, 'p_TcExc_ReInh':0.05, 'p_TcExc_L4Exc':0.04, 'p_TcExc_L4Inh':0.05, 
    'p_ReInh_ReInh':0.05, 'p_ReInh_TcExc':0.2, 
    'p_L5Exc_TcExc':0.025, 'p_L5Exc_L5Exc':0.05, 'p_L5Exc_L5Inh':0.05, 
    'p_L4Exc_L4Inh':0.01, 'p_L4Exc_L23Exc':0.01, 'p_L4Exc_L23Inh':0.01, 
    'p_L4Inh_L4Exc':0.01, 'p_L4Inh_L4Inh':0.05, 
    "p_L23Exc_L23Exc":0.05, "p_L23Exc_L23Inh":0.05, 
    "p_L23Inh_L23Exc":0.05, "p_L23Inh_L23Inh":0.05, 
    # connectivity parameters: afferent populations
    'p_BgExc_TcExc':0.1, #'p_BgExc_L4Exc':0.05, 
    'p_ModExc_TcExc':0.25, 'p_ModExc_ReInh':0.15, 'p_ModExc_L5Inh':0.05, 
}

t = ntwk.arange(int(Model['tstop']/Model['dt']))*Model['dt']
Model['Farray_BgExc'] = Model['F_BgExc']+0*t
Model['Farray_ModExc'] = ntwk.stim.waveform_library.varying_levels_function(\
                                                        t, [0, 10, 20], [0, 2e3, 6e3], 200)


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
                   reset="""V=%(Vreset)f*mV; w_adapt+= NM * %(b)f*pA""" % neuron_params)
    neurons.NM = 1.
    return neurons


NTWK = ntwk.build.populations(Model, Model['REC_POPS'],
                              AFFERENT_POPULATIONS=Model['AFF_POPS'],
                              custom_membrane_equation = custom_membrane_equation,
                              with_raster=True,
                              with_pop_act=True,
                              with_Vm=3,
                              verbose=False)

ntwk.build.recurrent_connections(NTWK, SEED=5, verbose=False)

# # # afferent excitation onto cortical excitation and inhibition
for i, tpop in enumerate(Model['REC_POPS']):
    for apop in Model['AFF_POPS']:
        ntwk.stim.construct_feedforward_input(NTWK, tpop, apop,
                                              t, Model['Farray_%s' % apop],
                                              verbose=False,
                                              SEED=int(37+i)%37)

ntwk.build.initialize_to_rest(NTWK)

def update_NM(NTWK):
    print('removing adaptation')
    for pop in NTWK['POPS']:
        pop.NM = 0.
    
network_sim = ntwk.collect_and_run(NTWK, verbose=False,
                                   INTERMEDIATE_INSTRUCTIONS=[{'time':2000, 'function':update_NM}])

ntwk.recording.write_as_hdf5(NTWK, filename='../data/thal-oscill.ntwk.h5')

# %%
data = ntwk.recording.load_dict_from_hdf5('../data/thal-oscill.ntwk.h5')

# ## plot
fig, AX = ntwk.plots.activity_plots(data, tzoom=[200, Model['tstop']],
                                    fig_args = dict(figsize=(3,1.5), dpi=75),
                                    axes_extents = dict(Aff=1, Raster=2, Rate=2, Vm=2),
                                    Vm_args=dict(lw=0.5, spike_peak=10, subsampling=1),
                                    pop_act_args=dict(smoothing=4, subsampling=4))


# %%
