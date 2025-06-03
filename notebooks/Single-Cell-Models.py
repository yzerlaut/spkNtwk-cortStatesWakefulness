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

# %% [markdown]
# # Single Cell Models
#
# > Based on the AdExp model, see [Brette & Gertner, 2005](https://journals.physiology.org/doi/full/10.1152/jn.00686.2005)

# %%
import sys
sys.path.append('../')
from neural_network_dynamics import ntwk
from neural_network_dynamics.utils import plot_tools as pt

import numpy as np

stim = {
    'amplitudes' : [250., 0, -250], # pA
    'durations' : [300., 500., 300.], # ms
    'delay' : 300,
}

def plot(t, Vm, I, spikes, params,
         spike_peak=10):

    fig, ax = pt.figure(figsize=(2,1.5), left=0, right=0, top=0, bottom=0)

    VmC = Vm.copy()
    for s in spikes:
        VmC[np.argmin((t-s)**2)] = spike_peak
    ax.plot(t, VmC, 'k-', lw=0.5)
        #ax.plot([s-1, s-1], [params['Vthre']+5*params['deltaV'], 10], 'k-', lw=0.5)
    ax.plot(t, VmC.min()-10+10*I/np.max(np.abs(I)), color='grey')

    pt.draw_bar_scales(ax, Ybar=10, Xbar=100, Xbar_label='100ms', Ybar_label='10mV', Ybar_label2='%ipA' % 250)
    ax.axis('off')
    return fig, ax


# %% [markdown]
# ## Thalamo-Cortical (TC) cells
#
# > excitatory

# %%
params = {
    'Gl':10., 'Cm':200.,'Trefrac':2.5,
    'El':-60., 'Vthre':-50., 'Vreset':-60., 'deltaV':2.5,
    'a': 40., 'b': 20., 'tauw':300., 'N':1}

t, Vm, I, spikes = ntwk.cells.pulse_protocols.run_sim(stim, params)
fig, ax = plot(t, Vm, I, spikes, params)

# %% [markdown]
# ## Reticular (RE) thalamic cells
#
# > inhibitory

# %%
params = {
    'Gl':5., 'Cm':200.,'Trefrac':2.5,
    'El':-60., 'Vthre':-50., 'Vreset':-60., 'deltaV':2.5,
    'a': 20., 'b': 10, 'tauw':400., 'N':1}

t, Vm, I, spikes = ntwk.cells.pulse_protocols.run_sim(stim, params)
fig, ax = plot(t, Vm, I, spikes, params)

# %% [markdown]
# ## Layer 4 cells
#
# > excitatory

# %%
params = {
    'Gl':10., 'Cm':200.,'Trefrac':2.5,
    'El':-60., 'Vthre':-50., 'Vreset':-60., 'deltaV':2.5,
    'a':0., 'b': 60., 'tauw':300., 'N':1}

t, Vm, I, spikes = ntwk.cells.pulse_protocols.run_sim(stim, params)
fig, ax = plot(t, Vm, I, spikes, params)

# %%
