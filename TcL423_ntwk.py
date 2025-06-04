"""
we start from the Layer423 network that we modifies
"""
from Layer423_ntwk import *

# change afferent network for TC network

Model['N_TcExc'] = Model['N_AffExc'] # we keep AffExc here

keys = list(Model.keys()).copy()
for key in keys:
    if len(key.split('L423'))==2 and len(key.split('Aff'))==2:
        Model[key.replace('Aff', 'Tc')] = Model[key]
        print(key.replace('Aff', 'Tc'),  Model[key])
        Model.pop(key, None)

# connect Aff to Thal
Model['Q_AffExc_TcExc'] = 4.
Model['p_AffExc_TcExc'] = 0.05

# Give Neuronal properties to TcExc neurons
params = {\
    # --> Excitatory population (TcExc, recurrent excitation)
    'TcExc_Gl':10., 'TcExc_Cm':200.,'TcExc_Trefrac':5.,
    'TcExc_El':-70., 'TcExc_Vthre':-50., 'TcExc_Vreset':-70., 'TcExc_delta_v':0.,
    'TcExc_a':0., 'TcExc_b': 0., 'TcExc_tauw':1e9, 'TcExc_deltaV':0,
}

Model.update(params)

# update afferent freqs params

Model['Faff1'] = 2.5
Model['Faff2'] = 10.

if __name__=='__main__':

    
    if not sys.argv[-1]=='plot':

        NTWK = ntwk.build.populations(Model, ['TcExc', 'L423Exc', 'L423Inh', 'DsInh'],
                                      AFFERENT_POPULATIONS=['AffExc'],
                                      with_raster=True, 
                                      with_pop_act=True,
                                      with_Vm=4,
                                      verbose=True)

        ntwk.build.recurrent_connections(NTWK, SEED=5,
                                         verbose=True)

        t_array = ntwk.arange(int(Model['tstop']/Model['dt']))*Model['dt']
        # # # afferent excitation onto thalamic excitation
        ntwk.stim.construct_feedforward_input(NTWK, 'TcExc', 'AffExc',
                                              t_array, waveform(t_array, Model),
                                              verbose=True, SEED=27)

        ntwk.build.initialize_to_rest(NTWK)

        ntwk.collect_and_run(NTWK, verbose=True)

        ntwk.recording.write_as_hdf5(NTWK, filename='data/TcL423.ntwk.h5')

    data = ntwk.recording.load_dict_from_hdf5('data/TcL423.ntwk.h5')

    # ## plot
    fig, _ = ntwk.plots.activity_plots(data, 
                                       pop_act_args=dict(smoothing=100, 
                                                         subsampling=2, log_scale=False),
                                       fig_args=dict(figsize=(2,0.4), dpi=75))
    ntwk.plots.few_Vm_plot(data, clip_spikes=True, vpeak=-10)

    plt.show()
