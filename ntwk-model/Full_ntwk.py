"""
we start from the Layer423 network that we modifies
"""
from TcL5_ntwk import *
from TcL423_ntwk import Model as Model2

# Model.update(Model2)

# update afferent freqs params


# Model['REC_POPS'] = ['TcExc', 'ReInh', 'L5Exc', 'L423Exc', 'L423Inh', 'DsInh']
# Model['AFF_POPS'] = ['BgExc', 'ModExc']

Model['tstop'] = 6e3
t = ntwk.arange(int(Model['tstop']/Model['dt']))*Model['dt']
Model['Farray_BgExc'] = ntwk.stim.waveform_library.varying_levels_function(\
                                                t, [0, 20], [0, 300], 300)
Model['Farray_ModExc'] = ntwk.stim.waveform_library.varying_levels_function(\
                                        t, [0, 8, 20], [0, 2e3, 4e3], 300)

        
if __name__=='__main__':

    
    if not sys.argv[-1]=='plot':

        ntwk.quick_run.simulation(Model, with_Vm=3, verbose=False,
                                  filename='data/full.ntwk.h5')

    data = ntwk.recording.load_dict_from_hdf5('data/full.ntwk.h5')

    # ## plot
    fig, _ = ntwk.plots.activity_plots(data, 
                                       pop_act_args=dict(smoothing=30, 
                                                         subsampling=2),
                                       fig_args=dict(figsize=(2,0.4), dpi=75))
    # ntwk.plots.few_Vm_plot(data, clip_spikes=True, vpeak=-10)

    plt.show()
