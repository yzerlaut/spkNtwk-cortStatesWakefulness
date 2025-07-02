"""

"""
from bTcL5_ntwk import *

Model['tstop'] = 1200.
Model['Farray_BgExc'] = ntwk.stim.waveform_library.varying_levels_function(\
                                                        t, [0, 20], [0, 300], 200)

for mod in np.linspace(2, 30, 10):

    Model['Farray_ModExc'] = ntwk.stim.waveform_library.varying_levels_function(\
                                                        t, [0, mod], [0, 600], 200)

    ntwk.quick_run.simulation(Model, with_Vm=3, verbose=False,
                              filename='data/temp.ntwk.h5')

    data = ntwk.recording.load_dict_from_hdf5('data/temp.ntwk.h5')

    print('Mod:', mod, 'Hz --> TcExc rate ', data['POP_ACT_TcExc'][-2000:].mean(), 'Hz')

