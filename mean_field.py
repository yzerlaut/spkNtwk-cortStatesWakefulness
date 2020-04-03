import sys, pathlib
import numpy as np
sys.path.append(str(pathlib.Path(__file__).resolve()))
# from neural_network_dynamics.mean_field import find_fp, solve_mean_field_first_order
from neural_network_dynamics.main import *

from model import Model, REC_POPS, AFF_POPS

# by waiting for the proper params scan, let's pick those of the Cell Rep study
for pop in REC_POPS:
    Model['COEFFS_%s' % pop] = np.load('neural_network_dynamics/configs/The_Spectrum_of_Asynch_Dyn_2018/COEFFS_RecExc.npy')

    
if __name__=='__main__':

    from graphs.my_graph import graphs
    mg = graphs()

    tstop, dt = 1, 5e-4

    DYN_SYSTEM = {
        'Exc': {'aff_pops':AFF_POPS, 'aff_pops_input_values':[3.], 'x0':1.},
        'RecInh': {'aff_pops':['AffExc'], 'aff_pops_input_values':[3.], 'x0':1.}
    }
    INPUTS = {
        'AffExc_RecExc':np.ones(int(tstop/dt))*3.,
        'AffExc_RecInh':np.ones(int(tstop/dt))*3.
        }
        
    tstop, dt = 2, 5e-4
    t = np.arange(int(tstop/dt))*dt

    DYN_SYSTEM = {
        'Exc': {'aff_pops':AFF_POPS, 'x0':1.},
        'oscillExc': {'aff_pops':AFF_POPS, 'x0':1.},
        'PvInh': {'aff_pops':AFF_POPS, 'x0':1.},
        'SstInh': {'aff_pops':AFF_POPS, 'x0':1.},
        'VipInh': {'aff_pops':AFF_POPS, 'x0':1.}}
    INPUTS = {}
    for aff in AFF_POPS:
        for pop in REC_POPS:
            INPUTS['%s_%s' % (aff, pop)] = 0.5*np.ones(len(t))

    CURRENT_INPUTS = {'oscillExc':190*(1+np.sin(2*np.pi*3.*t))} # very low current just for demo
    
    X0 = mean_field.find_fp(Model,
                            DYN_SYSTEM,
                            INPUTS=INPUTS,
                            CURRENT_INPUTS=CURRENT_INPUTS,
                            dt=dt,
                 tstop=tstop,
                 replace_x0=True)
    
    X = mean_field.solve_mean_field_first_order(Model, DYN_SYSTEM,
                                     INPUTS=INPUTS,
                                     CURRENT_INPUTS=CURRENT_INPUTS,
                                     dt=dt, tstop=tstop)

    # print(mean_field.get_full_statistical_quantities(Model, DYN_SYSTEM))
    
    fig, ax = mg.figure()
    ax.set_yscale('log')
    mg.plot(Y=[X['Exc'], X['oscillExc'], X['PvInh'], X['SstInh'], X['VipInh']],
            COLORS=[mg.g, mg.b, mg.r, mg.purple, mg.orange], ax=ax, axes_args=dict(yticks=[0.1,1.,10], ylim=[1e-3, 100]))
    mg.show()

