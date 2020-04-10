from datavyz.main import graph_env
import numpy as np

Umodel_params = {
    # input range
    'min_Fmodul_in_Hz':0.,
    'max_Fmodul_in_Hz':15.,    
    'decay_delta_Fmodul_in_Hz':2.,
    'rise_asynch_Fmodul_in_Hz':3.,
    # Vm
    'max_delta_envelope_mV':15.,
    'max_asynch_level_mV':20.,
    'max_gamma_envelope_mV':5.,
    'rise_asynch_envelope_Firing_in_Hz':3.,
    # rate
    'max_delta_envelope_Firing_in_Hz':1.,
    'decay_delta_envelope_Firing_in_Hz':3.,
    'sparse_regime_Firing_in_Hz':1e-2,
    'dense_regime_Firing_in_Hz':20.,
    'rise_asynch_envelope_Firing_in_Hz':3.}


def threshold_linear(x):
    return (np.sign(x)+1)/2.*x

class Umodel:

    def __init__(self, Params):

        for key, val in Params.items():
            setattr(self, key, val)
        self.Td = Params['decay_delta_Fmodul_in_Hz']
        self.Tr = Params['rise_asynch_Fmodul_in_Hz']

        self.F0 = 2*(self.Td+self.Tr)

        
    def show(self):

        pass

    def compute_delta_Vm(self, Faff):

        return self.max_delta_envelope_mV*\
            np.exp(-(Faff-self.min_Fmodul_in_Hz)/self.Td)

    def compute_gamma_Vm(self, Faff):

        norm_factor = np.exp((self.max_Fmodul_in_Hz-self.F0)/self.Tr)
        max_value = self.max_gamma_envelope_mV
        return max_value*np.exp((Faff-self.F0)/self.rise_asynch_Fmodul_in_Hz)\
            /norm_factor
    
        return self.max_delta_envelope_mV*\
            np.exp(-(Faff-self.min_Fmodul_in_Hz)/self.Td)
    
    def compute_asynch_Vm(self, Faff):

        norm_factor = np.exp((self.max_Fmodul_in_Hz-self.F0)/self.Tr)
        max_value = self.max_asynch_level_mV
        return max_value*np.exp((Faff-self.F0)/self.rise_asynch_Fmodul_in_Hz)\
            /norm_factor
    
    
    def illustration(self, graph, ax=None):

        if ax is None:
            fig, ax = graph.figure(left=.7, bottom=.7)
        else:
            fig = None

        F = np.linspace(self.min_Fmodul_in_Hz, self.max_Fmodul_in_Hz)

        COLORS = ['darkgreen', 'darkblue', 'firebrick']
        ax.plot(F, self.compute_asynch_Vm(F)+.5*self.compute_delta_Vm(F),
                COLORS[0], lw=3)
        ge.annotate(ax, r'$\langle V_m \rangle^{pyr}$',
                    (.42,.45), color=COLORS[0], size='large')
        
        ax.plot(F, self.compute_delta_Vm(F), color=COLORS[1], lw=3)
        ge.annotate(ax, '$\delta_{env}^{pyr}$',
                    (0.1,.65), color=COLORS[1], size='large')
        
        ax.plot(F, self.compute_gamma_Vm(F), color=COLORS[2], lw=3)
        ge.annotate(ax, '$\\gamma_{env}^{pyr}$',
                    (1.1,.3), color=COLORS[2], ha='right', size='large')
        
        ge.set_plot(ax, ylabel='depol. (mV)',
                    xlabel='modulatory\ninput (Hz)',
                    title='U-model of cortical states      ')
        ge.annotate(fig, 'a', (0,1), va='top', bold=True, size='x-large')
        # ge.legend(ax)
        return fig, ax


if __name__=='__main__':

    um = Umodel(Umodel_params)
    ge = graph_env()
    fig,_ = um.illustration(ge)
    
    ge.show()
    
