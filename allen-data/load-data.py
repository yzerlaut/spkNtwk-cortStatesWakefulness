# %%

import sys
sys.path += ['../..', './Network_State_Index/']

import numpy as np

import plot_tools as pt
pt.set_style('dark-notebook')

from allensdk.brain_observatory.ecephys.ecephys_project_cache import\
    EcephysProjectCache
data_directory = os.path.join(os.path.expanduser('~'),
                              'Downloads',
                              'ecephys_cache_dir')

manifest_path = os.path.join(data_directory, "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
all_sessions = cache.get_session_table() # get all sessions

# let's filter the sessions 
sessions = all_sessions[(all_sessions.sex == 'M') & \
                        (all_sessions.full_genotype.str.find('wt/wt') > -1) & \
                        #(all_sessions.session_type == 'brain_observatory_1.1') & \
                        (all_sessions.session_type == 'functional_connectivity') & \
                        ([(('VISp' in acronyms) and ('LGd' in acronyms))\
                            for acronyms in all_sessions.ecephys_structure_acronyms])]
print(30*'--'+'\n--> Number of sessions with the desired characteristics: ' + str(len(sessions))+'\n'+30*'--')

# %%

import time
from scipy.interpolate import interp1d
   
V1_Allen_params = {'pLFP_band':[40,140], # Hz -- paper's values 
                   'delta_band':[4,8], # Hz -- the delta peaks at 6hz, not 3hz..
                   'Tsmoothing':30e-3,  # s -- slightly lower smoothing because the 42ms smoothing of S1 smooth out a bit too much the delta in V1
                   'alpha_rate':2} # the rate is a virtually noiseless signal (it defines Network States), so no need to be >2 (see paper's derivation)


class Data:
    """
    an object to load, format and process the data
    we use the Allen SDK to fetch the V1 channels
    
    we format things so that data are accessible as: "data.QUANTITY" with time sampling data.t_QUANTITY (e.g. "data.pLFP", "data.t_pLFP")
    """
    
    def __init__(self, 
                 session_index=0,
                 reduced=False,
                 t0=115*60, 
                 duration=20*60, # 20 min by default
                 init=[],#['pop_act', 'pLFP', 'NSI'], # for a full init
                 verbose=False):
        """
        loading data:
        - the Allen NWB files are loaded by default, select the session according to the index of the "sessions" above
        - for initial troubleshooting, there is a "demo" option that loads a lightweight data sample provided in the repo
        - for faster analysis alter (to loop over the data), we can load the "reduced" version of the data of interest (generated below)


        can load a subset using the "t0" and "duration" args
        """
        
        self.session_index = session_index
        if reduced:
            rdata = np.load('reduced_data/Allen_FC_session%i.npy' % (\
                            self.session_index+1), allow_pickle=True).item()
            for key in rdata:
                setattr(self, key, rdata[key]) # sets LFP, pLFP, pop_act, running_speed
            for key in ['LFP', 'pLFP', 'pop_act']:
                setattr(self, 't_%s'%key, np.arange(len(getattr(self,key)))/getattr(self,'%s_sampling_rate'%key)+self.t0)
        else:
            # -------------------------------------------------------- # 
            # -- using the Allen SDK to retrieve and cache the data -- #
            # -------------------------------------------------------- # 

            print('loading session #%i [...]' % (1+session_index))
            tic = time.time()
            # we load a single session
            session = cache.get_session_data(\
                sessions.index.values[session_index])
            
            # use the running timestamps to set start and duration in the data object
            self.t0 = np.max([t0,
                              session.running_speed.start_time.values[0]])
            self.duration = np.min([duration,
                        session.running_speed.end_time.values[-1]-self.t0])

            ###########################################################
            ##                  Behavioral Data                      ##
            ###########################################################

            # let's fetch the running speed
            cond = (session.running_speed.end_time.values>self.t0) &\
                (session.running_speed.start_time.values<(self.t0+self.duration))
            self.t_running_speed = .5*(session.running_speed.start_time.values[cond]+\
                                       session.running_speed.end_time.values[cond])
            self.running_speed = session.running_speed.velocity[cond]

            try:
                pupil = session.get_pupil_data()
                self.pupil_area = np.pi*pupil['pupil_height'].values/2.*pupil['pupil_width'].values/2.
                self.t_pupil_area = pupil.index.values
            except BaseException:
                self.pupil_area, self.t_pupil_area = None, None                
                
            ###########################################################
            ##               V1 activity (units + LFP)               ##
            ###########################################################

            # let's fetch the isolated single units in V1
            V1_units = session.units[session.units.ecephys_structure_acronym == 'VISp'] # V1==VISp
            self.V1_RASTER = []
            for i in V1_units.index:
                cond = (session.spike_times[i]>=self.t0) & (session.spike_times[i]<(self.t0+self.duration))
                self.V1_RASTER.append(session.spike_times[i][cond])

            # let's fetch the V1 probe --> always on "probeC"
            V1probe_id = session.probes[session.probes.description == 'probeC'].index.values[0]

            # -- let's fetch the lfp data for that probe and that session --
            # let's fetch the all the channels falling into V1 domain
            self.V1_channel_ids = session.channels[(session.channels.probe_id == V1probe_id) & \
                          (session.channels.ecephys_structure_acronym.isin(['VISp']))].index.values

            # limit LFP to desired times and channels
            # N.B. "get_lfp" returns a subset of all channels above
            self.lfp_slice_V1 = session.get_lfp(V1probe_id).sel(time=slice(self.t0,
                                                                         self.t0+self.duration),
                                                              channel=slice(np.min(self.V1_channel_ids), 
                                                                            np.max(self.V1_channel_ids)))
            self.Nchannels_V1 = len(self.lfp_slice_V1.channel) # store number of channels with LFP in V1
            self.lfp_sampling_rate = session.probes.lfp_sampling_rate[V1probe_id] # keeping track of sampling rate

            ###########################################################
            ##               LGN activity (units + LFP)               ##
            ###########################################################

            # let's fetch the isolated single units in the dLG thalamus
            dLG_units = session.units[session.units.ecephys_structure_acronym == 'dLG'] # 
            self.dLG_RASTER = []
            for i in dLG_units.index:
                cond = (session.spike_times[i]>=self.t0) & (session.spike_times[i]<(self.t0+self.duration))
                self.dLG_RASTER.append(session.spike_times[i][cond])

            # let's fetch the dLG probe --> always on "probeC"
            dLGprobe_id = session.probes[session.probes.description == 'probeC'].index.values[0]

            # -- let's fetch the lfp data for that probe and that session --
            # let's fetch the all the channels falling into dLG domain
            self.dLG_channel_ids = session.channels[(session.channels.probe_id == dLGprobe_id) & \
                          (session.channels.ecephys_structure_acronym.isin(['VISp']))].index.values

            # limit LFP to desired times and channels
            # N.B. "get_lfp" returns a subset of all channels above
            self.lfp_slice_dLG = session.get_lfp(dLGprobe_id).sel(time=slice(self.t0,
                                                                         self.t0+self.duration),
                                                              channel=slice(np.min(self.dLG_channel_ids), 
                                                                            np.max(self.dLG_channel_ids)))
            self.Nchannels_dLG = len(self.lfp_slice_dLG.channel) # store number of channels with LFP in dLG
            self.lfp_sampling_rate = session.probes.lfp_sampling_rate[dLGprobe_id] # keeping track of sampling rate
            print('data successfully loaded in %.1fs' % (time.time()-tic))
              
        for key in init:
            getattr(self, 'compute_%s' % key)()
            

    def update_t0_duration(self, t0, duration):
        t0 = t0 if (t0 is not None) else self.t0
        duration = duration if (duration is not None) else self.duration
        return t0, duration
    
        
    def compute_pop_act(self, RASTER,
                        pop_act_bin=5e-3,
                        pop_act_smoothing=V1_Allen_params['Tsmoothing']):
        """
        we bin spikes to compute population activity
        """
        print(' - computing pop_act from raster [...]') 
        t_pop_act = self.t0+np.arange(int(self.duration/pop_act_bin)+1)*pop_act_bin
        pop_act = np.zeros(len(t_pop_act)-1)

        for i, spikes in enumerate(RASTER):
            pop_act += np.histogram(spikes, bins=t_pop_act)[0]
        pop_act /= (len(RASTER)*pop_act_bin)

        self.t_pop_act = .5*(t_pop_act[1:]+t_pop_act[:-1])
        self.pop_act = nsi.gaussian_filter1d(pop_act, 
                                             int(pop_act_smoothing/pop_act_bin)) # filter from scipy
        self.pop_act_sampling_rate = 1./pop_act_bin
        print(' - - > done !')             
        
    def compute_NSI(self, quantity='pLFP',
                    low_freqs = np.linspace(*V1_Allen_params['delta_band'], 6),
                    p0_percentile=1.,
                    alpha=2.87,
                    T_sliding_mean=500e-3,
                    with_subquantities=True,
                    verbose=True):
        """
        ------------------------------
            HERE we use the NSI API
        ------------------------------
        """
        if verbose:
            print(' - computing NSI for "%s" [...]' % quantity) 
        setattr(self, '%s_0' % quantity, np.percentile(getattr(self, quantity), p0_percentile))
        
        lfe, sm, NSI = nsi.compute_NSI(getattr(self, quantity),
                                       getattr(self, '%s_sampling_rate' % quantity),
                                       low_freqs = low_freqs,
                                       p0=getattr(self, '%s_0' % quantity),
                                       alpha=alpha,
                                       T_sliding_mean=T_sliding_mean, 
                                       with_subquantities=True) # we fetch also the NSI subquantities (low-freq env and sliding mean), set below !
        setattr(self, '%s_low_freq_env' % quantity, lfe)
        setattr(self, '%s_sliding_mean' % quantity, sm)
        setattr(self, '%s_NSI' % quantity, NSI)
        if verbose:
            print(' - - > done !') 
        
    def validate_NSI(self, quantity='pLFP',
                     Tstate=200e-3,
                     var_tolerance_threshold=None,
                     verbose=True):
        """
        ------------------------------
            HERE we use the NSI API
        ------------------------------
        """
        if verbose:
            print(' - validating NSI for "%s" [...]' % quantity) 
        
        if var_tolerance_threshold is None:
            # by default the ~noise level evaluated as the first percentile
            var_tolerance_threshold = getattr(self, '%s_0' % quantity)
 
        vNSI = nsi.validate_NSI(getattr(self, 't_%s' % quantity),
                                getattr(self, '%s_NSI' % quantity),
                                Tstate=Tstate,
                                var_tolerance_threshold=var_tolerance_threshold)
    
        setattr(self, 'i_%s_vNSI' % quantity, vNSI)
        setattr(self, 't_%s_vNSI' % quantity, getattr(self, 't_%s' % quantity)[vNSI])
        setattr(self, '%s_vNSI' % quantity, getattr(self, '%s_NSI' % quantity)[vNSI])
        if verbose:
            print(' - - > done !')
        
    def plot(self, quantity, 
             t0=None, duration=None,
             ax=None, label='',
             subsampling=1,
             color='k', ms=0, lw=1, alpha=1):
        """
        a general plot function for the quantities of this object
        
        quantity as a string (e.g. "pLFP" or "running_speed")
        """
        
        t0, duration = self.update_t0_duration(t0, duration)
        
        try:
            if ax is None:
                fig, ax =plt.subplots(1, figsize=(8,3))
            else:
                fig = None
            t = getattr(self, 't_'+quantity.replace('_NSI','').replace('_low_freq_env','').replace('_sliding_mean',''))
            signal = getattr(self, quantity)
            cond = (t>t0) & (t<(t0+duration))
            ax.plot(t[cond][::subsampling], signal[cond][::subsampling], color=color, lw=lw, ms=ms, marker='o', alpha=alpha)
            ax.set_ylabel(label)
            return fig, ax
        except BaseException as be:
            print(be)
            print('%s not a recognized attribute to plot' % quantity)
            return None, None
        
    def save_reduced_data(self):
        rData = dict(\
            V1_Allen_params=V1_Allen_params,
            lfp_sampling_rate=self.lfp_sampling_rate,
            running_speed=self.running_speed,
            #
            V1_RASTER=self.V1_RASTER,
            V1_channel_ids=self.V1_channel_ids,
            lfp_slice_V1=self.lfp_slice_V1,
            #
            dLG_RASTER=self.dLG_RASTER,
            dLG_channel_ids=self.dLG_channel_ids,
            lfp_slice_dLG=self.lfp_slice_dLG,
            )
        np.save('./reduced_data/Allen_FC_session%i.npy' % (\
                                            self.session_index+1), rData)

        
# a tool very useful to 
def resample_trace(old_t, old_data, new_t):
    func = interp1d(old_t, old_data, kind='nearest', fill_value="extrapolate")
    return func(new_t)

# %%
for i in range(len(sessions))[:1]:
    data = Data(i)
    data.save_reduced_data()
# %%
data = Data(0, reduced=False)

# %%
