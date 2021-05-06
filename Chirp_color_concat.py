# -*- coding: utf-8 -*-
"""
Created on Mon May 18 09:01:43 2020

@author: Admin
"""

for stimid in sel_stims:
    print("Computing ISI and SPIKE distance matrix for stimid: %d" % stimid)
    

    sts = Stimuli['SpikeTrains'][stimid][conditions]
    flat_sts = sts_trial_pairs_for_dy(sts)
    del sts
#%%
sts_chirp = Stimuli['SpikeTrains'][1]#[conditions]
sts_color = Stimuli['SpikeTrains'][10]#[conditions]   
st_ch_c=[]

trial_times=[]
trial_times  = np.asarray(trial_times) / (Sampling/1000)
st2 = spk.SpikeTrain(trial_times, stim_dur/(Sampling/1000))
for cll in range(len(conditions)):
    ST2=[]
    for chc in range(10):
        st2 = spk.SpikeTrain(trial_times, stim_dur/(Sampling/1000))
        if len(sts_chirp[cll][chc].spikes)>0:
            st2.spikes=np.hstack((sts_chirp[cll][chc].spikes,sts_color[cll][chc].spikes+sts_chirp[cll][chc].spikes[-1]))
            st2.t_end=sts_chirp[0][0].t_end+sts_color[0][1].t_end
            ST2.append(st2)
            
            
        else:
            trial_times=[]
            trial_times  = np.asarray(trial_times) / (Sampling/1000)
            st2 = spk.SpikeTrain(trial_times, stim_dur/(Sampling/1000))
            st2.t_end=sts_chirp[0][0].t_end+sts_color[0][1].t_end

            ST2.append(st2)
#        del st2
            
    st_ch_c.append(ST2)
#    if len(ST2)>0:

Stimuli['SpikeTrains'].append(np.asarray(st_ch_c))


#%%
plt.figure(figsize=(12,100))
ntrials =10
for i,e in enumerate(bad_cells_idx):
    for ii,st in enumerate(Stimuli['SpikeTrains'][18][e][:]):
        plt.plot(st, np.ones(len(st))+ii+i*ntrials,'k|',ms=2, lw=4)
    plt.plot((0,44000),((i+1)*ntrials+0.5,(i+1)*ntrials+0.5),'grey')
plt.title('inValid units')
#%%
for ii,st in enumerate(Stimuli['SpikeTrains'][18][44][:]):
    plt.plot(st, np.ones(len(st))+ii+i*ntrials,'k|',ms=2, lw=4)
