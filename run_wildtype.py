# -*- coding: utf-8 -*-
"""
Created on Thu May  6 12:28:10 2021

@author: Admin
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 11:39:57 2020

@author: Admin
"""



# Recording

# Recording
notebooks_path = "C:/Users/hamed/Desktop/Hamed/github/Spike_distance_clistering/Bionic_Visison_Clustering/"
data_path = "C:/Users/hamed/Desktop/Hamed/github/Spike_distance_clistering/Bionic_Visison_Clustering/Matlab_data/"

data_dir ="C:/Users/hamed/Desktop/Hamed/github/Spike_distance_clistering/Bionic_Visison_Clustering/Matlab_data/"
#filenames=['2019_07_21_wl_secondmouce.mat','2018_12_12_left_nasal.mat','2020_01_17_rhalf1.mat','2018_11_28_WT_Left.mat','2018_11_29_WT1_Left.mat','2019_07_17_mrg.mat','2019_07_16_lw.mat','2020_01_25_r1.mat','2020_02_04_l1_before.mat','2020_02_06_r1_before.mat','2020_01_25_left.mat','2020_01_16_wl.mat']
filenames=['2020_01_17_rhalf1.mat','2020_01_25_r1.mat','2020_02_04_l1_before.mat','2020_02_06_r1_before.mat','2020_01_25_left.mat','2020_01_16_wl.mat'
           ,'2020_06_03_l2.mat','2020_06_17_r1.mat','2020_06_18_r2.mat','2020_06_18_r1.mat','2020_06_18_l1.mat'
           ,'2020_06_19.mat','2020_06_19_r2.mat','2020_07_22.mat','2020_07_23_l1.mat','2020_07_23_r1.mat'
           ,'2020_08_03_l1.mat','2020_08_03_r1.mat','2020_08_03_r2.mat','2020_08_04_r1.mat','2020_08_04_r2.mat',
           '2020_08_l1.mat']
#sf=data_dir+filenames


# stimulus information
#report_filenames  = [ data_path+"2019_07_17_mrg_report.txt" ]
#trigger_filenames = [ data_path+"trigger_2019_07_17_mrg.mat" ]
report_filenames  = [#data_path+"2019_07_21_wl_secondmouce_report.txt",
        #data_path+"2018_12_12_left_nasal_report.txt",
        data_path+"2020_01_17_rhalf1_report.txt",
                     
#        data_path+"2018_11_28_WT_Left_report.txt", 
#        data_path+"2018_11_29_WT1_Left_report.txt",
#        data_path+"2019_07_17_mrg_report_old.txt",data_path+"2019_07_16_lw_report.txt",
        data_path+"2020_01_25_r1_report.txt"
                     ,data_path+"2020_02_04_l1_before_report.txt",data_path+"2020_02_06_r1_before_report.txt"
                     ,data_path+"2020_01_25_left_report.txt",data_path+"2020_01_16_wl_report.txt"
                     ,data_path+"2020_06_03_l2_report.txt",data_path+"2020_06_17_r1_report.txt"
                     ,data_path+'2020_06_18_r2_report.txt',data_path+'2020_06_18_r1_report.txt'
                     ,data_path+'2020_06_18_l1_report.txt',data_path+'2020_06_19_report.txt'
                     ,data_path+'2020_06_19_r2_report.txt',data_path+'2020_07_22_report.txt'
                     ,data_path+'2020_07_23_l1_report.txt',data_path+'2020_07_23_r1_report.txt'
                     ,data_path+'2020_08_03_l1_report.txt',data_path+'2020_08_03_r1_report.txt'
                     ,data_path+'2020_08_03_r2_report.txt',data_path+'2020_08_04_r1_report.txt'
                     ,data_path+'2020_08_04_r2_report.txt',data_path+'2020_08_l1_report.txt']

trigger_filenames = [#data_path+"2019_07_21_wl_secondmouce_trigger.mat",
#        data_path+"2018_12_12_left_nasal_trigger.mat",
        data_path+"2020_01_17_rhalf1_trigger.mat",
#        data_path+"2018_11_28_WT_Left_trigger.mat", 
#        data_path+"2018_11_29_WT1_Left_trigger.mat", 
#        data_path+"trigger_2019_07_17_mrg.mat",data_path+"2019_07_16_lw_trigger.mat" ,
        data_path+"2020_01_25_r1_trigger.mat"
                     ,data_path+"2020_02_04_l1_before_trigger.mat" ,data_path+"2020_02_06_r1_before_trigger.mat"
                     ,data_path+"2020_01_25_left_trigger.mat",data_path+"2020_01_16_wl_trigger.mat"
                     ,data_path+"2020_06_03_l2_trigger.mat",data_path+"2020_06_17_r1_trigger.mat"
                     ,data_path+'2020_06_18_r2_trigger.mat',data_path+'2020_06_18_r1_trigger.mat'
                     ,data_path+'2020_06_18_l1_trigger.mat',data_path+'2020_06_19_trigger.mat'
                     ,data_path+'2020_06_19_r2_trigger.mat',data_path+'2020_07_22_trigger.mat'
                     ,data_path+'2020_07_23_l1_trigger.mat',data_path+'2020_07_23_r1_trigger.mat'
                     ,data_path+'2020_08_03_l1_trigger.mat',data_path+'2020_08_03_r1_trigger.mat'
                     ,data_path+'2020_08_03_r2_trigger.mat',data_path+'2020_08_04_r1_trigger.mat'
                     ,data_path+'2020_08_04_r2_trigger.mat',data_path+'2020_08_l1_trigger.mat']
#
# Set to False to reload the spike data from the original hdf5 files 
LOAD_STORED_SPIKES = False

# Set to False to re-compute the distance matrices
LOAD_STORED_DISTANCES = True


# Unit selection

# estimate the spatial spread of spikes for each unit
# poorly sorted units have a wide spread and/or high eccentricity

# exclude units with insufficient spike counts (per trial)
MIN_SPIKES_FF = 1 # min spikes in Full Field stimulus
MIN_SPIKES_CHIRP =1# min spikes in Chirp stimulus


# set to True to save the figures generated below
SAVE_FIGS = True


import os.path, sys
from os.path import dirname, join as pjoin
import scipy.io as sio

from herdingspikes import *
from spikeclass_metrics import *

import numpy as np
import seaborn as sns
import matplotlib as matplotlib

import re
import h5py
import pyspike as spk

import joblib

import scipy.cluster as cluster
import scipy.spatial.distance as distance
from scipy.spatial.distance import squareform
import sklearn.metrics.cluster as metrics
from sklearn.metrics import confusion_matrix

from sta import *
from spikeutilities import *

# plot parameters
rcParams = {
   'axes.labelsize': 12,
   'font.size': 10,
   'legend.fontsize': 12,
   'xtick.labelsize': 12,
   'ytick.labelsize': 12,
   'text.usetex': False,
   'figure.figsize': [4, 2.5] # instead of 4.5, 4.5
   }

get_cluster_inds = lambda C, cl: np.where(C==cl)[0]
data_path=data_dir
i = 0
base_path = data_path + filenames[i]
base_path = base_path.replace('.hdf5','__save_.npy')
mkp = lambda n: base_path.replace('_.npy','_'+n+'.npy')



# Stuff that will be filled from spikeclass
Times = {}
ClusterIDs = {}
ClusterLoc = {}
ClusterSizes = {}
# Stuff we want to compute
sCorrs = {}
ClusterEvals = {}
Sampling = 1#7062.058198545425 # copied from previous output
cell_names=[]
dsi_cels=[]
esta={}

E_STA=[]
esta_all=[]
if LOAD_STORED_SPIKES == False:
    ncl={}
    filenames1=filenames[0:10]
    for i,f in enumerate(filenames):
        mat_fname = pjoin(data_dir, f)
        mat_contents = sio.loadmat(mat_fname)

#        O = spikeclass(sf)
        Times[i] = mat_contents['data']*Sampling
        ClusterIDs[i]= mat_contents['ClusterIDs']
        ncl[i] = len(unique(ClusterIDs[i]))
        
        esta_fname = pjoin(data_dir, 'e_STA '+f)
        try:
            ESTA=sio.loadmat(esta_fname)['Data'][0]
            esta[i]= ESTA[0:-1]#['e_sta']# trigerss are removed
            esta_all=np.hstack((esta_all,esta[i]))

        except IOError:
            esta[i]= [None] * ncl[i]
            print( "No e-sta" , f)
            esta_all=np.hstack((esta_all,[None] * ncl[i]))

        print("Number of units: %d" % ncl[i],f)

        E_STA.append(np.asarray(esta[i]))
        cell_names.append(np.asarray(mat_contents['Chname']))
        if '2019_07_21_wl_secondmouce.mat' in f:# this dataset does not have moving bar stimulus manualy set to zero
            dsi_cels=np.hstack((dsi_cels,np.ones(ncl[i], dtype=bool)))
          
#            dsi_cels.append(np.ones(ncl[i], dtype=bool))
        else: 
            dsi_cels=np.hstack((dsi_cels,np.zeros(ncl[i], dtype=bool)))



Cell_names = []
i=-1
for sublist in cell_names:
    i=i+1
    
    for val in sublist[0]:
        Cell_names.append(filenames[i][:-4]+"_"+val[0])
Cell_names=np.asarray(Cell_names)  


#%% PSD

psd_all= [None] * len(Cell_names) 
freq_details=[None] * len(Cell_names) 
#%% find significant stas
from scipy import stats
from scipy import signal

stas_dis=[]
with_sta_cells=np.zeros(len(esta_all),dtype=bool)
#valid_units = np.ones(ncl, dtype=bool)
high_snr=np.zeros(len(esta_all),dtype=bool)
p_values=np.zeros(len(esta_all),dtype=bool)
mean_var_ratio=[]
min_maz=np.zeros(len(esta_all))
for i in range(len(esta_all)):
    if not(esta_all[i]==None):
        if len(esta_all[i][2])>1:
            esta_i=np.asarray(esta_all[i][3].flatten())

            mean_var_ratio.append(abs(np.average(esta_all[i][3],axis=0)/var(esta_all[i][3],axis=0)))
            with_sta_cells[i]=True
#            peaks=np.hstack((np.argmax(esta_i),np.argmax(-esta_i)))
#            peak_sta=esta_i[peaks[np.argmax([np.linalg.norm((esta_i[peaks][0]-mean(esta_i))),np.linalg.norm((esta_i[peaks][1]-mean(esta_i)))])]]
#            pck=peaks[np.argmax(np.linalg.norm((esta_i[peaks],mean(esta_i))))]
#            high_snr[i]=np.linalg.norm(peak_sta-mean(esta_i))>4*std(esta_ii)
            k2, p = stats.normaltest(esta_i)
            p_values[i]=p<.001



            freqsl, psd = signal.welch(esta_all[i][3].flatten(),25, nperseg=25)            
            psd_all[i]= psd
            
            norm_psd=psd/sum(psd);
            spsd=norm_psd
            max_arg=[]
            for ps in range(len(freqsl)):
                if sum(spsd)>=.5:
                    max_arg.append(np.argmax(spsd))
        
                    spsd=np.where(spsd==spsd[np.argmax(spsd)], 0, spsd) 
            roi=freqsl[np.min(max_arg)],freqsl[np.max(max_arg)]
            freq_details[i]=np.hstack((freqsl[np.argmax(psd)],    roi))
    
    #plt.hist(np.asarray(mean_var_ratio).flatten(), bins=55);
with_sta_cells=( (p_values)&(with_sta_cells) )   
psd_all=np.asarray(psd_all)
freq_details=np.asarray(freq_details)# peak frequancy and bandwidth at 50% of peak
#%% plot good and bad sta
import scipy.stats

gsta_matrix=[]
goodstas=esta_all[with_sta_cells]
for ii in range(len(goodstas)):
    if  not(goodstas[ii]==None):
        if len(goodstas[ii][2])>1:
            stime=goodstas[ii][2]
            gsta_matrix.append(asarray(goodstas[ii][3].flatten()))
            
bsta_matrix=[]
bad_trl_sta=[]    
badstas=esta_all[~with_sta_cells]
for ii in range(len(badstas)):
    if  not(badstas[ii]==None):
        if len(badstas[ii][2])>1:
            stime=badstas[ii][2]
            bsta_matrix.append(asarray(badstas[ii][3].flatten()))



plt.figure(figsize=(10,18))
plt.subplot(121)
plt.ylabel('Unit #',fontsize=14)
plt.xlabel('Time (s)')

plt.imshow(scipy.stats.zscore(gsta_matrix, axis=1, ddof=0),aspect='auto', extent=[-1,1,1,len(gsta_matrix)])            
plt.subplot(122)

plt.imshow(scipy.stats.zscore(bsta_matrix, axis=1, ddof=0),aspect='auto',extent=[-1,1,1,len(bsta_matrix)])  
plt.xlabel('Time (s)')
plt.rcParams.update({'font.size': 14})

if SAVE_FIGS:
   plt.savefig('good_bad_sta_wild_type.pdf', bbox_inches='tight')
#%%      organising spike trains
Stimuli = {}
timeStampMatrices = {}
trg={}
for ix,(rf,tf) in enumerate(zip(report_filenames,trigger_filenames)):
    print("Reading file set %s: %s, %s" % (ix,rf,tf))
    Stimuli[ix], timeStampMatrices[ix] = read_stimuli_info(rf,tf)
    trg[ix] = loadmat(tf)
#timeStampMatrices[ix]=timeStampMatrices[ix]*50000
    
for ix in range(len(Stimuli)):    
    stim_ntrials = []
    for i,name in enumerate(Stimuli[ix]['Name']):
        nstim1 = Stimuli[ix]['Nstim1'][i]
        ntrials = 0.1
        
    
        if 'Fullfield' in name:
          
            ntrials = nstim1
        elif 'chirp2' in name:
            ntrials = nstim1
        elif 'color' in name:
            ntrials = nstim1
     #       
        elif 'movingbar' in name:
            ntrials = nstim1
        elif 'Bar' in name:
            ntrials = 5.0
        else:
            print("Unknown stimulus name: %s." % name)
            raise Exception
    
        stim_ntrials.append(int(ntrials))
    Stimuli[ix]['NTrials'] = stim_ntrials
stim_durs=np.array([4,32,4,4,4,4,4,4,4,4,12]) *Sampling   
SpikeTimes = {}
for ix in ClusterIDs:
  SpikeTimes[ix] = []
  for cl in range(ncl[ix]):
    cl_spikes = np.where(ClusterIDs[ix]==cl+1)[0]
    cl_times  = np.unique(Times[ix][cl_spikes])
    SpikeTimes[ix].append(cl_times)

STss=[]  
for stimid in range(11):
    stim_trains_all= []

    stim_trains = []

    # 
    for ix in range(len(ClusterIDs)):
        STs = []
            #stimid=1
        # figure out how long each stimulus is
        n_trials    = Stimuli[ix]['NTrials'][stimid]
        stim_img_n  = Stimuli[ix]['Nstim1'][stimid] / n_trials
        stim_img_ms = Stimuli[ix]['Nrefresh'][stimid] * (1000/60)
        stim_dur    =stim_durs[stimid] #np.ceil(stim_img_ms * (Sampling/1000) * stim_img_n)
        # get the stimulus start times and reshape to [n_trials,-1]
        #    stim_start_end = get_stimtimes(stimid, Stimuli[ix], timeStampMatrices[ix]).reshape([n_trials,-1])
        if stimid==0:        
            stim_start_end=np.array(trg[ix].get('Fullfield')).flatten().reshape([n_trials,-1])*Sampling
            n_trials=60

        elif stimid==1:
            stim_start_end=np.array(trg[ix].get('chirp2')).flatten().reshape([n_trials,-1])*Sampling
            n_trials=10
        elif stimid==2:
            stim_start_end=np.array(trg[ix].get('d0')).flatten().reshape([n_trials,-1])*Sampling
        elif stimid==3:
            stim_start_end=np.array(trg[ix].get('d45')).flatten().reshape([n_trials,-1])*Sampling
        elif stimid==4:
            stim_start_end=np.array(trg[ix].get('d90')).flatten().reshape([n_trials,-1])*Sampling
        elif stimid==5:
            stim_start_end=np.array(trg[ix].get('d135')).flatten().reshape([n_trials,-1])*Sampling
        elif stimid==6:
            stim_start_end=np.array(trg[ix].get('d180')).flatten().reshape([n_trials,-1])*Sampling
        elif stimid==7:
            stim_start_end=np.array(trg[ix].get('d225')).flatten().reshape([n_trials,-1])*Sampling
        elif stimid==8:
            stim_start_end=np.array(trg[ix].get('d270')).flatten().reshape([n_trials,-1])*Sampling
        elif stimid==9:
            stim_start_end=np.array(trg[ix].get('d315')).flatten().reshape([n_trials,-1])*Sampling
        elif stimid==10:
            stim_start_end=np.array(trg[ix].get('color')).flatten().reshape([n_trials,-1])*Sampling
            n_trials=30    
        for cl in range(ncl[ix]):
          cl_trains = []
          # use pre-filtered cluster times, avoids doing it every time
          cl_times = SpikeTimes[ix][cl]
          for tx in range(n_trials):
            s0 = stim_start_end[tx,0]
            s1 = s0 + stim_dur
            trial_filter = np.where((cl_times >= s0) & (cl_times <= s1))[0]
            trial_times  = cl_times[trial_filter] - s0
            trial_times  = trial_times / (Sampling/1000)
            st = spk.SpikeTrain(trial_times, stim_dur/(Sampling/1000))
            cl_trains.append(st)
          stim_trains.append(cl_trains)
          stim_trains_all.append(cl_trains)
        STs.append(np.asarray(stim_trains))
     # Stimuli[ix]['SpikeTrains'] = STs
    del STs
    STss.append(np.asarray(stim_trains_all))  
Stimuli['SpikeTrains'] = STss


total = 0
for i in ncl.values():
      total += i
print(total)
    
#values = asarray(ncl.values())
plot(spk.psth(Stimuli['SpikeTrains'][0][0],100).y)

print(len(esta_all))
#%% Concatenate chirp anc color
ncl=len(   stim_trains_all )
sts_flash = Stimuli['SpikeTrains'][0]#[conditions]

sts_chirp = Stimuli['SpikeTrains'][1]#[conditions]
sts_color = Stimuli['SpikeTrains'][10]#[conditions]   
st_ch_c=[]

trial_times=[]
trial_times  = np.asarray(trial_times) / (Sampling/1000)
#st2 = spk.SpikeTrain(trial_times, stim_dur/(Sampling/1000))
for cll in range(ncl):
    ST2=[]
    for chc in range(10):
        st2 = spk.SpikeTrain(trial_times, 48000/(Sampling/1000))
        if len(sts_chirp[cll][chc].spikes)>0:
#            st2.spikes=np.hstack((sts_chirp[cll][chc].spikes,sts_color[cll][chc].spikes+sts_chirp[cll][chc].spikes[-1]))
            st2.spikes=np.hstack((sts_flash[cll][chc].spikes, sts_chirp[cll][chc].spikes+4000,sts_color[cll][chc].spikes+36000))

            st2.t_end=sts_flash[0][0].t_end+sts_chirp[0][0].t_end+sts_color[0][1].t_end
            ST2.append(st2)           
           
        else:
            trial_times=[]
            trial_times  = np.asarray(trial_times) / (Sampling/1000)
            st2 = spk.SpikeTrain(trial_times, 48000/(Sampling/1000))
            st2.t_end=sts_flash[0][0].t_end+sts_chirp[0][0].t_end+sts_color[0][1].t_end

            ST2.append(st2)
#        del st2
            
    st_ch_c.append(ST2)
#    if len(ST2)>0:

Stimuli['SpikeTrains'].append(np.asarray(st_ch_c))
from scipy.stats.stats import pearsonr

#%% function to remove outliers
def Remove_outliers(Stimuli):
    stimuli=Stimuli['SpikeTrains'][0]
    diffs=np.array([])
    for i,st in enumerate(stimuli):
        diff_cell=np.array([])
        for j,s in enumerate(st):
            diff_cell=np.append(diff_cell,np.diff(s)/1000)
        if len(diff_cell)>0:    
            diffs=np.append(diffs,np.median(diff_cell))
        else:
            diffs=np.append(diffs,0)
    
        
    sums=np.array([])
    for i,st in enumerate(stimuli):
        mean_cell=np.array([])
        for j,s in enumerate(st):
            mean_cell=np.append(mean_cell,len(s))
        sums=np.append(sums,sum(mean_cell)/len(st))
           
        
    sum_zs=(sums[:]-np.mean(sums))/np.std(sums)
    diff_zs=(diffs[:]-np.mean(diffs))/np.std(diffs)
    
    
    data_array_std = np.std(diffs)
    data_array_mean = np.mean(diffs)
    sigma_stdev = (2*data_array_std)+data_array_mean
    upper_limit = data_array_mean+data_array_std
    lower_limit = data_array_mean-data_array_std
    
    condition_diff = (diffs>0)& (diffs<upper_limit)&(diffs>0)
    condition_0ff_diff=~condition_diff
    
    data_array_std = np.std(sums)
    data_array_mean = np.mean(sums)
    sigma_stdev = (2*data_array_std)+data_array_mean
    upper_limit = data_array_mean+sigma_stdev
    lower_limit = data_array_mean-sigma_stdev
    
    condition_sums=(sums>0)& (sums<upper_limit)&(sums>10)
    condition_sums_off=~condition_sums
    return condition_sums,condition_diff

#%% function to compute correlation between stimulus and response
def pearsonr_score(Stimuli):
    stimuli=Stimuli['SpikeTrains'][0]
    
    b1 = np.full(1,0)
    b2 = np.ones(19)
    b4 =np.ones(1)*0
    
    b3 = np.ones(19)*0
    org_signal_interval = np.array([*b1,*b2,*b3,*b4])
    plot(org_signal_interval)
    
    PSTH = np.empty((len(stimuli),40))
    pearson_corr=np.empty((len(stimuli)))
    
    for i in range(len(stimuli)):
        PSTH[i,:]=spk.psth(Stimuli['SpikeTrains'][0][i],100).y
        pearson_corr[i] = pearsonr(PSTH[i,:],org_signal_interval)[0]
        #plot(pearson_corr)   
    return pearson_corr
    
cond_sum,cond_diff=Remove_outliers(Stimuli)
pearson_corr=pearsonr_score(Stimuli)  
#%%  remove low quality data based on varian ration index

valid_units = np.ones(ncl, dtype=bool)
print("Valid units: %d" %np.sum(valid_units))
def psth_trl(spike_trains, bin_size):
    bin_count = int((spike_trains.t_end - spike_trains.t_start) /
                    bin_size)
    bins = np.linspace(spike_trains.t_start, spike_trains.t_end,
                       bin_count+1)

    # N = len(spike_trains)
    combined_spike_train = spike_trains.spikes
#    for i in range(1, len(spike_trains)):
#        combined_spike_train = np.append(combined_spike_train,
#                                         spike_trains[i].spikes)

    vals, edges = np.histogram(combined_spike_train, bins, density=False)
    #bin_size = edges[1]-edges[0]
    return 1000*(vals/bin_size)#PieceWiseConstFunc(edges, vals) 

empty_trials = {}
# spike count in each full field trial
empty_trials[0] = np.asarray([np.median([len(s) for s in st])<MIN_SPIKES_FF for st in Stimuli['SpikeTrains'][0]])
# spike count in each chirp trial
empty_trials[1] = np.asarray([np.median([len(s) for s in st])<MIN_SPIKES_CHIRP for st in Stimuli['SpikeTrains'][1]])
#empty_trials[1]=empty_trials[0] 

ff=[]
var_idx = np.empty(ncl)

for cel in range(ncl):# CELL NUMBER
    st=Stimuli['SpikeTrains'][1][cel]
    psthtrl=[]
    for s in range(len(st)):# TRIAL NUMBER
        #ffs.append(np.asarray(st.flatten()[s]))
        #ff=np.concatenate((ff,np.asarray(st.flatten()[s])),axis=0)
        if len(st[s])>0:
            psthtrl.append(psth_trl(st[s],225))    
    var_idx[cel]=mean(np.var(psthtrl,axis=0,ddof=1))/var(psthtrl)
    del psthtrl
plt.hist(var_idx, bins = 550)
plt.xlabel('Variacne ratio')
plt.ylabel('Occurrence')
axvline(x=1,linewidth=1, color='r',linestyle='--')

good_cells=(var_idx<1)#&( var_idx>0.6)
good_cellsts_idx = np.where(((~empty_trials[1]&(~empty_trials[0]))&(good_cells)))[0][:]
bad_cells_idx = np.where(((~good_cells)|(empty_trials[1])|(empty_trials[0])))[0][:]
good_cells_idx = np.where(((~empty_trials[1]&(~empty_trials[0]))&(good_cells)))[0][:]


conditions =((~empty_trials[1]&(~empty_trials[0]))&(good_cells)&(with_sta_cells)&(cond_diff)&(abs(pearson_corr)>.1))

print("Valid units remaining: %d" %np.sum(conditions))
np.save('conditions_wildtype',conditions)
#%% plot psth
valids=np.where(conditions)
celss=zip(*valids)

PSTH = np.empty((size(valids),320))

plt.figure(figsize=(15,10))

for i,ii in enumerate(celss):
#    plot(spk.psth(Stimuli['SpikeTrains'][1][ii],100).y)
    PSTH[i,:]=spk.psth(Stimuli['SpikeTrains'][1][ii],100).y
PSTH_norm = PSTH/np.max(PSTH, axis=1).reshape(-1,1)

#plt.figure(figsize=(15,10))
#plt.imshow(PSTH_norm[np.argsort(np.mean(PSTH_norm,axis=1))], aspect='auto')


#%%

# select full field and chirp
sel_stims = [0,1,11]

def flat_sts_for_dy(st):
    flat = []
    for i in range(st.shape[0]):
      for j in range(i+1,st.shape[0]):
        flat.append(st[[i,j],:].flatten())
    return flat

    # For the _fullfield_ and _chirp_ stimuli only.


def sts_trial_pairs_for_dy(st):
    flat_pairs = []
    for i in range(st.shape[0]):
        for j in range(i+1,st.shape[0]):
            flat_pairs.append((st[i,:],st[j,:]))
    return flat_pairs

def compute_SPIKE_on_flat_pair(pair):
    sti = pair[0]
    stj = pair[1]
    assert sti.shape[0] == stj.shape[0]
    ds = []
    for i in range(sti.shape[0]):
        for j in range(i+1,sti.shape[0]):
            ds.append(spk.spike_distance([sti[i], stj[j]]))
    return np.average(ds)

def compute_ISI_on_flat_pair(pair):
    sti = pair[0]
    stj = pair[1]
    assert sti.shape[0] == stj.shape[0]
    ds = []
    for i in range(sti.shape[0]):
        for j in range(i+1,sti.shape[0]):
            ds.append(spk.isi_distance([sti[i], stj[j]]))
    return np.average(ds)

def spikeRates(st_s):
    sr = []
    func_c = lambda s: np.count_nonzero(s.spikes)
    for st in st_s:
        cs = [c for c in map(func_c, st)]
        hzs = np.divide(cs,(st[0].t_end-st[0].t_start)/1000)
        sr.append( np.average(hzs) )
    return np.asarray(sr)

if LOAD_STORED_DISTANCES == False:

    SPIKE_dist_ys = []
    ISI_dist_ys   = []

    for stimid in sel_stims:
        print("Computing ISI and SPIKE distance matrix for stimid: %d" % stimid)

        sts = Stimuli['SpikeTrains'][stimid][conditions]
        flat_sts = sts_trial_pairs_for_dy(sts)
        del sts

        SPIKE_dy = joblib.Parallel(n_jobs=-1,verbose=1)(joblib.delayed(compute_SPIKE_on_flat_pair)(st) for st in flat_sts)
        print(np.asarray(SPIKE_dy).shape)
        ISI_dy   = joblib.Parallel(n_jobs=-1,verbose=1)(joblib.delayed(compute_ISI_on_flat_pair)(st) for st in flat_sts)
        print(np.asarray(ISI_dy).shape)

        print("ISI and SPIKE complete for stimid: %d" % stimid)

        assert(distance.is_valid_y(SPIKE_dy))
        assert(distance.is_valid_y(ISI_dy))

        SPIKE_dist_ys.append(SPIKE_dy)
        ISI_dist_ys.append(ISI_dy)

        del SPIKE_dy
        del ISI_dy

    SPIKE_dist_ys = np.asarray(SPIKE_dist_ys)
    ISI_dist_ys   = np.asarray(ISI_dist_ys)

    print(SPIKE_dist_ys.shape)
    print(ISI_dist_ys.shape)
    
    np.save('SPIKE_dist_wildtype',SPIKE_dist_ys)
    np.save('ISI_dist_wildtype',ISI_dist_ys)
else:
    SPIKE_dist_ys = np.load('SPIKE_dist_wildtype.npy')
    ISI_dist_ys   = np.load('ISI_dist_wildtype.npy')
    

#%% compute dsi and distance matrix

#ncl=ncl[0]
OOi = np.zeros(ncl)
for cl in range(ncl):
    # FF is 4000 ms, take PSTH by spliting time into two parts
    r = spk.psth(Stimuli['SpikeTrains'][0][cl],100).y
    ron=np.max(r[:20])
    roff=np.max(r[20:])
#    oo = r[[0,1]]
    oo = [ron,roff]
    OOi[cl] = (oo[0]-oo[1])/(oo[0]+oo[1])
# fix NaNs to 0.0
OOi[np.where(np.isnan(OOi))] = 0.0

    
plt.hist(OOi, bins =40)
#plt.show()

ds_stim={}
ds_stim['SpikeTrains'] = Stimuli['SpikeTrains'][2:10]
ds_dirs = ['0', '45', '90', '135', '180', '225', '270', '315']
ds_max = []
ds_avg = []
for sts in ds_stim['SpikeTrains']:
    ds_max.append( maxSpikeRate(sts, ms=200) )
    ds_avg.append( spikeRates(sts) )
ds_max = np.asarray(ds_max)
ds_avg = np.asarray(ds_avg)
DSi = []
for cl in range(ncl):
    DSi.append(computeDSi(ds_max[:,cl],ds_avg[:,cl],cl))
DSi = np.asarray(DSi)
# fix NaNs
DSi[np.isnan(DSi[:,0]),0] = 0    
DSi[dsi_cels==1]=0# this dataset does not have moving bar stimulus manualy set to zero

# re-create distance matrices, excluding filtered units
conditions_all = np.copy(conditions)

def get_tri_sq(m):
    return m[np.tril_indices(int(np.sqrt(2*m.shape[0]+1/4)+1/2),-1)]

ISI_dist_ys_valid = {}
SPIKE_dist_ys_valid = {}
for i in range(3):
    ISI_dist_ys_valid[i] = np.copy(ISI_dist_ys[i])
    SPIKE_dist_ys_valid[i] =  np.copy(SPIKE_dist_ys[i])

    invalid_sta=np.array([])
    bad_inds = np.isin(np.where(conditions)[0], invalid_sta)

    mask = np.ones(squareform(ISI_dist_ys_valid[i]).shape, dtype=bool)
    mask[bad_inds,:] = 0
    mask[:,bad_inds] = 0
    n_remain = mask.shape[0]-np.sum(bad_inds)
    new_distances = squareform(ISI_dist_ys_valid[i])[mask].reshape((n_remain,n_remain))
    ISI_dist_ys_valid[i] = new_distances[np.triu_indices_from(new_distances,1)]
    new_distances = squareform(SPIKE_dist_ys_valid[i])[mask].reshape((n_remain,n_remain))
    SPIKE_dist_ys_valid[i] = new_distances[np.triu_indices_from(new_distances,1)]



plt.figure()
sns.distplot(OOi, label='all')
sns.distplot(OOi[conditions_all], label='valid')
plt.xlabel('Bias index')
plt.legend()

plt.figure()
sns.distplot(DSi[:,0],label='all')
sns.distplot(DSi[conditions_all,0], label='valid')
plt.xlabel('Direction selectivity')
plt.legend()
#

#%%
# obtain Gap statistic
tss = np.arange(0.3,3,0.01) # threshold values to test
gaps = np.empty((3,2,tss.shape[0]))

NCs_gap = np.empty((3,2,tss.shape[0]))

ncls = np.arange(3,61,1)
metric_scores = np.empty((3,4,len(ncls)))
links_ward_ISI   = []
links_ward_SPIKE = []
mk_l = lambda dy: cluster.hierarchy.linkage(dy, method='ward')
for i in range(3):
    links_ward_ISI.append(mk_l(ISI_dist_ys_valid[i]))
    links_ward_SPIKE.append(mk_l(SPIKE_dist_ys_valid[i]))

for s in (0,1,2):
    for i_d,d in enumerate((ISI_dist_ys_valid[s],SPIKE_dist_ys_valid[s])):
        Nc, Wk, Nc_shuff, Wk_shuff, Dk, Dk_shuff, ts = eval_gap_scores(d,tss)
        gaps[s,i_d,:] = np.log(Wk_shuff)-np.log(Wk)
        NCs_gap[s,i_d,:] = Nc
        
    isidm = distance.squareform(ISI_dist_ys_valid[s])
    spkdm = distance.squareform(SPIKE_dist_ys_valid[s])
    for i,t in enumerate(ncls):
        fcls_isi = cluster.hierarchy.fcluster(links_ward_ISI[s], t=t, criterion='maxclust')
        fcls_spk = cluster.hierarchy.fcluster(links_ward_SPIKE[s], t=t, criterion='maxclust')
        metric_scores[s,0,i] = metrics.adjusted_mutual_info_score(fcls_isi,fcls_spk)
        metric_scores[s,1,i] = metrics.completeness_score(fcls_isi,fcls_spk)
        metric_scores[s,2,i] = metrics.adjusted_rand_score(fcls_isi,fcls_spk)  
    
for i,t in enumerate(ncls):
    fcls_0 = cluster.hierarchy.fcluster(links_ward_ISI[0], t=t, criterion='maxclust')
    fcls_1 = cluster.hierarchy.fcluster(links_ward_ISI[1], t=t, criterion='maxclust')
    metric_scores[0,3,i] = metrics.adjusted_mutual_info_score(fcls_0,fcls_1)
    fcls_0 = cluster.hierarchy.fcluster(links_ward_SPIKE[0], t=t, criterion='maxclust')
    fcls_1 = cluster.hierarchy.fcluster(links_ward_SPIKE[1], t=t, criterion='maxclust')
    metric_scores[1,3,i] = metrics.adjusted_mutual_info_score(fcls_0,fcls_1)
    fcls_0 = cluster.hierarchy.fcluster(links_ward_SPIKE[0], t=t, criterion='maxclust')
    fcls_1 = cluster.hierarchy.fcluster(links_ward_SPIKE[2], t=t, criterion='maxclust')
    metric_scores[2,3,i] = metrics.adjusted_mutual_info_score(fcls_0,fcls_1)
    
plt.rcParams.update({'font.size': 12})
plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
plt.rc('ytick', labelsize=10)    # fontsize of the tick labels


plt.figure(figsize=(12,4))
ax = plt.subplot(121)
s_labels = ('Flash','Chirp','Flash_Chirp_color')
d_labels = ('ISI distance','SPIKE distance','SPIKE distance 2')

gaps[gaps == np.Inf] = np.NaN# I added 2022

with plt.rc_context(rcParams):
    for s in (0,1,2):
        for d in (0,1):
            p = plt.plot(NCs_gap[s,d,:],gaps[s,d,:], label=s_labels[s]+'; '+d_labels[d])
            print(s_labels[s]+' '+d_labels[d]+' gap stat peak at '+str(NCs_gap[s,d,np.nanargmax(gaps[s,d])])+' clusters')
            print([s,d ,NCs_gap[s,d,np.nanargmax(gaps[s,d])],np.nanargmax(gaps[s,d])])

            plt.vlines(NCs_gap[s,d,np.nanargmax(gaps[s,d])],0,gaps[s,d,np.nanargmax(gaps[s,d])],linestyles='--',colors=p[0].get_c())
plt.xlim((0,40))
plt.ylim((0,0.7))
plt.legend(frameon=False,loc = 'upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel('Number of clusters')
plt.ylabel('Gap statistic')

ax = plt.subplot(122)
m = 0# use mutual info
with plt.rc_context(rcParams):
    for s in (0,1,2):
        p = plt.plot(ncls,metric_scores[s,m,:], label=s_labels[s])
        print(s_labels[s]+' mi score peak at '+str(ncls[np.argmax(metric_scores[s,m,:])])+' clusters')
        plt.vlines(ncls[np.argmax(metric_scores[s,m,:])],0,metric_scores[s,m,np.argmax(metric_scores[s,m,:])],linestyles='--',colors=p[0].get_c())
#    for s in (0,1,2):
#        p = plt.plot(ncls,metric_scores[s,3,:], label=d_labels[s])
#        print(d_labels[s]+' mi score peak at '+str(ncls[np.argmax(metric_scores[s,m,:])])+' clusters')
#        plt.vlines(ncls[np.argmax(metric_scores[s,3,:])],0,metric_scores[s,3,np.argmax(metric_scores[s,3,:])],linestyles='--',colors=p[0].get_c())
plt.xlim((0,60))
#plt.ylim((0,1))

plt.legend(frameon=False,loc = 'lower right')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel('Number of clusters')
plt.ylabel('Adjusted mutual information');

t_best_ff = ncls[np.argmax(metric_scores[0,m,:])]
t_best_chirp = ncls[np.argmax(metric_scores[1,m,:])]
t_best_ff_chirp_color = ncls[np.argmax(metric_scores[2,m,:])]

if SAVE_FIGS:
    plt.savefig('clusters_comparison_mixed_wildtype.pdf', bbox_inches='tight')    


  
#%% plot results

stim = 2# choose the stimulus number: 0 is flash, 1 is chirp, 2 combination of all
t = t_best_ff_chirp_color
st_name='All'
fcls = []
fcls.append(cluster.hierarchy.fcluster(links_ward_ISI[stim], t=t, criterion='maxclust'))
fcls.append(cluster.hierarchy.fcluster(links_ward_SPIKE[stim], t=t, criterion='maxclust'))

cnf_matrix = confusion_matrix(fcls[0], fcls[1], labels=np.arange(1,t+1))
normed_cnf = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

with plt.rc_context(rcParams):
    plt.figure(figsize=(3,3))
    plt.imshow(normed_cnf[np.argmax(normed_cnf,axis=0),:],origin='lower')
    plt.axis('equal')
    cb = plt.colorbar()
    cb.set_label('Cluster overlap');
    plt.xlabel('ISI clusters')
    plt.ylabel('SPIKE clusters')
#    plt.xticks((0,10,20))
#    plt.yticks((0,10,20))
    
if SAVE_FIGS:
    plt.savefig('clusters_MI_mixed_ovelap_'+st_name+'.pdf', bbox_inches='tight')    


def seriation(Z,N,cur_index):
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index-N,0])
        right = int(Z[cur_index-N,1])
        return (seriation(Z,N,left) + seriation(Z,N,right))
    
def compute_serial_matrix(dist_mat,method="ward"):
    sq_dist_mat = squareform(dist_mat)
    N = sq_dist_mat.shape[0]
    res_linkage = cluster.hierarchy.linkage(dist_mat, method=method)
    #linkage(flat_dist_mat, method=method,preserve_input=True)
    res_order = seriation(res_linkage, N, N + N-2)
    seriated_dist = np.zeros((N,N))
    a,b = np.triu_indices(N,k=1)
    seriated_dist[a,b] = sq_dist_mat[ [res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b,a] = seriated_dist[a,b]
    
    return seriated_dist, res_order, res_linkage

#fcls_ditancematrix = cluster.hierarchy.fcluster(res_linkage[np.asarray(res_order),:], t=t, criterion='maxclust')

sorted_normed_cnf, res_order, res_linkage = compute_serial_matrix(SPIKE_dist_ys_valid[stim],method='ward')
with plt.rc_context(rcParams):
    plt.figure(figsize=(3,3))
    plt.imshow(sorted_normed_cnf, cmap=plt.cm.CMRmap, origin='lower')
    cb = plt.colorbar()
    cb.set_label('SPIKE distance')
    plt.axis('equal')
    plt.xlabel('Cell #')
    plt.ylabel('Cell #')

#    plt.xticks((0,40,00))
#    plt.yticks((0,40,80))
    
if SAVE_FIGS:
    plt.savefig('clusters_distance_matrix_clustered_SPIKE_mixed_wildtype'+st_name+'.pdf', bbox_inches='tight')    
    




#%% start clustering

new_distances=SPIKE_dist_ys_valid[stim]
#new_distances=ISI_dist_ys_valid[1]
l = cluster.hierarchy.linkage(new_distances, method='ward')
fcls = cluster.hierarchy.fcluster(l, t=t, criterion='maxclust')
n_flat_clusters = np.unique(fcls).shape[0]

print("Distance %.2f\nNumber of (flat) clusters: %d" % (t,n_flat_clusters))
silhouettes = metrics.silhouette_samples(distance.squareform(new_distances),fcls,metric='precomputed')
print("Mean Silhouette Coefficient: %.2f" % np.average(silhouettes))
Cells_names_and_clusters=list(zip(Cell_names[conditions_all],fcls))
### plot dendrogram

plt.figure(figsize=(16,3))

## set things up for coloured dendrogram
n = 41
pal = sns.diverging_palette(180,359,sep=1,n=n)
OOi_cspace = np.linspace(-1,1,n)
OOi_c_func = lambda i: pal[np.searchsorted(OOi_cspace,OOi[conditions_all][i])]
DSi_cspace = np.linspace(0,1,n)
DSi_c_func = lambda i: pal[np.searchsorted(DSi_cspace,DSi[conditions_all][i,0])]

def create_colors_for_linkage(Z,data_len,base_col_func):
    colors = []
    for i1,i2,d,c in Z:
        if i1 >= data_len:
            c1 = colors[int(i1)-data_len]
        else:
            c1 = base_col_func(int(i1))
            
        if i2 >= data_len:
            c2 = colors[int(i2)-data_len]
        else:
            c2 = base_col_func(int(i2))
        new_c = sns.blend_palette([c1,c2],n_colors=3).as_hex()[1]
        colors.append(new_c)
    return colors

cs = create_colors_for_linkage(l,l.shape[0]+1,OOi_c_func)

# calculate labels
n = n_flat_clusters
T = np.unique(fcls)
#labels=list('' for i in range(10*n))
labels=list('' for i in range(len(l)+1))
for i in range(n):
    labels[i]=str(i)+ ',' + str(T[i])

with plt.rc_context({'lines.linewidth': 2, 'font.size':10}):
    dend = cluster.hierarchy.dendrogram(l, p=n, no_labels=False, leaf_font_size=10, color_threshold=t, 
                                        distance_sort='ascending', link_color_func=lambda k: cs[k-l.shape[0]-1], 
                                        truncate_mode='lastp', labels = labels, show_leaf_counts=True)

if SAVE_FIGS:
    plt.savefig('clustered_dendrogram_'+st_name+'.pdf', bbox_inches='tight')    
    

# compute averages for each cluster
unit_mean_OOi = np.zeros(np.unique(fcls).shape[0])
#unit_mean_DSi = np.zeros(np.unique(fcls).shape[0])
for c in range(np.unique(fcls).shape[0]):
    inds = np.where(fcls == c+1)[0]
    unit_mean_OOi[c] = np.mean(OOi[conditions_all][inds])    
#    unit_mean_DSi[c] = np.mean(DSi[conditions_all][inds][:,0])
    
max_num_clustersx = n_flat_clusters
#show_order = np.argsort(unit_mean_OOi)
show_order = np.unique(fcls)[::-1]-1
max_num_clusters = np.unique(fcls).shape[0]

palette = sns.hls_palette(max_num_clusters,l=0.6,s=0.6)    
    
from scipy.interpolate import make_interp_spline
import matplotlib.gridspec as gridspec

plt.figure(figsize=(14,max_num_clustersx*1))
# plt.figure(figsize=(9,max_num_clustersx*1.7/4.8))
plot_stims = [0,1,10]
fs = 12#24
labels=list('' for i in range(n))
for i in range(n):
    labels[i] = '#' + str(T[i]) + '\n(' + str(np.count_nonzero(np.where(fcls==T[i]))) + ')'
    
# plot_widths = [1.2,0.8,3,0.4,0.4, 0.4, 0.4]
plot_widths = [3.2,.8,5,1.8,0.5,2, .5]

gs = gridspec.GridSpec(max_num_clustersx, 7, width_ratios = plot_widths, wspace=0.1, hspace=0.1)

bins = 20 # ms
cids = np.where(conditions_all)[0]
has_sta = np.zeros_like(conditions_all).astype(dtype=bool)
#has_sta[STAs['units'].value] = True
sta_inds = np.zeros(np.sum(conditions_all), dtype(int))
#sta_inds[has_sta[conditions_all]] = np.where(np.isin(STAs['units'].value, np.where(conditions_all&has_sta)[0]))[0]
has_sta = has_sta[conditions_all]
#mean_rf_size = np.median((np.abs(STAs['fits'][has_sta,3]),np.abs(STAs['fits'][has_sta,3])))
mean_rf_size=np.zeros(np.sum(conditions_all), dtype(int))
ylims = [(0,40),(0,10)]
Mean_sta_visual=[]    
for i, c in enumerate(show_order):

    n_units = np.where(fcls == c+1)[0].shape[0]
    for ci, stimid in enumerate(plot_stims):
        sts = Stimuli['SpikeTrains'][stimid][conditions_all][np.where(fcls == c+1)]
        if stimid == 0:
            txt = "Cluster %d (%d units)" % (c+1,n_units)
            t_ooi  = OOi[conditions_all][np.where(fcls == c+1)]
 #           txt = "Avg. OOi: %.2f±%.2f   " % (np.average(t_ooi),np.std(t_ooi))
        elif stimid == 1:
            t_sils = silhouettes[np.where(fcls == c+1)]
            t_ooi  = OOi[conditions_all][np.where(fcls == c+1)]
            txt = "Avg. OOi: %.2f±%.2f    Avg. Silhouette Coeff.: %.2f±%.2f" % (np.average(t_ooi),np.std(t_ooi),np.average(t_sils),np.std(t_sils))
            txt = ""
        elif stimid == 5:
            t_dsi = DSi[conditions_all][np.where(fcls == c+1)][:,0]
            txt = "Avg. DSi: %.2f±%.2f" % (np.median(t_dsi),np.std(t_dsi))
        else:
            txt = ""
        if (ci == 0) & (i==0):
            txt = 'B'
        elif (ci == 1) & (i==0):
            txt = 'C'
        elif (ci == 2) & (i==0):
            txt = 'D'
        else:
            txt = None
        with plt.rc_context({'font.size':fs, 'axes.titleweight': 'bold'}):
            ax = plt.subplot(gs[i,ci+1])
            plotPSTHs(ax,sts,txt,bins,palette[c], show_sd=False, lw=1)
            plt.xticks(())
            plt.yticks(())
            if stimid == 0:
                xy,ys = spk.psth(sts.flatten(), 20).get_plottable_data()
                ys = ys / sts.shape[0]
                plt.text(2200, .7*max(ys), " %.2f   " % (np.mean(t_ooi)), fontsize=8)

            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)

        
    with plt.rc_context({'font.size':fs, 'axes.titleweight': 'bold'}):
        ax = plt.subplot(gs[i,4])
        t_dsi = DSi[conditions_all][np.where(fcls == c+1)][:,0]
        sns.distplot(t_dsi,bins=np.arange(0,0.8,0.1),kde=True, norm_hist=True,
                    )
        
        t_dsi_all = DSi[conditions_all][:,0]
        sns.distplot(t_dsi_all,bins=np.arange(0,0.8,0.1),kde=True, norm_hist=True,
                     kde_kws={"color": "k"},
                     hist_kws={"color": "k","alpha": .2})
        
        
        if i == 0:
            txt = 'E'
        else:
            txt = ''
        med_t_dsi_all = np.median(DSi[conditions_all][:,0])
        med_t_dsi= np.median(DSi[conditions_all][np.where(fcls == c+1)][:,0])

#        plt.plot((med_t_dsi_all, med_t_dsi_all),ax.get_ylim(),'k')
#        plt.plot((med_t_dsi, med_t_dsi),ax.get_ylim(),'r')

        
        
        ax.set_title(txt)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_xlim((0,1.0))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if i==len(show_order)-1:
            plt.xlabel('DSI', fontsize=8)
            
            
    with plt.rc_context({'font.size':fs, 'axes.titleweight': 'bold'}):
        ax = plt.subplot(gs[i,5])
#        t_dsi = DSi[conditions_all][np.where(fcls == c+1)][:,0]
#        sns.distplot(t_dsi,bins=np.arange(0,0.8,0.1),kde=True, norm_hist=True)
        if i == 0:
            txt = 'F'
        else:
            txt = ''
#        t_dsi = np.median(DSi[conditions_all][:,0])
        trl_sta=[] 
        trl_sta_raw=[]
        stas=esta_all[conditions_all][np.where(fcls == c+1)]
        for ii in range(len(stas)):
            if  not(stas[ii]==None):
                if len(stas[ii][2])>1:
                    stime=stas[ii][2]
                    ssta=stas[ii][3]
    
                    xnew = np.linspace(stime.min(), stime.max(), 200)  
                    
                    sta_smooth = make_interp_spline(stime.flatten(), ssta.flatten())(xnew)
                    sta_smooth=(sta_smooth-mean(sta_smooth))/std(sta_smooth,ddof=0)
                    trl_sta.append(sta_smooth)
                    
                    sta_up_smple = make_interp_spline(stime.flatten(), ssta.flatten())( xnew)
                    trl_sta_raw.append(sta_up_smple)

                    
                    plt.plot(xnew,sta_smooth,linewidth=.8)
                    plt.axhline(y=0, color='k', linestyle='--',linewidth=.8)
#                    ax.set_ylim((stime.min(),stime.max()))
                    plt.axvline(x=.020, color='k', linestyle='--',linewidth=.8)
                
#                plt.plot(stas[i][2],(stas[i][3]-mean(stas[i][3]))/max(abs(stas[i][3])))
        if trl_sta:
            Mean_sta_visual.append(mean(trl_sta_raw,axis=0))
            plt.plot(xnew,mean(trl_sta,axis=0),linewidth=1.5,color='k')
        ax.set_title(txt)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_xlim((-1,0.5))

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        alpha =.5
        if i==len(show_order)-1:
            plt.xlabel('esta', fontsize=8)    


        with plt.rc_context({'font.size':fs, 'axes.titleweight': 'bold'}):
            ax = plt.subplot(gs[i,6])
    #        t_dsi = DSi[conditions_all][np.where(fcls == c+1)][:,0]
    #        sns.distplot(t_dsi,bins=np.arange(0,0.8,0.1),kde=True, norm_hist=True)
            if i == 0:
                txt = 'E'
            else:
                txt = ''
    #        t_dsi = np.median(DSi[conditions_all][:,0])
            clnames=Cell_names[conditions_all][np.where(fcls == c+1)]
            recording_names=[]
            for idss, rcrdings in enumerate(clnames):
                recording_names.append(clnames[idss][:-4])
                
            unqstrgns=np.unique(recording_names)    
            rcrding_numbers=[]
            for unqnbr in unqstrgns:
                rcrding_numbers.append(recording_names.count(unqnbr))
                
    
            # Plot
            theme = plt.get_cmap('jet') 
    #        ax.set_prop_cycle("color", [theme(1. * i / len(sizes))
    #                             for i in range(len(sizes))])
            plt.pie(rcrding_numbers, shadow=True, startangle=90)

        
ax = plt.subplot(gs[:,0])
p = ax.get_position()
p.x1 = p.x1-0.02
ax.set_position(p)
with plt.rc_context({'lines.linewidth': 2, 'font.size':fs, 'axes.titleweight': 'bold'}):
    dend = cluster.hierarchy.dendrogram(l, p=n, no_labels=False, leaf_font_size=11, color_threshold=t, 
                                        distance_sort='none', link_color_func=lambda k: cs[k-l.shape[0]-1], 
                                        truncate_mode='lastp', show_leaf_counts=True, orientation='left')

    ax.set_title("A")
    ax.set_yticklabels(labels)
    ax.set_xticks(())
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

if SAVE_FIGS:
    plt.savefig('clusters_summary__wt_bi'+st_name+'.pdf', bbox_inches='tight')
    
    plt.savefig('clusters_summary__wt_bi'+st_name+'.png', bbox_inches='tight')
    plt.savefig('clusters_summary__wt_bi'+st_name+'.svg', bbox_inches='tight')
#%% find the width and latancy of each sta (average of cluster)
from tabulate import tabulate
from spk_trains_maker import find_peaks_latency_STA

D1_D2=find_peaks_latency_STA(Mean_sta_visual,'fig3.5')
#STA_width_latency_func(Mean_sta_visual)

print(tabulate(np.round(D1_D2,2), headers=["cl#","w1-peak", "w2-trough", "l1-peak","l2-trough"]))
with open('table_STA_latency_width_fig3.5.txt', 'w') as f:
    f.write(tabulate(np.round(D1_D2,2)))
#%%   
    
with open("clusetr_numbers_wildtype.txt", 'w') as output:
    for row in Cells_names_and_clusters:
        output.write(str(row) + '\n')      
#%
#t_best_ff
#t_best_chirp 
#t_best_ff_chirp_color 
#stim=0
        
psths = [] 
for sts in Stimuli['SpikeTrains'][11][conditions_all]:
    xs,ys = getPSTHs((sts,),bs=50)
    psths.append(ys[0])
psths = np.array(psths)    
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.preprocessing import robust_scale
from matplotlib.colors import ListedColormap
plt.figure(figsize=(24,15))
psths_norm = psths/np.max(psths, axis=1).reshape(-1,1)
plt.imshow(psths_norm)

#t_ = t_best_ff
#new_distances_ = SPIKE_dist_ys_valid[stim]
#l_ = cluster.hierarchy.linkage(new_distances_, method='ward')
#fcls_ = cluster.hierarchy.fcluster(l_, t=t_, criterion='maxclust')
n_flat_clusters_ = np.unique(fcls).shape[0]
show_order_ = np.unique(fcls)[::-1]-1

model = TSNE(n_components=2, random_state=0, perplexity=30,n_iter=5000)#,init='pca')
proj = model.fit_transform(psths) 

#import umap
#model = umap.UMAP() 
#proj = model.fit_transform(psths) 

with plt.rc_context(rcParams):
    plt.figure(figsize=(14,5))
    ax = plt.subplot(121)
    # ax.set_facecolor((0.3,0.3,0.3))
    s = plt.scatter(proj[:,0],proj[:,1],s=16,lw=0,c=show_order_[fcls-1],
                    cmap=ListedColormap(sns.hls_palette(n_flat_clusters_,l=0.6,s=0.6).as_hex()))
    cb = plt.colorbar(s)
    cb.set_label('Cluster')
    # plt.axis('equal')
    plt.grid(False)
    ax = plt.subplot(122)
    # ax.set_facecolor((0.3,0.3,0.3))
    p = plt.scatter(proj[:,0],proj[:,1],s=16,lw=0,c=np.array(((OOi[conditions_all]))),
                    cmap=sns.diverging_palette(180,359,sep=1,n=32,as_cmap=True))
    cb = plt.colorbar(p)
    cb.set_label('Bias index')
    # plt.axis('equal')
    plt.grid(False)

if SAVE_FIGS:
    plt.savefig('tsne_SPIKE_6_wildtpe'+st_name+'.pdf', bbox_inches='tight')
    



#%% ESTA
    
stas_3=np.asarray([row[3][:25].flatten() for row in esta_all[conditions_all]])
zssta=[(row-mean(row))/std(row,ddof=0).flatten() for row in stas_3]
import sklearn.metrics

zssta_dstance=sklearn.metrics.pairwise_distances(zssta, Y=None, metric='euclidean')
zssta_dstance_flat=zssta_dstance[np.triu_indices(len(zssta), k = 1)]
zssta_dstance_flat=zssta_dstance_flat/zssta_dstance_flat.max()

plt.imshow(squareform(zssta_dstance_flat), interpolation='nearest', cmap=plt.cm.gnuplot2,
               vmin=0)
Nc, Wk, Nc_shuff, Wk_shuff, Dk, Dk_shuff, ts = eval_gap_scores(zssta_dstance_flat,tss) # threshold values to test

gapss = np.log(Wk_shuff)-np.log(Wk)
NCs_gaps = Nc

plt.figure()
p = plt.plot(NCs_gaps,gapss)

plt.vlines(NCs_gaps[np.argmax(gapss)],0,gapss[np.argmax(gapss)],linestyles='--',colors=p[0].get_c())
plt.xlim((0,40))
plt.ylim((0,1.1))
plt.legend(frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel('Number of clusters')
plt.ylabel('Gap statistic');
if SAVE_FIGS:
    plt.savefig('sta_Gapstatistic_healthy_shortsta.pdf', bbox_inches='tight')

cluster_number=NCs_gaps[np.argmax(gapss)]



sorted_normed_cnf, res_order, res_linkage = compute_serial_matrix(zssta_dstance_flat,method='ward')
with plt.rc_context(rcParams):
    plt.figure(figsize=(10,10))
    plt.imshow(sorted_normed_cnf, cmap=plt.cm.CMRmap, origin='lower')
    cb = plt.colorbar()
    cb.set_label('euclidean distance')
    plt.axis('equal')




#%%
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 10:45:00 2020

@author: Admin
"""
from scipy.spatial.distance import pdist

l = cluster.hierarchy.linkage(zssta, "ward")
t=cluster_number
fcls_sta = cluster.hierarchy.fcluster(l, t=t, criterion='maxclust')
n = 41
pal = sns.diverging_palette(180,359,sep=1,n=n)
OOi_cspace = np.linspace(-1,1,n)
OOi_c_func = lambda i: pal[np.searchsorted(OOi_cspace,OOi[conditions_all][i])]
DSi_cspace = np.linspace(0,1,n)
DSi_c_func = lambda i: pal[np.searchsorted(DSi_cspace,DSi[conditions_all][i,0])]

cs = create_colors_for_linkage(l,l.shape[0]+1,OOi_c_func)

c, coph_dists = cluster.hierarchy.cophenet(l, pdist(zssta))
c
silhouettes = metrics.silhouette_samples(distance.squareform(pdist(zssta)),fcls_sta,metric='precomputed')
print("Mean Silhouette Coefficient: %.2f" % np.average(silhouettes))

n_flat_clusters = np.unique(fcls_sta).shape[0]
n = n_flat_clusters
T = np.unique(fcls_sta)
labels=list('' for i in range(20*n))
for i in range(n):
    labels[i]=str(i)+ ',' + str(T[i])
show_order = np.unique(fcls_sta)[::-1]-1
max_num_clusters = np.unique(fcls_sta).shape[0]

palette = sns.hls_palette(max_num_clusters,l=0.6,s=0.6)    

max_num_clustersx=t

# plt.figure(figsize=(9,max_num_clustersx*1.7/4.8))
plt.figure(figsize=(14,max_num_clustersx*1))
# plt.figure(figsize=(9,max_num_clustersx*1.7/4.8))

plot_stims = [0,1,10]

fs = 12#24

labels=list('' for i in range(n))
for i in range(n):
    labels[i] = '#' + str(T[i]) + '\n(' + str(np.count_nonzero(np.where(fcls_sta==T[i]))) + ')'
    
# plot_widths = [1.2,0.8,3,0.4,0.4, 0.4, 0.4]
plot_widths = [2,2,2,8,3,.7,2, 0.8]

gs = gridspec.GridSpec(max_num_clustersx, 8, width_ratios = plot_widths, wspace=0.1, hspace=0.1)

bins = 20 # ms

cids = np.where(conditions_all)[0]
has_sta = np.zeros_like(conditions_all).astype(dtype=bool)
#has_sta[STAs['units'].value] = True
sta_inds = np.zeros(np.sum(conditions_all), dtype(int))
#sta_inds[has_sta[conditions_all]] = np.where(np.isin(STAs['units'].value, np.where(conditions_all&has_sta)[0]))[0]
has_sta = has_sta[conditions_all]
#mean_rf_size = np.median((np.abs(STAs['fits'][has_sta,3]),np.abs(STAs['fits'][has_sta,3])))
mean_rf_size=np.zeros(np.sum(conditions_all), dtype(int))
ylims = [(0,40),(0,10)]
peak_frq_table=[]
Mean_sta_raw=[]
Mean_sta_upsmpled=[]    
for i, c in enumerate(show_order):

    n_units = np.where(fcls_sta == c+1)[0].shape[0]
    for ci, stimid in enumerate(plot_stims):
        sts = Stimuli['SpikeTrains'][stimid][conditions_all][np.where(fcls_sta == c+1)]
        if stimid == 0:
            txt = "Cluster %d (%d units)" % (c+1,n_units)
            t_ooi  = OOi[conditions_all][np.where(fcls_sta == c+1)]
        elif stimid == 1:
            t_sils = silhouettes[np.where(fcls_sta == c+1)]
            
            txt = "Avg. OOi: %.2f±%.2f    Avg. Silhouette Coeff.: %.2f±%.2f" % (np.average(t_ooi),np.std(t_ooi),np.average(t_sils),np.std(t_sils))
        elif stimid == 5:
            t_dsi = DSi[conditions_all][np.where(fcls_sta == c+1)][:,0]
            txt = "Avg. DSi: %.2f±%.2f" % (np.average(t_dsi),np.std(t_dsi))
        else:
            txt = "E"
        if (ci == 0) & (i==0):
            txt = 'C'
        elif (ci == 1) & (i==0):
            txt = 'D'
        elif (ci == 2) & (i==0):
            txt = 'E'
        else:
            txt = None
        with plt.rc_context({'font.size':fs, 'axes.titleweight': 'bold'}):
            ax = plt.subplot(gs[i,ci+2])
            plotPSTHs(ax,sts,txt,bins,palette[c], show_sd=False, lw=1)

            plt.xticks(())
            plt.yticks(())
            if stimid == 0:            
                xy,ys = spk.psth(sts.flatten(), 20).get_plottable_data()
                ys = ys / sts.shape[0]
                plt.text(2500, .7*max(ys), " %.2f   " % (np.mean(t_ooi)), fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if (i==len(show_order)-1):
                plt.xlabel('time (s)', fontsize=10)
                if stimid == 0:
                    xticks([0,2000,4000], ['0', '2', '4'])
                elif stimid == 10:
                    xticks([0,3000,6000,9000,12000], ['0', '3', '6', '9', '12'])
                elif stimid == 1:
                    xticks([0,2000,5000,8000,10000,28000,32000], ['0', '2', '5', '8','10','28','32'])
                for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(10)
        
    with plt.rc_context({'font.size':fs, 'axes.titleweight': 'bold'}):
        ax = plt.subplot(gs[i,5])
        t_dsi = DSi[conditions_all][np.where(fcls_sta == c+1)][:,0]
        sns.distplot(t_dsi,bins=np.arange(0,0.8,0.1),kde=True, norm_hist=True,
                    )
        
        t_dsi_all = DSi[conditions_all][:,0]
        sns.distplot(t_dsi_all,bins=np.arange(0,0.8,0.1),kde=True, norm_hist=True,
                     kde_kws={"color": "k"},
                     hist_kws={"color": "k","alpha": .2})
        if i == 0:
            txt = 'F'
        else:
            txt = ''
#        t_dsi = np.median(DSi[conditions_all][:,0])
#        plt.plot((t_dsi, t_dsi),ax.get_ylim(),'r')
        ax.set_title(txt)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_xlim((0,1.0))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if i==len(show_order)-1:
            plt.xlabel('DSI', fontsize=8)
            
            
    with plt.rc_context({'font.size':fs, 'axes.titleweight': 'bold'}):
        ax = plt.subplot(gs[i,1])
#        t_dsi = DSi[conditions_all][np.where(fcls == c+1)][:,0]
#        sns.distplot(t_dsi,bins=np.arange(0,0.8,0.1),kde=True, norm_hist=True)
        if i == 0:
            txt = 'B'
        else:
            txt = ''
#        t_dsi = np.median(DSi[conditions_all][:,0])
        trl_sta=[]
        trl_sta_raw=[]
        stas=esta_all[conditions_all][np.where(fcls_sta == c+1)]
        for ii in range(len(stas)):
            if  not(stas[ii]==None):
                if len(stas[ii][2])>1:
                    stime=stas[ii][2]
                    ssta=stas[ii][3]
    
                    xnew = np.linspace(stime.min(), stime.max(), 200)  
                    sta_smooth = make_interp_spline(stime.flatten(), ssta.flatten())( xnew)
                    sta_smooth=(sta_smooth-mean(sta_smooth))/std(sta_smooth,ddof=0)
                    trl_sta.append(sta_smooth)
                    trl_sta_raw.append(ssta.flatten())
                    
                    plt.plot(xnew,sta_smooth,linewidth=.8,alpha=.7)
                    plt.axhline(y=0, color='k', linestyle='--',linewidth=.8)
#                    ax.set_ylim((stime.min(),stime.max()))
                    plt.axvline(x=.020, color='k', linestyle='--',linewidth=.8)
                
#                plt.plot(stas[i][2],(stas[i][3]-mean(stas[i][3]))/max(abs(stas[i][3])))
        if trl_sta:
            Mean_sta_raw.append(mean(trl_sta_raw,axis=0))
            Mean_sta_upsmpled.append(mean(trl_sta,axis=0))

            plt.plot(xnew,mean(trl_sta,axis=0),linewidth=1.5,color='k')
        ax.set_title(txt)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_xlim((-1,0.5))

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        alpha =.5
        if i==len(show_order)-1:
            plt.xlabel('Time (s)', fontsize=10) 
            ax.set_xticks(([-1,0,.5]))
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(10)   
           
            
            
    with plt.rc_context({'font.size':fs, 'axes.titleweight': 'bold'}):
        ax = plt.subplot(gs[i,6])
#        t_dsi = DSi[with_sta_cells][np.where(fcls == c+1)][:,0]
#        sns.distplot(t_dsi,bins=np.arange(0,0.8,0.1),kde=True, norm_hist=True)
        if i == 0:
            txt = 'G'
        else:
            txt = ''
#        t_dsi = np.median(DSi[with_sta_cells][:,0])
        trl_psd=[]    
        psds=psd_all[conditions_all][np.where(fcls_sta == c+1)]
        freq_dtls=freq_details[conditions_all][np.where(fcls_sta == c+1)]
        peak_frq=np.mean(freq_dtls)
        bwidth=peak_frq[2]-peak_frq[1]
        peak_frq_table.append(np.hstack((peak_frq,bwidth)))
        for ii in range(len(psds)):
            if  any(psds[ii]):
                if len(psds[ii])>1:
                    sfreq=freqsl
                    spsd=psds[ii]
    
                    xnew = np.linspace(sfreq.min(), sfreq.max(), 20)  
                    
                    psd_smooth = spsd/np.max(spsd)#spline(sfreq.flatten(), spsd.flatten(), xnew)
                    #psd_smooth=(psd_smooth-mean(psd_smooth))/std(psd_smooth,ddof=0)
                    trl_psd.append(psd_smooth)
                    
                    plt.plot(sfreq,psd_smooth,linewidth=.8,alpha=.7)
                    plt.axhline(y=0, color='k', linestyle='--',linewidth=.8)
                    plt.text(7, .7*max(psd_smooth), " %.2f   " % (peak_frq[0]), fontsize=8)

#                    ax.set_ylim((stime.min(),stime.max()))
#                    plt.axvline(x=.020, color='k', linestyle='--',linewidth=.8)
                
#                plt.plot(stas[i][2],(stas[i][3]-mean(stas[i][3]))/max(abs(stas[i][3])))
        if trl_sta:
            plt.plot(sfreq,mean(trl_psd,axis=0),linewidth=1.5,color='k')
        ax.set_title(txt)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_xlim((0,12))

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        alpha =.5



        if i==len(show_order)-1:
            plt.xlabel('frequency (hz)', fontsize=10) 
            ax.set_xticks(([0,6,12]))
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(10)             
            


        with plt.rc_context({'font.size':fs, 'axes.titleweight': 'bold'}):
            ax = plt.subplot(gs[i,7])
            
    #        t_dsi = DSi[conditions_all][np.where(fcls == c+1)][:,0]
    #        sns.distplot(t_dsi,bins=np.arange(0,0.8,0.1),kde=True, norm_hist=True)
            if i == 6:
                txt = 'G'
            else:
                txt = 'G'
    #        t_dsi = np.median(DSi[conditions_all][:,0])
            clnames=Cell_names[conditions_all][np.where(fcls_sta == c+1)]
            recording_names=[]
            for idss, rcrdings in enumerate(clnames):
                recording_names.append(clnames[idss][:-4])
                
            unqstrgns=np.unique(recording_names)    
            rcrding_numbers=[]
            for unqnbr in unqstrgns:
                rcrding_numbers.append(recording_names.count(unqnbr))
                
    
            # Plot
            theme = plt.get_cmap('jet') 
    #        ax.set_prop_cycle("color", [theme(1. * i / len(sizes))
    #                             for i in range(len(sizes))])
            plt.pie(rcrding_numbers, shadow=True, startangle=90)





        
ax = plt.subplot(gs[:,0])
p = ax.get_position()
p.x1 = p.x1-0.02
ax.set_position(p)
with plt.rc_context({'lines.linewidth': 3, 'font.size':fs, 'axes.titleweight': 'bold'}):
    dend = cluster.hierarchy.dendrogram(l, p=n, no_labels=False, leaf_font_size=11, color_threshold=t, 
                                        distance_sort='none', link_color_func=lambda k: cs[k-l.shape[0]-1], 
                                        truncate_mode='lastp', show_leaf_counts=True, orientation='left')

    ax.set_title("A")
    ax.set_yticklabels(labels)
    ax.set_xticks(())
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

if SAVE_FIGS:
   plt.savefig('clusters_e_STA_wt_shortsta_psdfull.pdf', bbox_inches='tight')
   plt.savefig('clusters_e_STA_wt_shortsta_psdfull.png', bbox_inches='tight')
   plt.savefig('clusters_e_STA_wt_shortsta_psdfull.svg', bbox_inches='tight')

    
    
    
    
Cells_names_and_clusters_sta=list(zip(Cell_names[conditions_all],fcls_sta))
    
with open("clusetr_numbers_sta.txt", 'w') as output:
    for row in Cells_names_and_clusters_sta:
        output.write(str(row) + '\n')      
#%%
D1_D2=find_peaks_latency_STA(Mean_sta_raw,'fig3_raw.8')


#STA_width_latency_func(Mean_sta_visual)

print(tabulate(np.round(D1_D2,2), headers=["cl#","w1-peak", "w2-trough", "l1-peak","l2-trough"]))
with open('table_STA_latency_width_fig3.8_raw.txt', 'w') as f:
    f.write(tabulate(np.round(D1_D2,2)))


#%% make table of requancies
from tabulate import tabulate

np.median(peak_frq_table,axis=0)

print(tabulate(np.round(peak_frq_table,2), headers=["peak freq", "f1 50%", "f2 50%","bw"]))
with open('table_wt_freqs.txt', 'w') as f:
    f.write(tabulate(np.round(peak_frq_table,2)))
        #%% PCA

# Z-score the features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(stas_3)
zstas_3 = scaler.transform(stas_3)

from sklearn.decomposition import PCA
pca = PCA(n_components=15)
X_r = pca.fit(zssta).transform(zssta)
plt.rcParams.update({'font.size': 18})
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels


show_order_ = np.unique(fcls_sta)[::-1]-1

plt.figure()

plt.figure(figsize=(14,10))

s = plt.scatter(X_r[:,0],X_r[:,1],s=16,lw=2,c=show_order_[fcls_sta-1],
                cmap=ListedColormap(sns.hls_palette(n_flat_clusters_,l=0.6,s=0.6).as_hex()))
cb = plt.colorbar(s)
cb.set_label('Cluster')
plt.title('PCA e-STAs') 
plt.xlabel('PC1')
plt.ylabel('PC2')
print(sum(pca.explained_variance_ratio_))   

#plt.ylim([-10,10])
#if SAVE_FIGS:
#    plt.savefig('PCA_projection_STA'+st_name+'.pdf', bbox_inches='tight')


plt.figure(figsize=(14,5), dpi=100)
axes=plt.subplot(121)
axes.scatter(X_r[:,0], X_r[:,1],s=16,lw=0, c=show_order_[fcls_sta-1],
             cmap=ListedColormap(sns.hls_palette(n_flat_clusters_,l=0.6,s=0.6).as_hex()))
axes.set_xlabel('PC1')
axes.set_ylabel('PC2')
#axes.set_title('PC1 vs PC2')
axes.axvline(c='grey', lw=1)
axes.axhline(c='grey', lw=1)

#plt.ylim([-10,10])


axes=plt.subplot(122)
axes.scatter(X_r[:,0], X_r[:,2],s=16,lw=0, c=show_order_[fcls_sta-1],
             cmap=ListedColormap(sns.hls_palette(n_flat_clusters_,l=0.6,s=0.6).as_hex()))
axes.set_xlabel('PC1')
axes.set_ylabel('PC3')
#axes.set_title('PC1 vs PC3')
axes.axvline(c='grey', lw=1)
axes.axhline(c='grey', lw=1)
#plt.rcParams.update({'font.size': 18})
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels


#plt.ylim([-10,10])
if SAVE_FIGS:
    plt.savefig('STA_Nornalized_PCA_projection'+st_name+'.pdf', bbox_inches='tight')
    plt.savefig('STA_Nornalized_PCA_projection'+st_name+'.svg', bbox_inches='tight')

#%%
#fnames=Cell_names
#Fnames=np.array([len(fnames)])
#Fnames=[]
#for i,names in enumerate(fnames):
#    Fnames.append(fnames[i][:-4])
#Fnames=np.array(Fnames)    
#unqnames = np.unique(Fnames[conditions_all][np.where(fcls_sta == c+1)])

        #%% t-sne for pca

   
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.preprocessing import robust_scale
from matplotlib.colors import ListedColormap

#t_ = cluster_number
#l_ = cluster.hierarchy.linkage(zssta, method='ward')
#fcls_ = cluster.hierarchy.fcluster(l_, t=t_, criterion='maxclust')
n_flat_clusters_ = np.unique(fcls_sta).shape[0]
show_order_ = np.unique(fcls_sta)[::-1]-1

model = TSNE(n_components=2, random_state=0, perplexity=20,n_iter=5000)#,init='pca')
proj = model.fit_transform(zssta) 

#import umap
#model = umap.UMAP() 
#proj = model.fit_transform(psths) 

with plt.rc_context(rcParams):
    plt.figure(figsize=(14,5))

    ax = plt.subplot(121)
    # ax.set_facecolor((0.3,0.3,0.3))
    s = plt.scatter(proj[:,0],proj[:,1],s=16,lw=0,c=show_order_[fcls_sta-1],
                    cmap=ListedColormap(sns.hls_palette(n_flat_clusters_,l=0.6,s=0.6).as_hex()))
#    plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
#    plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
    plt.rcParams.update({'font.size': 18})



    cb = plt.colorbar(s)
    cb.set_label('Cluster')
    plt.title('t-SNE embeddings')

    # plt.axis('equal')
    plt.grid(False)
    ax = plt.subplot(122)
    # ax.set_facecolor((0.3,0.3,0.3))
    p = plt.scatter(proj[:,0],proj[:,1],s=16,lw=0,c=np.array(((OOi[conditions_all]))),
                    cmap=sns.diverging_palette(180,359,sep=1,n=32,as_cmap=True))
    cb = plt.colorbar(p)
    cb.set_label('Bias index')
    # plt.axis('equal')

    plt.grid(False)
    plt.title('t-SNE embeddings')
    plt.rcParams.update({'font.size': 18})
#    plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
#    plt.rc('ytick', labelsize=14)    # fontsize of the tick labels


if SAVE_FIGS:
        plt.savefig('STA_tsne_normalized_esta'+st_name+'.pdf', bbox_inches='tight')
        plt.savefig('STA_tsne_normalized_esta'+st_name+'.svg', bbox_inches='tight')

            #%%
#
#   
#from sklearn.manifold import TSNE
#from sklearn.preprocessing import normalize
#from sklearn.preprocessing import robust_scale
#from matplotlib.colors import ListedColormap
#
##t_ = cluster_number
##l_ = cluster.hierarchy.linkage(zssta, method='ward')
##fcls_ = cluster.hierarchy.fcluster(l_, t=t_, criterion='maxclust')
#n_flat_clusters_ = np.unique(fcls_sta).shape[0]
#show_order_ = np.unique(fcls_sta)[::-1]-1
#
#model = TSNE(n_components=2, random_state=0, perplexity=20,n_iter=5000)#,init='pca')
#proj = model.fit_transform(zssta) 
#
##import umap
##model = umap.UMAP() 
##proj = model.fit_transform(psths) 
#
#with plt.rc_context(rcParams):
#    plt.figure(figsize=(14,5))
#
#    ax = plt.subplot(121)
#    # ax.set_facecolor((0.3,0.3,0.3))
#    s = plt.scatter(proj[:,0],proj[:,1],s=16,lw=0,c=show_order_[fcls_sta-1],
#                    cmap=ListedColormap(sns.hls_palette(n_flat_clusters_,l=0.6,s=0.6).as_hex()))
##    plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
##    plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
#    plt.rcParams.update({'font.size': 18})
#
#
#
#    cb = plt.colorbar(s)
#    cb.set_label('Cluster')
#    plt.title('t-SNE embeddings')
#
#    # plt.axis('equal')
#    plt.grid(False)
#    ax = plt.subplot(122)
#    # ax.set_facecolor((0.3,0.3,0.3))
#    p = plt.scatter(proj[:,0],proj[:,1],s=16,lw=0,c=np.array(((OOi[conditions_all]))),
#                    cmap=sns.diverging_palette(180,359,sep=1,n=32,as_cmap=True))
#    cb = plt.colorbar(p)
#    cb.set_label('Bias index')
#    # plt.axis('equal')
#
#    plt.grid(False)
#    plt.title('t-SNE embeddings')
#    plt.rcParams.update({'font.size': 18})
##    plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
##    plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
#
#
#if SAVE_FIGS:
#    plt.savefig('clusters_tsne_esta.pdf', bbox_inches='tight')
#    
#    
#        
    
        
        #%%
#from iteration_utilities import deepflatten
#
#def Remove_outliers(stimuli):
#    stimuli=Stimuli['SpikeTrains'][1]
#    diffs=np.array([])
#    for i,st in enumerate(stimuli):
#        diff_cell=np.array([])
#        for j,s in enumerate(st):
#            diff_cell=np.append(diff_cell,np.diff(s)/1000)
#        if len(diff_cell)>0:    
#            diffs=np.append(diffs,np.median(diff_cell))
#        else:
#            diffs=np.append(diffs,0)
#    
#        
#    sums=np.array([])
#    for i,st in enumerate(stimuli):
#        mean_cell=np.array([])
#        for j,s in enumerate(st):
#            mean_cell=np.append(mean_cell,len(s))
#        sums=np.append(sums,sum(mean_cell)/len(st))
#           
#        
#    sum_zs=(sums[:]-np.mean(sums))/np.std(sums)
#    diff_zs=(diffs[:]-np.mean(diffs))/np.std(diffs)
#    
#    
#    data_array_std = np.std(diffs)
#    data_array_mean = np.mean(diffs)
#    sigma_stdev = (2*data_array_std)+data_array_mean
#    upper_limit = data_array_mean+data_array_std
#    lower_limit = data_array_mean-data_array_std
#    
#    condition_diff = (diffs>0)& (diffs<upper_limit)&(diffs>0)
#    condition_0ff_diff=~condition_diff
#    
#    data_array_std = np.std(sums)
#    data_array_mean = np.mean(sums)
#    sigma_stdev = (2*data_array_std)+data_array_mean
#    upper_limit = data_array_mean+sigma_stdev
#    lower_limit = data_array_mean-sigma_stdev
#    
#    condition_sums=(sums>0)& (sums<upper_limit)&(sums>10)
#    condition_sums_off=~condition_sums
#    return condition_sums,condition_diff
##valid_units = np.ones(len(diff_zs), dtype=bool)
##valid_units(diff_zs)
##%%
##
#hist_diffs = np.histogram(diffs, bins=333)
#hist_sum = np.histogram(sums, bins=333)
#
#plt.figure(figsize=(15,10))
#plt.plot(hist_diffs[1][:-1],hist_diffs[0])
#plt.plot(hist_sum[1][:-1],hist_sum[0])
#plt.figure(figsize=(15,10))
#
#plt.plot(diffs[condition_diff],sums[condition_diff],'o')
#ratios=diffs/sums
#plt.figure(figsize=(15,10))
#
#plt.plot(ratios)
#hist_ration = np.histogram(ratios, bins=333)
#plt.figure(figsize=(15,10))
#plt.plot(hist_ration[1][:-1],hist_ration[0])
##%%
##celss=np.transpose(np.where((condition_0ff_diff)&(condition_sums_off)))
#valids=np.where(abs(ratios)<1)
#celss=zip(*valids)
#
#PSTH = np.empty((size(valids),320))
#
#plt.figure(figsize=(15,10))
#
#for i,ii in enumerate(celss):
##    plot(spk.psth(Stimuli['SpikeTrains'][1][ii],100).y)
#    PSTH[i,:]=spk.psth(Stimuli['SpikeTrains'][1][ii],100).y
#PSTH_norm = PSTH/np.max(PSTH, axis=1).reshape(-1,1)
#
#plt.figure(figsize=(15,10))
#plt.imshow(PSTH_norm[np.argsort(np.mean(PSTH_norm,axis=1))], aspect='auto')
##plt.imshow(PSTH_norm, aspect='auto')
#
##%%
#valids=np.where((condition_diff)&(condition_sums))
#celss=zip(*valids)
#
#PSTH = np.empty((size(valids),320))
#
#plt.figure(figsize=(15,10))
#
#for i,ii in enumerate(celss):
##    plot(spk.psth(Stimuli['SpikeTrains'][1][ii],100).y)
#    PSTH[i,:]=spk.psth(Stimuli['SpikeTrains'][1][ii],100).y
#PSTH_norm = PSTH/np.max(PSTH, axis=1).reshape(-1,1)
#
#plt.figure(figsize=(15,10))
#plt.imshow(PSTH_norm[np.argsort(np.mean(PSTH_norm,axis=1))], aspect='auto')
#
##%%
#valids=np.where((~condition_diff)|(~condition_sums))
#celss=zip(*valids)
#
#PSTH = np.empty((size(valids),320))
#
#plt.figure(figsize=(15,10))
#
#for i,ii in enumerate(celss):
##    plot(spk.psth(Stimuli['SpikeTrains'][1][ii],100).y)
#    PSTH[i,:]=spk.psth(Stimuli['SpikeTrains'][1][ii],100).y
#PSTH_norm = PSTH/np.max(PSTH, axis=1).reshape(-1,1)
#
#plt.figure(figsize=(15,10))
#plt.imshow(PSTH_norm[np.argsort(np.mean(PSTH_norm,axis=1))], aspect='auto')
# test for push