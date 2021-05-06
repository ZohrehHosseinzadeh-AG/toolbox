# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 15:26:06 2020

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 15:09:44 2020

@author: Admin
"""



# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 14:52:31 2019

@author: Admin
"""
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
# Recording

# Recording
#filenames=['2019_07_21_wl_secondmouce.mat','2018_12_12_left_nasal.mat','2020_01_17_rhalf1.mat','2018_11_28_WT_Left.mat','2018_11_29_WT1_Left.mat','2019_07_17_mrg.mat','2019_07_16_lw.mat','2020_01_25_r1.mat','2020_02_04_l1_before.mat','2020_02_06_r1_before.mat','2020_01_25_left.mat','2020_01_16_wl.mat']
filenames=['2020_02_07_rd10_l1_after.mat','2020_02_07_rd10_r1_after.mat','2020_02_05_rd10_l1_after.mat','2020_02_05_rd10_r1_after.mat']
#sf=data_dir+filenames


# stimulus information
#report_filenames  = [ data_path+"2019_07_17_mrg_report.txt" ]
#trigger_filenames = [ data_path+"trigger_2019_07_17_mrg.mat" ]
#%%
def spk_trains_maker(filenames,report_filenames,trigger_filenames,stims):
    notebooks_path = "C:/Users/Admin/Desktop/hennig_project/download/rgcclassification/sample_dara_hamed/"

    data_dir ="C:/Users/Admin/Desktop/hennig_project/download/rgcclassification/sample_dara_hamed/"

    data_path = "C:/Users/Admin/Desktop/hennig_project/download/rgcclassification/sample_dara_hamed/"



    # Set to False to reload the spike data from the original hdf5 files 
    LOAD_STORED_SPIKES = False
    
    # Set to False to re-compute the distance matrices
    LOAD_STORED_DISTANCES = False
    
    
    # Unit selection
    
    # estimate the spatial spread of spikes for each unit
    # poorly sorted units have a wide spread and/or high eccentricity
    EVAL_THRES = 0.17 # threshold for average spread
    ECC_THRES = 0.85 # threshold for eccentricity
    
    # exclude units with insufficient spike counts (per trial)
    MIN_SPIKES_FF = 1 # min spikes in Full Field stimulus
    MIN_SPIKES_CHIRP =20# min spikes in Chirp stimulus
    
    # set to True to apply an additional fiilter based on the STA
    # note this excludes units without a valid STA
    # we found not all RGCs have a clean STA, so this will exclude valid neurons
    # doing this reduces the number of clusters considerably
    FILTER_STA = False
    STA_MAX_DIST = 0.7
    STA_MAX_ASYM = 0.9
    
    # set to True to save the figures generated below
    SAVE_FIGS = False
    
  
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
    #esta= sio.loadmat('e_STA 2019_07_17_mrg.mat')
    #Data=esta['Data']
    #times=Data['time_sta']
    #e_sta=Data['e_sta']
    E_STA=[]
    esta_all={}
    if LOAD_STORED_SPIKES == False:
        ncl={}
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
    
            print("Number of units: %d" % ncl[i])
    
            E_STA.append(np.asarray(esta[i]))
            cell_names.append(np.asarray(mat_contents['Chname']))
            if '2019_07_21_wl_secondmouce.mat' in f:# this dataset does not have moving bar stimulus manualy set to zero
                dsi_cels=np.hstack((dsi_cels,np.ones(ncl[i], dtype=bool)))
              
    #            dsi_cels.append(np.ones(ncl[i], dtype=bool))
            else: 
                dsi_cels=np.hstack((dsi_cels,np.zeros(ncl[i], dtype=bool)))
    
    #            dsi_cels.append(np.zeros(ncl[i], dtype=bool))  
    esta_all=np.delete(esta_all, 0)
      
            
    #        ClusterLoc[i] = O.ClusterLoc()
    #        ClusterSizes[i] = O.ClusterSizes()
    #flattened = [val[0] for sublist in cell_names for val[0] in sublist]
    
    Cell_names = []
    i=-1
    for sublist in cell_names:
        i=i+1
        
        for val in sublist[0]:
            Cell_names.append(filenames[i][:-4]+"_"+val[0])
    Cell_names=np.asarray(Cell_names)   
    
    

         
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
    stim_durs=np.array([4,12,4,4,4,4,4,4,4,4,12]) *Sampling   
    SpikeTimes = {}
    for ix in ClusterIDs:
      SpikeTimes[ix] = []
      for cl in range(ncl[ix]):
        cl_spikes = np.where(ClusterIDs[ix]==cl+1)[0]
        cl_times  = np.unique(Times[ix][cl_spikes])
        SpikeTimes[ix].append(cl_times)
    
    STss=[] 
#    stims=[0] 
    for stimid in stims:
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
    
    #        elif stimid==10:
    #            stim_start_end=np.array(trg[ix].get('chirp2')).flatten().reshape([n_trials,-1])*Sampling
    #            n_trials=10
    #        elif stimid==2:
    #            stim_start_end=np.array(trg[ix].get('d0')).flatten().reshape([n_trials,-1])*Sampling
    #        elif stimid==3:
    #            stim_start_end=np.array(trg[ix].get('d45')).flatten().reshape([n_trials,-1])*Sampling
    #        elif stimid==4:
    #            stim_start_end=np.array(trg[ix].get('d90')).flatten().reshape([n_trials,-1])*Sampling
    #        elif stimid==5:
    #            stim_start_end=np.array(trg[ix].get('d135')).flatten().reshape([n_trials,-1])*Sampling
    #        elif stimid==6:
    #            stim_start_end=np.array(trg[ix].get('d180')).flatten().reshape([n_trials,-1])*Sampling
    #        elif stimid==7:
    #            stim_start_end=np.array(trg[ix].get('d225')).flatten().reshape([n_trials,-1])*Sampling
    #        elif stimid==8:
    #            stim_start_end=np.array(trg[ix].get('d270')).flatten().reshape([n_trials,-1])*Sampling
    #        elif stimid==9:
    #            stim_start_end=np.array(trg[ix].get('d315')).flatten().reshape([n_trials,-1])*Sampling
            elif stimid==1:
                try:
                    stim_start_end=np.array(trg[ix].get('color')).flatten().reshape([n_trials,-1])*Sampling
                    n_trials=14
                except IOError:
                    stim_start_end=np.ones([14,1])*11910
                    print( "No e-color" , f)


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
    Stimuli_rd_after=Stimuli
#    np.save('Stimuli_rd_after.npy',Stimuli_rd_after)
    return Stimuli, esta_all, Cell_names             
        
#%%
    




# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 19:41:44 2020

@author: Admin
"""
def plot_clustrs(stimuli,fcls,OOi,conditions_all,estas,Cell_names,plot_stims,name):
    SAVE_FIGS=True
    # compute averages for each cluster
    unit_mean_OOi = np.zeros(np.unique(fcls).shape[0])
    #unit_mean_DSi = np.zeros(np.unique(fcls).shape[0])
    for c in range(np.unique(fcls).shape[0]):
        inds = np.where(fcls == c+1)[0]
        unit_mean_OOi[c] = np.mean(OOi[conditions_all][inds])    
    #    unit_mean_DSi[c] = np.mean(DSi[conditions_all][inds][:,0])
        
    max_num_clustersx = n_flat_clusters
    # show_order = np.argsort(unit_mean_OOi)
    show_order = np.unique(fcls)[::-1]-1
    max_num_clusters = np.unique(fcls).shape[0]
    
    palette = sns.hls_palette(max_num_clusters,l=0.6,s=0.6)    
        
    from scipy.interpolate import spline
    
    
    
    import matplotlib.gridspec as gridspec
    
    plt.figure(figsize=(14,max_num_clustersx*1))
    # plt.figure(figsize=(9,max_num_clustersx*1.7/4.8))
    
#    plot_stims = [0]
    
    fs = 12#24
    
    labels=list('' for i in range(n))
    for i in range(n):
        labels[i] = '#' + str(T[i]) + '\n(' + str(np.count_nonzero(np.where(fcls==T[i]))) + ')'
        
    # plot_widths = [1.2,0.8,3,0.4,0.4, 0.4, 0.4]
    plot_widths = [3,4,12,3,1,.02, .02]
    
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
        
    for i, c in enumerate(show_order):
    
        n_units = np.where(fcls == c+1)[0].shape[0]
        for ci, stimid in enumerate(plot_stims):
            sts = stimuli['SpikeTrains'][stimid][conditions_all][np.where(fcls == c+1)]
            if stimid == 0:
                txt = "Cluster %d (%d units)" % (c+1,n_units)
                t_ooi  = OOi[conditions_all][np.where(fcls == c+1)]
            elif stimid == 1:
                t_sils = silhouettes[np.where(fcls == c+1)]
                
                txt = "Avg. OOi: %.2f±%.2f    Avg. Silhouette Coeff.: %.2f±%.2f" % (np.average(t_ooi),np.std(t_ooi),np.average(t_sils),np.std(t_sils))
            elif stimid == 5:
                t_dsi = DSi[conditions_all][np.where(fcls == c+1)][:,0]
                txt = "Avg. DSi: %.2f±%.2f" % (np.average(t_dsi),np.std(t_dsi))
            else:
                txt = ""
            if (ci == 0) & (i==0):
                txt = 'B'
            elif (ci == 1) & (i==0):
                txt = 'C'
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
                    #plt.text(3000, .8*max(ys), " %.2f   " % (np.mean(t_ooi)), fontsize=9)
                ax.spines['top'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['right'].set_visible(False)
                if (i==len(show_order)-1):
                    plt.xlabel('time (s)', fontsize=10)
                    if stimid == 0:
                        xticks([0,2000,4000], ['0', '2', '4'])
                    elif stimid == 1:
                        xticks([0,3000,6000,9000,12000], ['0', '3', '6', '9', '12'])
                    for tick in ax.xaxis.get_major_ticks():
                        tick.label.set_fontsize(10)                
    
    #        
    #    with plt.rc_context({'font.size':fs, 'axes.titleweight': 'bold'}):
    #        ax = plt.subplot(gs[i,4])
    #        t_dsi = DSi[conditions_all][np.where(fcls == c+1)][:,0]
    #        sns.distplot(t_dsi,bins=np.arange(0,0.8,0.1),kde=True, norm_hist=True)
    #        if i == 0:
    #            txt = 'D'
    #        else:
    #            txt = ''
    #        t_dsi = np.median(DSi[conditions_all][:,0])
    #        plt.plot((t_dsi, t_dsi),ax.get_ylim(),'r')
    #        ax.set_title(txt)
    #        ax.set_xticks(())
    #        ax.set_yticks(())
    #        ax.set_xlim((0,1.0))
    #        ax.spines['top'].set_visible(False)
    #        ax.spines['right'].set_visible(False)
    #        if i==len(show_order)-1:
    #            plt.xlabel('DSI', fontsize=8)
                
                
        with plt.rc_context({'font.size':fs, 'axes.titleweight': 'bold'}):
            ax = plt.subplot(gs[i,3])
    #        t_dsi = DSi[conditions_all][np.where(fcls == c+1)][:,0]
    #        sns.distplot(t_dsi,bins=np.arange(0,0.8,0.1),kde=True, norm_hist=True)
            if i == 0:
                txt = 'D'
            else:
                txt = ''
    #        t_dsi = np.median(DSi[conditions_all][:,0])
            trl_sta=[]    
            stas=estas[conditions_all][np.where(fcls == c+1)]
            for ii in range(len(stas)):
                if  not(stas[ii]==None):
                    if len(stas[ii][2])>1:
                        stime=stas[ii][2]
                        ssta=stas[ii][3]
        
                        xnew = np.linspace(stime.min(), stime.max(), 200)  
                        
                        sta_smooth = spline(stime.flatten(), ssta.flatten(), xnew)
                        sta_smooth=(sta_smooth-mean(sta_smooth))/std(sta_smooth,ddof=0)
                        trl_sta.append(sta_smooth)
                        
                        plt.plot(xnew,sta_smooth,linewidth=.8)
                        plt.axhline(y=0, color='k', linestyle='--',linewidth=.8)
    #                    ax.set_ylim((stime.min(),stime.max()))
                        plt.axvline(x=.020, color='k', linestyle='--',linewidth=.8)
                    
    #                plt.plot(stas[i][2],(stas[i][3]-mean(stas[i][3]))/max(abs(stas[i][3])))
            if trl_sta:
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
                plt.xlabel('time (s)', fontsize=10)
                ax.set_xticks(([-1,-.5,0]))
                for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(10)
                
                
                
                
                
    #    with plt.rc_context({'axes.facecolor': 'grey','axes.edgecolor': 'grey','font.size':fs, 'axes.titleweight': 'bold'}):
    #        ax = plt.subplot(gs[i,3])
    #        n_units = np.where(fcls == c+1)[0].shape[0]
    #        ax.scatter(ClusterLoc[0][0,conditions_all][np.where(fcls != c+1)],
    #                   ClusterLoc[0][1,conditions_all][np.where(fcls != c+1)], c='w', s=3)
    #        ax.scatter(ClusterLoc[0][0,conditions_all][np.where(fcls == c+1)],
    #                   ClusterLoc[0][1,conditions_all][np.where(fcls == c+1)], s=3, c=palette[c])
    #        if i == 0:
    #            txt = 'D'
    #        else:
    #            txt = ''
    #        ax.set_title(txt)
    #        ax.set_xlim((-1,65))
    #        ax.set_ylim((-1,65))
    #        ax.set_xticks(())
    #        ax.set_yticks(())
#    #        ax.set_aspect(1)
    
    
        with plt.rc_context({'font.size':fs, 'axes.titleweight': 'bold'}):
            ax = plt.subplot(gs[i,4])
    #        t_dsi = DSi[conditions_all][np.where(fcls == c+1)][:,0]
    #        sns.distplot(t_dsi,bins=np.arange(0,0.8,0.1),kde=True, norm_hist=True)
            if i == 0:
                txt = 'E'
            else:
                txt = ''








        clnames=Cell_names[conditions_all][np.where(fcls == c+1)]
        recording_names=[]
        for idss, rcrdings in enumerate(Cell_names):
            recording_names.append(Cell_names[idss][:-4])
            
        recording_names_cluster=[]
        for idss, rcrdings in enumerate(clnames):
            recording_names_cluster.append(clnames[idss][:-4])
            
        unqstrgns_cluster=np.unique(recording_names_cluster)    
        unqstrgns=np.unique(recording_names)    
        
        rcrding_numbers=[]
        for unqnbr in unqstrgns:
            rcrding_numbers.append(recording_names_cluster.count(unqnbr))
        
        cmap = plt.cm.prism

        plt.pie(rcrding_numbers, shadow=True, startangle=90)















#            clnames=Cell_names[conditions_all][np.where(fcls == c+1)]
#            recording_names=[]
#            for idss, rcrdings in enumerate(clnames):
#                recording_names.append(clnames[idss][:-4])
#                
#            unqstrgns=np.unique(recording_names)    
#            rcrding_numbers=[]
#            for unqnbr in unqstrgns:
#                rcrding_numbers.append(recording_names.count(unqnbr))
#                
#    
#            # Plot
#            theme = plt.get_cmap('jet') 
#
#            plt.pie(rcrding_numbers, shadow=True, startangle=90)
    
    
    
    
    
    
#        with plt.rc_context({'font.size':fs, 'axes.titleweight': 'bold'}):
#            ax = plt.subplot(gs[i,4])
#    #        t_dsi = DSi[conditions_all][np.where(fcls == c+1)][:,0]
#    #        sns.distplot(t_dsi,bins=np.arange(0,0.8,0.1),kde=True, norm_hist=True)
#            if i == 0:
#                txt = 'E'
#            else:
#                txt = ''
#    #        t_dsi = np.median(DSi[conditions_all][:,0])
#            clnames=Cell_names[conditions_all][np.where(fcls == c+1)]
#            recording_names=[]
#            for idss, rcrdings in enumerate(clnames):
#                recording_names.append(clnames[idss][:-4])
#            rcrding_numbers=len(np.unique(recording_names))
#            sizes = np.ones(rcrding_numbers)
#    
#            # Plot
#            plt.pie(sizes, shadow=True, startangle=90) 
#            ax.set_title(txt)
    
    
    
    
    #    with plt.rc_context({'font.size':fs, 'axes.titleweight': 'bold'}):
    #        inds = fcls==c+1
    #        usable_stas = sta_inds[has_sta&inds]
    #        if len(usable_stas)>0:
    #            ax = plt.subplot(gs[i,5])
    #            sx = np.mean((np.abs(STAs['fits'][usable_stas,3]),np.abs(STAs['fits'][usable_stas,4])),0)
    #            sns.distplot(sx[(sx<10)&(sx>0)],bins=np.arange(0,6,0.5),kde=True, norm_hist=True)
    #            plt.plot((mean_rf_size,mean_rf_size),ax.get_ylim(),'r')
    #            if i == 0:
    #                txt = 'F'
    #            else:
    #                txt = ''
    #            ax.set_title(txt)
    #            ax.set_xticks(())
    #            ax.set_yticks(())
    #            ax.set_xlim((1,6))
    #            ax.spines['top'].set_visible(False)
    #            ax.spines['right'].set_visible(False)
    #        if i==len(show_order)-1:
    #            plt.xlabel('RF size', fontsize=8)
    
    #    with plt.rc_context({'font.size':fs, 'axes.titleweight': 'bold'}):
    #        ax = plt.subplot(gs[i,6])
    #        if len(usable_stas)>0:
    ##            tmp_sta = np.array([get_sta_at_peak(STAs['STAs'][i]) for i in usable_stas])
    #            x = np.arange(tmp_sta.shape[1])
    #            plt.fill_between(x, np.mean(tmp_sta,0)-np.std(tmp_sta,0), np.mean(tmp_sta,0)+np.std(tmp_sta,0), color=palette[c])
    #            plt.plot(x,np.mean(tmp_sta,0),'k')
    #        if i == 0:
    #            txt = 'G'
    #        else:
    #            txt = ''
    #        ax.set_title(txt)
    #        ax.set_xticks(())
    #        ax.set_yticks(())
    #        plt.ylim((-0.3,0.3))
    #        ax.spines['top'].set_visible(False)
    #        ax.spines['bottom'].set_visible(False)
    #        ax.spines['left'].set_visible(False)
    #        ax.spines['right'].set_visible(False)
            
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
#        plt.savefig('clusters_summary_chirp_color_SPIKE_rd10_neworder_AFTER_FLASH.pdf', bbox_inches='tight')
        plt.savefig('clusters_summary_flash_color_SPIKE_'+name+'.pdf', bbox_inches='tight')
        plt.savefig('clusters_summary_flash_color_SPIKE_'+name+'.svg', bbox_inches='tight')


    #%%eSTA
def eata_clustsering(Stimuli,fcls,conditions_all,esta_all,psd_all,freqsl,Cell_names,plot_stims,t,name,OOi, plot_psd,SAVE_FIGS):
    import sklearn.metrics





   
    freq_details=[None] * len(Cell_names) 
    
    for i in range(len(psd_all)): 
            if not(esta_all[i]==None):
                if len(esta_all[i][2])>1:
                    norm_psd=psd_all[i]/sum(psd_all[i]);
                    spsd=norm_psd
                    max_arg=[]
                    for ps in range(len(freqsl)):
                        if sum(spsd)>=.5:
                            max_arg.append(np.argmax(spsd))
                            spsd=np.where(spsd==spsd[np.argmax(spsd)], 0, spsd) 
                    roi=freqsl[np.min(max_arg)],freqsl[np.max(max_arg)]
                    freq_details[i]=np.hstack((freqsl[np.argmax(psd_all[i])],    roi))
    
    freq_details=np.asarray(freq_details)# peak frequancy and bandwidth at 50% of peak















    zssta_dstance=sklearn.metrics.pairwise_distances(zssta, Y=None, metric='euclidean')
    zssta_dstance_flat=zssta_dstance[np.triu_indices(len(zssta), k = 1)]
    zssta_dstance_flat=zssta_dstance_flat/zssta_dstance_flat.max()

    # compute averages for each cluster
    unit_mean_OOi = np.zeros(np.unique(fcls).shape[0])
    for c in range(np.unique(fcls).shape[0]):
        inds = np.where(fcls == c+1)[0]
        unit_mean_OOi[c] = np.mean(OOi[conditions_all][inds]) 
 #   show_order = np.argsort(unit_mean_OOi)

    show_order = np.unique(fcls)[::-1]-1

#    plt.figure()
#
#    plt.imshow(squareform(zssta_dstance_flat), interpolation='nearest', cmap=plt.cm.gnuplot2,   vmin=0)
    Nc, Wk, Nc_shuff, Wk_shuff, Dk, Dk_shuff, ts = eval_gap_scores(zssta_dstance_flat,tss) # threshold values to test
    
    gapss = np.log(Wk_shuff)-np.log(Wk)
    NCs_gaps = Nc
    
    plt.figure()
    p = plt.plot(NCs_gaps,gapss)
    
    plt.vlines(NCs_gaps[np.argmax(gapss)],0,gapss[np.argmax(gapss)],linestyles='--',colors=p[0].get_c())
    plt.xlim((0,40))
    plt.ylim((0,1))
    plt.legend(frameon=False)
    plt.xlabel('Number of clusters')
    plt.ylabel('Gap statistic');

    peak_frq_table=[]
 
#    cluster_number=NCs_gaps[np.argmax(gapss)]

    max_num_clustersx=cluster_number
    
    # plt.figure(figsize=(9,max_num_clustersx*1.7/4.8))
    plt.figure(figsize=(14,max_num_clustersx*1))
    # plt.figure(figsize=(9,max_num_clustersx*1.7/4.8))
    
    #plot_stims = [0,1]
    
    fs = 12#24
    
    labels=list('' for i in range(n))
    for i in range(n):
        labels[i] = '#' + str(T[i]) + '\n(' + str(np.count_nonzero(np.where(fcls==T[i]))) + ')'
        
    # plot_widths = [1.2,0.8,3,0.4,0.4, 0.4, 0.4]
    if plot_psd:
        plot_widths = [2.5,2,2.3,7.5,1.7,.7, 0.04]
        name=name+' psd'
    else:
        plot_widths = [2.5,2,2.3,7.5,.7,.04, 0.04]
    
    gs = gridspec.GridSpec(max_num_clustersx, 7, width_ratios = plot_widths, wspace=0.1, hspace=0.1)
    
    bins = 20 # ms
    
    cids = np.where(conditions_all)[0]
    has_sta = np.zeros_like(conditions_all).astype(dtype=bool)
    #has_sta[STAs['units'].value] = True
    sta_inds = np.zeros(np.sum(conditions_all), dtype(int))
    #sta_inds[has_sta[with_sta_cells]] = np.where(np.isin(STAs['units'].value, np.where(with_sta_cells&has_sta)[0]))[0]
    has_sta = has_sta[conditions_all]
    #mean_rf_size = np.median((np.abs(STAs['fits'][has_sta,3]),np.abs(STAs['fits'][has_sta,3])))
    mean_rf_size=np.zeros(np.sum(conditions_all), dtype(int))
    ylims = [(0,40),(0,10)]
        
    for i, c in enumerate(show_order):
    
        n_units = np.where(fcls == c+1)[0].shape[0]
        for ci, stimid in enumerate(plot_stims):
            sts = Stimuli['SpikeTrains'][stimid][conditions_all][np.where(fcls == c+1)]
            if stimid == 0:
                txt = "Cluster %d (%d units)" % (c+1,n_units)
                t_ooi  = OOi[conditions_all][np.where(fcls == c+1)]
            elif stimid == 1:
                t_sils = silhouettes[np.where(fcls == c+1)]
                
                txt = "Avg. OOi: %.2f±%.2f    Avg. Silhouette Coeff.: %.2f±%.2f" % (np.average(t_ooi),np.std(t_ooi),np.average(t_sils),np.std(t_sils))
            elif stimid == 5:
                t_dsi = DSi[conditions_all][np.where(fcls == c+1)][:,0]
                txt = "Avg. DSi: %.2f±%.2f" % (np.average(t_dsi),np.std(t_dsi))
            else:
                txt = ""
            if (ci == 0) & (i==0):
#                txt = 'G'#after
                txt = 'C'
            elif (ci == 1) & (i==0):
 #               txt = 'E'
                txt = 'D'
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
                    plt.text(2500, .8*max(ys), " %.2f   " % (np.mean(t_ooi)), fontsize=9)
                ax.spines['top'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['right'].set_visible(False)
                if (i==len(show_order)-1):
                    plt.xlabel('time (s)', fontsize=10)
                    if stimid == 0:
                        xticks([0,2000,4000], ['0', '2', '4'])
                    elif stimid == 1:
                        xticks([0,3000,6000,9000,12000], ['0', '3', '6', '9', '12'])
                    for tick in ax.xaxis.get_major_ticks():
                        tick.label.set_fontsize(10)

    #    with plt.rc_context({'font.size':fs, 'axes.titleweight': 'bold'}):
    #        ax = plt.subplot(gs[i,5])
    ##        t_dsi = DSi[with_sta_cells][np.where(fcls == c+1)][:,0]
    #        sns.distplot(t_dsi,bins=np.arange(0,0.8,0.1),kde=True, norm_hist=True)
    #        if i == 0:
    #            txt = 'D'
    #        else:
    #            txt = ''
    #        t_dsi = np.median(DSi[with_sta_cells][:,0])
    #        plt.plot((t_dsi, t_dsi),ax.get_ylim(),'r')
    #        ax.set_title(txt)
    #        ax.set_xticks(())
    #        ax.set_yticks(())
    #        ax.set_xlim((0,1.0))
    #        ax.spines['top'].set_visible(False)
    #        ax.spines['right'].set_visible(False)
    #        if i==len(show_order)-1:
    #            plt.xlabel('DSI', fontsize=8)
                
                
        with plt.rc_context({'font.size':fs, 'axes.titleweight': 'bold'}):
            ax = plt.subplot(gs[i,1])
    #        t_dsi = DSi[with_sta_cells][np.where(fcls == c+1)][:,0]
    #        sns.distplot(t_dsi,bins=np.arange(0,0.8,0.1),kde=True, norm_hist=True)
            if i == 0:
#                txt = 'F'#after
                txt = 'B'
            else:
                txt = ''
    #        t_dsi = np.median(DSi[with_sta_cells][:,0])
            trl_sta=[]    
            stas=esta_all[conditions_all][np.where(fcls == c+1)]
            for ii in range(len(stas)):
                if  not(stas[ii]==None):
                    if len(stas[ii][2])>1:
                        stime=stas[ii][2]
                        ssta=stas[ii][3]
        
                        xnew = np.linspace(stime.min(), stime.max(), 200)  
                        
                        sta_smooth = spline(stime.flatten(), ssta.flatten(), xnew)
                        sta_smooth=(sta_smooth-mean(sta_smooth))/std(sta_smooth,ddof=0)
                        trl_sta.append(sta_smooth)
                        
                        plt.plot(xnew,sta_smooth,linewidth=.8,alpha=.7)
                        plt.axhline(y=0, color='k', linestyle='--',linewidth=.8)
    #                    ax.set_ylim((stime.min(),stime.max()))
                        plt.axvline(x=.020, color='k', linestyle='--',linewidth=.8)
                    
    #                plt.plot(stas[i][2],(stas[i][3]-mean(stas[i][3]))/max(abs(stas[i][3])))
            if trl_sta:
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
                plt.xlabel('time (s)', fontsize=10)
                ax.set_xticks(([-1,-.5,0]))
                for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(10)

    
 

        with plt.rc_context({'font.size':fs, 'axes.titleweight': 'bold'}):
            if plot_psd:
                pieplace=5
            else:
                pieplace=4
            ax = plt.subplot(gs[i,pieplace])
    #        t_dsi = DSi[conditions_all][np.where(fcls == c+1)][:,0]
    #        sns.distplot(t_dsi,bins=np.arange(0,0.8,0.1),kde=True, norm_hist=True)
            if i == 0:
#                txt = 'F'#after
                txt = 'B'

            else:
                txt = ''







        clnames=Cell_names[conditions_all][np.where(fcls== c+1)]
        recording_names=[]
        for idss, rcrdings in enumerate(Cell_names):
            recording_names.append(Cell_names[idss][:-4])
            
        recording_names_cluster=[]
        for idss, rcrdings in enumerate(clnames):
            recording_names_cluster.append(clnames[idss][:-4])
            
        unqstrgns_cluster=np.unique(recording_names_cluster)    
        unqstrgns=np.unique(recording_names)    
        
        rcrding_numbers=[]
        for unqnbr in unqstrgns:
            rcrding_numbers.append(recording_names_cluster.count(unqnbr))
        
        cmap = plt.cm.prism

        plt.pie(rcrding_numbers, shadow=True, startangle=90)






        if plot_psd:
       
      
            with plt.rc_context({'font.size':fs, 'axes.titleweight': 'bold'}):
                ax = plt.subplot(gs[i,4])
        #        t_dsi = DSi[with_sta_cells][np.where(fcls == c+1)][:,0]
        #        sns.distplot(t_dsi,bins=np.arange(0,0.8,0.1),kde=True, norm_hist=True)
                if i == 0:
#                    txt = 'H'#after
                    txt = 'E'

                else:
                    txt = ''
        #        t_dsi = np.median(DSi[with_sta_cells][:,0])
                trl_psd=[]    
                psds=psd_all[conditions_all][np.where(fcls == c+1)]
                freq_dtls=freq_details[conditions_all][np.where(fcls == c+1)]
                freq_dtls=[x for x in freq_dtls if x is not None]

                peak_frq=np.mean(freq_dtls,axis=0)
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
                            plt.text(9, .7*max(psd_smooth), " %.2f   " % (peak_frq[0]), fontsize=9)
        #                    ax.set_ylim((stime.min(),stime.max()))
                            #plt.axvline(x=.020, color='k', linestyle='--',linewidth=.8)
                        
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
                if i==len(show_order_sta)-1:
                    plt.xlabel('frequency (hz)', fontsize=10) 
                    ax.set_xticks(([0,6,12]))
                    for tick in ax.xaxis.get_major_ticks():
                        tick.label.set_fontsize(10)

                

#            clnames=Cell_names[conditions_all][np.where(fcls == c+1)]
#            recording_names=[]
#            for idss, rcrdings in enumerate(clnames):
#                recording_names.append(clnames[idss][:-4])
#                
#            unqstrgns=np.unique(recording_names)    
#            rcrding_numbers=[]
#            for unqnbr in unqstrgns:
#                rcrding_numbers.append(recording_names.count(unqnbr))
#                
#    
#            # Plot
#            theme = plt.get_cmap('jet') 
#
#            plt.pie(rcrding_numbers, shadow=True, startangle=90)







           
    ax = plt.subplot(gs[:,0])
    p = ax.get_position()
    p.x1 = p.x1-0.02
    ax.set_position(p)
    with plt.rc_context({'lines.linewidth': 4, 'font.size':fs, 'axes.titleweight': 'bold'}):
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
       plt.savefig('clusters_e_STA_color_mixed_'+name+'.pdf', bbox_inches='tight')
       plt.savefig('clusters_e_STA_color_mixed_'+name+'.svg', bbox_inches='tight')
        
        
        
    Cells_names_and_clusters_sta=list(zip(Cell_names[conditions_all],fcls))
        
    with open("clusetr_numbers_sta_rd_"+name+".txt", 'w') as output:
        for row in Cells_names_and_clusters_sta:
            output.write(str(row) + '\n')      
    
        

            
            #% make table of requancies
    from tabulate import tabulate
    
    print(tabulate(np.round(peak_frq_table,2), headers=["peak freq", "f1 50%", "f2 50%","bw"]))
    with open('table_rd10_freqs_'+name+'.txt', 'w') as f:
        f.write(tabulate(np.round(peak_frq_table,2)))
    
    
    return peak_frq_table