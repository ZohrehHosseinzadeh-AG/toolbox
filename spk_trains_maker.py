# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 16:36:02 2021

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
from scipy.special import ndtr as ndtr

from sta import *
from spikeutilities import *
def spk_trains_maker(filenames,report_filenames,trigger_filenames,stims):
    notebooks_path = "C:/Users/Admin/Desktop/hennig_project/download/rgcclassification/sample_dara_hamed/"

    data_dir ="C:/Users/Admin/Desktop/hennig_project/download/rgcclassification/sample_dara_hamed/"

    data_path = "C:/Users/Admin/Desktop/hennig_project/download/rgcclassification/sample_dara_hamed/"

    from os.path import dirname, join as pjoin


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


#%% chech if the sta is significant

def sta_sif(esta_i):

           
 #   esta_i=np.asarray(esta_all[i][3].flatten())
    esta_i_basline=esta_i[30:]
    esta_i_evoked=esta_i[:26]
    esta_i_p=np.hstack((esta_i_basline, np.max(esta_i_evoked)))
    esta_i_t=np.hstack((esta_i_basline, np.min(esta_i_evoked))) 

    df= pd.DataFrame(esta_i_p,columns=['Data_p'])
    df['Data_t']=esta_i_t    
    df['Data_zscore_p']=(df['Data_p']-df['Data_p'].mean())/df['Data_p'].std(ddof=0)
    df['Data_zscore_t']=(df['Data_t']-df['Data_t'].mean())/df['Data_t'].std(ddof=0)
    df['pval_p']=(1-ndtr(df['Data_zscore_p']))
    df['pval_t']=(1-ndtr(-df['Data_zscore_t']))   
    alpha=.001
    
    df['statistically_signicicance']=(df.pval_p.iloc[-1]<alpha) |(df.pval_t.iloc[-1]<alpha)#.astype(int) .astype(int)

    sta_sig_condition=False
    if np.nansum(df['statistically_signicicance'])>0:
        sta_sig_condition=True
    return sta_sig_condition    


#%%
def plot_clustrs(stimuli,silhouettes,fcls,l,t,cs,OOi,conditions_all,estas,Cell_names,plot_stims,name):
    SAVE_FIGS=False
    # compute averages for each cluster
    unit_mean_OOi = np.zeros(np.unique(fcls).shape[0])
    #unit_mean_DSi = np.zeros(np.unique(fcls).shape[0])
    for c in range(np.unique(fcls).shape[0]):
        inds = np.where(fcls == c+1)[0]
        unit_mean_OOi[c] = np.mean(OOi[conditions_all][inds])    
    #    unit_mean_DSi[c] = np.mean(DSi[conditions_all][inds][:,0])
    n = np.unique(fcls).shape[0]    
    max_num_clustersx = n
    # show_order = np.argsort(unit_mean_OOi)
    show_order = np.unique(fcls)[::-1]-1
    max_num_clusters = np.unique(fcls).shape[0]
    
    palette = sns.hls_palette(max_num_clusters,l=0.6,s=0.6)    
        
    from scipy.interpolate import make_interp_spline
    import matplotlib.gridspec as gridspec
    
    plt.figure(figsize=(14,max_num_clustersx*1))

    fs = 12#24
    T = np.unique(fcls)

    labels=list('' for i in range(n))
    for i in range(n):
        labels[i] = '#' + str(T[i]) + '\n(' + str(np.count_nonzero(np.where(fcls==T[i]))) + ')'

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

                
                
        with plt.rc_context({'font.size':fs, 'axes.titleweight': 'bold'}):
            ax = plt.subplot(gs[i,3])

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
                        
                        sta_smooth = make_interp_spline(stime.flatten(), ssta.flatten())(xnew)
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

    
        with plt.rc_context({'font.size':fs, 'axes.titleweight': 'bold'}):
            ax = plt.subplot(gs[i,4])

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



#%%
def esta_clustsering(Stimuli,show_order_sta,silhouettes,palette,zssta,fcls,l,cs,conditions_all,esta_all,psd_all,freqsl,Cell_names,plot_stims,t,name,OOi, plot_psd,SAVE_FIGS):
    import sklearn.metrics
    import matplotlib.gridspec as gridspec
    from scipy.interpolate import make_interp_spline






   
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
    tss = np.arange(0.3,3,0.01) # threshold values to test

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

    max_num_clustersx=t
    
    # plt.figure(figsize=(9,max_num_clustersx*1.7/4.8))
    plt.figure(figsize=(14,max_num_clustersx*1))
    # plt.figure(figsize=(9,max_num_clustersx*1.7/4.8))
    
    #plot_stims = [0,1]
    
    fs = 12#24
    n = np.unique(fcls).shape[0]    
    T = np.unique(fcls)
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
    Mean_sta_raw=[]    
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
                        trl_sta_raw.append(ssta.flatten())
                        
                        plt.plot(xnew,sta_smooth,linewidth=.8,alpha=.7)
                        plt.axhline(y=0, color='k', linestyle='--',linewidth=.8)
    #                    ax.set_ylim((stime.min(),stime.max()))
                        plt.axvline(x=.020, color='k', linestyle='--',linewidth=.8)
                    
    #                plt.plot(stas[i][2],(stas[i][3]-mean(stas[i][3]))/max(abs(stas[i][3])))
            if trl_sta:
                Mean_sta_raw.append(mean(trl_sta_raw,axis=0))

                plt.plot(xnew,mean(trl_sta,axis=0),linewidth=1.5,color='k')
                p2=sta_sif((mean(trl_sta_raw,axis=0)))
                if p2:
                    plt.text(.23, .85*max(mean(trl_sta,axis=0)), "*", fontsize=12)
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
        
    #%% find the width and latancy of each sta (average of cluster)
    from tabulate import tabulate
    from spk_trains_maker import find_peaks_latency_STA
    
   # D1_D2=find_peaks_latency_STA(Mean_sta_raw,stime,'rd10sta_latency'+name)
    D1_D2=STA_latency_easy(Mean_sta_raw,stime,'rd10sta_latency'+name)
    #STA_width_latency_func(Mean_sta_visual)
    
    print(tabulate(np.round(D1_D2,2), headers=["cl#","w1-peak", "w2-trough", "l1-peak","l2-trough"]))
    with open('table_STA_latency_width_rd10sta_latency' +name+'.txt', 'w') as f:
        f.write(tabulate(np.round(D1_D2,2)))    
        
    Cells_names_and_clusters_sta=list(zip(Cell_names[conditions_all],fcls))
        
    with open("clusetr_numbers_sta_rd_"+name+".txt", 'w') as output:
        for row in Cells_names_and_clusters_sta:
            output.write(str(row) + '\n')      
    
        

            
    #%% make table of frequancies
    
    print(tabulate(np.round(peak_frq_table,2), headers=["peak freq", "f1 50%", "f2 50%","bw"]))
    with open('table_rd10_freqs_'+name+'.txt', 'w') as f:
        f.write(tabulate(np.round(peak_frq_table,2)))
    
    
    return peak_frq_table
#%%
matplotlib.use('TkAgg')
import numpy as np
from scipy.interpolate import make_interp_spline
#Mean_sta=Mean_sta_visual
#plot(goodstas[0][3])

def find_peaks_latency_STA(Mean_sta,stime,figname):
    tabel=[]
    upsampling=5
    binlenght=40/upsampling
    shift=5
    upshift=upsampling*shift
    for c in range(len(Mean_sta)):
        len_sta=int(len(Mean_sta[0])/2+shift)
        rsta=Mean_sta[c][:len_sta].flatten()# includes 100 ms before zero
        xnew = np.linspace(stime[:len_sta].min(), stime[:len_sta].max(), upsampling*len(rsta)) # five times upsampling
                    
        sta_smooth = make_interp_spline(stime[:len_sta].flatten(), rsta.flatten())(xnew)
        rsta=np.flip(sta_smooth,axis=0)
        msta=np.mean(Mean_sta[c][len_sta:])
        
        from scipy.signal import chirp, find_peaks, peak_widths
        import matplotlib.pyplot as plt
        
        peaks, v_peaks = find_peaks(rsta)
        results_half = peak_widths(rsta, peaks, rel_height=0.5)
        results_full = peak_widths(rsta, peaks, rel_height=1)
        
        #Ppeaks, _ = find_peaks(rsta)
        #Npeaks, _ = find_peaks(-rsta)
        
        Ppeaks=np.argmax(rsta[upshift:])
        Npeaks=np.argmin(rsta[upshift:])

        D1=Ppeaks*binlenght#-shift*binlenght# remove 100 ms before zero
        D2=Npeaks*binlenght#-shift*binlenght# remove 100ms before zero
        thrp=msta+abs(rsta[Ppeaks+upshift]-msta)/2
        
        for iP1 in range(len(rsta)-Ppeaks):
            if rsta[Ppeaks+upshift+iP1]<thrp:
               break
        if Npeaks>0:   
            for iP2 in range(Ppeaks+upshift):
                if rsta[Ppeaks+upshift-iP2]<thrp:
                    break
        else:
            iP2=-1 
            
        w1=abs((Ppeaks+iP2/2)-(Ppeaks-iP1/2))*binlenght
        
        thrn=msta-abs(rsta[Npeaks+upshift]-msta)/2
        
        if Npeaks>0:
            for iN1 in range(Npeaks+upshift):
                if rsta[Npeaks+upshift-iN1]>thrn:
                   break   
        else:
            iN1=-1   
        
        for iN2 in range(len(rsta)-Npeaks):
            if rsta[Npeaks+upshift+iN2]>thrn:
                break 
        w2=abs((Npeaks+iN2/2)-(Npeaks-iN1/2))*binlenght
        
        t1=np.arange(0, len(rsta), step=50)*binlenght
        t=(t1-upshift*binlenght)/1000
        fig, ax =plt.subplots()

        plt.plot(rsta)
        plt.hlines(msta-abs(rsta[Npeaks+upshift]-msta)/2, xmin = Npeaks+upshift-iN1, xmax = Npeaks+upshift+iN2,color="C1") # width of negative peak
        plt.hlines(msta+abs(rsta[Ppeaks+upshift]-msta)/2, xmin = Ppeaks+upshift-iP2, xmax = Ppeaks+upshift+iP1,color="C30")  # width of positive peak
        
        plt.vlines(Ppeaks+upshift, ymin =msta , ymax = rsta[Ppeaks+upshift],color="C5")
        plt.vlines(Npeaks+upshift, ymin = rsta[Npeaks+upshift], ymax = msta,color="C5")
        
        plt.hlines(msta, xmin = 0, xmax = len(rsta),linestyles='dashed')
        plt.vlines(upshift, ymin = msta, ymax = rsta[Ppeaks+upshift],linestyles='dashed')


        
        plt.text(Ppeaks+upshift+5, thrp, " %.0f   " % (w1), fontsize=11)
        plt.text(Npeaks+upshift+5, thrn, " %.0f   " % (w2), fontsize=11)
        
        plt.plot(Ppeaks+upshift, rsta[Ppeaks+upshift], "x")
        plt.plot(Npeaks+upshift, rsta[Npeaks+upshift], "x")
        plt.title(np.str(len(Mean_sta)-c))
        plt.xticks(np.arange(0, len(rsta), step=50),t)

        #labels = [item.get_text() for item in ax.get_xticklabels()]
        

        #t2=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

        #t2=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
#        t1=np.arange(shift, len(rsta), step=3)
#        t2=(t1-2)*binlenght/1000
        #plt.xlabel('time s')
        #labels = tt

        #ax.set_xticklabels(labels)
        #xticks(tt)
        plt.savefig('STA_width_latency'+figname+np.str(len(Mean_sta)-c)+'.pdf', bbox_inches='tight')       
       
       
     
            

#        D1w=  iN*40    # width of positive peak
#        D2w= ((Npeaks+iN)-(Ppeaks+iP))*40# width of negative peak
#        
#        plt.plot(rsta)
#        plt.hlines(msta, xmin = Ppeaks+iP, xmax = Npeaks+iN,color="C1") # width of negative peak
#        plt.hlines(msta, xmin = 0, xmax = Ppeaks+iP,color="C30")  # width of positive peak
#        
#        plt.vlines(Ppeaks, ymin =msta , ymax = rsta[Ppeaks],color="C5")
#        plt.vlines(Npeaks, ymin = rsta[Npeaks], ymax = msta,color="C5")
#        
#        plt.text(Ppeaks, msta, " %.0f   " % (D1w), fontsize=11)
#        plt.text(Npeaks, msta, " %.0f   " % (D2w), fontsize=11)
#        
#        plt.plot(Ppeaks, rsta[Ppeaks], "x")
#        plt.plot(Npeaks, rsta[Npeaks], "x")
#        plt.title(np.str(c))
#        plt.savefig('STA_width_latency '+np.str(c)+'.pdf', bbox_inches='tight')

        plt.show()
        tabel.append((len(Mean_sta)-c,w1,w2,D1,D2))
    np.save('latncy_widths_STA_'+figname,tabel)
   
    #return tabel


#%%
def STA_latency_easy(Mean_sta,stime,figname):

   # Mean_sta=Mean_sta_raw
    import matplotlib
    matplotlib.use('TkAgg')
    import numpy as np
    from scipy.interpolate import make_interp_spline
    from scipy.signal import chirp, find_peaks, peak_widths
    import matplotlib.pyplot as plt
    tabel=[]
    upsampling=5
    binlenght=40/upsampling
    shift=5
    upshift=upsampling*shift
    for c in range(len(Mean_sta)):
        len_sta=int(len(Mean_sta[c])/2+shift)

        len_sta=int(len(Mean_sta[c])/2+shift)
        rsta=Mean_sta[c][:len_sta].flatten()# includes 100 ms before zero
        xnew = np.linspace(stime[:len_sta].min(), stime[:len_sta].max(), upsampling*len(rsta)) # five times upsampling
                    
        sta_smooth = make_interp_spline(stime[:len_sta].flatten(), rsta.flatten())(xnew)
        rsta=np.flip(sta_smooth,axis=0)
        msta=np.mean(Mean_sta[c][len_sta:])
        #rsta-=np.mean(rsta)
        peaksP, v_peaks = find_peaks(rsta[upshift:])
        peaksN, v_peaks = find_peaks(-rsta[upshift:])
        fig = plt.figure(figsize=(12,8))

        plt.plot( rsta)
        
        plt.plot(peaksP+upshift,rsta[peaksP+upshift], "x",color="green",linewidth=4, markersize=12)   
        plt.plot(peaksN+upshift,rsta[peaksN+upshift], "x",color="red",linewidth=4, markersize=12)   
        plt.hlines(msta, xmin = 0, xmax = len(rsta),linestyles='dashed')
        
        plt.vlines(upshift, ymin = msta, ymax = rsta[np.argmax(rsta)],linestyles='dashed')

        #plt.hlines(0, xmin = 0, xmax = len(rsta),color="C1") # width of negative peak
        
        
        matplotlib.use('TkAgg')
        plt.title('sta', fontweight ="bold")
        
        
        print("After 3 clicks :")
        x1 = plt.ginput(2)
        print(x1)
        x1=np.array(x1)
        x1 = x1.T[0]-upshift
        Ppeaks=min(peaksP, key=lambda x:abs(x-x1[0]))
        Npeaks=min(peaksN, key=lambda x:abs(x-x1[1]))
        #Ppeaks=x[0][0]#np.argmax(rsta[upshift:]
        #Npeaks=x[1][0]#np.argmin(rsta[upshift:])

        D1=Ppeaks*binlenght#-shift*binlenght# remove 100 ms before zero
        D2=Npeaks*binlenght#-shift*binlenght# remove 100ms before zero
        thrp=msta+abs(rsta[Ppeaks+upshift]-msta)/2
        
        for iP1 in range(len(rsta)-Ppeaks):
            if rsta[Ppeaks+upshift+iP1]<thrp:
               break
        if Npeaks>0:   
            for iP2 in range(Ppeaks+upshift):
                if rsta[Ppeaks+upshift-iP2]<thrp:
                    break
        else:
            iP2=-1 
            
        w1=abs((Ppeaks+iP2/2)-(Ppeaks-iP1/2))*binlenght
        
        thrn=msta-abs(rsta[Npeaks+upshift]-msta)/2
        
        if Npeaks>0:
            for iN1 in range(Npeaks+upshift):
                if rsta[Npeaks+upshift-iN1]>thrn:
                   break   
        else:
            iN1=-1   
        
        for iN2 in range(len(rsta)-Npeaks):
            if rsta[Npeaks+upshift+iN2]>thrn:
                break 
        w2=abs((Npeaks+iN2/2)-(Npeaks-iN1/2))*binlenght
        
        t1=np.arange(0, len(rsta), step=50)*binlenght
        t=(t1-upshift*binlenght)/1000
        fig, ax =plt.subplots()

        plt.plot(rsta)
        plt.hlines(msta-abs(rsta[Npeaks+upshift]-msta)/2, xmin = Npeaks+upshift-iN1, xmax = Npeaks+upshift+iN2,color="C1") # width of negative peak
        plt.hlines(msta+abs(rsta[Ppeaks+upshift]-msta)/2, xmin = Ppeaks+upshift-iP2, xmax = Ppeaks+upshift+iP1,color="C30")  # width of positive peak
        
        plt.vlines(Ppeaks+upshift, ymin =msta , ymax = rsta[Ppeaks+upshift],color="C5")
        plt.vlines(Npeaks+upshift, ymin = rsta[Npeaks+upshift], ymax = msta,color="C5")
        
        plt.hlines(msta, xmin = 0, xmax = len(rsta),linestyles='dashed')
        plt.vlines(upshift, ymin = msta, ymax = rsta[np.argmax(rsta)],linestyles='dashed')
        
        
        
        plt.text(Ppeaks+upshift+5, thrp, " %.0f   " % (w1), fontsize=11)
        plt.text(Npeaks+upshift+5, thrn, " %.0f   " % (w2), fontsize=11)
        
        plt.plot(Ppeaks+upshift, rsta[Ppeaks+upshift], "x")
        plt.plot(Npeaks+upshift, rsta[Npeaks+upshift], "x")
        plt.title(np.str(len(Mean_sta)-c))
        plt.xticks(np.arange(0, len(rsta), step=50),t)
        # thismanager = plt.get_current_fig_manager()
        # thismanager.window.SetPosition((59999999990, 9999915))
        plt.show()
        
        plt.savefig('STA_width_latency_new'+figname+np.str(len(Mean_sta)-c)+'.pdf', bbox_inches='tight')       

        plt.show()
        tabel.append((len(Mean_sta)-c,w1,w2,D1,D2))
        np.save('latncy_widths_STA_new'+figname,tabel)
   
    return tabel
        # results_half = peak_widths(np.sin(t), peaks, rel_height=0.5)
        # results_full = peak_widths(np.sin(t), peaks, rel_height=1)