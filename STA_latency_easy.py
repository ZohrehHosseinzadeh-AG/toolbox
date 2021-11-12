
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