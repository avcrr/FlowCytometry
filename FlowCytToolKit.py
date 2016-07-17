"""
Created on Thu Jul  7 11:16:52 2016

@author: avcarr
"""

from os import chdir,listdir,getcwd,rename
from sys import stdout
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
import numpy as np
from scipy.stats import linregress
from uncertainties import ufloat
from FlowCytometryTools import FCMeasurement,FCPlate, ThresholdGate
from palettable.tableau import Tableau_20
from palettable.colorbrewer.qualitative import Set3_12, Accent_3, Dark2_7,Pastel2_4
plt.style.use('ggplot')

global ssc_low
global ssc_high
global fsc_low
global fsc_high
global fitc_low
global fitc_high
global percp_low
global percp_high
global fsc_gate
global ssc_gate
global fitc_gate
global total_gate
global time_gate
global ch

ssc_low = 3500
ssc_high = 9000

fsc_low = 2000
fsc_high = 9900

fitc_low = 0
fitc_high = 9900

percp_low = 0
percp_high = 9900

time_low = 500
time_high = 2000
ch = ['FSC-A','SSC-A','FITC-A','PerCP-Cy5-5-A','Time']

ssc_gate = ThresholdGate(ssc_low,'SSC-A',region ='above') & ThresholdGate(ssc_high,'SSC-A',region ='below')  
fsc_gate = ThresholdGate(fsc_low,'FSC-A',region ='above') & ThresholdGate(fsc_high,'FSC-A',region ='below') 
fitc_gate = ThresholdGate(fitc_low,'FITC-A',region ='above') & ThresholdGate(fitc_high,'FITC-A',region ='below')
percp_gate = ThresholdGate(percp_low,'PerCP-Cy5-5-A',region ='above') & ThresholdGate(percp_high,'PerCP-Cy5-5-A',region ='below')
time_gate = ThresholdGate(time_low,'Time',region ='above') & ThresholdGate(time_high,'Time',region ='below') 
total_gate = ssc_gate & fsc_gate

def heatmap(data, title = ''):
    x = []
    y = []
    plt.figure(figsize = (12,8))
    for xid in range(12):
        for yid in range(8):
            for c in range(int(data.iloc[yid, xid])):
                x.append(xid)
                y.append(8-yid)
    plt.yticks(np.linspace(1.5,8.5, 9), ['A','B','C','D','E','F','G','H'][::-1])
    plt.xticks(np.linspace(0.5,12.5,14), ['1','2','3','4','5','6','7','8','9','10','11','12'])
    plt.tick_params(axis='x',which='both',bottom='off',top='off', labelbottom='off',labeltop='on'  )
    plt.tick_params(axis='y',which='both',left='off',right='off')
    plt.ylim(1,8)
    plt.grid(b ='off',which = 'both')
    plt.hist2d(x, y, bins=[12,8])
    plt.colorbar()
    plt.text(0,8.4,title,size = 18)
    plt.tight_layout()
    plt.show()    
    
def hist_series(plate,pltmap,ch):
    plt.figure(figsize = (10,10))
    for i,well in enumerate(pltmap):   
        try:
            gated_smp = plate[well].gate(total_gate)[ch].values
            hist = plt.hist(gated_smp,bins=100)
            if i == 0:
                chmax = max(hist[0])*100
                plt.xlim(0.8*min(hist[1]),1.1*max(hist[1]))
                plt.ylim(0,1.05*max(hist[0]))
        except:
            pass
    plt.xlabel('hlog(%s)'%(ch))
    plt.ylabel('Counts')
    if ch == 'SSC-A':
        plt.plot([ssc_low,ssc_low],[0,chmax],'k--')
        plt.plot((ssc_high,ssc_high),(0,chmax),'k--')
    if ch == 'FSC-A':
        plt.plot((fsc_low,fsc_low),(0,chmax),'k--')
        plt.plot((fsc_high,fsc_high),(0,chmax),'k--')
    if ch == 'FITC-A':
        plt.plot((fitc_low,fitc_low),(0,chmax),'k--')
        plt.plot((fitc_high,fitc_high),(0,chmax),'k--')
        
def plate_hist(plate,channel = 'SSC-A'):
    # Plot FSC and SSC Distributions with Gating
    plt.figure(figsize = (20,20))
    plt.title(plate.ID)
    if channel == 'SSC-A':
        gate = ssc_gate
        lim = (ssc_low*0.8,ssc_high*1.2)
    elif channel == 'FSC-A':
        gate = fsc_gate
        lim = (fsc_low*0.8,fsc_high*1.2)
    elif channel == 'FITC-A':
        gate = fitc_gate
        lim = (fitc_low*0.8,fitc_high*1.2)
    elif channel == 'PerCP-Cy5-5-A':
        gate = percp_gate
        lim = (percp_low*0.8,percp_high*1.2)
    plate.plot(channel,bins=100, color ='green',xlim = lim,ylim =(0,20000), gates = gate)  
    
def hist2d(smp,channels = ['FSC-A','SSC-A']):
    temp_gate = ThresholdGate(1,ch[0],region ='above') & ThresholdGate(1,ch[1],region ='above') 
    fig = plt.figure(figsize = (8,8))
    if (channels[0] == 'FSC-A' and channels[1] == 'SSC-A') or (channels[1] == 'FSC-A' and channels[0] == 'SSC-A'):
        ssc = smp.gate(temp_gate)[ch[1]]
        fsc = smp.gate(temp_gate)[ch[0]]
        H, xedges, yedges, img = plt.hist2d(ssc,fsc,bins=100, norm = LogNorm())
        colorbar = fig.colorbar(img)
        plt.xlabel('hlog(ssc)') 
        plt.ylabel('hlog(fsc)') 
        plt.plot((ssc_low,ssc_low),(fsc_low,fsc_high),'k--')
        plt.plot((ssc_high,ssc_high),(fsc_low,fsc_high),'k--')
        plt.plot((ssc_low,ssc_high),(fsc_low,fsc_low),'k--')
        plt.plot((ssc_low,ssc_high),(fsc_high,fsc_high),'k--')
    elif (channels[0] == 'FITC-A' and channels[1] == 'SSC-A') or (channels[1] == 'FITC-A' and channels[0] == 'SSC-A'):
        ssc = smp.gate(temp_gate)[ch[1]]
        fitc = smp.gate(temp_gate)[ch[2]]
        H, xedges, yedges, img = plt.hist2d(ssc,fitc,bins=100, norm = LogNorm())
        colorbar = fig.colorbar(img)
        plt.xlabel('hlog(ssc)') 
        plt.ylabel('hlog(fitc)') 
        plt.plot((ssc_low,ssc_low),(fitc_low,fitc_high),'k--')
        plt.plot((ssc_high,ssc_high),(fitc_low,fitc_high),'k--')
        plt.plot((ssc_low,ssc_high),(fitc_low,fitc_low),'k--')
        plt.plot((ssc_low,ssc_high),(fitc_high,fitc_high),'k--')
    elif (channels[0] == 'FITC-A' and channels[1] == 'FSC-A') or (channels[1] == 'FITC-A' and channels[0] == 'FSC-A'):
        fitc = smp.gate(temp_gate)[ch[2]]
        fsc = smp.gate(temp_gate)[ch[0]]
        H, xedges, yedges, img = plt.hist2d(fsc,fitc,bins=100, norm = LogNorm())
        colorbar = fig.colorbar(img)
        plt.xlabel('hlog(fsc)') 
        plt.ylabel('hlog(fitc)') 
        plt.plot((fsc_low,fsc_low),(fitc_low,fitc_high),'k--')
        plt.plot((fsc_high,fsc_high),(fitc_low,fitc_high),'k--')
        plt.plot((fsc_low,fsc_high),(fitc_low,fitc_low),'k--')
        plt.plot((fsc_low,fsc_high),(fitc_high,fitc_high),'k--')
    elif (channels[0] == 'FITC-A' and channels[1] == 'PerCP-Cy5-5-A') or (channels[1] == 'FITC-A' and channels[0] == 'PerCP-Cy5-5-A'):
        percp = smp.gate(temp_gate)[ch[3]]
        fitc = smp.gate(temp_gate)[ch[2]]
        H, xedges, yedges, img = plt.hist2d(percp,fitc,bins=100, norm = LogNorm())
        colorbar = fig.colorbar(img)
        plt.xlabel('hlog(percp)')
        plt.ylabel('hlog(fitc)') 
        plt.plot((percp_low,percp_low),(fitc_low,fitc_high),'k--')
        plt.plot((percp_high,percp_high),(fitc_low,fitc_high),'k--')
        plt.plot((percp_low,percp_high),(fitc_low,fitc_low),'k--')
        plt.plot((percp_low,percp_high),(fitc_high,fitc_high),'k--')
    else:
        print 'Wrong Input!'


def readasc(ascfile):
    data = pd.read_csv(ascfile, engine='python')
    if data.iloc[0, 0] != '0s':
        time = data.applymap(lambda x: re.sub('[\(, \)]', '', x.split()[0]) if type(x) == str else x)
        data = data.applymap(lambda x: re.sub('\(.*?\)', '', x) if type(x) == str else x)
        data, meta = data[~pd.isnull(data.A1)], data[pd.isnull(data.A1)]
        data.index = data.index * int(time['A1'][1][:-1])
        data.index = np.round(data.index/3600.,1)
    else:
        data, meta = data[~pd.isnull(data.A1)], data[pd.isnull(data.A1)]
        time = data.iloc[:, 0].apply(lambda x: int(re.sub('s', '', x)))
        data.index = time
        data.index = np.round(data.index/3600.,1)
    meta = meta.iloc[:, 0]
    meta.reset_index(drop=True, inplace=True)
    data.meta = meta
    data.drop(data.columns[[0, -1]], axis=1, inplace=True)
    data.index.name = 'time'
    data.columns.name = 'well'
    return data

def getabs384(time,ascdata,file_num = 0,reps = 2,odd = True):
    rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    cols = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    t =  pd.DataFrame(data = -1, index = rows, columns = cols)  
    decimal = abs(time - round(time))
    if decimal>0.3 and decimal < 0.7:
        time = time -decimal + 0.5
    elif decimal < 0.3:
        time = time -decimal 
    else:
        time = time - decimal +1
    index = np.where(ascdata[file_num].index == time)[0][0]
    col384 = 0
    col96 = 0
    if odd:
        for i,j in enumerate(ascdata[file_num].iloc[index,:]):
            j = float(j)
            if (i + 1)%2 != 0:
                row = (i-16*col384)/2
                if t.iloc[row,col96] == -1:
                    t.iloc[row,col96] = j
                elif reps == 2:
                    t.iloc[row,col96] = (float(t.iloc[row,col96]) + j)/2.
            if (i+1)%16 == 0:
                col384 += 1
            if (i+1)%32 == 0:
                col96 += 1
    else:
        for i,j in enumerate(ascdata[file_num].iloc[index,:]):
            j = float(j)
            if (i + 1)%2 == 0:
                row = (i-16*col384)/2
                if t.iloc[row,col96] == -1:
                    t.iloc[row,col96] = j
                elif reps ==2:
                    t.iloc[row,col96] = (float(t.iloc[row,col96]) + j)/2.
            if (i+1)%16 == 0:
                col384 += 1
            if (i+1)%32 == 0:
                col96 += 1 
    return t

def kineticabs(f):
    try:
        excel = pd.read_excel(f, sep = '/t')
        excel.columns = excel[31:32].values[0]
        excel.index = excel['Cycle Nr.'].values
        excel= excel[32:42]
        excel = excel.iloc[:,1:]
        plate = pd.DataFrame(data = None, index = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], columns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        for col in plate.columns.values:
             for row in plate.index.values:
                 well = row + str(col)
                 plate.loc[row,col] = np.mean(excel[well][:3])
        return plate
    except:
        print 'Sorry, no deal...'
        
def load_data(path,plate_name = 'None',channels = ['SSC-A','FSC-A']):
    #Formatting File names for FCPlate Parser Import
    chdir(path)
    files = listdir(getcwd())
    try:
        files.remove('.DS_Store')
    except:
        pass
    for i,name in enumerate(files):
        if name[-6] == '0':
            newname = plate_name + '_Well_' + name[-7] + name[-5:-4] + '_' + name[-7:-4] + name[-4:]
        else:
            newname = plate_name + '_Well_' + name[-7:-4] + '_' + name[-7:-4] + name[-4:]
        rename(name,newname)
    #import Data as plate
    files = listdir(getcwd())
    try:
        files.remove('.DS_Store')
    except:
        pass
    plate = FCPlate.from_files(ID = plate_name, datafiles = files,parser = 'name')
    trans = plate.transform('hlog',channels = channels, b = 10)
    chdir(r'../')
    print '%s Data Loaded!'%(plate_name)
    return trans
    
def plate_counts(plate,channels = ['FSC-A','SSC-A']):
    #Plate Event Counts
    rows = plate.row_labels
    cols = plate.col_labels
    plate_counts = pd.DataFrame(data = None, index = rows, columns = cols)
    progress = 0.0
    print 'Counting Events...'
    for well in plate:
        plate_counts.loc[well[0],int(well[1:])] =  evs_count(plate[well],channels)
        progress+=1
        update_progress(progress/float(len(plate)))   
    return plate_counts

def time_counts(plate,channel ='SSC-A',trim = False):
    gate = fsc_gate & ssc_gate
    rows = plate.row_labels
    cols = plate.col_labels
    plate_counts = pd.DataFrame(data = None, index = rows, columns = cols)
    progress = 0.0
    print 'Counting Events...%s'%(plate.ID)
    for well in plate:
        try:
            if trim:
                channel_vals = plate[well].gate(gate&time_gate)[channel].values
                time  = plate[well].gate(gate&time_gate)['Time'].values/100
            else:
                channel_vals = plate[well].gate(gate)[channel].values
                time  = plate[well].gate(gate)['Time'].values/100
            counts, xedges, yedges = np.histogram2d(time,channel_vals,bins = 100)
            evs = []
            for i in range(len(xedges)-1):
                evs.append(sum(counts[i])/(xedges[i+1]-xedges[i]))
        except:
            evs = [0,0]
        plate_counts.loc[well[0],int(well[1:])] = np.mean(evs)
        progress+=1
        update_progress(progress/float(len(plate)))
        plate_counts.ID = plate.ID
    return plate_counts
    
def get_od(step,length,od_init):
    dil = []
    dil.append(1)
    for i in range(length-1):
        dil.append(dil[-1]*step)
    dil = np.array(dil)
    od = od_init*dil
    return od

def fsc_index(smp):
    try:
        gated_fsc = smp.gate(fsc_gate & ssc_gate)[ch[0]]
        fsc_median = np.median(gated_fsc)
        fsc_dev = np.std(gated_fsc)
        raw_fsc= smp[ch[0]]
        fsc_index = raw_fsc[raw_fsc>fsc_median - 4*fsc_dev][raw_fsc<fsc_median+4*fsc_dev].index
        return fsc_index
    except:
        return [0]
    
def ssc_index(smp):
    try:
        gated_ssc = smp.gate(fsc_gate & ssc_gate)[ch[1]]
        ssc_median = np.median(gated_ssc)
        ssc_dev = np.std(gated_ssc)
        raw_ssc = smp[ch[1]]
        ssc_index = raw_ssc[raw_ssc>ssc_median - 4*ssc_dev][raw_ssc<ssc_median+4*ssc_dev].index
        return ssc_index
    except:
        return [0]
def fitc_index(smp):
    try:
        gated_fitc = smp.gate(total_gate)[ch[2]]
        fitc_median = np.median(gated_fitc)
        fitc_dev = np.std(gated_fitc)
        raw_fitc= smp[ch[2]]
        fitc_index = raw_fitc[raw_fitc>fitc_median - 4*fitc_dev][raw_fitc<fitc_median+4*fitc_dev].index
        return fitc_index
    except:
        return [0]
        
def evs_count(smp, channels = ['FSC-A','SSC-A']):
    index_call = {'FITC-A':fitc_index(smp),'SSC-A':ssc_index(smp),'FSC-A':fsc_index(smp)}
    indexes = []
    for ch in channels:
        indexes.append(index_call[ch])
    count = 0
    if len(indexes) == 3:
        for i in indexes[0]:
            if i in indexes[1] and i in indexes[2]:
                count +=1
    elif len(indexes) == 2:
        for i in indexes[0]:
            if i in indexes[1]:
                count +=1
    elif len(indexes) == 1:
        count = len(indexes[0])
    return count
            
def update_progress(progress):
    barLength = 10
    status = ""
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    stdout.write(text)
    stdout.flush()
