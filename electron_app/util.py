
# coding: utf-8

# coding: utf-8
import numpy as np

# graph layout
import seaborn as sns
# sns.set(style = 'white')
import pylab

# show graphs
import matplotlib
import matplotlib.pyplot as plt

# use data
import pandas as pd

# curve fitting
import scipy.optimize

# differential & integrate
from sympy import *

import os
import sys

def file_error():
	print("File style is not correct. Please check your index.(ex. Ww Wd W1h W3d)")
	exit(1)

def arrange(rawdf, dt, S):
    # rawdataをコピー
    tmp =rawdf.copy()

    weight = {}
    for x in tmp.columns:
        if x == "number" or x == "group" or x == "V" or x == "Ww" or x == "Wd":
            pass
        else:
            if x[-1] =="d":
                day = float(x[1:].rstrip("d"))
            elif x[-1] == "h":
                day = float(x[1:].rstrip("h"))/24
            else:
                file_error()
            weight[day]=x

    print(weight)
    
    # S: 被補修面表面積    V2: 試験体体積     depth: 試験体厚さ 
    tmp['depth'] = tmp['V']/S
    
    # amount: 含水量   ratio: 含水率     eta: ボルツマン変換 (λ = x / sqrt(t) )
    for t in weight.keys():
        #max_water_amount: 'Ww'-'Wd'
        tmp['max_water_amount'] = tmp['Ww']-tmp['Wd']
        if t is 0:
            tmp['amount_d0'] = 0
            tmp['ratio_d0'] =  1-( tmp['amount_d'+str(t)] / tmp['max_water_amount'] )
            tmp['eta_d0'] = tmp['depth']/(2*np.sqrt(dt))
        else:
            #amount_dt: 'W0'-'Wt'
            #ratio_dt: 1-(amount_dt/amount_d0)
            tmp['amount_d'+str(t)] = tmp[weight[0]]-tmp[weight[t]]
            tmp['ratio_d'+str(t)] = 1-( tmp['amount_d'+str(t)] / tmp['max_water_amount'] )
            tmp['eta_d'+str(t)] = tmp['depth']/np.sqrt(t)
    
    # グループ名のリストを作成
    groups=[]
    for group in rawdf['group']:
        if not group in groups:
            groups.append(group)
            
    # 乾燥日数のシリーズを作成
    days = list(weight.keys())
    days = sorted(days)
    
    return tmp, groups, days, weight

def select_data(src_df, days, group):
    # pickup group data
    df = pd.DataFrame(columns=['eta','ratio'])
    for i in days:
        tmp = src_df.loc[src_df['group']==group, ['eta_d'+str(i), 'ratio_d'+str(i)]]
        tmp.columns=['eta','ratio']
        df = df.append(tmp)
    return df


# In[35]:

def show_graph_each(filename, group, point_x, point_y, line_x, line_y):
#     show graph
    plt.clf()
    plt.plot(point_x, point_y, 'ko')
    plt.plot(line_x, line_y, 'k-')

    plt.xlim([0,10])
    plt.ylim([0,1.2])

    plt.title(group)
    plt.xlabel('eta')
    plt.ylabel('R')
    plt.savefig(filename)


# In[71]:
def make_legends(data):
    colorlist = {}
    linelist = {}
    point = {}
    point_line = {}
    facecolor = {}
    for elem in data.keys():
        if "marker" in elem :
            print(data[elem][-1])
            group_name = elem.rstrip("_marker")
            point[group_name] = data[elem][0]

            if data[elem][-1] == "w":
                facecolor[group_name] = "white"
            elif data[elem][-1] == "b":
                facecolor[group_name] = "black"

    for elem in data.keys():  
        if "line" in elem :
            group_name = elem.rstrip("_line")

            if data[elem] == "h":
                colorlist[group_name] = str('#FF9400')
                linelist[group_name] = str('-')
                point_line[group_name] = str( point[group_name] + '-')

            elif data[elem] == "c":
                colorlist[group_name] = str('#0159B4')
                linelist[group_name] = str('-')
                point_line[group_name] = str( point[group_name] + '-')

            else:
                colorlist["others"] = str('#0159B4')
                linelist["others"] = str(':')
                point_line[group_name] = str( point[group_name] + ':')

    legends_dict = {"facecolor":facecolor, "colorlist":colorlist, "linelist":linelist, "point":point, "point_line":point_line}

    return legends_dict

def show_graph_all(group, point_x, point_y, line_x, line_y, color, legends_dict ,watercontent=False):
#     show graph
    if color is True:
        line = {'A0':'k-', 'A1':'g-', 'A2':'r-', 'A3':'y-', 'A4':'b-',
                'B0':'k--', 'B1':'g--', 'B2':'r--', 'B3':'y--', 'B4':'b--',
                'A5':'r-', 'A6':'b-',
                'B5':'r--', 'B6':'b--'
               }
        plt.plot(line_x, line_y, line[group], label=group)
        plt.legend(loc='best')
    else:
        if watercontent is True:
            if group in legends_dict["linelist"].keys():
                plt.plot(point_x, point_y, legends_dict["point_line"][group],markersize=5.5,
                     markeredgewidth=1, markeredgecolor="black",
                     markerfacecolor=legends_dict["facecolor"][group], color=legends_dict["colorlist"][group]
                        ,label=group)
            else:
                plt.plot(point_x, point_y, legends_dict["point_line"][group],markersize=5.5,
                     markeredgewidth=1, markeredgecolor="black",
                     markerfacecolor=legends_dict["facecolor"][group], color=legends_dict["colorlist"]['others']
                        ,label=group)
        else:
            
            plt.plot(point_x, point_y, legends_dict["point"][group],markersize=5.5,
                     markeredgewidth=1, markeredgecolor="black",
                     markerfacecolor=legends_dict["facecolor"][group] ,label=group)
            
            if group in legends_dict["linelist"].keys():
                plt.plot(line_x, line_y, legends_dict["linelist"][group], color=legends_dict["colorlist"][group])
            else:
                plt.plot(line_x, line_y, legends_dict["linelist"]['others'], color=legends_dict["colorlist"]['others'])
        
#         <setting location of legend>
        plt.legend(loc="best",ncol=2)
        

    plt.hold(True)

def make_costdict(group_data):
    for elem in group_data.keys():
        cost_dict = {}
        if "cost" in elem :
            cost_dict[elem.rstrip("_cost")] = group_data[elem]
    return cost_dict

def cost_vs_diffcoef(cost_dict):
    pass
