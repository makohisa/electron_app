
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
import util
import json


# data setting ###########################################################

argvs = sys.argv
argc = len(argvs)

project_name = argvs[1]
rawdata_path = argvs[2]

results_path = 'results'

with open('./data.json') as data_file:    
    group_data = json.load(data_file)

##########################################################################



# path setting ###########################################################

abs_path = os.path.abspath(os.path.dirname(__file__))

results_path = os.path.join(abs_path, results_path)
results_path = os.path.join(results_path, project_name)
if os.path.isdir(results_path) is False:
    os.makedirs(results_path)
    
watercontent_file = 'time_R.csv'
watercontent_path = os.path.join(results_path, watercontent_file)

curvefit_dir = os.path.join(results_path, 'curve_fit')
if os.path.isdir(curvefit_dir) is False:
    os.mkdir(curvefit_dir)

timeR_color_path = os.path.join(results_path, 'time_R_color')
timeR_nocolor_path = os.path.join(results_path, 'time_R_no_color')

RK_color_path = os.path.join(results_path, 'R_D_color')
RK_nocolor_path = os.path.join(results_path, 'R_D_no_color')

param_path =  os.path.join(results_path, 'param.csv')
diffcoef_path = os.path.join(results_path, 'diffcoef.csv')
diffcoef_img_path = os.path.join(results_path, 'diffcoef')

##########################################################################


#dpi設定
fine=300
pylab.figure(figsize=(10, 4),dpi=fine)

#W0時点の乾燥時間dtを定義
dt=1*np.exp(-9223372036854775808)

#  重量測定をcsvファイルから読み込み
rawdata_df=pd.read_csv(rawdata_path)

#csvファイルから読み込んだデータを編集(dataframe : data_df)
util.arrange(rawdata_df, dt)
data_df, groups, days ,weight = util.arrange(rawdata_df, dt)
data_df = data_df.dropna()

#legendを設定
legends_dict = util.make_legends(group_data)

def watercontent(data_df, days, color, legends_dict):
    watercontent_df = pd.DataFrame(index=days, columns=groups)
    for group in groups:
        
        df = pd.DataFrame(index=days, columns=['ratio'])
        
        for i in days:
            # 乾燥日数ごとに平均の含水率算出
            tmp = data_df.loc[data_df['group']==group, ['ratio_d'+str(i)]]
            df.loc[i, ['ratio']] = tmp.mean()['ratio_d'+str(i)]
            # show graph
        
        util.show_graph_all(group, point_x = df.index, point_y = df['ratio'],
                       line_x = df.index, line_y = df['ratio'], color=color, watercontent=True, legends_dict=legends_dict)
        
        plt.xlim([0, max(days)+1])
        if min(df['ratio']) < 0.5 :
            plt.ylim([0, 1])
        else:
            plt.ylim([0.5, 1])

        plt.title('Relationship between time and R')
        plt.xlabel('time [day]')
        plt.ylabel('R')
        if color is True:
            plt.savefig(timeR_color_path, dpi=fine)
        else:
            plt.savefig(timeR_nocolor_path, dpi=fine)
            
        watercontent_df[group]=df['ratio']
    
    return watercontent_df


watercontent_df = watercontent(data_df, days, color=False, legends_dict=legends_dict)
watercontent_df.to_csv(watercontent_path)


def curve_fitting(data, group, parameter_initial):
    # func に指定した双曲線でフィッティング
    # group : dataのグループ
    # parameter_initial : 初期パラメータ (a, b, n) 
    # parameter_optional : フィッティング後パラメータ (a, b, n) 
    
    # tmpの点(xdata, ydata)
    xdata = data['eta']
    ydata = data['ratio']

    # 双曲線
    def func(x, a, b, f):
        with np.errstate(invalid ='ignore'):
            return  1 + f - (a/(x+b)**2)

    #     parameter の範囲を指定してscipyのcurve_fitでフィッティング
    #    parameter :      0<a<20    0<b<20   0<n<30
    
    paramater_optimal, covariance = scipy.optimize.curve_fit(
        func, 
        xdata, 
        ydata,
        bounds=(0, [5., 5., 1.])
    )
    
    # show graph
    x=np.linspace(0,10,50)
    y=func(x, paramater_optimal[0], paramater_optimal[1], paramater_optimal[2])
    util.show_graph_each(filename=os.path.join(curvefit_dir, group), group=group, point_x = xdata, point_y = ydata, line_x = x, line_y = y)
    
    return paramater_optimal


# In[75]:

def make_param_df(data_df, days):    
    
    # パラメータを入れるDataFrame　"param" 作成(dataframe : param_df)
    param_df=pd.DataFrame(index=groups, columns=['a','b','f'])

    for group in groups:
        group_df = util.select_data(data_df, days, group)
        param_df.loc[group]=curve_fitting(group_df, group,  parameter_initial =np.array([20, 20, 50]))
    return param_df



def functionK():
    #拡散係数の算出

    x,a,b,f,R = symbols('x a b f R')

    #  η(R)
    f0 = 1 + f - (a/(x+b)**2) - R
    y = solve(Eq(f0, 0), x)

    #  dC/dλ   
    f1 = diff(1 + f - (a/(x+b)**2), x)
    f1 = f1.subs([(x, y[0])])
    f1 = simplify(f1)

    # K
    f2 = integrate(y[0], R)
    f2 = simplify(f2.subs([(R, 1)])-f2.subs([(R, R)]))

    fK = simplify((2*f2)/f1)

    return fK



def solve_K(data, group, param_df, diffcoef_df, fK, color, legends_dict):

    x,a,b,f,R = symbols('x a b f R')
    fK2 = fK.subs([
            (a, float(param_df.loc[group,['a']])),
            (b, float(param_df.loc[group,['b']])),
            (f, float(param_df.loc[group,['f']]))
        ])
    print(fK)
    print(fK2)

    # 含水率が0.5のときの拡散係数
    diffcoef_df.loc[group,['Formula']] = fK.subs([
            (a, float(param_df.loc[group,['a']])),
            (b, float(param_df.loc[group,['b']])),
            (f, float(param_df.loc[group,['f']]))
        ])
    print(param_df.loc[group,['f']])
    diffcoef_df.loc[group,['D(R=0.5)']] = fK2.subs([(R, 0.5)]).round(4)
    diffcoef_df.loc[group,['D(R=0.99)']] = fK2.subs([(R, 0.99)]).round(4)

    # show graph
    xdata = list(data['ratio'])[4:]

    ydata = []
    for i in xdata:
        ydata.append(fK2.subs([(R, i)]))

    x=list(np.linspace(0,0.95,100))
    y=[]
    for i in x:
        y.append(fK2.subs([(R, i)]))

#         <梗概用>
#         if group == 'A0' or group == 'A2' or group == 'A3' or group=='A4' or group == 'B0':
#             show_graph_all(group=group, point_x = xdata, point_y = ydata, line_x = x, line_y = y, color=color)

#         <本文用>

    util.show_graph_all(group=group, point_x = xdata, point_y = ydata, line_x = x, line_y = y, legends_dict=legends_dict, color=color)

    plt.title('Relationship between R and K')
    plt.ylabel('D [cm2/day]')  
    plt.xlabel('R')
    plt.yscale('log')
    plt.ylim(ymin=0)
    plt.xlim([0,0.95])

    if color is True:
        plt.savefig(RK_color_path, dpi=fine)
    else:
        plt.savefig(RK_nocolor_path, dpi=fine)


def make_diffcoef_df(data_df, param_df, color, legends_dict):    
    # 含水率が0.5のときの拡散係数のDataFrame作成(dataframe : diffcoef_df)
    diffcoef_df = pd.DataFrame( index = groups, columns=['Formula', 'D(R=0.5)','D(R=0.99)'] )
    fK=functionK()

    plt.clf()
    for group in groups:
        group_df = util.select_data(src_df=data_df, days=days, group=group)
        solve_K(group_df, group, param_df, diffcoef_df, fK, color=color, legends_dict=legends_dict)
            
    make_diffcoef_img(diffcoef_df)

    return diffcoef_df

def make_diffcoef_img(diffcoef_df):
    plt.clf()
    d_array = diffcoef_df['D(R=0.5)']
    left = np.arange(1, len(groups)+1)

    plt.bar(left, d_array, tick_label=groups, align="center")
    plt.title("D at R = 0.5")
    plt.xlabel("group")
    plt.ylabel("D [cm2/day]")

    for x,y in zip(left, d_array):
        plt.text(x,y,y, ha='center', va='bottom')
    plt.savefig(diffcoef_img_path, dpi=fine)


def make_cost_vs_diffcoef_df(diffcoef_df):    
    # 含水率が0.5のときの拡散係数のDataFrame作成(dataframe : diffcoef_df)
    diffcoef_df = pd.DataFrame( index = groups, columns=['Formula', 'D(R=0.5)','D(R=0.99)'] )
    fK=functionK()

    plt.clf()
    for group in groups:
        group_df = util.select_data(src_df=data_df, days=days, group=group)
        solve_K(group_df, group, param_df, diffcoef_df, fK, color=color, legends_dict=legends_dict)
            
    make_diffcoef_img(diffcoef_df)

    return diffcoef_df

def make_diffcoef_img(diffcoef_df):
    plt.clf()
    d_array = diffcoef_df['D(R=0.5)']
    left = np.arange(1, len(groups)+1)

    plt.bar(left, d_array, tick_label=groups, align="center")
    plt.title("D at R = 0.5")
    plt.xlabel("group")
    plt.ylabel("D [cm2/day]")

    for x,y in zip(left, d_array):
        plt.text(x,y,y, ha='center', va='bottom')
    plt.savefig(diffcoef_img_path, dpi=fine)

# In[79]:

param_df=make_param_df(data_df, days)

diffcoef_df=make_diffcoef_df(data_df, param_df, color=False, legends_dict=legends_dict)

#パラメータをcsvファイルに出力
param_df.to_csv(param_path)

#拡散係数をcsvファイルに出力
diffcoef_df.to_csv(diffcoef_path)

print("end")
