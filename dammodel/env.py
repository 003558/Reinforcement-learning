# %% [markdown]
"""
# # **強化学習エージェント**
# CTI kokubunkenISP | Kensuke Matsuda  
# ■計算の流れ  
    def __init__(self):
        # アクションの数の設定
        # (ex)ACTION_NUM=3
        self.action_space = gym.spaces.Discrete(ACTION_NUM) 
        # 状態空間の設定 
        # (ex)状態が3つの時で、それぞれの状態が正規化されている場合、LOW=[0,0,0]、HIGH=[1,1,1]
        self.observation_space = gym.spaces.Box(low=LOW, high=HIGH)
    def _reset(self):
        # シミュレータの初期化処理
        return observation
    def _step(self):
        # ステップを進める処理
        return observation, reward, done, {} #infoの部分は使わなかったので{}を返却
    def _render(self):
        # シミュレータ環境の描画処理    
    def _observe(self):
        # 状態の設定
    def _get_reward(self):
        # 報酬の設定
    def _is_done(self):
        # 終了判定

"""

#%%
# coding: utf-8
import yaml
import copy
import sys
import random
import datetime
from boto3.resources.model import Action
import gym
import numpy as np
import gym.spaces
import subprocess        # import文なので次以降の例では省略します
import pandas as pd
import csv
from datetime import datetime
from scipy import signal, interpolate
import matplotlib.pyplot as plt
import os
from io import StringIO
import math
from ctypes import *
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d
# グラフ用のフォントの設定
from matplotlib.font_manager import FontProperties
# x軸の日付表示
import matplotlib.dates as mdates
# グラフのサイズ変更
from matplotlib import gridspec
# グラフ用のフォントの設定
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
fp1 = FontProperties(fname="C:\Windows\Fonts\msgothic.ttc", size=15)
fp2 = FontProperties(fname="C:\Windows\Fonts\msgothic.ttc", size=12)
# 実行時の警告を表示しない
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.simplefilter('ignore')


class dam(gym.Env):
    def __init__(self):
        super().__init__()
        self.outf      = '01_Output_train' # 出力フォルダ
        self.out_n     = 20   #行動： 一定流量の選択数、但し書き水位の選択数
        self.features  = 24   #状態(実績流入量【〇～現時刻】、予測雨量【１～72】、実績雨量【〇～現時刻】、実績貯水位【現時刻】)：
        self.Qinmax     = 500 #流入量のMAX値 正規化に使用
        self.Vmax       = 12936 * 1000 #サーチャージ水位の貯水量 正規化に使用
        self.Rmax       = 20 #雨量の正規化に使用
        #self.low       = [0 for _ in range(100)]  #入力する状態（流入量0～12hr、放流量0hr、貯水位0hr）の最小値  正規化するので0
        #self.high      = [1 for _ in range(100)]  #入力する状態（流入量0～12hr、放流量0hr、貯水位0hr）の最大値  正規化するので1
        self.Qb = 0.8    #維持流量!!!
        self.Vcon = 0.3   #!!!コンジットゲート開閉速度(m/min)
        #self.Vcre = 0.3   #!!!クレストゲート開閉速度(m/min)
        self.count = 0     #学習回数
        
        #各種データのファイル名設定
        self.path_Qcon = './nomura_dam/Qcon.csv'   #開度別コンジットゲート放流テーブル!!!
        self.path_Qcre = './nomura_dam/Qcre.csv'  #開度別クレストゲート放流テーブル!!!
        self.path_gensoku = './nomura_dam/gensoku.csv'    #放流の原則テーブル!!!
        self.path_HV = './nomura_dam/HV.csv'      #HVテーブル!!!
        self.path_jis = './nomura_dam/'      #実績!!!
        self.path_rain_p = './nomura_dam/'      #予測雨量!!!

        self.path_res= './nomura_dam/res/res.csv'                #利水計算結果
        self.path_grhp='./nomura_dam/res/praph.png'              #利水計算結果グラフ
        
        with open('./cnt.yaml', 'r+', encoding='utf-8') as f:
            inp = yaml.safe_load(f)
        self.past = inp['past']  
        flgs      = inp['train_flg']
        tmp_list = [item for item in flgs if item != 0]
        self.kozui_no = len(tmp_list)
        # ダム用inputファイル（/input/dam_observed.csv   /input/qincal.csv）の作成
        # DQN_input.csvより、flg1の洪水を抽出
        #self.damfile = DQNmakefile.makeinput()
            
        
        #self.damfile._dropflg()    #flgが1のものだけ
        # action_space, observation_space, reward_range を設定する
        # space = spaces.Discrete(8) # Set with 8 elements {0, 1, 2, ..., 7}
        # 範囲[low、high]の連続値、Float型のn次元配列。 gym.spaces.Box(low=-100, high=100, shape=(2,))
        # 範囲[0、n-1]の離散値、   Int型の数値。           gym.spaces.Discrete(4) 
        #self.action_space      = gym.spaces.Discrete(self.out_n)
        #self.action_space      = gym.spaces.MultiDiscrete([self.out_n, self.out_n])  
        #self.action_space      = gym.spaces.MultiDiscrete([[0,self.out_n], [0,self.out_n]])

        
        
        self.Qcon_table = pd.read_csv(self.path_Qcon)
        self.Qcre_table = pd.read_csv(self.path_Qcre)
        self.gensoku    = pd.read_csv(self.path_gensoku)
        self.HV_table   = pd.read_csv(self.path_HV)

        #コンジットゲート補間関数作成
        con_open_list     = self.Qcon_table.iloc[0,1:].values.tolist()
        con_H_list        = self.Qcon_table.iloc[1:,0].values.tolist()
        con_dataset       = self.Qcon_table.iloc[1:,1:].values.tolist()
        self.interpolation_con = RegularGridInterpolator((con_H_list, con_open_list), con_dataset, method='linear')
        self.con_open_max = max(con_open_list)

        #クレストゲート補間関数作成
        #cre_open_list     = self.Qcre_table.iloc[0,1:].values.tolist()
        #cre_H_list        = self.Qcre_table.iloc[1:,0].values.tolist()
        #cre_dataset       = self.Qcre_table.iloc[1:,1:].values.tolist()
        #self.interpolation_cre = RegularGridInterpolator((cre_H_list, cre_open_list), cre_dataset, method='linear')

        #HV補間関数作成
        HVH_list          = self.HV_table.iloc[0:,0].values.tolist()
        HVV_list          = (self.HV_table.iloc[0:,1]*1000).values.tolist()
        self.interpolation_HV  = interp1d(HVH_list, HVV_list, kind="linear")
        self.interpolation_VH  = interp1d(HVV_list, HVH_list, kind="linear")
        
        #self.rain_p_list = self.rain_p.values[int(self.past/3)][1:] #予測雨量

        #放流量分割数
        self.Qout_list_div = 20

        #acttableに従った行動の設定
        action = 0
        self.Qout=0.0

        #actionの取りうる値の空間
        self.action_space      = gym.spaces.Discrete(self.out_n)    # 行動の選択数
        print('self.action_space',self.action_space)
        #観測データの取りうる値の空間
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.features,), dtype=np.float32)  # 状態の上限、加減の設定
        print('self.observation_space',self.observation_space)
        #報酬の上限
        self.reward_range = [-100., 100.] # WIN or LOSE
        self._reset()           # ■笂ｵ洪水イベントを抽出
        action = 0              # acttable
        self._step(action)      # ■笂ｶ上記のactionで初回1ステップ実行
        
        
    """■笂ｵ諸々の変数を初期化して洪水イベントを取得"""
    def _reset(self):                                                                                    # 洪水イベントを抽出
        try_no = self.count % (self.kozui_no-1)
        self.jis        = pd.read_csv(self.path_jis+"jisseki_tmp_{}.csv".format(try_no))
        self.rain_p     = pd.read_csv(self.path_rain_p+"rain_tmp_p_{}.csv".format(try_no))
        self.Qin_peak = self.jis['Qin'].max()
        self.max_steps = len(self.jis)-504

        #目標水位（制限水位）
        date_ini = datetime.strptime(self.jis['datetime'].values[0],'%Y/%m/%d %H:%M') #初期日時
        month = date_ini.month
        day = date_ini.day
        if datetime(1900,month,day)>=datetime(1900,7,15) and datetime(1900,month,day)<=datetime(1900,10,15):
            self.Hs = 166.2
        else:
            self.Hs = 169.4
        self.Vs = self.interpolation_HV(self.Hs)

        #使用予測雨量算定用
        hour = date_ini.hour
        minute = date_ini.minute
        self.def_rain = 18 - (hour%3)*6 + int(minute/10)

        self.steps = self.past * 6 - 1 #6時間step数
        self.Qout     = self.jis['Qout'].values[self.steps]
        self.Qin      = self.jis['Qin'].values[self.steps]
        #self.H        = self.jis['H'].values[self.steps]
        #self.H        = self.jis['H'].values[self.steps]
        self.H        = self.Hs - random.uniform(0, 4) #(self.count % 5)
        self.Hini     = self.H
        self.V        = self.interpolation_HV(self.H)
        self.Qout_pre = self.Qout
        self.Qin_pre  = self.Qin
        self.H_pre    = self.H
        self.V_pre    = self.V
        self.Vini     = self.V
        self.con_open = 0.0
        self.peak_flg = 0
        self.Hs_flg   = 0
        #self.cre_open = 0.0
        self.episode_reward = 0

        #output用
        self.output_Qin_o  = []
        self.output_Qout_o = []
        self.output_H_o    = []
        self.output_Qout_p = []
        self.output_H_p    = []
        self.output_time   = []
        self.output_rain   = []
               
        self.reward   = 0
        self.done     = False
        return self._observe()


    """■笂ｶ1ステップ進める処理を記述。戻り値は observation, reward, done(ゲーム終了したか)
        強化学習はここ_stepを繰り返す。ただしイベントが終われば_resetで次のイベントに進む"""
    def _step(self, action):# 初回1ステップ実行
        
        #現時刻のダム諸量
        self.H_pre = self.H  
        self.V_pre =  self.interpolation_HV(self.H)
        self.Qout_pre = self.Qout
        self.Qin_pre  = self.Qin
        self.rate_Qout_max = self.Qout_pre * 2.0

        #acttableに従った行動の設定
        self.action  = action
        if self.Hs <= self.H:
            if self.Qin >= 300.0:
                if self.Qout >= 300.0:
                    self.Qout = min((self.Qin-300.0)*0.790 + 300.0, self.Qin)
                else:
                    Qout_gensoku_max = self.gensoku[(self.gensoku['Qout_d']<=self.Qout) & (self.gensoku['Qout_u']>self.Qout)]['dQout'].values[0]
                    self.Qout = min(self.Qin, Qout_gensoku_max+self.Qout, 299.99)
            else:
            #    if self.Qout > 300.0:
            #        self.Qout = min((self.Qin-300.0)*0.790 + 300.0, self.Qin)
            #    else:
            #        Qout_gensoku_max = self.gensoku[(self.gensoku['Qout_d']<=self.Qout) & (self.gensoku['Qout_u']>self.Qout)]['dQout'].values[0]
            #        self.Qout = min(self.Qin, Qout_gensoku_max+self.Qout, 299.99)
                self.Qout = self.Qin
        else:
            #次ステップ放流可能リスト作成
            if self.Qout >= 300:
                self.Qout = 300.0
            else:
                Qout_gensoku_max = self.gensoku[(self.gensoku['Qout_d']<=self.Qout) & (self.gensoku['Qout_u']>self.Qout)]['dQout'].values[0]
                #Qout_list_min    = max(self.Qb, min(self.Qin, self.interpolation_con([[self.H, self.con_open]])[0], self.Qout)) #下げる場合考慮したほうがいい？
                Qout_list_min    = min(self.Qin, max(self.Qb, self.Qout))
                #Qout_list_max    = min(self.Qin, self.interpolation_con([[self.H, min(self.con_open+self.Vcon*10,self.con_open_max)]])[0], Qout_gensoku_max+self.Qout)
                Qout_list_max    = min(self.rate_Qout_max, max(Qout_list_min, min(self.Qin, self.interpolation_con([[self.H, min(self.con_open+self.Vcon*10,self.con_open_max)]])[0], Qout_gensoku_max+self.Qout)))
                self.Qout_list   = np.linspace(Qout_list_min, Qout_list_max, self.Qout_list_div)
                self.Qout        = self.Qout_list[self.action]
        
        #計算諸量
        if self.Qin_peak == self.Qin:
            self.peak_flg = 1
        self.Qin = self.jis['Qin'].values[self.steps+1]
        self.rain_p_list = self.rain_p.values[int((self.steps-self.def_rain)/18)] #予測雨量
        self.con_open = np.linspace(self.con_open, min(self.con_open+self.Vcon*10,self.con_open_max), self.Qout_list_div)[self.action]

        #次ステップ水位算定
        self.V     = self.V_pre + (self.Qin - self.Qout) * 600
        self.H     =  self.interpolation_VH(self.V)

        #if (self.steps>=self.max_steps-1) or ((self.peak_flg == 1) and (self.Hs <= self.H)):
        #if (self.steps>=self.max_steps-1) or ((self.Qin < self.Qout) and (self.Hs <= self.H) and (self.Qin<300)) or ((self.Qin <= self.Qout) and (self.peak_flg == 1)):
        if (self.steps>=self.max_steps-1) or ((self.Qin > 300.0) and (self.Hs <= self.H)) or ((self.Qin <= self.Qout) and (self.peak_flg == 1)):
            self.done = 1

        if self.Hs <= self.H and self.Hs_flg == 0:
            self.Hs_flg = 1
        elif self.Hs_flg == 1:
            self.Hs_flg = 2
        else:
            self.Hs_flg = 0

              
        #output用
        self.output_Qin_o.append(self.Qin)
        self.output_Qout_o.append(self.jis['Qout'].values[self.steps])
        self.output_H_o.append(self.jis['H'].values[self.steps])
        self.output_H_p.append(self.H)
        self.output_time.append(self.jis['datetime'].values[self.steps])
        if ((self.Qin <= self.Qout) and (self.Hs <= self.H)) or ((self.Qin <= self.Qout) and (self.peak_flg == 1)):
            self.output_Qout_p.append(self.Qin)
        else:
            self.output_Qout_p.append(self.Qout)
        self.output_rain.append(self.jis['r'].values[self.steps])

        #self.outfile_dqn.reset_index(drop=True,inplace=True)
        #self.outfile = self.outfile_dqn
        # ③ターン数を進める
        self.observation = self._observe()     #状態の算定
        self.reward      = self._get_reward()  #報酬の算定
        self.done        = self._is_done()     #終わったら1
        self.steps += 1
        """■笂ｷ笂ｸ笂ｹの繰り返し"""
        return self.observation, self.reward, self.done, {}


    # 実行時のコマンドラインへの出力処理
    # human の場合はコンソールに出力。ansiの場合は StringIO を返す
    def _render(self, mode='human', close=False):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write(f'steps：{"{:0=4}".format(self.steps)}、reward：{"{:05.1f}".format(self.reward)}、action：{"{:0=2}".format(self.action)}\n')#, observe:{self.observe}\n')
        return outfile

    def _close(self):
        pass

    def _seed(self, seed=None):
        pass

    """■笂ｷ状態の算定"""
    def _observe(self):
        # 状態読み込み（流入量0hr、雨量0hr、貯水位0hr） 正規化の実施
        # ■DQN時の状態（初回は操作規則と同じ）
        self.obs_dqn = np.hstack([self.jis['Qin'].values[self.steps]/self.Qinmax,#ダム流入量
            sum(np.array(self.jis['r'].values[self.steps-self.past:self.steps])/self.Rmax),#流域平均雨量
            np.array([self.Hs-self.H]),#ダム貯水位
            np.array([self.Qout])/self.Qinmax,#ダム放流量
            #np.array(self.rain_p.values[int((self.steps-self.def_rain)/18),1:])/self.Rmax])#予測雨量
            #np.array(self.rain_p.values[int((self.steps-self.def_rain)/18),1:]/self.Rmax*random.uniform(0.9,1.1))])#予測雨量
            #np.array(self.rain_p.values[int((self.steps-self.def_rain)/18),1:])/self.Rmax])#予測雨量
            np.array(self.jis['r'].values[self.steps+18])/self.Rmax,
            np.array(self.jis['r'].values[self.steps+36])/self.Rmax,
            np.array(self.jis['r'].values[self.steps+54])/self.Rmax,
            np.array(self.jis['r'].values[self.steps+72])/self.Rmax,
            np.array(self.jis['r'].values[self.steps+90])/self.Rmax,
            np.array(self.jis['r'].values[self.steps+108])/self.Rmax,
            np.array(self.jis['r'].values[self.steps+126])/self.Rmax,
            np.array(self.jis['r'].values[self.steps+144])/self.Rmax,
            np.array(self.jis['r'].values[self.steps+162])/self.Rmax,
            np.array(self.jis['r'].values[self.steps+180])/self.Rmax,
            np.array(self.jis['r'].values[self.steps+198])/self.Rmax,
            np.array(self.jis['r'].values[self.steps+216])/self.Rmax,
            np.array(self.jis['r'].values[self.steps+252])/self.Rmax,
            np.array(self.jis['r'].values[self.steps+288])/self.Rmax,
            np.array(self.jis['r'].values[self.steps+324])/self.Rmax,
            np.array(self.jis['r'].values[self.steps+360])/self.Rmax,
            np.array(self.jis['r'].values[self.steps+396])/self.Rmax,
            np.array(self.jis['r'].values[self.steps+432])/self.Rmax,
            np.array(self.jis['r'].values[self.steps+468])/self.Rmax,
            np.array(self.jis['r'].values[self.steps+504])/self.Rmax])#予測雨量（実績）
        return self.obs_dqn
    
    """■笂ｸ報酬を返却"""
    def _get_reward(self):
        reward_tmp = 0
        #if self.done != 1:  #放流量増加率による報酬
        #    self.rate_Qout = self.Qout / self.Qout_pre
        #    if self.rate_Qout >= 2.0:
        #        reward_tmp -= pow(self.rate_Qout,2.0)/10.0
        #else:  #回復量による報酬
        #    if self.H_pre >= self.Hs:
        #        reward_tmp -= (self.H_pre-self.Hs) 
        #    elif self.H_pre < self.Hs:
        #        reward_tmp -= (self.Hs-self.H_pre)

        #self.rate_Qout = self.Qout / self.Qout_pre
        #if self.rate_Qout >= 2.0:
        #    reward_tmp -= pow(self.rate_Qout,2.0)/10.0
        #if self.H_pre >= self.Hs:
        #    reward_tmp -= (self.H_pre-self.Hs)/len(self.jis)
        #elif self.H_pre < self.Hs:
        #    reward_tmp -= (self.Hs-self.H_pre)/len(self.jis)
        
        #self.rate_Qout = self.Qout / self.Qout_pre
        #if self.rate_Qout >= 2.0:
        #    reward_tmp -= pow(self.rate_Qout,2.0)/25
        #if self.H_pre >= self.Hs:
        #    reward_tmp -= (self.H_pre-self.Hs)
        #elif self.H_pre < self.Hs:
        #    reward_tmp += (self.H-self.H_pre)*25
        #if self.Hs_flg == 1:
        #    if self.Qin==self.Qout:
        #        reward_tmp += 100
        #    else:
        #        reward_tmp -= 100
        #if self.done==1:
        #    if self.H <= self.Hs:
        #        reward_tmp -= 100
        
        self.rate_Qout = self.Qout / self.Qout_pre
        if self.rate_Qout > 2.0:
            reward_tmp -= (self.rate_Qout-2.0)/10
        #if self.H_pre >= self.Hs:
        #    reward_tmp -= (self.H-self.H_pre)
        #if self.H_pre < self.Hs:
        #    reward_tmp += (self.H-self.H_pre)*25
        
        #if self.H >= self.Hs:
        #    reward_tmp -= 5
        #elif self.H < self.Hs and self.H >= self.Hs+0.05:
        #    reward_tmp += 1
        #elif self.H < self.Hs+0.5 and self.H >= self.Hs+0.1:
        #    reward_tmp += 0.11
        
        #Hr = (self.Hs - self.H) / (self.Hs - self.Hini)
        #Qr = self.Qout / self.Qin
        #re = Hr - Qr + 1
        #if re > 1 and self.Hs>self.H:
        #    reward_tmp += abs(1 - re) * 10
        #elif re <= 1 and self.Hs>self.H:
        #    reward_tmp += re * 10
        
        if self.H >= self.Hs and self.Qin <= 300:
            reward_tmp -= 10
        if self.H < self.Hs:
            #reward_tmp += pow((self.Hs - self.Hini) / (self.Hs - self.H), 2)
            #reward_tmp += abs(1 / (self.Hs - self.H))
            reward_tmp += abs(1 / (self.Hs - self.H))
        
        self.episode_reward += reward_tmp
        return reward_tmp

    
    """■笂ｹ終了判定とグラフ化"""
    def _is_done(self):
        if (self.done==1):

#            self.obs_all       = []
#            self.obs_all.append([self.output_time,
#                self.output_H_o,
#                self.output_H_p,
#                self.output_Qin_o,
#                self.output_Qout_o,
#                self.output_Qout_p,
#                self.Qout_list,
#                self.output_rain])
#
#            o_mx = 0
#            self.qout_o_tmp = copy.deepcopy(self.output_Qout_o)
#            for n in range(len(self.output_Qout_o)):
#                if n == 0:
#                    self.qout_o_tmp[0] = np.nan
#                else:
#                    if self.output_Qout_o[n]/self.output_Qout_o[n-1] > 2.0:
#                        self.qout_o_tmp[n] = self.output_Qout_o[n]
#                        self.qout_o_tmp[n-1] = self.output_Qout_o[n-1]
#                    else:
#                        self.qout_o_tmp[n] = np.nan
#                    o_mx = max(o_mx, self.output_Qout_o[n]/self.output_Qout_o[n-1])
#            self.qout_p_tmp = []
#            p_mx = 0
#            self.qout_p_tmp = copy.deepcopy(self.output_Qout_p)
#            for n in range(len(self.output_Qout_p)):
#                if n == 0:
#                    self.qout_p_tmp[0] = np.nan
#                else:
#                    if self.output_Qout_p[n]/self.output_Qout_p[n-1] > 2.0:
#                        self.qout_p_tmp[n] = self.output_Qout_p[n]
#                        self.qout_p_tmp[n-1] = self.output_Qout_p[n-1]
#                    else:
#                        self.qout_p_tmp[n] = np.nan
#                    p_mx = max(p_mx, self.output_Qout_p[n]/self.output_Qout_p[n-1])
#            
#            self.df_obs_all = pd.DataFrame(self.obs_all, columns=['date', 'H_org', 'H_dqn', 'QinDam', 'QoutDam_org', 'QoutDam_dqn', 'out_list', 'rain'])
#            self.df_obs_all = self.df_obs_all.set_index('date')
#            self.df_obs_all.to_csv("check.csv")
#            #self.df_obs_all.to_csv(f'./{self.outf}//csv_{self.infilename}_reward{"{:05.1f}".format(self.reward)}_調節開始流量{"{:04}".format(self.kaishiQ)}_増放流開始水位{"{:.2f}".format(self.tadashiH)}_org被害額{"{:06.0f}".format(self.Qout_damage_org1+self.Qout_damage_org2)}_dqn被害額{"{:06.0f}".format(self.Qout_damage_dqn1+self.Qout_damage_dqn2)}.csv')
#            #self.df_obs_all = self.df_obs_all[35:35+self.steps+288]
#                                    
#            #jisseki
#            fig = plt.figure(figsize=(16.0, 8.0))
#            grid = plt.GridSpec(4, 2, wspace=0.2, hspace=0.1)
#            ax1 = plt.subplot(grid[1:, 0])
#            ln1=ax1.plot(self.df_obs_all.index.values[0], self.df_obs_all['QinDam'].values[0], label='ダム流入量Q')
#            ln1=ax1.plot(self.df_obs_all.index.values[0], self.df_obs_all['QoutDam_org'].values[0], label='ダム放流量Q(実績)')
#            ln1=ax1.plot(self.df_obs_all.index.values[0], self.qout_o_tmp, color='red', label='放流量２倍以上')
#            #ln1.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=12)
#            ax2 = ax1.twinx()
#            ln2=ax2.plot(self.df_obs_all.index.values[0], self.df_obs_all['H_org'].values[0],label='ダム水位H(実績)',color='black', zorder=1)
#            ln2=ax2.plot([self.df_obs_all.index.values[0][0],self.df_obs_all.index.values[0][-1]], [self.Hs, self.Hs],label='洪水貯留準備水位',color='peachpuff')
#            h1, l1 = ax1.get_legend_handles_labels()
#            h2, l2 = ax2.get_legend_handles_labels()
#            ax1.set_ylim(0, 500, 100)                                                            # y軸範囲の指定
#            ax1.yaxis.set_ticks(np.arange(0, 500+0.0001, 100))                                 # y軸範囲の指定
#            ax1.xaxis.set_ticks(np.arange(0, len(self.output_Qout_p), step=72))
#            ax2.set_ylim(155, 180, 5)                                                            # y軸範囲の指定
#            ax2.yaxis.set_ticks(np.arange(155, 180+0.0001, 5))                                 # y軸範囲の指定
#            ax1.set_xlabel('time')
#            ax1.set_ylabel(r'流量Q(m3/s)')
#            ax1.grid(True)
#            ax2.set_ylabel(r'水位H(m)')
#            ax1.legend(h1+h2, l1+l2, bbox_to_anchor=(0, 1), loc='upper left', borderaxespad=0)
#            #handler2, label2 = ax1.get_legend_handles_labels()
#            #ax1.legend(handler2, label2, prop=fp2)                                                               # 凡例の表示
#            #ax1.xaxis.set_major_locator(mdates.DayLocator(bymonthday=None, interval=1, tz=None))
#            #ax1.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 6), tz=None))
#            #axs[1].xaxis.set_minor_locator(mdates.DayLocator(bymonthday=None, interval=1, tz=None))
#            #ax1.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m-%d"))
#            #ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
#            ax1.axes.xaxis.set_visible(True)
#            ax1.xaxis.grid(b=True,which='major',color = "gray", alpha = 0.8)
#            ax1.xaxis.grid(b=True,which='minor',color = "lightgray", alpha = 0.8, linestyle = "--")
#            ax1.yaxis.grid(b=True,which='major',color = "gray", alpha = 0.8)
#
#            fig.text(0.47, 0.67, f'放流量最大増加率：{"{:.2f}".format(o_mx)}', ha='right', va='top', color="red")
#            
#            #yosoku
#            ax3 = plt.subplot(grid[1:, 1])
#            ln1=ax3.plot(self.df_obs_all.index.values[0], self.df_obs_all['QinDam'].values[0], label='ダム流入量Q')
#            ln1=ax3.plot(self.df_obs_all.index.values[0], self.df_obs_all['QoutDam_dqn'].values[0], label='ダム放流量Q(AI操作)')
#            ln1=ax3.plot(self.df_obs_all.index.values[0], self.qout_p_tmp, color='red', label='放流量２倍以上')
#            #ln1.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=12)
#            ax4 = ax3.twinx()
#            ln2=ax4.plot(self.df_obs_all.index.values[0], self.df_obs_all['H_dqn'].values[0],label='ダム水位H(AI操作)',color='black')
#            ln2=ax4.plot([self.df_obs_all.index.values[0][0],self.df_obs_all.index.values[0][-1]], [self.Hs, self.Hs],label='洪水貯留準備水位',color='peachpuff')
#            h1, l1 = ax3.get_legend_handles_labels()
#            h2, l2 = ax4.get_legend_handles_labels()
#            ax3.set_ylim(0, 500, 100)                                                            # y軸範囲の指定
#            ax3.yaxis.set_ticks(np.arange(0, 500+0.0001, 100))                                 # y軸範囲の指定
#            #ax3.xaxis.set_ticks(np.arange(0, len(self.output_Qout_p), step=72), np.arange(0, int(len(self.output_Qout_p)/72)+1, step=1))
#            ax3.xaxis.set_ticks(np.arange(0, len(self.output_Qout_p), step=72))
#            ax4.set_ylim(155, 180, 5)                                                            # y軸範囲の指定
#            ax4.yaxis.set_ticks(np.arange(155, 180+0.0001, 5))                                 # y軸範囲の指定
#            ax3.set_xlabel('time')
#            ax3.set_ylabel(r'流量Q(m3/s)')
#            ax3.grid(True)
#            ax4.set_ylabel(r'水位H(m)')
#            ax3.legend(h1+h2, l1+l2, bbox_to_anchor=(0, 1), loc='upper left', borderaxespad=0)
#            #handler2, label2 = ax3.get_legend_handles_labels()
#            #ax3.legend(handler2, label2, prop=fp2)                                                               # 凡例の表示
#            #ax3.xaxis.set_major_locator(mdates.DayLocator(bymonthday=None, interval=1, tz=None))
#            #ax3.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 6), tz=None))
#            #axs[1].xaxis.set_minor_locator(mdates.DayLocator(bymonthday=None, interval=1, tz=None))
#            #ax3.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m-%d"))
#            #ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
#            ax3.axes.xaxis.set_visible(True)
#            ax3.xaxis.grid(b=True,which='major',color = "gray", alpha = 0.8)
#            ax3.xaxis.grid(b=True,which='minor',color = "lightgray", alpha = 0.8, linestyle = "--")
#            ax3.yaxis.grid(b=True,which='major',color = "gray", alpha = 0.8)
#
#            fig.text(0.90, 0.67, f'放流量最大増加率：{"{:.2f}".format(p_mx)}', ha='right', va='top', color="red")
#            
#            #fig.text(0.9, 0.94, f'調節開始流量：{self.kaishiQ}(m3/s)  増放流開始水位：{"{:.2f}".format(self.tadashiH)}(m)', ha='right', va='top')
#            #fig.text(0.9, 0.91, f'①被害軽減額(ダム直下～小田川合流)：{"{:06.0f}".format((self.Qout_damage_org1) - (self.Qout_damage_dqn1))}(百万円)  ②被害軽減額(小田川合流～河口)：{"{:06.0f}".format((self.Qout_damage_org2) - (self.Qout_damage_dqn2))}(百万円)  被害軽減額(①＋②)：{"{:06.0f}".format((self.Qout_damage_org1+self.Qout_damage_org2) - (self.Qout_damage_dqn1+self.Qout_damage_dqn2))}(百万円)', ha='right', va='top')
#            
#            #ax5 = plt.subplot(grid[0, 0])
#            #ln5 = ax5.bar(self.df_obs_all.index.values[0], self.df_obs_all['rain'].values[0], width=1.0, label='rain')
#            #ax5.set_ylim(0, 50, 10)                                                            # y軸範囲の指定
#            #ax5.yaxis.set_ticks(np.arange(0, 50+0.0001, 10))                                 # y軸範囲の指定
#            #ax5.axes.xaxis.set_visible(False)
#            #ax5.invert_yaxis()
#            #ax6 = plt.subplot(grid[0, 1])
#            #ln6 = ax6.bar(self.df_obs_all.index.values[0], self.df_obs_all['rain'].values[0], width=1.0, label='rain')
#            #ax6.set_ylim(0, 50, 10)                                                            # y軸範囲の指定
#            #ax6.yaxis.set_ticks(np.arange(0, 50+0.0001, 10))                                 # y軸範囲の指定
#            #ax6.axes.xaxis.set_visible(False)
#            #ax6.invert_yaxis()
#            
#            fig.text(0.90, 0.75, f'報酬：{"{:.2f}".format(self.episode_reward)}', ha='right', va='top', color="red")
#
#            plt.tight_layout()
#            plt.savefig(f'./out/tmp{self.count}.png', dpi=300, bbox_inches="tight", pad_inches=0.05)
#            plt.close()
#
#
#            #if os.path.exists(f'./{self.outf}/out.csv'):
#            #    with open(f'./{self.outf}/out.csv', 'a', newline='') as f:
#            #        writer = csv.writer(f)
#            #        out=[datetime.now(), self.infilename, self.reward, self.H_org_max1, self.H_dqn_max1, self.kaishiQ, self.tadashiH, self.Qout_org_max1, self.Qout_damage_org1, self.Qout_org_max2, self.Qout_damage_org1+self.Qout_damage_org2, self.Qout_dqn_max1, self.Qout_damage_dqn1, self.Qout_dqn_max2, self.Qout_damage_dqn1+self.Qout_damage_dqn2, ((self.Qout_damage_org1) - (self.Qout_damage_dqn1)), ((self.Qout_damage_org2) - (self.Qout_damage_dqn2)), ((self.Qout_damage_org1+self.Qout_damage_org2) - (self.Qout_damage_dqn1+self.Qout_damage_dqn2))]
#            #        writer.writerow(out)
#            #else:
#            #    with open(f'./{self.outf}/out.csv', 'w', newline='') as f:
#            #        writer = csv.writer(f)
#            #        out=[datetime.now(), self.infilename, self.reward, self.H_org_max1, self.H_dqn_max1, self.kaishiQ, self.tadashiH, self.Qout_org_max1, self.Qout_damage_org1, self.Qout_org_max2, self.Qout_damage_org1+self.Qout_damage_org2, self.Qout_dqn_max1, self.Qout_damage_dqn1, self.Qout_dqn_max2, self.Qout_damage_dqn1+self.Qout_damage_dqn2, ((self.Qout_damage_org1) - (self.Qout_damage_dqn1)), ((self.Qout_damage_org2) - (self.Qout_damage_dqn2)), ((self.Qout_damage_org1+self.Qout_damage_org2) - (self.Qout_damage_dqn1+self.Qout_damage_dqn2))]
#            #        writer.writerow(['date', '洪水No', '報酬', '貯水位(操作規則)', '貯水位(AI操作)', '最適洪水調節開始流量', '最適増放流開始水位', 'ダムピーク放流量(操作規則時)', 'ダム下流被害額(操作規則)', '大津ピーク流量(操作規則時)', '大津被害額(操作規則)','ダムピーク放流量(AI操作時)', 'ダム下流被害額(AI操作)','大津ピーク流量(AI操作時)', '大津被害額(AI操作)', '①被害軽減額(ダム直下～小田川合流)', '②被害軽減額(小田川合流～河口)', '被害軽減額(①＋②)'])
#            #        #out=[datetime.now(), self.infilename, self.reward, self.kaishiQ, self.tadashiH, self.Qout_damage_org1, self.Qout_damage_dqn1, self.Qout_damage_org1+self.Qout_damage_org2, self.Qout_damage_dqn1+self.Qout_damage_dqn2, ((self.Qout_damage_org1) - (self.Qout_damage_dqn1)), ((self.Qout_damage_org2) - (self.Qout_damage_dqn2)), ((self.Qout_damage_org1+self.Qout_damage_org2) - (self.Qout_damage_dqn1+self.Qout_damage_dqn2)), self.Qout_org_max1, self.Qout_dqn_max1, self.Qout_org_max2, self.Qout_dqn_max2]
#            #        #writer.writerow(['date', '洪水No', '報酬', '洪水調節開始流量', '増放流開始水位', 'ダム下流被害額(操作規則)', 'ダム下流被害額(AI操作)', '大津被害額(操作規則)', '大津被害額(AI操作)', '①被害軽減額(ダム直下～小田川合流)', '②被害軽減額(小田川合流～河口)', '被害軽減額(①＋②)', 'ダム放流量(操作規則時)','ダム放流量(AI操作時)','大津流量(操作規則時)','大津流量(AI操作時)'])
#            #        writer.writerow(out)
#            #self.steps    = 1

            self.count += 1
            return True
        else:
            return False


#テスト計算用のインスタンス
class damtest(dam):
    def __init__(self):
        super().__init__()
        self.outf      = '01_Output_test' # 出力フォルダ

# %%
