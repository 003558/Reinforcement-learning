# %% [markdown]

# # **強化学習DQNモデル**
# CTI kokubunkenISP | Kensuke Matsuda  
# =====================【gym・tensorflow・kerasによる強化学習モデル】====================  
# ==    ①プログラム内容     ：DNNを活用したQ学習を実施する。       
# ==    ②プログラミング言語 ：Python3.6.7                       
# ==    ③AIライブラリ       ：動作確認Tensorflow1.14.0  keras2.2.4  keras-rl0.4.2  gym0.18 
# ==                          (Tensorflow2とkeras-rl＆gymの相性は悪そう。。。)
# ==    ④解析モデル         ：DQN、DoubleDQN(DDQN)、DuelingDQN
# =====================================================================================  
# ■計算の流れ  
#     - 1 Experience Replay用のメモリ確保  
#     - 2 行動ポリシーの設定        
#     - 3 エージェントの交差検証学習      
#        ⇒  環境モデルの読み込みと深層学習部分の構築
#        ⇒  エージェントの構築        
#        ⇒  エージェントのコンパイル
#           ⇒  エージェントの学習
#           ⇒  エージェントの検証        
# ■更新履歴               
#     - 21.01 DQN実装             
#     - 21.02 交差検証仕様の実装
#     - dummy  
#     - dummy  
#     - dummy  

import yaml
import dammodel
import datetime
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
import gym
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import shutil
random.seed(1)
#  plot_model：モデルの可視化
from keras.utils.vis_utils import plot_model
#from dammodel import DQNmakefile
# plot_modelのpy_dotエラー対策用
try:
    # pydot-ng is a fork of pydot that is better maintained.
    import pydot_ng as pydot
except ImportError:
    # pydotplus is an improved version of pydot
    try:
        import pydotplus as pydot
    except ImportError:
        # Fall back on pydot if necessary.
        try:
            import pydot
        except ImportError:
            pydot = None
# 実行時の警告を表示しない
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.simplefilter('ignore')



##########################################
#Experience Replay用のメモリ
##########################################
"""
パラメータ	                必須            意味
limit	                    ○              メモリ（collections.deque）の上限サイズ。action（行動）、reward（報酬）、observations（観測）のそれぞれにdequeが用意される。上限を超えると古いデータが書き換えられていく
window_length	            ○              観測を何個連結して処理するか。例えば時系列の複数の観測をまとめて1つの状態とする場合に利用。
ignore_episode_boundaries                  エピソードの境界を無視するかどうか（別のエピソードの経験を利用する場合はTrue）。
"""
from rl.memory import SequentialMemory
#memory = SequentialMemory(limit=50000, window_length=1, ignore_episode_boundaries=False)
#エピソード（洪水）は複数利用するためTrue
memory = SequentialMemory(limit=500000, window_length=1, ignore_episode_boundaries=True)


##########################################
#行動ポリシーの設定
##########################################
"""
LinearAnnealedPolicyは引数に指定したポリシーについて、attrに指定したパラメータをvalue_maxからvalue_minまでnb_steps回目までに線形に変化させるというものです。
value_testについてはテスト時の固定的なパラメータ値となります。
BoltzmannGumbelQPolicyについてはトレーニング時のみに利用できるポリシーとのことです。
パラメータ	必須	意味
eps		           ランダムな行動を選択する確率（イプシロン）。値が大きいほど探索に重きを置いて行動を決定する。
tau		           ボルツマン分布を利用したソフトマックス手法のQ値を割る値。
clip		       Q値をtauで割った後にクリップする範囲。
C		           ガンベル分布の乗数betaの計算式beta = C/np.sqrt(action_counts)で利用。
"""
#from rl.policy import LinearAnnealedPolicy
#from rl.policy import SoftmaxPolicy
from rl.policy import EpsGreedyQPolicy
#from rl.policy import GreedyQPolicy
#from rl.policy import BoltzmannQPolicy
#from rl.policy import MaxBoltzmannQPolicy
#from rl.policy import BoltzmannGumbelQPolicy

#policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05, nb_steps=1000)
#policy = SoftmaxPolicy()
policy = EpsGreedyQPolicy(eps=0.1)
#policy = GreedyQPolicy()
#policy = BoltzmannQPolicy(tau=1., clip=(-500., 500.))
#policy = MaxBoltzmannQPolicy(eps=.1, tau=1., clip=(-500., 500.))
#policy = BoltzmannGumbelQPolicy(C=1.0)
#%%

def make_env_model_agent(ENV_NAME):
    ##########################################
    #環境モデルの読み込みと深層学習部分の構築
    ##########################################
    #train用のDQN_input_run.csv作成
    env = gym.make(ENV_NAME)
    nb_actions = env.action_space.n # 行動の数
    #print('env.action_space.n', env.action_space.n)
    print('env.observation_space.shape', env.observation_space.shape)
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(300))
    model.add(Activation('relu'))
    model.add(Dense(300))
    model.add(Activation('relu'))
    model.add(Dense(300))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))

    ##########################################
    #エージェントの構築  
    ##########################################
    """
    パラメータ	        必須	     意味
    nb_actions	          ○         行動空間の次元数。環境により決まるためこの段階で変更することはない。
    memory	              ○         事前に作ったExperience Replay用のメモリ。
    gamma		                    行動価値関数の式に出てくる割引率。将来の価値をどれくらい考慮するかを決める（値が大きいほど将来の価値を大きく反映する）。
                                    1未満の値を設定する。
    batch_size		                学習のバッチサイズ
    nb_steps_warmup	                ウォームアップステップ数。学習の初期は安定しないため、学習率を徐々に上げていく期間。
    train_interval		            トレーニングのインターバル。train_intervalステップ毎に学習が実行される。
    memory_interval	                Experience Replay用のメモリにデータを貯めるインターバル。memory_intervalステップ毎にメモリに経験が保存される。
    target_model_update	            0以上の値。1未満の値の場合はSoft updateと呼ばれ、(1 - target_model_update) * old + target_model_update * newの式で重みが更新される。
                                    1以上の値の場合はHard updateと呼ばれ、int(target_model_update)ステップごとに重みが完全に更新される。
    delta_range		                非推奨のパラメータ。delta_clipを代わりに使ってくれとのこと。もし値が設定された場合はdelta_clip = delta_range[1]となる。
    delta_clip		                Huber損失のデルタ値。
    custom_model_objects            ターゲットモデルを生成する際のオプション。詳細はこちらを参照。
    model	             ○	        事前に作ったDNNのモデル。
    policy		                    行動ポリシー。設定しない場合はEpsGreedyQPolicyが適用される。
    test_policy		                テスト時の行動ポリシー。設定しない場合はGreedyQPolicyが適用される。
    enable_double_dqn	            DoubleDQN(DDQN)を適用するかどうか。
    enable_dueling_network          DuelingDQNを適用するかどうか。Trueにすると出力層の前の層にDueling Networkを挿入してくれる。
    dueling_type		            Duelingネットワークのタイプ（enable_dueling_networkがTrueのときに使用）。
                                    行動価値関数 Q(s,a)を状態価値関数V(s)とアドバンテージA(s,a)から計算する際の計算方法を決める。avg、max、naiveが選択できるが、
                                    avgが推奨されているとのこと。具体的な式は以下を参照。
    """
    from rl.agents.dqn import DQNAgent
    agent = DQNAgent(nb_actions=nb_actions, memory=memory, gamma=1.0, batch_size=32, nb_steps_warmup=1000,
                    train_interval=1, memory_interval=1, target_model_update=10000,
                    delta_range=None, delta_clip=np.inf, custom_model_objects={}, 
                    model=model, policy=policy, test_policy=None, enable_double_dqn=True,
                    enable_dueling_network=True, dueling_type='avg')

    ##########################################
    #エージェントのコンパイル
    ##########################################
    """
    パラメータ     必須        意味
    optimizer        ○        DNN重み更新の最適化手法。利用可能な手法はKerasのoptimizers。更新式はこの記事に整理されている。Adamであれば学習率(Learning rate: lr)は1e-3が論文で推奨されているらしい
    metrics		              評価関数のリストで、Kerasの評価関数が使える。または自作のlambda y_true, y_pred: metricの形式で指定できる。指定の有無にかかわらずmean_qという評価関数が追加される。
    """
    from keras.optimizers import Adam
    agent.compile(optimizer=Adam(lr=1e-3), metrics=['mse'])
    try:plot_model(model, to_file="agent.png", show_shapes=True)  #モデル構造の保存
    except:pass

    return env, agent





##########################################
#エージェントの学習
##########################################
"""
パラメータ	      必須	      意味
env       	        ○	     事前に作った強化学習環境。
nb_steps	        ○	     シミュレーションのステップ数。１ステップは、１回の観測、行動、報酬の獲得。
action_repetition	         行動の繰り返し回数。2以上にすると、観測なしに同じ行動をaction_repetition回繰り返す。
callbacks		             コールバックのリスト。rl.callbacks.Callbackを継承したコールバックが使える。ステップの開始・終了、エピソードの開始・終了、アクションの開始・終了がイベントとなる。
verbose		                 ロギングのモード。0でロギングなし、1でlog_intervalステップ毎にロギング、2でエピソード毎にロギング。具体的にはrl.callbacks.TrainIntervalLoggerが呼ばれる。
visualize		             可視化のEnable。具体的にはrl.callbacks.Visualizerが呼ばれる。環境でrender(mode='human'))が実装されていないと例外が起きる。
nb_max_start_steps	         この値を最大値とする乱数値回目からステップを開始する。シミュレーションの開始位置を変動させたい場合に利用。この間はExperience Replayに記録されない。
start_step_policy	         nb_max_start_stepsが0でない場合に有効。nb_max_start_stepsまでの間にどのような行動ポリシーとするかをlambda observation: actionの形式で指定。Noneの場合はランダムに行動が選択される。
log_interval		         ロギングするステップ間隔。
nb_max_episode_steps         1エピソードにおける最大のステップ数。
このパラメータを見ればわかるように、Keras-RLではエピソード数を決めるのではなく、
ステップ数と1エピソードにおける最大のステップ数で「最小のエピソード数」が決まります。
つまり、nb_steps / nb_max_episode_stepsが最小エピソード数となります。
例えばnb_steps=3000、nb_maxepisode_steps=300の場合は、最低3000/300=10回のエピソードが繰り返されます。
ただし、nb_max_episode_stepsよりも少ないステップ数でエピソードが終了することもあるので、
エピソード数は最大でnb_steps回となります（ほぼありえませんが、毎回1ステップ目でシミュレーションが終了する場合）。
"""

# 交差検証開始
ENV_NAME_TEST  = 'damtest-v0'
ENV_NAME       = 'dam-v0'
all_kouzui_flg = 0# 0で交差検証、1で全洪水対象
train_flg      = 1# 1でtrainあり 0で検証のみ
#nb_steps=120000
#nb_steps=8000

#学習期間

with open('./cnt.yaml', 'r+', encoding='utf-8') as f:
    inp = yaml.safe_load(f)
past       = inp['past']    
TERM       = inp['TERM']
NB_steps   = inp['NB_steps']
start_date = inp['start_date']
flgs       = inp['train_flg']

jis_path     = "./nomura_dam/jisseki.csv"
jis_tmp_path = "./nomura_dam/jisseki_tmp.csv"
jis_df = pd.read_csv(jis_path)

rain_p_path     = "./nomura_dam/rain_p.csv"
rain_p_tmp_path = "./nomura_dam/rain_tmp_p.csv"
rain_p_df       = pd.read_csv(rain_p_path)

print(jis_df)
if all_kouzui_flg==0:    #####交差検証
    print('交差検証による訓練＆検証計算の実施')
    kensho_no = 0
    for flg_1 in flgs:
        try_no = 0
        if flg_1:
            continue        #####検証洪水の場合に以下の処理
        for flg_2 in flgs:
            term = TERM[try_no]
            nb_steps = NB_steps[try_no]
            Q_tmp = jis_df[(pd.to_datetime(jis_df['datetime'])>=start_date[try_no]-datetime.timedelta(hours=past)) & \
                           (pd.to_datetime(jis_df['datetime'])<=start_date[try_no]+datetime.timedelta(minutes=term*2))]
            Q_tmp.to_csv(jis_tmp_path, index=False)
            rain_p_tmp = rain_p_df[(pd.to_datetime(rain_p_df['datetime'])>=start_date[try_no]-datetime.timedelta(hours=past)) & \
                                   (pd.to_datetime(rain_p_df['datetime'])<=start_date[try_no]+datetime.timedelta(minutes=term*2))]
            rain_p_tmp.to_csv(rain_p_tmp_path, index=False)
            #########################################
            #エージェントの訓練
            ##########################################
            if train_flg and flg_2:
                env, agent = make_env_model_agent(ENV_NAME)
                if try_no > 0:
                    agent.load_weights(f'./dammodel/agent_{ENV_NAME}_weights.h5f')
                history = agent.fit(env=env, nb_steps=nb_steps, action_repetition=1, callbacks=None, verbose=0,
                            visualize=True, nb_max_start_steps=0, start_step_policy=None, log_interval=100,
                            nb_max_episode_steps=None)
                agent.save_weights(f'./dammodel/agent_{ENV_NAME}_weights.h5f', overwrite=True)
                #plt.subplot(2,1,1)
                #plt.plot(history.history["nb_episode_steps"], label='episord_steps')
                #plt.ylabel("step")
                #plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=4)
                #plt.subplot(2,1,2)
                #plt.plot(history.history["episode_reward"], label='reward')
                #plt.plot(pd.Series(history.history["episode_reward"]).rolling(len(flgs)-1).mean(), label='reward(moving average)')
                #plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=4)
                #plt.xlabel("episode")
                #plt.ylabel("reward")
                #plt.savefig(f'./dammodel/episode_steps_reward{try_no}.png', dpi=300, bbox_inches="tight", pad_inches=0.05)
                #plt.close()
                try_no += 1
        ##########################################
        #エージェントの検証計算
        ##########################################
        term = TERM[try_no]
        nb_steps = NB_steps[try_no]
        agent.save_weights(f'./dammodel/agent_{ENV_NAME}_{kensho_no}_weights.h5f', overwrite=True)
        Q_tmp = jis_df[(pd.to_datetime(jis_df['datetime'])>=start_date[kensho_no]-datetime.timedelta(hours=past)) & \
                       (pd.to_datetime(jis_df['datetime'])<=start_date[kensho_no]+datetime.timedelta(minutes=term*2))]
        Q_tmp.to_csv(jis_tmp_path, index=False)
        rain_p_tmp = rain_p_df[(pd.to_datetime(rain_p_df['datetime'])>=start_date[kensho_no]-datetime.timedelta(hours=past)) & \
                               (pd.to_datetime(rain_p_df['datetime'])<=start_date[kensho_no]+datetime.timedelta(minutes=term*2))]
        rain_p_tmp.to_csv(rain_p_tmp_path, index=False)
        env_test, agent_test = make_env_model_agent(ENV_NAME_TEST)
        agent_test.load_weights(f'./dammodel/agent_{ENV_NAME}_{kensho_no}_weights.h5f')
        predict= agent_test.test(env_test, nb_episodes=1, visualize=False)
        kensho_no += 1
else:    #####全洪水対象
    print('全洪水を対象とした訓練＆検証計算の実施')
    kensho_no = 0
    for flg_1 in flgs:
        try_no = 0
        for flg_2 in range(len(flgs)):
            term = TERM[try_no]
            nb_steps = NB_steps[try_no]
            Q_tmp = jis_df[(pd.to_datetime(jis_df['datetime'])>=start_date[try_no]-datetime.timedelta(hours=past)) & \
                           (pd.to_datetime(jis_df['datetime'])<=start_date[try_no]+datetime.timedelta(minutes=term*2))]
            Q_tmp.to_csv(jis_tmp_path, index=False)
            rain_p_tmp = rain_p_df[(pd.to_datetime(rain_p_df['datetime'])>=start_date[try_no]-datetime.timedelta(hours=past)) & \
                                   (pd.to_datetime(rain_p_df['datetime'])<=start_date[try_no]+datetime.timedelta(minutes=term*2))]
            rain_p_tmp.to_csv(rain_p_tmp_path, index=False)
            ##########################################
            #エージェントの訓練
            ##########################################
            if train_flg and flg_2:
                env, agent = make_env_model_agent(ENV_NAME)
                if try_no > 0:
                    agent.load_weights(f'./dammodel/agent_{ENV_NAME}_weights.h5f')
                history = agent.fit(env=env, nb_steps=nb_steps, action_repetition=1, callbacks=None, verbose=0,
                            visualize=True, nb_max_start_steps=0, start_step_policy=None, log_interval=100,
                            nb_max_episode_steps=None)#None)
                agent.save_weights(f'./dammodel/agent_{ENV_NAME}_weights.h5f', overwrite=True)
                plt.subplot(2,1,1)
                plt.plot(history.history["nb_episode_steps"], label='episord_steps')
                plt.ylabel("step")
                plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=4)
                plt.subplot(2,1,2)
                plt.plot(history.history["episode_reward"], label='reward')
                plt.plot(pd.Series(history.history["episode_reward"]).rolling(len(flgs)).mean(), label='reward(moving average)')
                plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=4)
                plt.xlabel("episode")
                plt.ylabel("reward")
                plt.savefig(f'./dammodel/episode_steps_reward{try_no}.png', dpi=300, bbox_inches="tight", pad_inches=0.05)
                plt.close()
                try_no += 1
        ##########################################
        #エージェントの検証計算
        ##########################################
        env_test, agent_test = make_env_model_agent(ENV_NAME_TEST)
        agent_test.load_weights(f'./dammodel/agent_{ENV_NAME}_{kensho_no}_weights.h5f')
        predict= agent_test.test(env_test, nb_episodes=len(flgs), visualize=False)
        kensho_no += 1

        # %%
