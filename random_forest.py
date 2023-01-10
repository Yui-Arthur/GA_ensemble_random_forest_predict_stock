#雙均線策略
from operator import itemgetter
import random
import matplotlib.pyplot as plt
import numpy as np
import queue
import pandas as pd
import pandas as pd
import random,time
import threading
from datetime import datetime, date
from io import StringIO
from multiprocessing import Pool
import math
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go


def init_population(pop_size):
    dna_length  = 39
    population = []
    np.random.seed(1108)   #隨機種子
    for num in range(pop_size):
        chromosome = ''
        for len in range(dna_length):
            chromosome = chromosome+str(np.random.randint(0,2))
        population.append(chromosome)

    return population

def random_forest(training_data , training_data_label , random_forest_argument , drop_columns) :

    regressor = RandomForestRegressor( **random_forest_argument)

    training_data = training_data.drop(columns=drop_columns)


    regressor.fit(training_data, training_data_label)  #訓練 fit(Data, Label)
    
    score  = regressor.score(training_data, training_data_label)

    #print(score)
    return score
        
#解碼
def random_forest_decode(chromosome):
    
    '''
    Random Forest 39 bit
    n_estimators        : 31 ~ 158 = range 0~127 = 6bit         樹的數量
    max_depth           : 10 ~ 41 = range 0~31 = 4bit           最大深度
    max_features        : 0.0 ~ 100.0 = range 0 ~ 127 = 6 bit   每個決策最多用到的features數量
    max_samples         : 0.0 ~ 100.0 = range 0 ~ 127 = 6 bit   每棵樹使用的sample數量
    oob_score           : 0 or 1                                是否使用out of bag

    maket_value         : 0 or 1                                是否使用 市值(百萬元)
    close_price         : 0 or 1                                是否使用 收盤價(元)_年
    PE                  : 0 or 1                                是否使用 本益比
    PB_ratio            : 0 or 1                                是否使用 股價淨值比
    PS_ratio            : 0 or 1                                是否使用 股價營收比
    ROE                 : 0 or 1                                是否使用 M淨值報酬率─稅後
    ROA                 : 0 or 1                                是否使用 資產報酬率
    OPM                 : 0 or 1                                是否使用 營業利益率
    NPM                 : 0 or 1                                是否使用 利潤邊際
    D_E                 : 0 or 1                                是否使用 負債/淨值比
    Current_Ratio       : 0 or 1                                是否使用 M流動比率
    Quick_Ratio         : 0 or 1                                是否使用 M速動比率
    Inventory_Turnover  : 0 or 1                                是否使用 M存貨週轉率 (次)
    Receivables_Turnover: 0 or 1                                是否使用 M應收帳款週轉次
    Pre_Tax_Income      : 0 or 1                                是否使用 M營業利益成長率
    Net_Income          : 0 or 1                                是否使用 M稅後淨利成長率
    

    捨棄
    min_samples_leaf    : 0.0 ~ 100.0 = range 0 ~ 127 = 6 bit   葉子至少需要的sample數量        (X)算出來都是 0
    min_samples_split   : 0.0 ~ 100.0 = range 0 ~ 127 = 6 bit   內部節點需要的最小sample數量    (X)算出來都是 0
    '''
    n_estimators = int(chromosome[0:6], 2) + 31
    max_depth = int(chromosome[6:10], 2) + 10
    max_features = max( 1 / 127 * int(chromosome[10:16], 2) , 0.001 )
    max_samples = 1 / 128 * int(chromosome[16:22], 2) + 0.001
    oob_score = bool(chromosome[22]) 
    maket_value = int(chromosome[23]) 
    close_price = int(chromosome[24])         
    PE = int(chromosome[25])                  
    PB_ratio = int(chromosome[26])            
    PS_ratio = int(chromosome[27])            
    ROE = int(chromosome[28])                 
    ROA = int(chromosome[29])                 
    OPM = int(chromosome[30])                 
    NPM = int(chromosome[31])                 
    D_E = int(chromosome[32])                 
    Current_Ratio = int(chromosome[33])       
    Quick_Ratio = int(chromosome[34])         
    Inventory_Turnover = int(chromosome[35])  
    Receivables_Turnover = int(chromosome[36]) 
    Pre_Tax_Income = int(chromosome[37])     
    Net_Income = int(chromosome[38])          
    
    random_forest_argument = {
        "n_estimators" : n_estimators , 
        "max_depth" : max_depth , 
        "max_features" : max_features , 
        "max_samples" : max_samples , 
        "oob_score" : oob_score ,
        "n_jobs" : -1 , 
        "random_state" : 0 ,
    }
    feature_used = [maket_value , close_price , PE , PB_ratio , PS_ratio , ROE , ROA , OPM , NPM , D_E , Current_Ratio , Quick_Ratio , Inventory_Turnover , Receivables_Turnover , Pre_Tax_Income , Net_Income]
    feature_name = ["市值(百萬元)" , "收盤價(元)_年", "本益比" , "股價淨值比" , "股價營收比" , "M淨值報酬率─稅後",
        "資產報酬率ROA" , "營業利益率OPM" , "利潤邊際NPM" , "負債/淨值比" , "M流動比率" , "M速動比率" , "M存貨週轉率 (次)" ,
        "M應收帳款週轉次" , "M營業利益成長率" , "M稅後淨利成長率" ]
    drop_columns = []
    idx = 0
    for i in feature_used:
        if i == 0 :
            drop_columns.append(feature_name[idx])
        idx+=1

    if len(drop_columns) == 16 :
        del drop_columns[random.randint(0,15)]

    return random_forest_argument , drop_columns


def random_forest_Fitness_Process(chromosome , training_data , training_data_label , train_year):

    try:
        random_forest_argument , drop_columns = random_forest_decode(chromosome)
        score =  random_forest(training_data , training_data_label , random_forest_argument , drop_columns)
        return score
    except Exception as e:
        return e


def random_forest_validate (best_model ,drop_columns , val_data ,val_data_label , val_year , print_figure = False):

    # random_forest_argument , drop_columns  =  random_forest_decode(best_chromosome)
    # regressor = RandomForestRegressor( **random_forest_argument)
    # train_data = train_data.drop(columns=drop_columns)
    # regressor.fit(train_data , train_label)
    val_data = val_data.drop(columns=drop_columns)
    pred_label = best_model.predict(val_data)  #訓練 fit(Data, Label)
    #pred_label = np.array(pred_label)
    mse = mean_squared_error(val_data_label , pred_label)
    print("MSE ", mse)
    val_data_label = val_data_label.reset_index(drop=True)
    val_data_label.to_numpy()

    if print_figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=val_data.index, y=pred_label , name = "predict"))
        fig.add_trace(go.Scatter(x=val_data.index, y=val_data_label , name = "correct"))
        fig.show()

    # print(f"n_estimators : {random_forest_argument[0]} \nmax_depth : {random_forest_argument[1]}\nmax_features : {random_forest_argument[2]}\nmax_samples : {random_forest_argument[3]}\noob_score : {random_forest_argument[4]}")
    
    return pd.DataFrame(pred_label, columns = ['return'])

def get_best_model(best_chromosome ,train_data , train_label):

    random_forest_argument , drop_columns  =  random_forest_decode(best_chromosome)
    regressor = RandomForestRegressor( **random_forest_argument)
    train_data = train_data.drop(columns=drop_columns)
    regressor.fit(train_data , train_label)
    print(random_forest_argument)
    print(drop_columns)

    return regressor , drop_columns

def get_DNA_length():
    return 39



