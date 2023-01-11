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
    dna_length  = 36
    population = []
    np.random.seed(1108)   #隨機種子
    for num in range(pop_size):
        chromosome = ''
        for len in range(dna_length):
            chromosome = chromosome+str(np.random.randint(0,2))
        population.append(chromosome)

    return population

def adaBooest(training_data , training_data_label , adaBooest_n_estimators , decision_tree_argument  , drop_columns) :

    regressor = AdaBoostRegressor(
        DecisionTreeRegressor(**decision_tree_argument), n_estimators=adaBooest_n_estimators, random_state=0
    )

    training_data = training_data.drop(columns=drop_columns)


    regressor.fit(training_data, training_data_label)  #訓練 fit(Data, Label)
    
    score  = regressor.score(training_data, training_data_label)

    #print(score)
    return score
        
#解碼
def adaBoost_decode(chromosome):
    
    '''
    Random Forest 36 bit
    max_features        : 0.0 ~ 100.0 = range 0 ~ 127 = 6 bit   每個決策最多用到的features數量
    min_samples_leaf    : 0.0 ~ 100.0 = range 0 ~ 127 = 6 bit   葉子至少需要的sample數量      
    min_samples_split   : 0.0 ~ 100.0 = range 0 ~ 127 = 6 bit   內部節點需要的最小sample數量 

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
    

    棄用
    (X) n_estimators        : 1 ~ 128 = range 0~127 = 7bit          樹的數量        固定100
    (X) criterion           : 0 ~ 2  = 2 bit                        決策樹切分標準  absolute_error
    (X) max_depth           : 1 ~ 8 = range 0~7 = 3bit              最大深度        手動調整
    '''

    max_depth = int(chromosome[0:2], 2) + 1
    max_features = max( 1 / 127 * int(chromosome[2:8], 2) , 0.001 )
    min_samples_leaf = max( 1 / 127 * int(chromosome[8:14], 2) , 0.001 )
    min_samples_split = max( 1 / 127 * int(chromosome[14:20], 2) , 0.001 )

    maket_value = int(chromosome[20]) 
    close_price = int(chromosome[21])         
    PE = int(chromosome[22])                  
    PB_ratio = int(chromosome[23])            
    PS_ratio = int(chromosome[24])            
    ROE = int(chromosome[25])                 
    ROA = int(chromosome[26])                 
    OPM = int(chromosome[27])                 
    NPM = int(chromosome[28])                 
    D_E = int(chromosome[29])                 
    Current_Ratio = int(chromosome[30])       
    Quick_Ratio = int(chromosome[31])         
    Inventory_Turnover = int(chromosome[32])  
    Receivables_Turnover = int(chromosome[33]) 
    Pre_Tax_Income = int(chromosome[34])     
    Net_Income = int(chromosome[35])          
    
    # criterion_name = ["squared_error", "friedman_mse", "absolute_error"]

    # if criterion == 3:
    #     criterion = random.randint(0,2)

    adaBooest_n_estimators = 100
    decision_tree_argument = {
        "max_depth" : 5, 
        "max_features" : max_features , 
        "criterion" : "absolute_error",
        "min_samples_leaf" : min_samples_leaf,
        "min_samples_split" : min_samples_split,
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

    return adaBooest_n_estimators , decision_tree_argument  , drop_columns


def adaBooest_Fitness_Process(chromosome , training_data , training_data_label , train_year):

    try:
        adaBooest_n_estimators , decision_tree_argument  , drop_columns = adaBoost_decode(chromosome)
        score =  adaBooest(training_data , training_data_label , adaBooest_n_estimators , decision_tree_argument  , drop_columns)
        return score
    except Exception as e:
        return e


def adaBoost_validate (best_model ,drop_columns , val_data ,val_data_label , val_year , print_figure = False):


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

def get_best_model(best_chromosome ,train_data , train_label , train_year, save_chromosome):

    adaBooest_n_estimators , decision_tree_argument  , drop_columns = adaBoost_decode(best_chromosome)
    regressor = AdaBoostRegressor(
        DecisionTreeRegressor(**decision_tree_argument), n_estimators=adaBooest_n_estimators, random_state=0
    )
    train_data = train_data.drop(columns=drop_columns)
    regressor.fit(train_data , train_label)

    print(adaBooest_n_estimators)
    print(decision_tree_argument)
    print(drop_columns)

    if save_chromosome:
        save_best_chromosome(best_chromosome , train_year)

    return regressor , drop_columns

def get_DNA_length():
    return 36

def save_best_chromosome(best_chromosome , train_year):
    
    adaBooest_n_estimators , decision_tree_argument  , drop_columns = adaBoost_decode(best_chromosome)

    with open(f'abaBoost_Best_Chromosome/T{train_year}V{10-train_year}.txt' , 'w' ,encoding="utf8") as f:
        f.write(best_chromosome+"\n")
        f.write(f"adaBooest_n_estimators {adaBooest_n_estimators}\n" )
        f.write(str(decision_tree_argument)+"\n")
        f.write(str(drop_columns)+"\n")
    


