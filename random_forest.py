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



def random_forest(training_data , training_data_label , n_estimators , max_depth , min_samples_split , min_samples_leaf , max_features , max_samples , oob_score) :

    regressor = RandomForestRegressor(n_estimators = n_estimators ,
                                    max_depth = max_depth , 
                                    random_state=0 , 
                                    n_jobs = -1 , 
                                    max_samples = max_samples ,
                                    max_features = max_features , 
                                    oob_score= oob_score , 
                                    min_samples_split =  min_samples_split ,
                                    min_samples_leaf = min_samples_leaf )

    regressor.fit(training_data, training_data_label)  #訓練 fit(Data, Label)

    score  = regressor.score(training_data, training_data_label)

    print(score)
    return score
    # return (assets - init_money) / init_money * 100
    
#解碼
def random_forest_decode(chromosome):
    
    '''
    Random Forest 35 bit
    n_estimators        : 31 ~ 158 = range 0~127 = 6bit         樹的數量
    max_depth           : 10 ~ 41 = range 0~31 = 4bit           最大深度
    min_samples_split   : 0.0 ~ 100.0 = range 0 ~ 127 = 6 bit   內部節點需要的最小sample數量
    min_samples_leaf    : 0.0 ~ 100.0 = range 0 ~ 127 = 6 bit   葉子至少需要的sample數量
    max_features        : 0.0 ~ 100.0 = range 0 ~ 127 = 6 bit   每個決策最多用到的features數量
    max_samples         : 0.0 ~ 100.0 = range 0 ~ 127 = 6 bit   每棵樹使用的sample數量
    oob_score           : 0 or 1                                是否使用out of bag

    '''
    n_estimators = int(chromosome[0:6], 2) + 31
    max_depth = int(chromosome[6:10], 2) + 10
    min_samples_split = 1 / 128 * int(chromosome[10:16], 2) + 0.001
    min_samples_leaf = 1 / 128 * int(chromosome[16:22], 2) + 0.001
    max_features = max( 1 / 127 * int(chromosome[22:28], 2) , 0.001 )
    max_samples = 1 / 128 * int(chromosome[28:34], 2) + 0.001
    oob_score = int(chromosome[34], 2) 
    # number = [int(chromosome[0:8], 2) + 1 , int(chromosome[8:], 2) + 1]

    return n_estimators , max_depth , min_samples_split , min_samples_leaf , max_features , max_samples , oob_score

def random_forest_Fitness_Process(chromosome , training_data , training_data_label , train_year):


    n_estimators , max_depth , min_samples_split , min_samples_leaf , max_features , max_samples , oob_score = random_forest_decode(chromosome)
    score =  random_forest(training_data , training_data_label ,  n_estimators , max_depth , min_samples_split , min_samples_leaf , max_features , max_samples , oob_score)
    return score


def random_forest_validate ( best_chromosome , val_data ,val_data_label , train_year):
    n_estimators , max_depth , min_samples_split , min_samples_leaf , max_features , max_samples , oob_score =  random_forest_decode(best_chromosome)
    
    score =  random_forest( val_data ,val_data_label , n_estimators , max_depth , min_samples_split , min_samples_leaf , max_features , max_samples , oob_score)
    print(f"n_estimators : {n_estimators} \nmax_depth : {max_depth}\nmin_samples_split : {min_samples_split}\nmin_samples_leaf : {min_samples_leaf}\nmax_features : {max_features}\nmax_samples : {max_samples}\noob_score : {oob_score}")
    return score



