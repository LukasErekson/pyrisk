# model comparison
import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

import time
# from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import datetime as dt #used in filename creation
import os
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

#traceback included in experiment try-except block
import sys
import traceback
import json #not sure if this is actually needed

# PARAMETERS FOR ANALYSIS

path='pyrisk/02_nth_turn_datasets/'
path='pyrisk/01_nth_turn_datasets/'
path='hdfs/'
output_file_location_for_csv_results = 'results/'

# path='/Users/joeywilkes/senior-labs/402_project/data/01_big_nth_turn_datasets/big2/'
# output_file_location_for_csv_results = '../data/'

second_dataset = False #whether or not I'm running on second nth turn dataset or first
num_players = 6
small_scale = False
compare_models={'xgboost','adaboost','rf','softmax','softmax_grid','ovr','dt'} #select from these
compare_models={'adaboost','rf','softmax', 'knc'}
#compare_models={'knc'}

if 'softmax_grid' in compare_models:
    raise ValueError('softmax_grid should not be in compare_models until parameter grid works')
if 'xgboost' in compare_models:
    # should i use eval_metric? or just use .predict, and then use my own objective function
    raise ValueError('xgboost should not be in compare_models until I have a score function')
now = dt.datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("\n\ndate and time =", dt_string)

start = time.time()
# create features list
desired_features = ['Total Reinforcements', 'Troop Count', 'Country Count', 'player_cut_edges'
,'player_number_boundary_nodes','player_connected_components'
,'player_average_boundary_fortifications']
feat = []
for i in range(num_players):
    for j in desired_features:
        feat.append('Player ' + str(i) + ' ' +j)

if second_dataset:
    # this is according to second nth turn dataset
    possible_turn_datasets = np.arange(150,700,5)
else:
    # this is according to the first nth turn dataset
    possible_turn_datasets = np.array(
            [131, 136, 141, 146, 151, 156, 161, 166, 171, 176, 181, 186, 191,
           196, 201, 206, 211, 216, 221, 226, 231, 236, 241, 246, 251, 256,
           261, 266, 271, 276, 281, 286, 291, 296, 301, 306, 311, 316, 321,
           326, 331, 336, 341, 346, 351, 356, 361, 366, 371, 376, 381, 386,
           391, 396, 405, 410, 415, 420, 425, 430, 435, 440, 445, 450, 455,
           460, 465, 470, 475, 480, 485, 490, 495, 500, 505, 510, 515, 520,
           525, 530, 535, 540, 545, 550, 555, 560, 565, 570, 575, 580, 585,
           590, 595, 600, 605, 610, 615, 620, 625, 630, 635, 640, 645, 650,
           655, 660, 665, 670, 675, 680, 685, 690, 695, 700, 705, 710, 715,
           720, 725, 730])

if small_scale:
    #400-410 is about 327 test games
    turn_lower_bound,turn_upper_bound = 400,410
    # 55 prediction points in time if all datasets included
    if second_dataset:
        possible_turn_datasets = np.array([280, 360])
    else:
        possible_turn_datasets = np.array([281, 361])

else:
    #400-450 is 1514 test games
    turn_lower_bound,turn_upper_bound = 400,450
    # 63 prediction points in time if all datasets included

prediction_evaluation_points = np.where(possible_turn_datasets <= turn_upper_bound,1,0).sum()
print(f'there are {prediction_evaluation_points} points in time\n to predict on for games that end before {turn_upper_bound} turns')

# how many games included in test set
if second_dataset:
    turn_num = 170
else:
    turn_num = 131

temp = pd.read_hdf(path + f'big_{turn_num}_turn.hdf')
temp.drop(index=temp[temp.winner.isna()].index,inplace=True)
temp = temp.loc[(temp.total_numer_turns_in_game > turn_lower_bound) & (temp.total_numer_turns_in_game < turn_upper_bound)].copy()
print('number of games in test set',temp.shape[0])

## BUILD THE TRAIN SET
if small_scale:
    # simple test set
    turn_num = 306
    df = pd.read_hdf(path + f'big_{turn_num}_turn.hdf')
    print(df.shape)
    df.drop(index=df[df.winner.isna()].index,inplace=True)
    print(df.shape)
    # get the training data
    train = df.loc[(df.total_numer_turns_in_game < turn_lower_bound) | (df.total_numer_turns_in_game > turn_upper_bound)].copy()
    print(train.shape)
else:
    train = pd.DataFrame()
    for turn_num in [131, 181, 231, 281, 331, 381, 435, 485, 535, 585, 635, 685]:
        df1 = pd.read_hdf(path + f'big_{turn_num}_turn.hdf')
        df1.drop(index=df1[df1.winner.isna()].index,inplace=True)
        df1 = df1.loc[(df1.total_numer_turns_in_game < turn_lower_bound) | (df1.total_numer_turns_in_game > turn_upper_bound)].copy()
        temp = pd.DataFrame()
        min_wins = df1.winner.value_counts().min()

        #get an even number of wins from each player for each dataset
        for i in range(num_players):
            temp = pd.concat([temp,df1.loc[df1.winner == i].sample(min_wins)],ignore_index=True)
        #combine the datsets
        train = pd.concat([train,temp],ignore_index=True)

    print(train.shape)

# create several classifiers
if small_scale:

    if 'softmax' in compare_models:
        lr_soft = LogisticRegression(penalty='l2',C=0.01,multi_class='multinomial',solver='newton-cg',max_iter=10,
            n_jobs=-1)
        lr_soft.fit(train[feat],train.winner)

    if 'ovr' in compare_models:
        print('for ovr the logistic regression is the same as for small_scale = False')
        lr_ovr = LogisticRegression(penalty='l2',C=0.01,multi_class='ovr',solver='newton-cg',max_iter=10, n_jobs=-1)
        lr_ovr.fit(train[feat],train.winner)

    if 'adaboost' in compare_models:
        gbc = GradientBoostingClassifier(max_depth=20
                            ,min_samples_leaf=100
                            ,n_estimators=5)
        gbc.fit(train[feat],train.winner)

    if 'rf' in compare_models:
        rfc = RandomForestClassifier(n_estimators=5
            ,max_depth=15,max_features=5
            ,oob_score=True,n_jobs=-1,warm_start=False)
        rfc.fit(train[feat],train.winner)

    if 'xgboost' in compare_models:
        xg_params = {'alpha': 3.1623e-05, 'eta': 0.0178, 'gamma': 0.1, 'lambda': 0.0032
            ,'objective':"multi:softmax",'n_estimators':2,'n_jobs':8}
        xgb_model = xgb.XGBClassifier(**xg_params)
        xgb_model.fit(train[feat],train['winner'])
    
    if 'knc' in compare_models:
        pca = PCA(n_components=30)
        knc = KNeighborsClassifier(n_neighbors=30, n_jobs=-1)
        knc.fit(pca.fit_transform(train[feat]), train['winner'])

else:
    if 'softmax' in compare_models:
        lr_soft = LogisticRegression(penalty='l2',C=0.01,multi_class='multinomial',solver='newton-cg',
            max_iter=10, n_jobs=-1)
        lr_soft.fit(train[feat],train.winner)

    elif 'softmax_grid' in compare_models:
        #throws an error at the moment
        start_lr_soft_gridsearch = time.time()

        lr_soft_params = {'penalty':['l2','l1','elasitcnet']
            ,'C':np.logspace(10,-5,5)
            ,'max_iter':[int(1e3),int(1e5)]
            ,'multi_class':'multinomial'
            ,'solver':['newton-cg','sag','saga','lbfgs']
            ,'warm_start':False
            ,'l1_ratio':np.logspace(0,1,5)}

        lr_soft = LogisticRegression()
        lrclf = GridSearchCV(lr_soft,lr_soft_params,verbose=1,cv=5,n_jobs=-1)
        lrclf.fit(train[feat],train.winner)

        lr_time = time.time() - start_lr_soft_gridsearch
        total_time_lr_soft_gridsearch = f'it took {round(lr_time / 60,2)} minutes to do gridsearch\n'

        try:
            month, day = dt.datetime.now().month, dt.datetime.now().day
            hour, minute = dt.datetime.now().hour, dt.datetime.now().minute
            lrclf_title = f'softmax_gridsearch_results{month}_{day}_{hour}_{minute}.txt'
            with open(output_file_location_for_csv_results + lrclf_title,'w') as f:
                f.write(total_time_lr_soft_gridsearch)
                f.write(str(lr_soft.best_params_) + '\n' + str(lr_soft.best_estimator_) + '\n\n\n')
                f.write(str(lr_soft.cv_results_))
        except:
            pass

    if 'ovr' in compare_models:
        lr_ovr = LogisticRegression(penalty='l2',C=0.01,multi_class='ovr',solver='newton-cg',max_iter=10, n_jobs=-1)
        lr_ovr.fit(train[feat],train.winner)

    if 'adaboost' in compare_models:
        # gradient boosted classifier best params
        #{'max_depth': 130, 'min_samples_leaf': 50, 'n_estimators': 30}
        #(according to my gridsearch) in hw 154
        gbc = GradientBoostingClassifier(max_depth=130
                            ,min_samples_leaf=50
                            ,n_estimators=30)
        gbc.fit(train[feat],train.winner)

    if 'rf' in compare_models:
        #random forest best params
        #{'max_depth': 120, 'max_features': 21} #based upon hw153
        rfc = RandomForestClassifier(n_estimators=100
                ,max_depth=120,max_features=21
                ,oob_score=True,n_jobs=-1,warm_start=False)
        rfc.fit(train[feat],train.winner)

    if 'xgboost' in compare_models:
        #xgboost optimal parameters for 296 turn dataset
        xg_params = {'alpha': 3.1623e-05, 'eta': 0.0178
            , 'gamma': 0.1, 'lambda': 0.0032,'objective':"multi:softmax",'n_jobs':8}
        xgb_model = xgb.XGBClassifier(**xg_params)
        xgb_model.fit(train[feat],train['winner'])
        
    if 'knc' in compare_models:
        pca = PCA(n_components=30)
        knc = KNeighborsClassifier(n_neighbors=30, n_jobs=-1)
        knc.fit(pca.fit_transform(train[feat]), train['winner'])

if 'xgboost' in compare_models:
    #try saving the model
    try:
        month, day = dt.datetime.now().month, dt.datetime.now().day
        hour, minute = dt.datetime.now().hour, dt.datetime.now().minute
        xgb_save_model = f'xgboost_model_{month}_{day}_{hour}_{minute}.json'
        xgb_model.save_model(output_file_location_for_csv_results + xgb_save_model)
    except:
        traceback.print_exc()
        pass
        print('saving xgboost model didnt work, probably because of json format')

    try:
        month, day = dt.datetime.now().month, dt.datetime.now().day
        hour, minute = dt.datetime.now().hour, dt.datetime.now().minute
        xgb_save_model = f'xgboost_model_{month}_{day}_{hour}_{minute}.model'
        xgb_model.save_model(output_file_location_for_csv_results + xgb_save_model)
    except:
        traceback.print_exc()
        pass
        print('saving xgboost model didnt work, probably because of .model format')


#how to use feature importances, output to file?
#https://xgboost.readthedocs.io/en/latest/python/python_api.html
#what is num_round in xgb.train?

# PERFORM  THE EXPERIMENT
results = dict()
print('all turns are being considered for prediction')
turns_to_consider = possible_turn_datasets[::1]
for i,turn_num in enumerate(turns_to_consider):
    try:
        df = pd.read_hdf(path + f'big_{turn_num}_turn.hdf')

        num_games = df.shape[0]
        stalemates = len(df[df.winner.isna()].index)
        df.drop(index=df[df.winner.isna()].index,inplace=True)
        df = df.loc[(df.total_numer_turns_in_game > turn_lower_bound) & (df.total_numer_turns_in_game < turn_upper_bound)].copy()

        if i == 0:
            number_of_games_in_test_set = df.shape[0]
            print('total number of games in the train set',train.shape[0])
            print('total numer of games in test set is',number_of_games_in_test_set)

        if 'adaboost' in compare_models:
            #adaboost
            adaboost_score = gbc.score(df[feat],df.winner)
        else:
            adaboost_score = None

        if 'softmax' in compare_models:
            #softmax
            softmax_score = lr_soft.score(df[feat],df.winner)
        elif 'softmax_grid' in compare_models:
            softmax_score = lrclf.score(df[feat],df.winner)
        else:
            softmax_score = None


        if 'ovr' in compare_models:
            #one v rest
            one_v_rest_score = lr_ovr.score(df[feat],df.winner)
        else:
            one_v_rest_score = None

        if 'xgboost' in compare_models:
            #xgboost
            #is it .predict or .score
            #xgboost_score = xgb_model.predict(df[feat],df.winner)
            xgboost_score = None
        else:
            xgboost_score = None

        if 'dt' in compare_models:
            #decision tree
            decision_tree_score = None
        else:
            decision_tree_score = None

        if 'rf' in compare_models:
            #random forest
            random_forest_score = rfc.score(df[feat],df.winner)
        else:
            random_forest_score = None
            
        if 'knc' in compare_models:
            knc_score = knc.score(pca.transform(df[feat]), df.winner)
        else:
            knc_score = None

        results[i] = {'turn':turn_num
                     ,'adaboost_score':adaboost_score
                      ,'softmax_score':softmax_score
                      ,'one_v_rest_score':one_v_rest_score
                      ,'xgboost_score':xgboost_score
                      ,'decision_tree_score':decision_tree_score
                      ,'random_forest_score':random_forest_score
                      ,'knc_score':knc_score
                     ,'game_count_without_stalemates':num_games - stalemates
                     ,'stalemate':stalemates
                     ,'num_games':num_games
                     }

        print('done with turn',turn_num)
    except:
        #traceback.print_exc()
        pass

#PLOT & SAVE THE RESULTS
month, day = dt.datetime.now().month, dt.datetime.now().day
hour, minute = dt.datetime.now().hour, dt.datetime.now().minute
title = f'model_comparison_{month}_{day}_{hour}_{minute}'
keep = pd.DataFrame(results).T
keep.to_csv(output_file_location_for_csv_results + title + '.csv',index=False)

try:
    fig = plt.figure(figsize=(8,8))
    # print('turns in consideration',[x for x in turns_to_consider])
    plt.plot(keep['turn'],keep['adaboost_score'],label='Adaboost')
    plt.plot(keep['turn'],keep['softmax_score'],label='Softmax')
    plt.plot(keep['turn'],keep['one_v_rest_score'],label='One Vs. Rest')
    plt.plot(keep['turn'],keep['xgboost_score'],label='Xgboost')
    plt.plot(keep['turn'],keep['decision_tree_score'],label='Individual Tree')
    plt.plot(keep['turn'],keep['random_forest_score'],label='Random Forest')
    plt.plot(keep['turn'].keep['knc_score'],label='K-Neighbors')

    plt.title(f'Model Comparison\ntrained on {train.shape[0]} games,\ntested on {number_of_games_in_test_set} games\nwhere test games lasted between {turn_lower_bound}-{turn_upper_bound} turns')
    plt.ylabel('Performance')
    plt.xlabel('Turn Number')
    plt.grid()
    plt.legend(loc='best')
    title = f'model_comparison_{month}_{day}_{hour}_{minute}'
    plt.savefig(output_file_location_for_csv_results + title + '.png')
except:
    print('making the figure didnt work')
    traceback.print_exc()
    pass
#plt.savefig must be called before plt.show()
# plt.show()
print(f'it took {round((time.time() - start)/60,2)} minutes to execute the entire program\nWas it small_scale={small_scale}')
