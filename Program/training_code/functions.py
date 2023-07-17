# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 17:16:54 2022

@author: Elaine
"""

#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
import time
import logging



n_sample = 7


# In[]  
#做10 fold並產出ROC curve
def Print10foldScoreSmoteandDraw(classifier, sm, X, y, title, year, testsmResult=False, testParameters=True, threshold=0.4, xgbClassifier=False, testFI=False):
    #parameters
    cv = StratifiedKFold(n_splits=n_sample, shuffle = True, random_state = 0)   
    accs = []
    recs = []
    pres = []
    f1s = []
    aucs = []
    fig, ax = plt.subplots()
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs1 = []
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    fold_infos = dict()
    
    #feature importance default setting
    cols = X.columns
    feature_importances = dict()
    for i in cols:
        feature_importances[i] = 0    
    start = time.time()
    #10 fold
    for fold, (train_index, test_index) in enumerate(cv.split(X, y)):
        print("preceeding: ", f"{fold+1} fold")
        logging.info(f"preceeding: {fold+1} fold")
        X_train = X.loc[train_index]
        y_train = y.loc[train_index].values.ravel()
        X_test = X.loc[test_index]
        y_test = y.loc[test_index].values.ravel()
        
        #fit
        X_train_oversampled, y_train_oversampled = sm.fit_resample(X_train, y_train)
        classifier.fit(X_train_oversampled, y_train_oversampled)
        
        #測試SMOTE結果, 畫圖
        if testsmResult == True:
            result = [0, 0]
            for i in y_train_oversampled:
                if i == 0:
                    result[0] += 1
                else:
                    result[1] += 1
            
            plt.pie(result, labels = [0, 1], autopct = '%1.1f%%')
            testsmResult = False
            plt.show()
        
        #calculate feature importances: if classifier==xgb
        if xgbClassifier==True:
            fi = classifier.get_booster().get_score(importance_type = 'gain')
            for i, j in fi.items():
                feature_importances[i] += j

        #調整每回合判斷的threshold
        estimated_prob = classifier.predict_proba(X_test)[:,1]
        the_threshold = threshold
        predicted_y = []
        for p in estimated_prob:
            if p >= the_threshold:
                predicted_y.append(1)
            else:
                predicted_y.append(0)
        
        #calculate metrics
        accuracy = accuracy_score(y_test, predicted_y)
        precision = precision_score(y_test, predicted_y, average="macro")
        recall = recall_score(y_test, predicted_y, average="macro")
        f1 = f1_score(y_test, predicted_y, average="macro")
        accs.append(accuracy)
        recs.append(recall)
        pres.append(precision)
        f1s.append(f1)
        fp_rates, tp_rates, thresholds = roc_curve(y_test, estimated_prob)   #different length
        AUC = auc(fp_rates, tp_rates)
        aucs.append(AUC)
        
        fold_infos[fold+1] = [accuracy, precision, recall, f1, AUC]
        
        #draw line of each fold
        if testParameters==False:
            viz = RocCurveDisplay.from_estimator(
                classifier, 
                X_test,
                y_test,
                name="ROC fold {}".format(fold+1),
                alpha = 0.3, 
                lw = 1,
                ax = ax)
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs1.append(viz.roc_auc)
        
        #add up confusion matrix numbers of each fold
        cm = confusion_matrix(y_test, predicted_y)
        if len(cm[0] == 2):
            tp += cm[1][1]
            fp += cm[0][1]
            fn += cm[1][0]
            tn += cm[0][0]
        else:
            tn += cm[0]
        
        
    #performance metrixs results
    mean_acc = np.array(accs).mean()
    mean_recs = np.array(recs).mean()
    mean_precs = np.array(pres).mean()
    mean_f1s = np.array(f1s).mean()
    mean_aucs = np.array(aucs).mean()
    print("cross validation result:")
    print("Mean accuracy:" '%.5f' % mean_acc)
    print("Mean f1 score:" '%.5f' % mean_f1s)
    
    #confusion matrix
    print("confusion matrix:")
    print("                    true positive  true negative")
    print("predicted positive:            ", tp, "            ", fp)
    print("predicted negative:            ", fn, "            ", tn)
    
    #feauture importance
    #回報: 前五後五高feautres
    if xgbClassifier==True:
        for i,j in feature_importances.items():
            feature_importances[i] = j/10
        sorted_fi = sorted(feature_importances.items(), key=lambda x:x[1])
        #most important 5 features/least important 5 features:
        most_5 = sorted_fi[-10:]
        least_5 = sorted_fi[:6]
        if testFI==True:
            print("most_5: ", most_5)
            print()
            print("least_5: ", least_5)
    else:
        most_5 = np.nan
    
    #plotting: 最終測試完parameters之後
    print("start plotting:")
    if testParameters==False:
        #plot equal line
        ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="equal line", alpha=0.8)
        #plot mean line
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs1)
        ax.plot(
            mean_fpr,
            mean_tpr,
            color="b",
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )
        #general setting
        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            title=f"{title}_{year}",
        )
        ax.legend(loc="best", fontsize=5)
        plt.savefig(f"./figures/{title}_{year}.svg", format="svg")
        #plt.show()
    end = time.time()
    print("total time : ", end-start)
    logging.info(f"total time : {end-start}")
    cmall = [tp, fp, fn, tn]
    return mean_acc, mean_recs, mean_precs, mean_f1s, mean_aucs, cmall, most_5, fold_infos


def XwithEliminatedLeastFeature(orig_data, featureNames):
    col = [i for i in range(orig_data.shape[1])]
    col.pop(-1)
    X = orig_data.iloc[:, col]
    X = X.drop(featureNames, axis = 1)
    return X


#將一個時間點、三個model的結果統整於DataFrame上
def aTimeingThreePredictions(c_list, sm, X, y, timing, threshold):
    #一個時間點，儲存三個模型預測結果，結果內容：acc/precision/recall/f1/auc/cm/most important 3 fs
    big_result = pd.DataFrame()
    big_result["accuracy"] = float(0)
    big_result["precision"] = float(0)
    big_result["recall"] = float(0)
    big_result["f1"] = float(0)
    big_result["AUC"] = float(0)
    big_result["confusion_matrix"] = list()
    big_result["10 fold infos"] = dict()
    big_result["most important 10 features"] = list()

    #以三個不同的classifier去做預測
    title_names = ["ROC curve of XGB", "ROC curve of RF", "ROC curve of DTB"]
    now = ["XBG", "RF", "DTB"]
    for i in range(len(c_list)):
        print("now processing: ", now[i])
        logging.info(f"now processing: {now[i]}")
        if i == 0:
            xgb_yes = True
        else:
            xgb_yes = False
        #predict
        mean_acc, mean_recs, mean_precs, mean_f1s, mean_aucs, cmall, most_3, fold_infos = Print10foldScoreSmoteandDraw(c_list[i], sm, X, y, title_names[i], timing, testsmResult=False, testParameters=False, threshold=threshold, xgbClassifier=xgb_yes)
        big_result.loc[len(big_result.index)] = [mean_acc, mean_recs, mean_precs, mean_f1s, mean_aucs, cmall, fold_infos, most_3]
    
    #重新命名indeces
    index_names = [f"XGB_{timing}", f"RF_{timing}", f"DTB_{timing}"]
    big_result.set_index([index_names], inplace = True)
    
    return big_result


