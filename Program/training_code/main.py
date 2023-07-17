# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 13:04:41 2021

@author: User
"""


import pandas as pd
from functions_tmp import aTimeingThreePredictions
from cfg import DataPipeline_cfg as cfg
import warnings
import os
import logging


# In[] cfgs
researchTopicName = cfg["researchTopicName"]
c_list = cfg['c_list']
timing = cfg['timing']
sm = cfg["sm"]
threshold = cfg['threshold']
eliminatedFeatures = cfg['eliminatedFeatures']
file_names = cfg['file_names']


# In[] 產出一個類別檔案的不同時間段(四個長期or四個中短期)之X, y
def makeXandY(dataName):
    data = pd.read_csv(dataName)
    col = [i for i in range(data.shape[1])]
    col.pop(-1)
    X = data.iloc[:, col]
    y = data.iloc[:, -1]
    return X, y


def main_oneTypeofResearchTarget():
    print("main_oneTypeofResearchTarget")
    FORMAT = '%(asctime)s     %(levelname)s:  %(message)s'
    logging.basicConfig(
        format=FORMAT, filename=f"./logging_{researchTopicName}.txt", level=logging.DEBUG, filemode='w')
    
    allX = list()
    ally = list()

    for a_file in file_names:
        X, y = makeXandY(a_file)
        if eliminatedFeatures != None:
            X = X.drop(eliminatedFeatures, axis = 1) 
        allX.append(X)
        ally.append(y)
    
    total_result = pd.DataFrame()
    for i in range(len(timing)): 
        print("processing: ", timing[i])
        logging.info(f"processing: {timing[i]}")
        df = aTimeingThreePredictions(c_list, sm, allX[i], ally[i], timing[i], threshold)  #產出3 models combination
        total_result = pd.concat([total_result, df], axis = 0)                  #1個topic有12個entries
        print()
        
    total_result.to_csv(f"{researchTopicName}.csv", encoding='utf_8_sig', line_terminator='\n', index=True)
    return total_result
    
    
if __name__ == "__main__":
    current_path = os.getcwd()
    try:
        os.mkdir(os.path.join(current_path, "figures"))
    except:
        pass
    warnings.filterwarnings("ignore")
    total_result = main_oneTypeofResearchTarget()



