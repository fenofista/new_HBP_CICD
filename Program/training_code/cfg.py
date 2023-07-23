# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 19:29:03 2022

@author: Elaine
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier
#differnet SMOTE strategy
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTENC
    

# In[]
classifier = XGBClassifier(n_estimators = 1000, eval_metric='mlogloss')
classifier2 = RandomForestClassifier(n_estimators = 1100)
classifier3 = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators = 550)


# In[]
DataPipeline_cfg = {
    #default
    'researchTopicName': "1369 with temperature",
    'c_list':[classifier, classifier2, classifier3],
    'timing': ["1m_temp", '3m_temp', '6m_temp', '9m_temp'],
    'file_names': [
                   "cvd1m_temp.csv", "cvd3m_temp.csv", "cvd6m_temp.csv", "cvd9m_temp.csv"],
    #after testParameters.py
    "sm": BorderlineSMOTE(random_state = 0, m_neighbors=2, k_neighbors=1), 
    'threshold': 0.56,
    'eliminatedFeatures': None,
    }


