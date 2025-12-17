import os
import sys
import dill

import numpy as np
import pandas as pd

from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

def save_object(file_path,obj):
    try:
      dir_path=os.path.dirname(file_path)
      os.makedirs(dir_path,exist_ok=True)

      with open(file_path,"wb") as file_object:
         dill.dump(obj,file_object)

    except Exception as e:
      raise CustomException(e,sys)
    

train_evals={}
test_evals={}

def evaluate(x_train,y_train,x_test,y_test,models,params):
   
   for i in range(len(list(models))):
        mod=list(models.values())[i]
        mod_name=list(models.keys())[i]
        mod_params=list(params.values())[i]

        random=RandomizedSearchCV(estimator=mod,param_distributions=mod_params,cv=3)
        random.fit(x_train,y_train)

        mod.set_params(**random.best_params_)

        mod.fit(x_train,y_train)
        
        y_train_pred=mod.predict(x_train)
        y_test_pred=mod.predict(x_test)

        train_r2_score=r2_score(y_train,y_train_pred)
        test_r2_score=r2_score(y_test,y_test_pred)

        train_evals[mod_name]=train_r2_score
        test_evals[mod_name]=test_r2_score

        return test_evals
   

def load_pickle_file(file_path):
    try:
  
      with open(file_path,"rb") as file_obj:
         return dill.load(file_obj)
      

    except Exception as e:
        raise CustomException(e,sys)
         

