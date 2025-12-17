import os
import sys
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import evaluate,save_object

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

@dataclass 
class ModelTrainerConfig:
  model_trainer_file_path:str=os.path.join('artifacts','model_trainer.pkl')


class ModelTrainer:
  def __init__(self):
    self.model_trainer_config=ModelTrainerConfig()
    

  def initiate_model_trainer(self,train_arr,test_arr):

    logging.info("inside model trainer")

    try:


      x_train,y_train,x_test,y_test=(
        train_arr[:,:-1],
        train_arr[:,-1],
        test_arr[:,:-1],
        test_arr[:,-1]
      )

      models={
        "DecisionTree":DecisionTreeRegressor(),
        "Randomforest":RandomForestRegressor(),
        "LinearRegression":LinearRegression(),
        "Xgboost":XGBRegressor(),
        "Catboost":CatBoostRegressor(),
        "Adaboost":AdaBoostRegressor(),
        "Ridge":Ridge(),
        "Lasso":Lasso(),

      }



      params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Ridge":{
                    'alpha':[0.0001, 0.001, 1, 10, 100, 1000]
                },
                "Lasso":{
                    'alpha':[0.0001, 0.001, 1, 10, 100, 1000]
                }}


      test_evals=evaluate(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,params=params)

      best_r2_score=max(list(test_evals.values()))
      best_model_name=list(test_evals.keys())[
      list(test_evals.values()).index(best_r2_score)
      ]
      best_model=models[best_model_name]


      save_object(
        file_path=self.model_trainer_config.model_trainer_file_path,
        obj=best_model
      )

      if best_r2_score<0.6:
        print('no best model found')

      logging.info(f"Found the best model{best_model_name} with r2score {best_r2_score}")
      return best_r2_score

    except Exception as e:
      raise CustomException(e,sys)










