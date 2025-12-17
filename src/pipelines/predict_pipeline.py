from src.exception import CustomException
import pandas as pd
from src.utils import load_pickle_file
import sys




class PredictOutput:
  def __init__(self):
    pass

  def predict(self,df):
    try:

      preprocessor=load_pickle_file(file_path='artifacts\preprocessor.pkl')
      model=load_pickle_file(file_path='artifacts\model_trainer.pkl')

      prec=preprocessor.transform(df)
      pred=model.predict(prec)

      return pred
    except Exception as e:
        raise CustomException(e,sys)

class CustomData:

  def __init__(self,gender:str
               ,race_ethinicity:str
               ,Parental_Level_of_Education:str,
               Lunch_Type:str,
               Test_preparation_Course:str,
               Writing_Score:int,
               Reading_Score:int):
    
    self.gender=gender
    self.race_ethinicity=race_ethinicity
    self.Parental_Level_of_Education=Parental_Level_of_Education
    self.Lunch_Type=Lunch_Type
    self.Test_preparation_Course=Test_preparation_Course
    self.Writing_Score=Writing_Score
    self.Reading_Score=Reading_Score
    

  def input_data_to_dataframe(self):
    try:

      dict={"gender":[self.gender],
            "race_ethnicity":[self.race_ethinicity],
            "parental_level_of_education":[self.Parental_Level_of_Education],
            "lunch":[self.Lunch_Type],
            "test_preparation_course":[self.Test_preparation_Course],
            "reading_score":[self.Reading_Score],
            "writing_score":[self.Writing_Score]}
      
      input_df=pd.DataFrame(dict)

      return input_df
    
    except Exception as e:
      raise CustomException(e,sys)





