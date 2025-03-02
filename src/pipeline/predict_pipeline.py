import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','proprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        Administrative: int,
        Administrative_Duration: float,
        Informational: int,
        Informational_Duration: float,
        ProductRelated: int,
        ProductRelated_Duration: float,
        BounceRates: float,
        ExitRates: float,
        PageValues: float,
        SpecialDay: float,
        Month: float,
        OperatingSystems: int,
        Browser: int,
        Region: int,
        TrafficType: int,
        Weekend: int,
        Returning_Visitor: int):

        self.Administrative = Administrative
        self.Administrative_Duration = Administrative_Duration
        self.Informational = Informational
        self.Informational_Duration = Informational_Duration
        self.ProductRelated = ProductRelated
        self.ProductRelated_Duration = ProductRelated_Duration
        self.BounceRates = BounceRates
        self.ExitRates = ExitRates
        self.PageValues = PageValues
        self.SpecialDay = SpecialDay
        self.Month = Month
        self.OperatingSystems = OperatingSystems
        self.Browser = Browser
        self.Region = Region
        self.TrafficType = TrafficType
        self.Weekend = Weekend
        self.Returning_Visitor = Returning_Visitor

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Administrative": [self.Administrative],
                "Administrative_Duration": [self.Administrative_Duration],
                "Informational": [self.Informational],
                "Informational_Duration": [self.Informational_Duration],
                "ProductRelated": [self.ProductRelated],
                "ProductRelated_Duration": [self.ProductRelated_Duration],
                "BounceRates": [self.BounceRates],
                "ExitRates": [self.ExitRates],
                "PageValues": [self.PageValues],
                "SpecialDay": [self.SpecialDay],
                "Month": [self.Month],
                "OperatingSystems": [self.OperatingSystems],
                "Browser": [self.Browser],
                "Region": [self.Region],
                "TrafficType": [self.TrafficType],
                "Weekend": [self.Weekend],
                "Returning_Visitor": [self.Returning_Visitor]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)