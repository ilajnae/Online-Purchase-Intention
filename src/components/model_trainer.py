import os
import sys
from dataclasses import dataclass

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from src.components.data_transformation import DataTransformation

from sklearn.metrics import roc_auc_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "RandomForestClassifier": RandomForestClassifier(),
                "DecisionTreeClassifier": DecisionTreeClassifier(),
                "KNeighborsClassifier": KNeighborsClassifier(),
                "RidgeClassifier": RidgeClassifier(),
                "BernoulliNB": BernoulliNB(),
                "SVC": SVC(),
            }
            params = {
                "RandomForestClassifier": {
                    'n_estimators': [50, 100, 150],
                    'max_depth': [10, 20, None],
                    'criterion': ['gini', 'entropy'],
                    'max_features': ['sqrt', 'log2']
                },
                "DecisionTreeClassifier": {
                    'max_depth': [10, 20, None],
                    'criterion': ['gini', 'entropy'],
                    'splitter': ['best', 'random']
                },
                "KNeighborsClassifier": {
                    'n_neighbors': [3, 5, 7],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                },
                "RidgeClassifier": {
                    'alpha': [0.1, 1.0, 10.0],
                    'solver': ['auto', 'lsqr', 'sag']
                },
                "BernoulliNB": {
                    'alpha': [0.1, 0.5, 1.0],
                    'fit_prior': [True, False]
                },
                "SVC": {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto']
                }
            }


            report = {}

            for model_name, model in models.items():
                para = params[model_name]
                
                gs = GridSearchCV(model,para,cv=3)
                gs.fit(X_train, y_train)

                # Set the best parameters and retrain model
                model.set_params(**gs.best_params_)
                model.fit(X_train,y_train)

                # Predictions for ROC-AUC
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                train_roc_auc = roc_auc_score(y_train, y_train_pred)
                test_roc_auc = roc_auc_score(y_test, y_test_pred)

                report[model_name] = test_roc_auc
            print(report)

            # Identify the best model based on ROC-AUC score
            best_model_name = max(report, key=report.get)
            best_model_score = report[best_model_name]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No suitable model found with ROC-AUC score above 0.6")

            logging.info(f"Best model selected: {best_model_name} with score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            print(best_model_name)

            

            # Final evaluation on test data
            final_predictions = best_model.predict(X_test)
            roc_auc = roc_auc_score(y_test, final_predictions)

            return roc_auc

        except Exception as e:
            raise CustomException(e, sys)