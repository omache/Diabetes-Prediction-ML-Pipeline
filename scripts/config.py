from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import pandas as pd

DATA_PATH = 'data/diabetes.csv'

RANDOM_STATE = 34

CLASSIFICATION = True

DEBUG_NUMBER = 150

knn_params = {"n_neighbors": range(2, 50)}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "sqrt"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200],
                  "colsample_bytree": [0.5, 1]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500],
                   "colsample_bytree": [0.7, 1]}


gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 4, 5],
              "n_estimators": [1600, 1800, 2000],
              "subsample": [0.2, 0.3, 0.4],
              "max_features": ['sqrt']}

classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params),
               ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
               ('LightGBM', LGBMClassifier(), lightgbm_params)]

models_ = [('CART', DecisionTreeClassifier(), cart_params),
          ('RF', RandomForestClassifier(), rf_params),
          ('GBM', GradientBoostingClassifier(), gbm_params),
          ("XGBoost", XGBClassifier(objective='reg:squarederror'), xgboost_params),
          ("LightGBM", LGBMClassifier(), lightgbm_params)]

columns = ['name', 'model', 'params']

models_df = pd.DataFrame(models_, columns=columns)