# This document focuses on executing model prediction with

import os
                        
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score, log_loss

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, plot_importance
from xgboost import XGBClassifier

import optuna
from optuna.samplers import TPESampler

def main():

    print("this is the main")
    
    ## Read In Data
    train = pd.read_csv("train.csv", index_col = 0)
    x_test = pd.read_csv("test.csv", index_col = 0)
    target = "loan_status"
    y_train = train[target]
    x_train = train.drop(target, axis = 1)
    
    ## Define Data Preprocess PipeLine function
    def preprocessing (train, use_encoding = True, new_feature = True):
    
        categorical_columns = train.select_dtypes(include = ['object']).columns.tolist()
        
        if use_encoding:
            encoder = OrdinalEncoder()
            train[categorical_columns] = encoder.fit_transform(train[categorical_columns])
            #test[categorical_columns] = encoder.transform(test[categorical_columns])
    
        if new_feature:
            train['loan_to_income'] = train['loan_amnt']/train['person_income']
            train['loan_to_rate'] = train['loan_int_rate']/train['loan_amnt']
    
            #test['loan_to_income'] = test['loan_amnt']/test['person_income']
            #test['loan_to_rate'] = test['loan_int_rate']/test['loan_amnt']
            
        return train, categorical_columns
    
    
    x_train = x_train.head(10000)
    y_train = y_train.head(10000)
    
    
    ## Tunning Catboost, XGBoost, LightBoost using Optuna

    def model_pipeline (trial,model_type):
        
        use_encoding = True
        
        if model_type == 'xgb':
            
            params = {"verbosity": 0,
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "reg_lambda": trial.suggest_float("reg_lambda", 0.3, 6, log=True),
                "gamma": trial.suggest_float("gamma", 0.3, 5, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 4, log=True),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_weight": trial.suggest_int("min_child_weight", 12, 45),
                "subsample": trial.suggest_float("subsample", 0.8, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.92),
                "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 2, 5),
                "n_jobs": -1,
                "enable_categorical": True}
            
            model = XGBClassifier(** params)
            
        elif model_type == 'lxgb':
        
            params = {"objective": "binary",
                "metric": "auc",
                "verbose": -1,
                "n_jobs": -1,
                "random_state": 42,
                "num_leaves": trial.suggest_int("num_leaves", 10, 200),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_iterations": trial.suggest_int("numn_iterations", 10, 1000),
                "max_depth": trial.suggest_int("max_depth", 2, 10),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
                "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "subsample": trial.suggest_float("subsample", 0.25, 1.0)}
            
            model = LGBMClassifier(**params)
            
        elif model_type == 'catboost':
        
            use_encoding = False
            
            params = {"loss_function": "Logloss","eval_metric": "AUC",
                "verbose": False,
                "random_seed": 42,
                "depth": trial.suggest_int("depth", 2, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "iterations": trial.suggest_int("iterations", 10, 1000),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10, log=True),
                "subsample": trial.suggest_float("subsample", 0.25, 1.0),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0)}
            
            model = CatBoostClassifier(**params)
            
        return model, use_encoding
        
    def objective (trial, model, use_encoding):
    
        x_processed_train, cat_cols = preprocessing(x_train, use_encoding)
        
        cv = StratifiedKFold(5, shuffle = True, random_state = 42)
        cv_splits = cv.split(x_processed_train, y_train)
        
        scores = []

        for train_idx, val_idx in cv_splits:
            
            x_fold, x_val_fold = x_processed_train.iloc[train_idx], x_processed_train.iloc[val_idx]
            y_fold, y_val_fold = y_train.iloc[train_idx],y_train.iloc[val_idx]
            
            if use_encoding:
                model.fit(x_fold,y_fold)
            else:
                model.fit(x_fold,y_fold,cat_features = cat_cols)
                
            y_pred = model.predict_proba(x_val_fold)[:,1]
            score = roc_auc_score(y_val_fold,y_pred)
            
            scores.append(score)
        
        return np.mean(scores)
            
    
    def objective_pipeline (trial, model_type):
        
        model, use_encoding = model_pipeline(trial, model_type)
        
        return objective(trial, model, use_encoding)


    study_xgb = optuna.create_study(sampler=TPESampler(n_startup_trials=3, multivariate=True, seed=42), direction="maximize")
    study_xgb.optimize(lambda trial: objective_pipeline(trial, 'xgb'), n_trials=10)
    
    study_lxgb = optuna.create_study(sampler=TPESampler(n_startup_trials=3, multivariate=True, seed=42), direction="maximize")
    study_lxgb.optimize(lambda trial: objective_pipeline(trial, 'lxgb'), n_trials=10)
    
    study_cat = optuna.create_study(sampler=TPESampler(n_startup_trials=3, multivariate=True, seed=42), direction="maximize")
    study_cat.optimize(lambda trial: objective_pipeline(trial, 'catboost'), n_trials=10)
    
    ## Define Model Training function using the best parameters
    
    class model_train:
    
        def __init__ (self, model, seed = 42):
        
            self.model = model
            self.seed = seed
            
        def fit_predict (self, x_train, y_train, x_test):
        
            scores = []
            
            cv = StratifiedKFold(5, shuffle = True, random_state = self.seed)
            cv_splits = cv.split(x_train, y_train)
        
            model = self.model
    
            for train_idx, val_idx in cv_splits:
            
                x_fold, x_val_fold = x_train.iloc[train_idx], x_train.iloc[val_idx]
                y_fold, y_val_fold = y_train.iloc[train_idx],y_train.iloc[val_idx]
            
                model.fit(x_fold,y_fold)

                y_pred = model.predict_proba(x_val_fold)[:,1]
                score = roc_auc_score(y_val_fold,y_pred)
                
                print(score)
            
                scores.append(score)
                
            print("overall value" + str(np.mean(scores)))
            
            prediction = model.predict_proba(x_test)
            
            return prediction
            
    x_test = x_test.head(10000)
    
    final_train, cat_cols = preprocessing(x_train)
    final_test, cat_cols = preprocessing(x_test)
    
    xgb_best_params = study_xgb.best_trial.params
    xgb_model = XGBClassifier(**xgb_best_params)
    xgb_trainer = model_train(xgb_model)
    xgb_prediction = xgb_trainer.fit_predict(final_train, y_train, final_test)
    
    lxgb_best_params = study_lxgb.best_trial.params
    lxgb_model = LGBMClassifier(**lxgb_best_params)
    lxgb_trainer = model_train(lxgb_model)
    lxgb_prediction = lxgb_trainer.fit_predict(final_train, y_train, final_test)
    
    cat_best_params = study_cat.best_trial.params
    cat_model = CatBoostClassifier(**cat_best_params)
    cat_trainer = model_train(cat_model)
    cat_prediction = cat_trainer.fit_predict(final_train, y_train, final_test)
    
    ## Ensemble model prediction
    model_final = np.mean([xgb_prediction[:,1],lxgb_prediction[:,1],cat_prediction[:,1]], axis = 0)
    
    print("bbbbb")
    
    return 0

if __name__ == "__main__":
    main()
    
