import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from src.fairshap import FairSHAP

def cross_validate(model, X:pd.DataFrame, y:pd.Series, dataset_name, num_folds=5, matching_method='NN', threshold=0.05):  
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=1)  # 5-fold cross-validation
    i = 1

    for train_index, val_index in kf.split(X):
        print("-------------------------------------")
        print(f"-------------{i}th fold----------------")
        print("-------------------------------------")
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
        
        # Train the model
        model.fit(X_train_fold, y_train_fold)
        
        # Evaluate the model
        experiment = FairSHAP(
            model=model, 
            X_train=X_train_fold, 
            y_train=y_train_fold, 
            X_test=X_val_fold, 
            y_test=y_val_fold, 
            dataset_name=dataset_name, 
            matching_method=matching_method)
        experiment.run(ith_fold=i, threshold=threshold)
        i += 1
    print("Cross-validation completed.")
