"""
FairSHAP: Preprocessing for Fairness Through Attribution-Based Data Augmentation

Copyright (c) 2025 Zhu, Lin and Bian, Yijun and You, Lei

Licensed under the MIT License. See the LICENSE file in the project root for
license information.

This module implements the FairSHAP algorithm, which improves model fairness by
identifying and modifying features that contribute to unfair predictions across
privileged and unprivileged groups.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import os
import time
import copy
import logging
from sklearn.metrics import accuracy_score
from typing import Tuple, List, Dict
from src.matching.ot_matcher import OptimalTransportPolicy
from src.matching.nn_matcher import NearestNeighborDataMatcher
from src.attribution import FairnessExplainer
from src.composition.data_composer import DataComposer
from src.fairness_metrics.fairness_measures import fairness_value_function, grp1_DP, grp2_EO, grp3_PQP, marginalised_np_mat, perturb_numpy_ver
from src.fairness_metrics.metric_fair_multigroup import (
    marginalised_np_gen,                 # Generate confusion matrices for all sensitive attribute groups
    extGrp1_DP_sing, extGrp2_EO_sing,    # Multi-group Demographic Parity (DP) and Equalized Odds (EO) metrics
    extGrp3_PQP_sing, alterGrps_sing,    # Multi-group Predictive Quality Parity (PQP) and inter-group differences
    unpriv_group_one, unpriv_group_two,  # Binary group Demographic Parity (DP) and Equalized Odds (EO) metrics
    unpriv_group_thr, calc_fair_group    # Binary group Predictive Quality Parity (PQP) and fairness gap calculation
)
EPSILON = 1e-20


class FairSHAP:
    """
    FairSHAP: Preprocessing for Fairness Through Attribution-Based Data Augmentation
    
    This class implements the core FairSHAP algorithm which:
    1. Splits data by sensitive attributes and labels
    2. Matches instances between privileged and unprivileged groups
    3. Computes fairness-aware SHAP values for feature importance
    4. Iteratively modifies features to reduce unfairness
    5. Evaluates fairness metrics (DR, DP, EO, PQP) on modified data
    
    Args:
        model: A trained machine learning model with predict and predict_proba methods
        X_train: Training feature data
        y_train: Training labels
        X_test: Test feature data
        y_test: Test labels
        dataset_name: Name of the dataset 
                    ('german_credit', 
                    'compas', 'compas4race', 'compas4multirace',
                    'adult', 'adult4multirace',  
                    'census_income_kdd', 'census_income_kdd_multirace',
                    'default_credit')
        matching_method: Method for matching instances between groups ('NN' or 'OT')
                        'NN' = Nearest Neighbor, 'OT' = Optimal Transport
    
    Attributes:
        sensitive_attri: Name of the sensitive attribute (e.g., 'sex', 'race')
        gap: Step size for iterating through feature modifications
    """
    
    def __init__(self,
                 model,
                 X_train: pd.DataFrame,
                 y_train: pd.Series,
                 X_test: pd.DataFrame,
                 y_test: pd.Series,
                 dataset_name: str,
                 matching_method: str = 'NN',
                 ):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.dataset_name = dataset_name
        self.matching_method = matching_method

        # Dataset-specific configuration for sensitive attributes and iteration gap
        if self.dataset_name == 'german_credit':
            self.sensitive_attri = 'sex'
            self.gap = 1
            self.num_sensitive_values = 2
        elif self.dataset_name == 'adult':
            self.sensitive_attri = 'sex'
            self.gap = 1
            self.num_sensitive_values = 2
        elif self.dataset_name == 'compas':
            self.sensitive_attri = 'sex'
            self.gap = 10
            self.num_sensitive_values = 2
        elif self.dataset_name == 'compas4race':
            self.sensitive_attri = 'race'
            self.gap = 10
            self.num_sensitive_values = 2
        elif self.dataset_name == 'census_income_kdd':
            self.sensitive_attri = 'sex'
            self.gap = 1
            self.num_sensitive_values = 2
        elif self.dataset_name == 'default_credit':
            self.sensitive_attri = 'sex'
            self.gap = 1
            self.num_sensitive_values = 2
        # Multi-sensitive-value datasets
        elif self.dataset_name == 'compas4multirace':
            self.sensitive_attri = 'race'
            self.gap = 20
            self.num_sensitive_values = 6 
        elif self.dataset_name == 'adult4multirace':   
            self.sensitive_attri = 'race'
            self.gap = 20
            self.num_sensitive_values = 5  
        elif self.dataset_name == 'census_income_kdd_multirace':
            self.sensitive_attri = 'race'
            self.gap = 20
            self.num_sensitive_values = 5 
        else:
            raise ValueError(f'Dataset "{dataset_name}" is not supported')      
   

    def run(self, threshold: float = 0.05, ith_fold: int = 0, save_results: bool = True):
        """
        Execute the FairSHAP algorithm to mitigate unfairness through data modification.
        
        This method automatically detects whether the dataset has binary or multiple 
        sensitive attribute values and calls the appropriate processing pipeline.
        
        Args:
            threshold: Minimum SHAP value to consider for modification (default: 0.05)
                      Features with SHAP values below this are not modified
            ith_fold: The fold number for cross-validation (used in output filename, default: 0)
                     Set to 0 for single run without cross-validation
            save_results: Whether to save results to CSV file (default: True)
        
        Returns:
            dict: Dictionary containing the results with keys:
                - 'action_numbers': List of modification counts
                - 'accuracies': List of accuracy values
                - 'fairness_metrics': Dictionary of fairness metric lists
                - 'original_metrics': Dictionary of original model metrics
        """
        self.ith_fold = ith_fold
        
        # Determine processing method based on number of sensitive values
        if self.num_sensitive_values > 2:
            return self._run_multi_sensitive(threshold, ith_fold, save_results)
        else:
            return self._run_binary_sensitive(threshold, ith_fold, save_results)

    def _run_binary_sensitive(self, threshold: float = 0.05, ith_fold: int = 0, save_results: bool = True):
        """
        Execute FairSHAP for datasets with binary sensitive attributes.
        
        This method performs the following steps:
        1. Splits data into majority/minority groups and positive/negative labels
        2. Matches instances between groups using NN or OT
        3. Computes fairness SHAP values to identify unfair features
        4. Generates counterfactual data (q) for feature replacement
        5. Iteratively modifies features and retrains models
        6. Evaluates fairness metrics (DR, DP, EO, PQP) at each iteration
        7. Optionally saves results to CSV file
        
        Args:
            threshold: Minimum SHAP value to consider for modification (default: 0.05)
            ith_fold: The fold number for cross-validation (default: 0)
            save_results: Whether to save results to CSV (default: True)
        
        Returns:
            dict: Dictionary containing results
        """
        
        # Step 1: Split data into four groups (majority/minority × label0/label1)
        print(f"1. Split the {self.dataset_name} dataset into majority group and minority group according to the number of sensitive attribute, besides split by label 0 and label 1")
        X_train_majority_label0, y_train_majority_label0, X_train_majority_label1, y_train_majority_label1, X_train_minority_label0, y_train_minority_label0, X_train_minority_label1, y_train_minority_label1 = self._split_into_majority_minority_label0_label1()
        print(f'X_train_majority_label0 shape: {X_train_majority_label0.shape}')
        print(f'X_train_majority_label1 shape: {X_train_majority_label1.shape}')
        print(f'X_train_minority_label0 shape: {X_train_minority_label0.shape}')
        print(f'X_train_minority_label1 shape: {X_train_minority_label1.shape}')
        
        # Step 2: Initialize FairnessExplainer to compute fairness SHAP values
        print('2. Initialize FairnessExplainer')
        sen_att_name = [self.sensitive_attri]
        sen_att = [self.X_test.columns.get_loc(name) for name in sen_att_name]
        priv_val = [1]  # Privileged group value
        unpriv_dict = [list(set(self.X_test.values[:, sa])) for sa in sen_att]
        for sa_list, pv in zip(unpriv_dict, priv_val):
            sa_list.remove(pv)  # Remove privileged value to get unprivileged values
        fairness_explainer_original = FairnessExplainer(
                model=self.model, 
                sen_att=sen_att, 
                priv_val=priv_val, 
                unpriv_dict=unpriv_dict,
                )
        
        start_time = time.time()

        print('--------Next, we will modify the minority group--------')
        print('3(a). Match X_train_minority_label0 with X_train_majority_label0')
        print('3(b). Match X_train_minority_label1 with X_train_majority_label1')
        if self.matching_method == 'NN':
            matching_minority_label0 = NearestNeighborDataMatcher(X_labeled=X_train_minority_label0, X_unlabeled=X_train_majority_label0).match(n_neighbors=1)
            matching_minority_label1 = NearestNeighborDataMatcher(X_labeled=X_train_minority_label1, X_unlabeled=X_train_majority_label1).match(n_neighbors=1)
        elif self.matching_method == 'OT':
            matching_minority_label0 = OptimalTransportPolicy(X_labeled=X_train_minority_label0.values, X_unlabeled=X_train_majority_label0.values).match()
            matching_minority_label1 = OptimalTransportPolicy(X_labeled=X_train_minority_label1.values, X_unlabeled=X_train_majority_label1.values).match()
        else:
            raise ValueError('The matching method is not supported')
        print('4(a). Use FairSHAP to find suitable values from X_train_majority_label0 to replace data in X_train_minority_label0')
        fairness_shapley_minority_value_label0 = fairness_explainer_original.shap_values(
                                    X = X_train_minority_label0.values,
                                    Y = y_train_minority_label0.values,
                                    X_baseline = X_train_majority_label0.values,
                                    matching=matching_minority_label0,
                                    sample_size=2000,
                                    shap_sample_size="auto",
                                )
        X_change_minority_label0 = X_train_minority_label0.copy()
        X_base_minority_label0 = X_train_majority_label0
        print('4(b). Use FairSHAP to find suitable values from X_train_majority_label1 to replace data in X_train_minority_label1')
        fairness_shapley_minority_value_label1 = fairness_explainer_original.shap_values(
                                    X = X_train_minority_label1.values,
                                    Y = y_train_minority_label1.values,
                                    X_baseline = X_train_majority_label1.values,
                                    matching=matching_minority_label1,
                                    sample_size=2000,
                                    shap_sample_size="auto",
                                )
        X_change_minority_label1 = X_train_minority_label1.copy()
        X_base_minority_label1 = X_train_majority_label1
        print('5. Calculate varphi and q')
        fairness_shapley_minority_value = np.vstack((fairness_shapley_minority_value_label0, fairness_shapley_minority_value_label1))
        non_zero_count_minority = np.sum(fairness_shapley_minority_value > threshold)
        print(f"There are {non_zero_count_minority} SHAP values greater than {threshold} in X_train_minority")
        q_minority_label0 = DataComposer(
                        x_counterfactual=X_base_minority_label0.values, 
                        joint_prob=matching_minority_label0, 
                        method="max").calculate_q() 
        q_minority_label1 = DataComposer(
                        x_counterfactual=X_base_minority_label1.values, 
                        joint_prob=matching_minority_label1, 
                        method="max").calculate_q()
        q_minority = np.vstack((q_minority_label0, q_minority_label1))
        print('--------Next, we will modify the majority group--------')
        print('3(a). Match X_train_majority_label0 with X_train_minority_label0')
        print('3(b). Match X_train_majority_label1 with X_train_minority_label1')
        if self.matching_method == 'NN':
            matching_majority_label0 = NearestNeighborDataMatcher(X_labeled=X_train_majority_label0, X_unlabeled=X_train_minority_label0).match(n_neighbors=1)       
            matching_majority_label1 = NearestNeighborDataMatcher(X_labeled=X_train_majority_label1, X_unlabeled=X_train_minority_label1).match(n_neighbors=1)
        elif self.matching_method == 'OT':
            matching_majority_label0 = OptimalTransportPolicy(X_labeled=X_train_majority_label0.values, X_unlabeled=X_train_minority_label0.values).match()
            matching_majority_label1 = OptimalTransportPolicy(X_labeled=X_train_majority_label1.values, X_unlabeled=X_train_minority_label1.values).match()
        else:
            raise ValueError('The matching method is not supported')
        print('4(a). Use FairSHAP to find suitable values from X_train_minority_label0 to replace data in X_train_majority_label0')
        fairness_shapley_majority_value_label0 = fairness_explainer_original.shap_values(
                                    X = X_train_majority_label0.values,
                                    Y = y_train_majority_label0.values,
                                    X_baseline = X_train_minority_label0.values,
                                    matching=matching_majority_label0,
                                    sample_size=2000,
                                    shap_sample_size="auto",
                                )
        X_change_majority_label0 = X_train_majority_label0.copy()
        X_base_majority_label0 = X_train_minority_label0
        print('4(b). Use FairSHAP to find suitable values from X_train_minority_label1 to replace data in X_train_majority_label1')
        fairness_shapley_majority_value_label1 = fairness_explainer_original.shap_values(
                                    X = X_train_majority_label1.values,
                                    Y = y_train_majority_label1.values,
                                    X_baseline = X_train_minority_label1.values,
                                    matching=matching_majority_label1,
                                    sample_size=2000,
                                    shap_sample_size="auto",
                                )  
        X_change_majority_label1 = X_train_majority_label1.copy()
        X_base_majority_label1 = X_train_minority_label1
        end_time = time.time()
        print(f"执行时间: {end_time - start_time:.4f} 秒")
        
        print('5. Calculate varphi and q')
        # Select SHAP values greater than 0.1, set others to 0, then normalize
        fairness_shapley_majority_value = np.vstack((fairness_shapley_majority_value_label0, fairness_shapley_majority_value_label1))
        non_zero_count_majority =np.sum(fairness_shapley_majority_value > threshold)
        print(f"There are {non_zero_count_majority} SHAP values greater than {threshold} in X_train_majority")
        q_majority_label0 = DataComposer(
                        x_counterfactual=X_base_majority_label0.values, 
                        joint_prob=matching_majority_label0, 
                        method="max").calculate_q() 
        q_majority_label1 = DataComposer(
                        x_counterfactual=X_base_majority_label1.values, 
                        joint_prob=matching_majority_label1, 
                        method="max").calculate_q()
        q_majority = np.vstack((q_majority_label0, q_majority_label1))
        fairness_shapley_value = np.vstack((fairness_shapley_minority_value, fairness_shapley_majority_value))
        # varphi = fix_negative_probabilities_select_larger(fairness_shapley_value)
        varphi = np.where(fairness_shapley_value > threshold, fairness_shapley_value, 0)

        q = np.vstack((q_minority,q_majority))   # q_minority_label0 + q_minority_label1 + q_majority_label0 + q_majority_label1
        X_change = pd.concat([X_change_minority_label0, X_change_minority_label1, X_change_majority_label0, X_change_majority_label1], axis=0)
        non_zero_count = non_zero_count_majority + non_zero_count_minority
        print('6. Calculate accuracy, DR, DP, EO, PP of the original model on X_test')
        y_pred = self.model.predict(self.X_test)
        original_accuracy = accuracy_score(self.y_test, y_pred)
        original_DR = fairness_value_function(sen_att, priv_val, unpriv_dict, self.X_test.values, self.model)
        priv_idx = self.X_test[self.sensitive_attri].to_numpy().astype(bool)
        g1_Cm, g0_Cm = marginalised_np_mat(y=self.y_test, y_hat=y_pred, pos_label=1, priv_idx=priv_idx)
        original_DP = grp1_DP(g1_Cm, g0_Cm)[0]
        original_EO = grp2_EO(g1_Cm, g0_Cm)[0]
        original_PQP = grp3_PQP(g1_Cm, g0_Cm)[0]
        print(f'7. Start organizing modifications for the minority and majority groups and merge the new data; a total of {non_zero_count} data points modified; train a new model using the new training set')
        values_range = np.arange(1, non_zero_count, self.gap)
        accuracy_results = []
        DR_results = []
        DP_results = []
        EO_results = []
        PQP_results = []
        for action_number in values_range:
            # Step 1: Flatten varphi values and positions into one dimension
            flat_varphi = [(value, row, col) for row, row_vals in enumerate(varphi)
                        for col, value in enumerate(row_vals)]
            # Step 2: Sort by value in descending order
            flat_varphi_sorted = sorted(flat_varphi, key=lambda x: x[0], reverse=True)
            # Step 3: Select the top action_number positions
            top_positions = flat_varphi_sorted[:action_number]
            # Step 4: Replace values in X_change at top positions
            for value, row_idx, col_idx in top_positions:
                X_change.iloc[row_idx, col_idx] = q[row_idx, col_idx]
            x = X_change
            y = pd.concat([y_train_minority_label0, y_train_minority_label1, y_train_majority_label0, y_train_majority_label1], axis=0)
            # Step 6: Train the new model
            model_new = XGBClassifier()
            model_new.fit(x, y)
            # Step 7: Evaluate the new model's performance on DR, DP, EO, PQP
            new_DR = fairness_value_function(sen_att, priv_val, unpriv_dict, self.X_test.values, model_new)
            y_hat = model_new.predict(self.X_test)
            y_test = self.y_test
            new_accuracy = accuracy_score(self.y_test, y_hat)
            g1_Cm, g0_Cm = marginalised_np_mat(y_test, y_hat, 1, priv_idx)
            new_DP = grp1_DP(g1_Cm, g0_Cm)[0]
            new_EO = grp2_EO(g1_Cm, g0_Cm)[0]
            new_PQP = grp3_PQP(g1_Cm, g0_Cm)[0]
            accuracy_results.append(new_accuracy)
            DR_results.append(new_DR)
            DP_results.append(new_DP)
            EO_results.append(new_EO)
            PQP_results.append(new_PQP)
        if save_results:
            print('8. Save results to CSV file')
            df = pd.DataFrame({
                "action_number": values_range,  # Directly use values_range
                "new_accuracy": accuracy_results,
                "new_DR": DR_results,
                "new_DP": DP_results,
                "new_EO": EO_results,
                "new_PQP": PQP_results,
            })
            df.loc[-1] = ["original", original_accuracy, original_DR, original_DP, original_EO, original_PQP]  # Insert as first row
            df.index = df.index + 1  # Reindex
            df = df.sort_index()  # Ensure the 'original' row is at the top
            dataset_folder = os.path.join('saved_results', self.dataset_name)
            os.makedirs(dataset_folder, exist_ok=True)
            # Generate CSV filename
            # csv_filename = f"fairSHAP-SamplingExplainer-{self.fairshap_base}_{threshold}_{self.matching_method}_{self.ith_fold}-fold_results.csv"
            csv_filename = f"2-4_fairSHAP-KernelExplainer-{threshold}_{self.matching_method}_{self.ith_fold}-fold_results.csv"
            # csv_filename = f"fairSHAP-PermutationExplainer-{self.fairshap_base}_{threshold}_{self.matching_method}_{self.ith_fold}-fold_results.csv"
            csv_filepath = os.path.join(dataset_folder, csv_filename)
            # Save CSV
            df.to_csv(csv_filepath, index=False)
            print(f"CSV file saved: {csv_filepath}")
        else:
            print('8. Results not saved (save_results=False)')
            print("\n" + "="*80)
            print("FAIRSHAP RESULTS SUMMARY")
            print("="*80)
            print(f"\nDataset: {self.dataset_name}")
            print(f"Matching method: {self.matching_method}")
            print(f"Threshold: {threshold}")
            print(f"Total modifications tested: {len(values_range)}")

            print(f"\n{'Original Model Metrics':^80}")
            print("-"*80)
            print(f"  Accuracy: {original_accuracy:.4f}")
            print(f"  DR:       {original_DR:.4f}")
            print(f"  DP:       {original_DP:.4f}")
            print(f"  EO:       {original_EO:.4f}")
            print(f"  PQP:      {original_PQP:.4f}")

            if len(values_range) > 0:
                print(f"\n{'Final Model Metrics (after {values_range[-1]} modifications)':^80}")
                print("-"*80)
                print(f"  Accuracy: {accuracy_results[-1]:.4f}  (Δ = {accuracy_results[-1]-original_accuracy:+.4f})")
                print(f"  DR:       {DR_results[-1]:.4f}  (Δ = {DR_results[-1]-original_DR:+.4f})")
                print(f"  DP:       {DP_results[-1]:.4f}  (Δ = {DP_results[-1]-original_DP:+.4f})")
                print(f"  EO:       {EO_results[-1]:.4f}  (Δ = {EO_results[-1]-original_EO:+.4f})")
                print(f"  PQP:      {PQP_results[-1]:.4f}  (Δ = {PQP_results[-1]-original_PQP:+.4f})")

                best_dr_idx = DR_results.index(min(DR_results))
                print(f"\n{'Best Fairness Point (lowest DR)':^80}")
                print("-"*80)
                print(f"  Modifications: {values_range[best_dr_idx]}")
                print(f"  Accuracy:      {accuracy_results[best_dr_idx]:.4f}")
                print(f"  DR:            {DR_results[best_dr_idx]:.4f}")
                print(f"  DP:            {DP_results[best_dr_idx]:.4f}")
                print(f"  EO:            {EO_results[best_dr_idx]:.4f}")
                print(f"  PQP:           {PQP_results[best_dr_idx]:.4f}")

            print("\n" + "="*80)
            print("Note: Results not saved to CSV (save_results=False)")
            print("="*80 + "\n")

    def _run_multi_sensitive(self, threshold: float = 0.05, ith_fold: int = 0, save_results: bool = True):
        """
        Execute FairSHAP for datasets with multiple sensitive attribute values (e.g., 5-6 racial groups).
        
        This method handles datasets where the sensitive attribute has more than 2 distinct values.
        It processes each unprivileged group separately against the privileged group.
        
        Args:
            threshold: Minimum SHAP value to consider for modification (default: 0.05)
            ith_fold: The fold number for cross-validation (default: 0)
            save_results: Whether to save results to CSV (default: True)
        
        Returns:
            dict: Dictionary containing results with extended metrics
        """
        logging.basicConfig(
            force=True,
            level=logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)
        
        print(f"1. Split the {self.dataset_name} dataset into privileged group and unprivileged groups according to sensitive attribute values")
        splits = self._split_privileged_and_unprivileged()
        
        # Step 2: Initialize FairnessExplainer
        print('2. Initialize FairnessExplainer')
        sen_att_name = [self.sensitive_attri]
        sen_att = [self.X_test.columns.get_loc(name) for name in sen_att_name]
        priv_val = [1]
        unpriv_dict = [list(set(self.X_test.values[:, sa])) for sa in sen_att]
        
        for sa_list, pv in zip(unpriv_dict, priv_val):
            sa_list.remove(pv)
        
        fairness_explainer_original = FairnessExplainer(
                model=self.model, 
                sen_att=sen_att, 
                priv_val=priv_val, 
                unpriv_dict=unpriv_dict,
                )
        
        # Step 3: Calculate shapley values for all groups
        print('3. Calculate shapley values of the privileged group and unprivileged groups')
        start_time = time.time()
        
        # Call appropriate modification function based on number of races
        if self.num_sensitive_values == 5:
            X_base, Y_base, fairness_shapley_value, q = \
                self._modify_5_races(splits, fairness_explainer_original)
        elif self.num_sensitive_values == 6:
            X_base, Y_base, fairness_shapley_value, q = \
                self._modify_6_races(splits, fairness_explainer_original)
        else:
            raise ValueError(f'num_sensitive_values={self.num_sensitive_values} not supported for multi-sensitive processing')
        
        elapsed = time.time() - start_time
        print(f'Shapley values computed in {elapsed:.2f} seconds')
        
        # Filter SHAP values by threshold
        varphi = np.where(fairness_shapley_value > threshold, fairness_shapley_value, 0)
        non_zero_count = np.sum(fairness_shapley_value > threshold)
        
        # Step 4: Calculate original model metrics
        print('4. Calculate accuracy, DR, DP, EO, PQP of the original model on X_test')
        y_pred = self.model.predict(self.X_test)
        original_accuracy = accuracy_score(self.y_test, y_pred)
        original_DR = fairness_value_function(sen_att, priv_val, unpriv_dict, self.X_test.values, self.model)
        
        # Calculate multi-group fairness metrics
        y = self.y_test.to_numpy()
        hx = y_pred
        A = self.X_test[self.sensitive_attri].to_numpy()
        
        (
            original_dp_gap,
            original_eo_gap,
            original_pq_gap,
            original_dp_max,
            original_dp_avg,
            original_eo_max,
            original_eo_avg,
            original_pqp_max,
            original_pqp_avg,
        ) = evaluate(y=y, hx=hx, A=A, priv_val=1)
        
        # Step 5: Iteratively modify features and retrain
        print(f'5. Start organizing modifications; a total of {non_zero_count} data points modified; train new models')
        values_range = np.arange(1, non_zero_count, self.gap)
        accuracy_results = []
        DR_results = []
        dp_gap_results = []
        eo_gap_results = []
        pq_gap_results = []
        dp_max_results = []
        dp_avg_results = []
        eo_max_results = []
        eo_avg_results = []
        pqp_max_results = []
        pqp_avg_results = []
        
        for action_number in values_range:
            # Flatten and sort varphi values
            flat_varphi = [(value, row, col) for row, row_vals in enumerate(varphi)
                        for col, value in enumerate(row_vals)]
            flat_varphi_sorted = sorted(flat_varphi, key=lambda x: x[0], reverse=True)
            
            # Select top positions to modify
            top_positions = flat_varphi_sorted[:action_number]
            
            # Apply modifications
            X_change = copy.deepcopy(X_base)
            for value, row_idx, col_idx in top_positions:
                X_change.iloc[row_idx, col_idx] = q[row_idx, col_idx]
            
            x = X_change
            y_col = Y_base.to_numpy().reshape(-1, 1)
            
            # Train new model
            # Automatically detect model type and train accordingly
            model_class = type(self.model)
            model_new = model_class()
            model_new.fit(x, y_col.ravel())

            
            # Evaluate new model
            new_DR = fairness_value_function(sen_att, priv_val, unpriv_dict, self.X_test.values, model_new)
            y_hat = model_new.predict(self.X_test)
            y_test = self.y_test
            new_accuracy = accuracy_score(y_test, y_hat)
            
            # Calculate multi-group metrics
            (
                new_dp_gap,
                new_eo_gap,
                new_pq_gap,
                new_dp_max,
                new_dp_avg,
                new_eo_max,
                new_eo_avg,
                new_pqp_max,
                new_pqp_avg,
            ) = evaluate(y=y_test.to_numpy(), hx=y_hat, A=A, priv_val=1)
            
            # Store results
            accuracy_results.append(new_accuracy)
            DR_results.append(new_DR)
            dp_gap_results.append(new_dp_gap)
            eo_gap_results.append(new_eo_gap)
            pq_gap_results.append(new_pq_gap)
            dp_max_results.append(new_dp_max)
            dp_avg_results.append(new_dp_avg)
            eo_max_results.append(new_eo_max)
            eo_avg_results.append(new_eo_avg)
            pqp_max_results.append(new_pqp_max)
            pqp_avg_results.append(new_pqp_avg)
        
        # Step 6: Prepare results
        print('6. Prepare and optionally save results')
        
        results = {
            'action_numbers': list(values_range),
            'accuracies': accuracy_results,
            'fairness_metrics': {
                'DR': DR_results,
                'dp_gap': dp_gap_results,
                'eo_gap': eo_gap_results,
                'pq_gap': pq_gap_results,
                'dp_max': dp_max_results,
                'dp_avg': dp_avg_results,
                'eo_max': eo_max_results,
                'eo_avg': eo_avg_results,
                'pqp_max': pqp_max_results,
                'pqp_avg': pqp_avg_results
            },
            'original_metrics': {
                'accuracy': original_accuracy,
                'DR': original_DR,
                'dp_gap': original_dp_gap,
                'eo_gap': original_eo_gap,
                'pq_gap': original_pq_gap,
                'dp_max': original_dp_max,
                'dp_avg': original_dp_avg,
                'eo_max': original_eo_max,
                'eo_avg': original_eo_avg,
                'pqp_max': original_pqp_max,
                'pqp_avg': original_pqp_avg
            }
        }
        
        # Save to CSV if requested, otherwise print summary
        if save_results:
            df = pd.DataFrame({
                "action_number": values_range,
                "new_accuracy": accuracy_results,
                "new_DR": DR_results,
                "new_dp_gap": dp_gap_results,
                "new_eo_gap": eo_gap_results,
                "new_pq_gap": pq_gap_results,
                "new_dp_max": dp_max_results,
                "new_dp_avg": dp_avg_results,
                "new_eo_max": eo_max_results,
                "new_eo_avg": eo_avg_results,
                "new_pqp_max": pqp_max_results,
                "new_pqp_avg": pqp_avg_results,
            })
            
            # Insert original results as first row
            df.loc[-1] = ["original", original_accuracy, original_DR,
                          original_dp_gap, original_eo_gap, original_pq_gap,
                          original_dp_max, original_dp_avg,
                          original_eo_max, original_eo_avg,
                          original_pqp_max, original_pqp_avg]
            df.index = df.index + 1
            df = df.sort_index()
            
            # Save CSV
            dataset_folder = os.path.join('saved_results', self.dataset_name)
            os.makedirs(dataset_folder, exist_ok=True)
            
            if ith_fold > 0:
                csv_filename = f"fairSHAP-{threshold}_{self.matching_method}_{ith_fold}-fold_results.csv"
            else:
                csv_filename = f"fairSHAP-{threshold}_{self.matching_method}_results.csv"
            
            csv_filepath = os.path.join(dataset_folder, csv_filename)
            df.to_csv(csv_filepath, index=False)
            print(f"CSV file saved: {csv_filepath}")
        else:
            # Print results summary when not saving to CSV
            print("\n" + "="*80)
            print("FAIRSHAP RESULTS SUMMARY (Multi-Sensitive Values)")
            print("="*80)
            print(f"\nDataset: {self.dataset_name}")
            print(f"Number of sensitive values: {self.num_sensitive_values}")
            print(f"Matching method: {self.matching_method}")
            print(f"Threshold: {threshold}")
            print(f"Total modifications tested: {len(values_range)}")
            
            print(f"\n{'Original Model Metrics':^80}")
            print("-"*80)
            print(f"  Accuracy:  {original_accuracy:.4f}")
            print(f"  DR:        {original_DR:.4f}")
            print(f"\n  Binary Group Comparisons:")
            print(f"    DP gap:  {original_dp_gap:.4f}")
            print(f"    EO gap:  {original_eo_gap:.4f}")
            print(f"    PQ gap:  {original_pq_gap:.4f}")
            print(f"\n  Multi-Group Metrics:")
            print(f"    DP max:  {original_dp_max:.4f}  |  DP avg:  {original_dp_avg:.4f}")
            print(f"    EO max:  {original_eo_max:.4f}  |  EO avg:  {original_eo_avg:.4f}")
            print(f"    PQP max: {original_pqp_max:.4f}  |  PQP avg: {original_pqp_avg:.4f}")
            
            print(f"\n{'Final Model Metrics (after {values_range[-1]} modifications)':^80}")
            print("-"*80)
            print(f"  Accuracy:  {accuracy_results[-1]:.4f}  (Δ = {accuracy_results[-1]-original_accuracy:+.4f})")
            print(f"  DR:        {DR_results[-1]:.4f}  (Δ = {DR_results[-1]-original_DR:+.4f})")
            print(f"\n  Binary Group Comparisons:")
            print(f"    DP gap:  {dp_gap_results[-1]:.4f}  (Δ = {dp_gap_results[-1]-original_dp_gap:+.4f})")
            print(f"    EO gap:  {eo_gap_results[-1]:.4f}  (Δ = {eo_gap_results[-1]-original_eo_gap:+.4f})")
            print(f"    PQ gap:  {pq_gap_results[-1]:.4f}  (Δ = {pq_gap_results[-1]-original_pq_gap:+.4f})")
            print(f"\n  Multi-Group Metrics:")
            print(f"    DP max:  {dp_max_results[-1]:.4f}  (Δ = {dp_max_results[-1]-original_dp_max:+.4f})  |  DP avg:  {dp_avg_results[-1]:.4f}  (Δ = {dp_avg_results[-1]-original_dp_avg:+.4f})")
            print(f"    EO max:  {eo_max_results[-1]:.4f}  (Δ = {eo_max_results[-1]-original_eo_max:+.4f})  |  EO avg:  {eo_avg_results[-1]:.4f}  (Δ = {eo_avg_results[-1]-original_eo_avg:+.4f})")
            print(f"    PQP max: {pqp_max_results[-1]:.4f}  (Δ = {pqp_max_results[-1]-original_pqp_max:+.4f})  |  PQP avg: {pqp_avg_results[-1]:.4f}  (Δ = {pqp_avg_results[-1]-original_pqp_avg:+.4f})")
            
            # Find best fairness point
            best_dr_idx = DR_results.index(min(DR_results))
            print(f"\n{'Best Fairness Point (lowest DR)':^80}")
            print("-"*80)
            print(f"  Modifications: {values_range[best_dr_idx]}")
            print(f"  Accuracy:      {accuracy_results[best_dr_idx]:.4f}")
            print(f"  DR:            {DR_results[best_dr_idx]:.4f}")
            print(f"  DP gap:        {dp_gap_results[best_dr_idx]:.4f}")
            print(f"  EO gap:        {eo_gap_results[best_dr_idx]:.4f}")
            print(f"  PQ gap:        {pq_gap_results[best_dr_idx]:.4f}")
            print("\n" + "="*80)
            print("Note: Results not saved to CSV (save_results=False)")
            print("="*80 + "\n")
        
        return results

    def modify_binary_once(
        self, 
        threshold: float = 0.05,
        return_modified_features: bool = True
    ) -> Dict:
        """
        One-shot modification for datasets with binary sensitive attributes.
        
        Directly applies all modifications where SHAP values exceed the threshold,
        trains a new model once, and returns evaluation metrics.
        
        Args:
            threshold: Minimum SHAP value to consider for modification (default: 0.05)
                    Features with SHAP values above this will be replaced
            return_modified_features: Whether to return modified training data (default: True)
        
        Returns:
            dict: Dictionary containing:
                - 'original_metrics': Dict with accuracy, DR, DP, EO, PQP of original model
                - 'modified_metrics': Dict with accuracy, DR, DP, EO, PQP of modified model
                - 'num_modifications': Number of features modified
                - 'modified_X_train': Modified training features (if return_modified_features=True)
                - 'modified_y_train': Training labels (if return_modified_features=True)
                - 'modification_positions': List of (row, col) positions modified (if return_modified_features=True)
        """
        print("="*80)
        print(f"FairSHAP Binary One-Shot Modification (threshold={threshold})")
        print("="*80)
        
        # Step 1: Split data
        print(f"1. Split the {self.dataset_name} dataset into majority/minority groups by sensitive attribute and label")
        (X_train_majority_label0, y_train_majority_label0, 
        X_train_majority_label1, y_train_majority_label1,
        X_train_minority_label0, y_train_minority_label0, 
        X_train_minority_label1, y_train_minority_label1) = self._split_into_majority_minority_label0_label1()
        
        print(f'   X_train_majority_label0 shape: {X_train_majority_label0.shape}')
        print(f'   X_train_majority_label1 shape: {X_train_majority_label1.shape}')
        print(f'   X_train_minority_label0 shape: {X_train_minority_label0.shape}')
        print(f'   X_train_minority_label1 shape: {X_train_minority_label1.shape}')
        
        # Step 2: Initialize FairnessExplainer
        print('2. Initialize FairnessExplainer')
        sen_att_name = [self.sensitive_attri]
        sen_att = [self.X_test.columns.get_loc(name) for name in sen_att_name]
        priv_val = [1]
        unpriv_dict = [list(set(self.X_test.values[:, sa])) for sa in sen_att]
        for sa_list, pv in zip(unpriv_dict, priv_val):
            sa_list.remove(pv)
        
        fairness_explainer_original = FairnessExplainer(
            model=self.model, 
            sen_att=sen_att, 
            priv_val=priv_val, 
            unpriv_dict=unpriv_dict,
        )
        
        start_time = time.time()
        
        # Step 3: Process minority group
        print('3. Process minority group (match and compute SHAP values)')
        if self.matching_method == 'NN':
            matching_minority_label0 = NearestNeighborDataMatcher(
                X_labeled=X_train_minority_label0, 
                X_unlabeled=X_train_majority_label0
            ).match(n_neighbors=1)
            matching_minority_label1 = NearestNeighborDataMatcher(
                X_labeled=X_train_minority_label1, 
                X_unlabeled=X_train_majority_label1
            ).match(n_neighbors=1)
        elif self.matching_method == 'OT':
            matching_minority_label0 = OptimalTransportPolicy(
                X_labeled=X_train_minority_label0.values, 
                X_unlabeled=X_train_majority_label0.values
            ).match()
            matching_minority_label1 = OptimalTransportPolicy(
                X_labeled=X_train_minority_label1.values, 
                X_unlabeled=X_train_majority_label1.values
            ).match()
        else:
            raise ValueError('The matching method is not supported')
        
        # Compute SHAP values for minority group
        fairness_shapley_minority_value_label0 = fairness_explainer_original.shap_values(
            X=X_train_minority_label0.values,
            Y=y_train_minority_label0.values,
            X_baseline=X_train_majority_label0.values,
            matching=matching_minority_label0,
            sample_size=2000,
            shap_sample_size="auto",
        )
        
        fairness_shapley_minority_value_label1 = fairness_explainer_original.shap_values(
            X=X_train_minority_label1.values,
            Y=y_train_minority_label1.values,
            X_baseline=X_train_majority_label1.values,
            matching=matching_minority_label1,
            sample_size=2000,
            shap_sample_size="auto",
        )
        
        # Calculate q values for minority group
        q_minority_label0 = DataComposer(
            x_counterfactual=X_train_majority_label0.values, 
            joint_prob=matching_minority_label0, 
            method="max"
        ).calculate_q()
        
        q_minority_label1 = DataComposer(
            x_counterfactual=X_train_majority_label1.values, 
            joint_prob=matching_minority_label1, 
            method="max"
        ).calculate_q()
        
        fairness_shapley_minority_value = np.vstack((
            fairness_shapley_minority_value_label0, 
            fairness_shapley_minority_value_label1
        ))
        q_minority = np.vstack((q_minority_label0, q_minority_label1))
        
        # Step 4: Process majority group
        print('4. Process majority group (match and compute SHAP values)')
        if self.matching_method == 'NN':
            matching_majority_label0 = NearestNeighborDataMatcher(
                X_labeled=X_train_majority_label0, 
                X_unlabeled=X_train_minority_label0
            ).match(n_neighbors=1)
            matching_majority_label1 = NearestNeighborDataMatcher(
                X_labeled=X_train_majority_label1, 
                X_unlabeled=X_train_minority_label1
            ).match(n_neighbors=1)
        elif self.matching_method == 'OT':
            matching_majority_label0 = OptimalTransportPolicy(
                X_labeled=X_train_majority_label0.values, 
                X_unlabeled=X_train_minority_label0.values
            ).match()
            matching_majority_label1 = OptimalTransportPolicy(
                X_labeled=X_train_majority_label1.values, 
                X_unlabeled=X_train_minority_label1.values
            ).match()
        else:
            raise ValueError('The matching method is not supported')
        
        # Compute SHAP values for majority group
        fairness_shapley_majority_value_label0 = fairness_explainer_original.shap_values(
            X=X_train_majority_label0.values,
            Y=y_train_majority_label0.values,
            X_baseline=X_train_minority_label0.values,
            matching=matching_majority_label0,
            sample_size=2000,
            shap_sample_size="auto",
        )
        
        fairness_shapley_majority_value_label1 = fairness_explainer_original.shap_values(
            X=X_train_majority_label1.values,
            Y=y_train_majority_label1.values,
            X_baseline=X_train_minority_label1.values,
            matching=matching_majority_label1,
            sample_size=2000,
            shap_sample_size="auto",
        )
        
        # Calculate q values for majority group
        q_majority_label0 = DataComposer(
            x_counterfactual=X_train_minority_label0.values, 
            joint_prob=matching_majority_label0, 
            method="max"
        ).calculate_q()
        
        q_majority_label1 = DataComposer(
            x_counterfactual=X_train_minority_label1.values, 
            joint_prob=matching_majority_label1, 
            method="max"
        ).calculate_q()
        
        fairness_shapley_majority_value = np.vstack((
            fairness_shapley_majority_value_label0, 
            fairness_shapley_majority_value_label1
        ))
        q_majority = np.vstack((q_majority_label0, q_majority_label1))
        
        elapsed_time = time.time() - start_time
        print(f"   SHAP computation completed in {elapsed_time:.2f} seconds")
        
        # Step 5: Combine and filter by threshold
        print('5. Combine SHAP values and filter by threshold')
        fairness_shapley_value = np.vstack((
            fairness_shapley_minority_value, 
            fairness_shapley_majority_value
        ))
        varphi = np.where(fairness_shapley_value > threshold, fairness_shapley_value, 0)
        q = np.vstack((q_minority, q_majority))
        
        X_change = pd.concat([
            X_train_minority_label0, X_train_minority_label1,
            X_train_majority_label0, X_train_majority_label1
        ], axis=0)
        
        y_combined = pd.concat([
            y_train_minority_label0, y_train_minority_label1,
            y_train_majority_label0, y_train_majority_label1
        ], axis=0)
        
        non_zero_count = np.sum(varphi > 0)
        print(f'   Found {non_zero_count} features with SHAP values > {threshold}')
        
        # Step 6: Calculate original model metrics
        print('6. Calculate original model metrics on test set')
        y_pred = self.model.predict(self.X_test)
        original_accuracy = accuracy_score(self.y_test, y_pred)
        original_DR = fairness_value_function(sen_att, priv_val, unpriv_dict, self.X_test.values, self.model)
        
        priv_idx = self.X_test[self.sensitive_attri].to_numpy().astype(bool)
        g1_Cm, g0_Cm = marginalised_np_mat(y=self.y_test, y_hat=y_pred, pos_label=1, priv_idx=priv_idx)
        original_DP = grp1_DP(g1_Cm, g0_Cm)[0]
        original_EO = grp2_EO(g1_Cm, g0_Cm)[0]
        original_PQP = grp3_PQP(g1_Cm, g0_Cm)[0]
        
        print(f'   Original metrics - Accuracy: {original_accuracy:.4f}, DR: {original_DR:.4f}, '
            f'DP: {original_DP:.4f}, EO: {original_EO:.4f}, PQP: {original_PQP:.4f}')
        
        # Step 7: Apply all modifications at once
        print(f'7. Apply all {non_zero_count} modifications')
        
        # Flatten and sort varphi values
        flat_varphi = [(value, row, col) for row, row_vals in enumerate(varphi)
                    for col, value in enumerate(row_vals) if value > 0]
        flat_varphi_sorted = sorted(flat_varphi, key=lambda x: x[0], reverse=True)
        
        # Apply all modifications
        modification_positions = []
        for value, row_idx, col_idx in flat_varphi_sorted:
            X_change.iloc[row_idx, col_idx] = q[row_idx, col_idx]
            modification_positions.append((row_idx, col_idx))
        
        # Step 8: Train new model
        print('8. Train new model with modified data')
        model_new = XGBClassifier()
        model_new.fit(X_change, y_combined)
        
        # Step 9: Evaluate new model
        print('9. Evaluate modified model on test set')
        new_DR = fairness_value_function(sen_att, priv_val, unpriv_dict, self.X_test.values, model_new)
        y_hat = model_new.predict(self.X_test)
        new_accuracy = accuracy_score(self.y_test, y_hat)
        
        g1_Cm, g0_Cm = marginalised_np_mat(self.y_test, y_hat, 1, priv_idx)
        new_DP = grp1_DP(g1_Cm, g0_Cm)[0]
        new_EO = grp2_EO(g1_Cm, g0_Cm)[0]
        new_PQP = grp3_PQP(g1_Cm, g0_Cm)[0]
        
        print(f'   Modified metrics - Accuracy: {new_accuracy:.4f}, DR: {new_DR:.4f}, '
            f'DP: {new_DP:.4f}, EO: {new_EO:.4f}, PQP: {new_PQP:.4f}')
        
        # Prepare results
        results = {
            'original_metrics': {
                'accuracy': original_accuracy,
                'DR': original_DR,
                'DP': original_DP,
                'EO': original_EO,
                'PQP': original_PQP
            },
            'modified_metrics': {
                'accuracy': new_accuracy,
                'DR': new_DR,
                'DP': new_DP,
                'EO': new_EO,
                'PQP': new_PQP
            },
            'num_modifications': non_zero_count,
        }
        
        if return_modified_features:
            results['modified_X_train'] = X_change
            results['modified_y_train'] = y_combined
            results['modification_positions'] = modification_positions
        
        # Print summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Modifications applied: {non_zero_count}")
        print(f"Accuracy change: {original_accuracy:.4f} → {new_accuracy:.4f} "
            f"(Δ = {new_accuracy - original_accuracy:+.4f})")
        print(f"DR change:       {original_DR:.4f} → {new_DR:.4f} "
            f"(Δ = {new_DR - original_DR:+.4f})")
        print(f"DP change:       {original_DP:.4f} → {new_DP:.4f} "
            f"(Δ = {new_DP - original_DP:+.4f})")
        print(f"EO change:       {original_EO:.4f} → {new_EO:.4f} "
            f"(Δ = {new_EO - original_EO:+.4f})")
        print(f"PQP change:      {original_PQP:.4f} → {new_PQP:.4f} "
            f"(Δ = {new_PQP - original_PQP:+.4f})")
        print("="*80 + "\n")
        
        return results

    def modify_multi_once(
        self,
        threshold: float = 0.05,
        return_modified_features: bool = True
    ) -> Dict:
        """
        One-shot modification for datasets with multiple sensitive attribute values.
        
        Directly applies all modifications where SHAP values exceed the threshold,
        trains a new model once, and returns evaluation metrics including multi-group fairness.
        
        Args:
            threshold: Minimum SHAP value to consider for modification (default: 0.05)
                    Features with SHAP values above this will be replaced
            return_modified_features: Whether to return modified training data (default: True)
        
        Returns:
            dict: Dictionary containing:
                - 'original_metrics': Dict with accuracy, DR, and multi-group fairness metrics
                - 'modified_metrics': Dict with accuracy, DR, and multi-group fairness metrics
                - 'num_modifications': Number of features modified
                - 'modified_X_train': Modified training features (if return_modified_features=True)
                - 'modified_y_train': Training labels (if return_modified_features=True)
                - 'modification_positions': List of (row, col) positions modified (if return_modified_features=True)
        """
        print("="*80)
        print(f"FairSHAP Multi-Group One-Shot Modification (threshold={threshold})")
        print("="*80)
        
        # Step 1: Split data by privileged/unprivileged groups
        print(f'1. Split the {self.dataset_name} dataset by sensitive attribute values')
        splits = self._split_privileged_and_unprivileged()
        
        # Step 2: Initialize FairnessExplainer
        print('2. Initialize FairnessExplainer')
        sen_att_name = [self.sensitive_attri]
        sen_att = [self.X_test.columns.get_loc(name) for name in sen_att_name]
        priv_val = [1]
        unpriv_dict = [list(set(self.X_test.values[:, sa])) for sa in sen_att]
        for sa_list, pv in zip(unpriv_dict, priv_val):
            sa_list.remove(pv)
        
        fairness_explainer_original = FairnessExplainer(
            model=self.model, 
            sen_att=sen_att, 
            priv_val=priv_val, 
            unpriv_dict=unpriv_dict,
        )
        
        # Step 3: Compute SHAP values for all groups
        print('3. Compute SHAP values for privileged and unprivileged groups')
        start_time = time.time()
        
        if self.num_sensitive_values == 5:
            X_base, Y_base, fairness_shapley_value, q = \
                self._modify_5_races(splits, fairness_explainer_original)
        elif self.num_sensitive_values == 6:
            X_base, Y_base, fairness_shapley_value, q = \
                self._modify_6_races(splits, fairness_explainer_original)
        else:
            raise ValueError(f'num_sensitive_values={self.num_sensitive_values} not supported')
        
        elapsed = time.time() - start_time
        print(f'   SHAP computation completed in {elapsed:.2f} seconds')
        
        # Step 4: Filter SHAP values by threshold
        print('4. Filter SHAP values by threshold')
        varphi = np.where(fairness_shapley_value > threshold, fairness_shapley_value, 0)
        non_zero_count = np.sum(varphi > 0)
        print(f'   Found {non_zero_count} features with SHAP values > {threshold}')
        
        # Step 5: Calculate original model metrics
        print('5. Calculate original model metrics on test set')
        y_pred = self.model.predict(self.X_test)
        original_accuracy = accuracy_score(self.y_test, y_pred)
        original_DR = fairness_value_function(sen_att, priv_val, unpriv_dict, self.X_test.values, self.model)
        
        # Calculate multi-group fairness metrics
        y = self.y_test.to_numpy()
        hx = y_pred
        A = self.X_test[self.sensitive_attri].to_numpy()
        
        (
            original_dp_gap,
            original_eo_gap,
            original_pq_gap,
            original_dp_max,
            original_dp_avg,
            original_eo_max,
            original_eo_avg,
            original_pqp_max,
            original_pqp_avg,
        ) = evaluate(y=y, hx=hx, A=A, priv_val=1)
        
        print(f'   Original metrics - Accuracy: {original_accuracy:.4f}, DR: {original_DR:.4f}')
        print(f'   Original fairness - DP_gap: {original_dp_gap:.4f}, EO_gap: {original_eo_gap:.4f}, '
            f'PQ_gap: {original_pq_gap:.4f}')
        
        # Step 6: Apply all modifications at once
        print(f'6. Apply all {non_zero_count} modifications')
        
        # Flatten and sort varphi values
        flat_varphi = [(value, row, col) for row, row_vals in enumerate(varphi)
                    for col, value in enumerate(row_vals) if value > 0]
        flat_varphi_sorted = sorted(flat_varphi, key=lambda x: x[0], reverse=True)
        
        # Apply all modifications
        X_change = copy.deepcopy(X_base)
        modification_positions = []
        for value, row_idx, col_idx in flat_varphi_sorted:
            X_change.iloc[row_idx, col_idx] = q[row_idx, col_idx]
            modification_positions.append((row_idx, col_idx))
        
        # Step 7: Train new model
        print('7. Train new model with modified data')
        y_col = Y_base.to_numpy().reshape(-1, 1)
        
        # Automatically detect model type and train accordingly
        model_class = type(self.model)
        model_new = model_class()
        model_new.fit(X_change, y_col.ravel())
        
        # Step 8: Evaluate new model
        print('8. Evaluate modified model on test set')
        new_DR = fairness_value_function(sen_att, priv_val, unpriv_dict, self.X_test.values, model_new)
        y_hat = model_new.predict(self.X_test)
        new_accuracy = accuracy_score(self.y_test, y_hat)
        
        # Calculate multi-group metrics
        (
            new_dp_gap,
            new_eo_gap,
            new_pq_gap,
            new_dp_max,
            new_dp_avg,
            new_eo_max,
            new_eo_avg,
            new_pqp_max,
            new_pqp_avg,
        ) = evaluate(y=self.y_test.to_numpy(), hx=y_hat, A=A, priv_val=1)
        
        print(f'   Modified metrics - Accuracy: {new_accuracy:.4f}, DR: {new_DR:.4f}')
        print(f'   Modified fairness - DP_gap: {new_dp_gap:.4f}, EO_gap: {new_eo_gap:.4f}, '
            f'PQ_gap: {new_pq_gap:.4f}')
        
        # Prepare results
        results = {
            'original_metrics': {
                'accuracy': original_accuracy,
                'DR': original_DR,
                'dp_gap': original_dp_gap,
                'eo_gap': original_eo_gap,
                'pq_gap': original_pq_gap,
                'dp_max': original_dp_max,
                'dp_avg': original_dp_avg,
                'eo_max': original_eo_max,
                'eo_avg': original_eo_avg,
                'pqp_max': original_pqp_max,
                'pqp_avg': original_pqp_avg
            },
            'modified_metrics': {
                'accuracy': new_accuracy,
                'DR': new_DR,
                'dp_gap': new_dp_gap,
                'eo_gap': new_eo_gap,
                'pq_gap': new_pq_gap,
                'dp_max': new_dp_max,
                'dp_avg': new_dp_avg,
                'eo_max': new_eo_max,
                'eo_avg': new_eo_avg,
                'pqp_max': new_pqp_max,
                'pqp_avg': new_pqp_avg
            },
            'num_modifications': non_zero_count,
        }
        
        if return_modified_features:
            results['modified_X_train'] = X_change
            results['modified_y_train'] = Y_base
            results['modification_positions'] = modification_positions
        
        # Print summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Modifications applied: {non_zero_count}")
        print(f"\nAccuracy:  {original_accuracy:.4f} → {new_accuracy:.4f} "
            f"(Δ = {new_accuracy - original_accuracy:+.4f})")
        print(f"DR:        {original_DR:.4f} → {new_DR:.4f} "
            f"(Δ = {new_DR - original_DR:+.4f})")
        print(f"\nBinary Group Gaps:")
        print(f"  DP gap:  {original_dp_gap:.4f} → {new_dp_gap:.4f} "
            f"(Δ = {new_dp_gap - original_dp_gap:+.4f})")
        print(f"  EO gap:  {original_eo_gap:.4f} → {new_eo_gap:.4f} "
            f"(Δ = {new_eo_gap - original_eo_gap:+.4f})")
        print(f"  PQ gap:  {original_pq_gap:.4f} → {new_pq_gap:.4f} "
            f"(Δ = {new_pq_gap - original_pq_gap:+.4f})")
        print(f"\nMulti-Group Metrics:")
        print(f"  DP max:  {original_dp_max:.4f} → {new_dp_max:.4f} "
            f"(Δ = {new_dp_max - original_dp_max:+.4f})")
        print(f"  DP avg:  {original_dp_avg:.4f} → {new_dp_avg:.4f} "
            f"(Δ = {new_dp_avg - original_dp_avg:+.4f})")
        print(f"  EO max:  {original_eo_max:.4f} → {new_eo_max:.4f} "
            f"(Δ = {new_eo_max - original_eo_max:+.4f})")
        print(f"  EO avg:  {original_eo_avg:.4f} → {new_eo_avg:.4f} "
            f"(Δ = {new_eo_avg - original_eo_avg:+.4f})")
        print(f"  PQP max: {original_pqp_max:.4f} → {new_pqp_max:.4f} "
            f"(Δ = {new_pqp_max - original_pqp_max:+.4f})")
        print(f"  PQP avg: {original_pqp_avg:.4f} → {new_pqp_avg:.4f} "
            f"(Δ = {new_pqp_avg - original_pqp_avg:+.4f})")
        print("="*80 + "\n")
        
        return results

    def _split_into_majority_minority_label0_label1(self):
        """
        Split the training dataset into four groups based on sensitive attribute and labels.
        
        For binary sensitive attributes only.
        
        The data is divided into:
        - Majority group (larger count of sensitive attribute value)
        - Minority group (smaller count of sensitive attribute value)
        Each group is further split by:
        - Label 0 (negative class)
        - Label 1 (positive class)
        
        Returns:
            Tuple of 8 elements:
                - X_train_majority_label0: Features for majority group with label 0
                - y_train_majority_label0: Labels for majority group with label 0
                - X_train_majority_label1: Features for majority group with label 1
                - y_train_majority_label1: Labels for majority group with label 1
                - X_train_minority_label0: Features for minority group with label 0
                - y_train_minority_label0: Labels for minority group with label 0
                - X_train_minority_label1: Features for minority group with label 1
                - y_train_minority_label1: Labels for minority group with label 1
        """
        group_division = self.X_train[self.sensitive_attri].value_counts()
        
        # Split X_train into majority and minority groups
        if group_division[0] > group_division[1]:
            majority = self.X_train[self.sensitive_attri] == 0
            X_train_majority = self.X_train[majority]
            y_train_majority = self.y_train[majority]
            minority = self.X_train[self.sensitive_attri] == 1
            X_train_minority = self.X_train[minority]
            y_train_minority = self.y_train[minority]

        else:
            majority = self.X_train[self.sensitive_attri] == 1
            X_train_majority = self.X_train[majority]
            y_train_majority = self.y_train[majority]
            minority = self.X_train[self.sensitive_attri] == 0
            X_train_minority = self.X_train[minority]
            y_train_minority = self.y_train[minority]

        y_train_majority_label1 = y_train_majority[y_train_majority == 1]
        y_train_majority_label0 = y_train_majority[y_train_majority == 0]
        y_train_minority_label1 = y_train_minority[y_train_minority == 1]
        y_train_minority_label0 = y_train_minority[y_train_minority == 0]

        X_train_majority_label0 = X_train_majority.loc[y_train_majority_label0.index]
        X_train_majority_label1 = X_train_majority.loc[y_train_majority_label1.index]
        X_train_minority_label0 = X_train_minority.loc[y_train_minority_label0.index]
        X_train_minority_label1 = X_train_minority.loc[y_train_minority_label1.index]

        return X_train_majority_label0, y_train_majority_label0, X_train_majority_label1, y_train_majority_label1, X_train_minority_label0, y_train_minority_label0, X_train_minority_label1, y_train_minority_label1

    def _split_privileged_and_unprivileged(self):
        """
        Split training data by multiple sensitive attribute values.
        
        For multi-valued sensitive attributes (e.g., race with 5-6 categories).
        
        Divides self.X_train by sensitive_attri into:
        - privileged (value == 1)
        - each other value v as separate group
        - all non-1 values merged as 'unprivileged' group
        
        Within each group, further split by label 0/1.
        
        Returns:
            Dictionary with structure:
            {
                'privileged': {'label0': (X, y), 'label1': (X, y)},
                value1: {'label0': (X, y), 'label1': (X, y)},
                value2: {...},
                ...
                'unprivileged': {'label0': (X, y), 'label1': (X, y)}
            }
        """
        splits = {}
        
        # 1. Privileged group (value == 1)
        mask_priv = (self.X_train[self.sensitive_attri] == 1)
        X_priv = self.X_train[mask_priv]
        y_priv = self.y_train[mask_priv]
        splits['privileged'] = {
            'label0': (X_priv[y_priv == 0], y_priv[y_priv == 0]),
            'label1': (X_priv[y_priv == 1], y_priv[y_priv == 1])
        }
        
        # 2. Each non-1 value as separate group
        all_values = sorted(self.X_train[self.sensitive_attri].unique())
        for v in all_values:
            if v == 1:
                continue
            mask_v = (self.X_train[self.sensitive_attri] == v)
            X_v, y_v = self.X_train[mask_v], self.y_train[mask_v]
            splits[v] = {
                'label0': (X_v[y_v == 0], y_v[y_v == 0]),
                'label1': (X_v[y_v == 1], y_v[y_v == 1])
            }
        
        # 3. Merge all ≠1 values as "unprivileged" group
        mask_unpriv = (self.X_train[self.sensitive_attri] != 1)
        X_unpriv = self.X_train[mask_unpriv]
        y_unpriv = self.y_train[mask_unpriv]
        splits['unprivileged'] = {
            'label0': (X_unpriv[y_unpriv == 0], y_unpriv[y_unpriv == 0]),
            'label1': (X_unpriv[y_unpriv == 1], y_unpriv[y_unpriv == 1])
        }
        
        return splits
        
    def _modify_5_races(self, splits: Dict, fairness_explainer: FairnessExplainer):
        """
        Process datasets with 5 racial categories using aggregated unprivileged group approach.
        
        This method treats all unprivileged groups as a single aggregate group rather than
        processing each racial group separately. This simplifies the code and improves
        statistical stability.
        
        Args:
            splits: Dictionary containing data splits for each racial group
            fairness_explainer: FairnessExplainer instance
        
        Returns:
            Tuple: (X_base, Y_base, fairness_shapley_value, q)
        """
        # Extract privileged and aggregated unprivileged data
        X_priv_l1, y_priv_l1 = splits['privileged']['label1']
        X_priv_l0, y_priv_l0 = splits['privileged']['label0']
        X_unpriv_all_l1, y_unpriv_all_l1 = splits['unprivileged']['label1']
        X_unpriv_all_l0, y_unpriv_all_l0 = splits['unprivileged']['label0']
        
        # Modify privileged group using unprivileged as background
        fairness_shapley_pri, q_pri = self._modify_group(
            modify_data_x_label1=X_priv_l1,
            modify_data_y_label1=y_priv_l1,
            modify_data_x_label0=X_priv_l0,
            modify_data_y_label0=y_priv_l0,
            background_data_x_label1=X_unpriv_all_l1,
            background_data_y_label1=y_unpriv_all_l1,
            background_data_x_label0=X_unpriv_all_l0,
            background_data_y_label0=y_unpriv_all_l0,
            fairness_explainer=fairness_explainer
        )
        
        # Modify unprivileged group using privileged as background
        fairness_shapley_unpri, q_unpri = self._modify_group(
            modify_data_x_label1=X_unpriv_all_l1,
            modify_data_y_label1=y_unpriv_all_l1,
            modify_data_x_label0=X_unpriv_all_l0,
            modify_data_y_label0=y_unpriv_all_l0,
            background_data_x_label1=X_priv_l1,
            background_data_y_label1=y_priv_l1,
            background_data_x_label0=X_priv_l0,
            background_data_y_label0=y_priv_l0,
            fairness_explainer=fairness_explainer
        )
        
        # Combine results
        X_base = pd.concat([X_priv_l1, X_priv_l0, X_unpriv_all_l1, X_unpriv_all_l0], axis=0)
        Y_base = pd.concat([y_priv_l1, y_priv_l0, y_unpriv_all_l1, y_unpriv_all_l0], axis=0)
        fairness_shapley_value = np.vstack((fairness_shapley_pri, fairness_shapley_unpri))
        q = np.vstack((q_pri, q_unpri))
        
        return X_base, Y_base, fairness_shapley_value, q

    def _modify_6_races(self, splits: Dict, fairness_explainer: FairnessExplainer):
        """
        Process datasets with 6 racial categories using aggregated unprivileged group approach.
        
        This method treats all unprivileged groups as a single aggregate group rather than
        processing each racial group separately. This simplifies the code and improves
        statistical stability.
        
        Args:
            splits: Dictionary containing data splits for each racial group
            fairness_explainer: FairnessExplainer instance
        
        Returns:
            Tuple: (X_base, Y_base, fairness_shapley_value, q)
        """
        # Extract privileged and aggregated unprivileged data
        X_priv_l1, y_priv_l1 = splits['privileged']['label1']
        X_priv_l0, y_priv_l0 = splits['privileged']['label0']
        X_unpriv_all_l1, y_unpriv_all_l1 = splits['unprivileged']['label1']
        X_unpriv_all_l0, y_unpriv_all_l0 = splits['unprivileged']['label0']
        
        # Modify privileged group using unprivileged as background
        fairness_shapley_pri, q_pri = self._modify_group(
            modify_data_x_label1=X_priv_l1,
            modify_data_y_label1=y_priv_l1,
            modify_data_x_label0=X_priv_l0,
            modify_data_y_label0=y_priv_l0,
            background_data_x_label1=X_unpriv_all_l1,
            background_data_y_label1=y_unpriv_all_l1,
            background_data_x_label0=X_unpriv_all_l0,
            background_data_y_label0=y_unpriv_all_l0,
            fairness_explainer=fairness_explainer
        )
        
        # Modify unprivileged group using privileged as background
        fairness_shapley_unpri, q_unpri = self._modify_group(
            modify_data_x_label1=X_unpriv_all_l1,
            modify_data_y_label1=y_unpriv_all_l1,
            modify_data_x_label0=X_unpriv_all_l0,
            modify_data_y_label0=y_unpriv_all_l0,
            background_data_x_label1=X_priv_l1,
            background_data_y_label1=y_priv_l1,
            background_data_x_label0=X_priv_l0,
            background_data_y_label0=y_priv_l0,
            fairness_explainer=fairness_explainer
        )
        
        # Combine results
        X_base = pd.concat([X_priv_l1, X_priv_l0, X_unpriv_all_l1, X_unpriv_all_l0], axis=0)
        Y_base = pd.concat([y_priv_l1, y_priv_l0, y_unpriv_all_l1, y_unpriv_all_l0], axis=0)
        fairness_shapley_value = np.vstack((fairness_shapley_pri, fairness_shapley_unpri))
        q = np.vstack((q_pri, q_unpri))
        
        return X_base, Y_base, fairness_shapley_value, q

    def _modify_group(
        self,
        modify_data_x_label1: pd.DataFrame,
        modify_data_y_label1: pd.Series,
        modify_data_x_label0: pd.DataFrame,
        modify_data_y_label0: pd.Series,
        background_data_x_label1: pd.DataFrame,
        background_data_y_label1: pd.Series,
        background_data_x_label0: pd.DataFrame,
        background_data_y_label0: pd.Series,
        fairness_explainer: FairnessExplainer
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        General modification function for matching and computing SHAP values.
        
        This method performs the core FairSHAP computation for a group:
        1. Matches instances between modify group and background group (separately for label 0 and 1)
        2. Computes fairness SHAP values to identify unfair features
        3. Generates counterfactual feature values (q) for replacement
        
        This corresponds to the modify() function in modify_num_races.py.
        
        Args:
            modify_data_x_label1: Features to modify with label 1 (positive class)
            modify_data_y_label1: Labels for modify data with label 1
            modify_data_x_label0: Features to modify with label 0 (negative class)
            modify_data_y_label0: Labels for modify data with label 0
            background_data_x_label1: Background features with label 1
            background_data_y_label1: Background labels with label 1
            background_data_x_label0: Background features with label 0
            background_data_y_label0: Background labels with label 0
            fairness_explainer: FairnessExplainer instance
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - fairness_shapley_value: SHAP values indicating feature importance for fairness
                - q: Counterfactual feature values for replacement
        """
        # Match and compute SHAP for label 1 (positive class)
        if self.matching_method == 'NN':
            matching_label1 = NearestNeighborDataMatcher(
                X_labeled=modify_data_x_label1,
                X_unlabeled=background_data_x_label1
            ).match(n_neighbors=1)
        else:
            matching_label1 = OptimalTransportPolicy(
                X_labeled=modify_data_x_label1.values,
                X_unlabeled=background_data_x_label1.values
            ).match()
        
        fairness_shapley_label1 = fairness_explainer.shap_values(
            X=modify_data_x_label1.values,
            Y=modify_data_y_label1.values,
            X_baseline=background_data_x_label1.values,
            matching=matching_label1,
            sample_size=200,
            shap_sample_size="auto"
        )
        
        # Match and compute SHAP for label 0 (negative class)
        if self.matching_method == 'NN':
            matching_label0 = NearestNeighborDataMatcher(
                X_labeled=modify_data_x_label0,
                X_unlabeled=background_data_x_label0
            ).match(n_neighbors=1)
        else:
            matching_label0 = OptimalTransportPolicy(
                X_labeled=modify_data_x_label0.values,
                X_unlabeled=background_data_x_label0.values
            ).match()
        
        fairness_shapley_label0 = fairness_explainer.shap_values(
            X=modify_data_x_label0.values,
            Y=modify_data_y_label0.values,
            X_baseline=background_data_x_label0.values,
            matching=matching_label0,
            sample_size=200,
            shap_sample_size="auto"
        )
        
        # Stack SHAP values (label 1 on top, label 0 on bottom)
        fairness_shapley_value = np.vstack((fairness_shapley_label1, fairness_shapley_label0))
        
        # Calculate counterfactual values (q) for label 1
        q_label1 = DataComposer(
            x_counterfactual=background_data_x_label1.values,
            joint_prob=matching_label1,
            method="max"
        ).calculate_q()
        
        # Calculate counterfactual values (q) for label 0
        q_label0 = DataComposer(
            x_counterfactual=background_data_x_label0.values,
            joint_prob=matching_label0,
            method="max"
        ).calculate_q()
        
        # Stack q values (label 1 on top, label 0 on bottom)
        q = np.vstack((q_label1, q_label0))
        
        return fairness_shapley_value, q
        


def evaluate(y, hx, A, priv_val: int = 1):
    """
    Calculate single/multi-group fairness metrics.

    Parameters
    ----------
    y        : np.ndarray, shape (n,)   True labels
    hx       : np.ndarray, shape (n,)   Predicted labels (already discretized)
    A        : np.ndarray, shape (n,)   Sensitive attribute values
    priv_val : int, default 1           Which value is considered the privileged group

    Returns
    -------
    tuple(dp_gap, eo_gap, pq_gap,
          dp_max, dp_avg,
          eo_max, eo_avg,
          pq_max, pq_avg)
    """
    # 1) Multi-group confusion matrices (list[np.ndarray(4,)])
    gs_Cm, vA, idx, _ = marginalised_np_gen(y, hx, A, priv_val=priv_val)

    # 2) Binary group: privileged vs rest
    g1_Cm = gs_Cm[idx]                                  # privileged
    g0_Cm = np.sum(gs_Cm[:idx] + gs_Cm[idx + 1:], axis=0)  # rest
    dp_gap = calc_fair_group(*unpriv_group_one(g1_Cm, g0_Cm))
    eo_gap = calc_fair_group(*unpriv_group_two(g1_Cm, g0_Cm))
    pq_gap = calc_fair_group(*unpriv_group_thr(g1_Cm, g0_Cm))

    # 3) Multi-group: DP / EO / PQP
    idx_Sjs = [A == val for val in vA]

    # extGrp*_sing are wrapped by fantasy_timer, returns ((max, avg, alt), elapsed)
    (dp_max, dp_avg, _), _ = extGrp1_DP_sing(y, hx, idx_Sjs)
    (eo_max, eo_avg, _), _ = extGrp2_EO_sing(y, hx, idx_Sjs)
    (pq_max, pq_avg, _), _ = extGrp3_PQP_sing(y, hx, idx_Sjs)

    return (dp_gap, eo_gap, pq_gap,
            dp_max, dp_avg,
            eo_max, eo_avg,
            pq_max, pq_avg)
