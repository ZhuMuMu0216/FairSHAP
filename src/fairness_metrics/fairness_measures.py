import numpy as np
# ================================================================================
# Fairness Metrics
# ================================================================================

# Discriminative Risk (DR)
def fairness_value_function(sen_att, priv_val, unpriv_dict, X, model):
    """
    Compute the individual fairness measure: Discriminative Risk (DR).
    
    This function perturbs the sensitive attribute of each instance and measures
    how much the model's predictions change. It quantifies whether individuals 
    with identical non-sensitive features but different sensitive attributes 
    receive different predictions (indicating discrimination).
    
    Process:
    1. Perturb sensitive attributes: flip privileged ↔ unprivileged values
    2. Compare predictions before and after perturbation
    3. Return average prediction change across all instances
    
    Args:
        sen_att (list): Indices of sensitive attributes in feature matrix
        priv_val (list): Values indicating privileged group for each sensitive attribute
        unpriv_dict (list): List of unprivileged values for each sensitive attribute
        X (np.ndarray): Feature matrix (N samples × D features)
        model: Trained model with predict_proba method
    
    Returns:
        float: Mean absolute difference in positive class probabilities between
               original and perturbed data. Higher values indicate greater discrimination.
    """
    # Perturb sensitive attributes for all instances
    X_disturbed = perturb_numpy_ver(
        X=X,
        sen_att=sen_att,
        priv_val=priv_val,
        unpriv_dict=unpriv_dict,
        ratio=1.0,  # Perturb 100% of instances
    )
    # Get predictions for original and perturbed data
    fx = model.predict_proba(X)[:, 1]  # Original predictions
    fx_q = model.predict_proba(X_disturbed)[:, 1]  # Predictions after perturbation
    # Return average prediction change
    return np.mean(np.abs(fx - fx_q))

# Group fairness: Demographic Parity (DP)
def grp1_DP(g1_Cm, g0_Cm):
    """
    Compute Demographic Parity (DP) fairness metric.
    
    Demographic Parity measures the difference in positive prediction rates
    between privileged and unprivileged groups.
    
    Args:
        g1_Cm (tuple): Confusion matrix for privileged group (TP, FP, FN, TN)
        g0_Cm (tuple): Confusion matrix for unprivileged group (TP, FP, FN, TN)
    
    Returns:
        tuple: (DP_gap, priv_rate, unpriv_rate) where
               - DP_gap: Absolute difference in positive prediction rates
               - priv_rate: Positive prediction rate for privileged group
               - unpriv_rate: Positive prediction rate for unprivileged group
    """
    g1 = g1_Cm[0] + g1_Cm[1]
    g1 = zero_division(g1, sum(g1_Cm))
    g0 = g0_Cm[0] + g0_Cm[1]
    g0 = zero_division(g0, sum(g0_Cm))
    return abs(g0 - g1), float(g1), float(g0)

# Group fairness: Equality of Opportunity (EO)
def grp2_EO(g1_Cm, g0_Cm):
    """
    Compute Equalized Odds (EO) fairness metric.
    
    Equalized Odds measures the difference in true positive rates (TPR) between
    privileged and unprivileged groups. TPR = TP / (TP + FN) = recall.
    
    Args:
        g1_Cm (tuple): Confusion matrix for privileged group (TP, FP, FN, TN)
        g0_Cm (tuple): Confusion matrix for unprivileged group (TP, FP, FN, TN)
    
    Returns:
        tuple: (EO_gap, priv_tpr, unpriv_tpr) where
               - EO_gap: Absolute difference in true positive rates
               - priv_tpr: True positive rate for privileged group
               - unpriv_tpr: True positive rate for unprivileged group
    """
    g1 = g1_Cm[0] + g1_Cm[2]
    g1 = zero_division(g1_Cm[0], g1)
    g0 = g0_Cm[0] + g0_Cm[2]
    g0 = zero_division(g0_Cm[0], g0)
    return abs(g0 - g1), float(g1), float(g0)

# Group fairness: Predictive Parity (PQP)
def grp3_PQP(g1_Cm, g0_Cm):
    """
    Compute Predictive Equality (PE) or Predictive Parity fairness metric.
    
    Predictive Equality measures the difference in positive predictive values (precision)
    between privileged and unprivileged groups. Precision = TP / (TP + FP).
    
    Args:
        g1_Cm (tuple): Confusion matrix for privileged group (TP, FP, FN, TN)
        g0_Cm (tuple): Confusion matrix for unprivileged group (TP, FP, FN, TN)
    
    Returns:
        tuple: (PE_gap, priv_precision, unpriv_precision) where
               - PE_gap: Absolute difference in precision values
               - priv_precision: Precision for privileged group
               - unpriv_precision: Precision for unprivileged group
    """
    g1 = g1_Cm[0] + g1_Cm[1]
    g1 = zero_division(g1_Cm[0], g1)
    g0 = g0_Cm[0] + g0_Cm[1]
    g0 = zero_division(g0_Cm[0], g0)
    return abs(g0 - g1), float(g1), float(g0)

def zero_division(dividend, divisor):
    """
    Safe division that handles zero denominators.
    
    Args:
        dividend: Numerator
        divisor: Denominator
    
    Returns:
        float: dividend/divisor, or 0.0 if both are zero, or 10.0 if divisor is zero (and dividend is non-zero)
    """
    if divisor == 0 and dividend == 0:
        return 0.
    elif divisor == 0:
        return 10.  # Large value to indicate extreme case
    return dividend / divisor

def contingency_tab_bi(y, y_hat, pos=1):
    """
    Compute binary confusion matrix (TP, FP, FN, TN).
    
    Args:
        y (array-like): True labels
        y_hat (array-like): Predicted labels
        pos (int): The positive class label (default: 1)
    
    Returns:
        tuple: (TP, FP, FN, TN) - True Positives, False Positives, False Negatives, True Negatives
    """
    tp = np.sum((y == pos) & (y_hat == pos))
    fn = np.sum((y == pos) & (y_hat != pos))
    fp = np.sum((y != pos) & (y_hat == pos))
    tn = np.sum((y != pos) & (y_hat != pos))
    return tp, fp, fn, tn

def marginalised_np_mat(y, y_hat, pos_label=1, priv_idx=list()):
    """
    Compute confusion matrices for privileged and unprivileged groups.
    
    This function splits the true and predicted labels by privileged/unprivileged
    groups and computes separate confusion matrices for fairness analysis.
    
    Args:
        y (array-like): True labels
        y_hat (array-like): Predicted labels
        pos_label (int): The positive class label (default: 1)
        priv_idx (array-like): Boolean array indicating privileged group members
    
    Returns:
        tuple: (g1_Cm, g0_Cm) where
               - g1_Cm: Confusion matrix (TP, FP, FN, TN) for privileged group
               - g0_Cm: Confusion matrix (TP, FP, FN, TN) for unprivileged group
    """
    if isinstance(y, list) or isinstance(y_hat, list):
        y, y_hat = np.array(y), np.array(y_hat)

    g1_y = y[priv_idx]
    g0_y = y[~priv_idx]
    g1_hx = y_hat[priv_idx]
    g0_hx = y_hat[~priv_idx]

    g1_Cm = contingency_tab_bi(g1_y, g1_hx, pos_label)
    g0_Cm = contingency_tab_bi(g0_y, g0_hx, pos_label)
    return g1_Cm, g0_Cm


# ================================================================================
# Perturbation Functions
# ================================================================================

def perturb_pandas_ver(X, sen_att, priv_val, ratio=0.5):
    """
    Perturb sensitive attributes in a pandas DataFrame.
    
    This function creates a copy of the DataFrame and randomly flips the values
    of sensitive attributes. For each sensitive attribute:
    - If current value is NOT privileged: change to privileged value
    - If current value IS privileged: change to a random unprivileged value
    
    Args:
        X (pd.DataFrame): Input DataFrame with features
        sen_att (list): Column names of sensitive attributes
        priv_val (list): Privileged value for each sensitive attribute
        ratio (float): Probability of perturbing each attribute (default: 0.5)
    
    Returns:
        pd.DataFrame: Perturbed copy of the input DataFrame
    """
    unpriv_dict = [X[sa].unique().tolist() for sa in sen_att]
    for sa_list, pv in zip(unpriv_dict, priv_val):
        sa_list.remove(pv)

    X_qtb = X.copy()
    num, dim = len(X_qtb), len(sen_att)
    if dim > 1:
        new_attr_name = "-".join(sen_att)

    for i, ti in enumerate(X.index):
        prng = np.random.rand(dim)
        prng = prng <= ratio

        for j, sa, pv, un in zip(range(dim), sen_att, priv_val, unpriv_dict):
            if not prng[j]:
                continue

            if X_qtb.iloc[i][sa] != pv:
                X_qtb.loc[ti, sa] = pv
            else:
                X_qtb.loc[ti, sa] = np.random.choice(un)

    return X_qtb  # pd.DataFrame

def perturb_numpy_ver(X, sen_att, priv_val, unpriv_dict, ratio=0.5):
    """
    Perturb sensitive attributes in a NumPy array.
    
    This function creates a copy of the array and randomly flips the values
    of sensitive attributes. For each sensitive attribute:
    - If current value is NOT privileged: change to privileged value
    - If current value IS privileged: change to a random unprivileged value
    
    Args:
        X (np.ndarray): Input array (N samples × D features)
        sen_att (list): Column indices of sensitive attributes
        priv_val (list): Privileged value for each sensitive attribute
        unpriv_dict (list): List of unprivileged values for each sensitive attribute
        ratio (float): Probability of perturbing each attribute (default: 0.5)
    
    Returns:
        np.ndarray: Perturbed copy of the input array
    """
    X_qtb = X.copy()
    num, dim = len(X_qtb), len(sen_att)

    for i in range(num):
        prng = np.random.rand(dim)
        prng = prng <= ratio

        for j, sa, pv, un in zip(range(dim), sen_att, priv_val, unpriv_dict):
            if not prng[j]:
                continue

            if X_qtb[i, sa] != pv:
                X_qtb[i, sa] = pv
            else:
                X_qtb[i, sa] = np.random.choice(un)

    return X_qtb  # np.ndarray

if __name__ == "__main__":

  pairs = [(1,1), (1,0), (0,1), (0,0)]

  for i, (g1_y, g1_hx) in enumerate(pairs, 1):
      # Here we assume g1_y and g1_hx each contain a single sample;
      # wrap them as array([value])
      arr_y = np.array([g1_y])
      arr_hat = np.array([g1_hx])
      
      # Compute confusion matrix (tp, fp, fn, tn)
      g1_Cm = contingency_tab_bi(y=arr_y, y_hat=arr_hat, pos=1)
      
      # Assume g0_y and g0_hx are always 0, representing an empty g0
      # or a group with a single negative example.
      # In this demo we only focus on g1_Cm; g0_Cm is a placeholder.
      g0_Cm = (0,0,0,0)
      
      print(f"=== Test case {i}: (g1_y={g1_y}, g1_hx={g1_hx}) ===")
      print("g1_Cm =", g1_Cm)

      # -- Demographic Parity (DP) --
      #   DP = | P(f=1|g1) - P(f=1|g0) |
      #   Here g0_Cm=0, so P(f=1|g0)=0
      numerator = g1_Cm[0] + g1_Cm[1]  # tp + fp
      denominator = sum(g1_Cm)        # total samples = tp+fp+fn+tn
      g1_rate = zero_division(numerator, denominator)
      DP = abs(g1_rate - 0)
      
      # -- Equality of Opportunity (EO) --
      #   EO = | TPR(g1) - TPR(g0) |
      #   Here g0_Cm=0, so TPR(g0)=0
      tpr_denominator = g1_Cm[0] + g1_Cm[2]  # tp + fn = total actual positives
      g1_tpr = zero_division(g1_Cm[0], tpr_denominator)
      EO = abs(g1_tpr - 0)
      
      # -- Predictive Parity (PQP) --
      #   PQP = | Precision(g1) - Precision(g0) |
      #   Here g0_Cm=0, so Precision(g0)=0
      prec_denominator = g1_Cm[0] + g1_Cm[1]  # tp + fp = total predicted positives
      g1_prec = zero_division(g1_Cm[0], prec_denominator)
      PQP = abs(g1_prec - 0)

      # Print results
      print(f"DP:  {DP:.4f}")
      print(f"EO:  {EO:.4f}")
      print(f"PQP: {PQP:.4f}")
      print("------------------------------------")  
  pass
