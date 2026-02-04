import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Plot the values of various fairness metrics after FairSHAP modification, 
compared to their initial values before modification.
"""

def plot_fairness_improvement(
    folds,
    original_DR,         
    original_DP,         
    original_EO,         
    original_PQP,        
    original_recall,     # New: list of original recall values corresponding to each fold
    original_precision,  # New: list of original precision values corresponding to each fold
    original_sufficiency,# New: list of original sufficiency values corresponding to each fold
    stop_when_no_data=3,
    min_action=1,
    baseline=0.0,
    figsize=(15, 10),    # Adjusted default size to fit more subplots
    fill_alpha=0.2,
    fill_color='b'
):
    """
    Plot multiple subplots showing the difference between new metric values and original values
    (original_XXX) for DR, DP, EO, PQP, Recall, Precision, and Sufficiency,
    as a function of action_number with mean ± standard deviation.

    For each subplot:
      1) Subtract the original_XXX value from the new_XXX column in each fold.
      2) Iterate from action=1 up to max action, stopping if stop_when_no_data folds have no data.
      3) Compute mean and standard deviation across valid folds and plot.
      4) Add a reference line at y=baseline.

    Parameters:
    -----------
    1) folds : list of pd.DataFrame
       - Each DataFrame must contain the following columns:
         'action_number', 'new_DR', 'new_DP', 'new_EO', 'new_PQP',
         'new_recall', 'new_precision', 'new_sufficiency'

    2) original_DR, original_DP, original_EO, original_PQP,
       original_recall, original_precision, original_sufficiency : list of float
       - Original metric values for each fold. All lists must be of the same length as folds.

    Other parameters remain unchanged.
    """

    # ============== 1) Input validation ==============
    num_folds = len(folds)
    if not (len(original_DR) == len(original_DP) == len(original_EO) == 
            len(original_PQP) == len(original_recall) == len(original_precision) == 
            len(original_sufficiency) == num_folds):
        raise ValueError("All original_xxx lists must have the same length as folds.")

    # Bundle all metric information
    measures_info = [
        ("DR",          "new_DR",          original_DR),
        ("DP",          "new_DP",          original_DP),
        ("EO",          "new_EO",          original_EO),
        ("PQP",         "new_PQP",         original_PQP),
        ("Recall",      "new_recall",      original_recall),
        ("Precision",   "new_precision",   original_precision),
        ("Sufficiency", "new_sufficiency", original_sufficiency),
    ]

    # ============== 2) Prepare figure layout ==============
    n_metrics = len(measures_info)
    n_cols = 3  # 3 subplots per row
    n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division
    fig = plt.figure(figsize=figsize)

    # Loop through each metric and create a subplot
    for i, (measure_name, measure_col, original_list) in enumerate(measures_info, start=1):
        # 2.1) Convert action_number to numeric and subtract original value
        for df, orig_val in zip(folds, original_list):
            df['action_number'] = pd.to_numeric(df['action_number'], errors='coerce')
            df[measure_col] = df[measure_col] - orig_val

        # 2.2) Find maximum action number across all folds
        max_actions = []
        for df in folds:
            if not df.empty:
                max_val = df['action_number'].max()
                if pd.notna(max_val):
                    max_actions.append(max_val)
        if len(max_actions) == 0:
            print(f"Warning: All folds are empty or invalid for {measure_col}, skipping subplot.")
            continue
        overall_max_action = int(np.nanmax(max_actions))

        # 2.3) Collect values for each action number
        measure_values = {}
        for action in range(min_action, overall_max_action + 1):
            current_list = []
            count_no_data = 0
            for df in folds:
                row = df.loc[df['action_number'] == action, measure_col]
                if row.empty:
                    count_no_data += 1
                else:
                    current_list.append(row.values[0])
            if count_no_data >= stop_when_no_data:
                break
            measure_values[action] = current_list

        action_range = sorted(measure_values.keys())
        if len(action_range) == 0:
            print(f"Warning: No valid action numbers found for {measure_name} that meet stop_when_no_data threshold. Skipping.")
            continue

        # 2.4) Calculate mean and std
        means = []
        stds = []
        for action in action_range:
            vals = measure_values[action]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        means = np.array(means)
        stds = np.array(stds)

        # ============== 3) Plot on subplot ==============
        ax = fig.add_subplot(n_rows, n_cols, i)
        ax.set_title(f"{measure_name} Difference from Original")
        # 3.1) Baseline reference line
        ax.axhline(y=baseline, color='black', linewidth=2, linestyle='-', label=f'Baseline (y={baseline})')
        # 3.2) Mean line
        ax.plot(action_range, means, color=fill_color, label=f'Mean {measure_name} Gap')
        # 3.3) Shaded area for ±1 std dev
        ax.fill_between(
            action_range,
            means - stds,
            means + stds,
            alpha=fill_alpha,
            color=fill_color,
            label='±1 std dev'
        )
        ax.set_xlabel("Action Number")
        ax.set_ylabel(f"{measure_name} - Original")
        ax.grid(True)
        ax.legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()