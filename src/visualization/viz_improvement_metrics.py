import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter  # For curve smoothing

def plot_fairness_improvement(
    folds,
    original_accuracy,  
    original_DR,      
    original_DP,      
    original_EO,      
    original_PQP,     
    stop_when_no_data=3,
    min_action=1,
    baseline=0.0,
    figsize=None,  
    fill_alpha=0.2,
    fill_color='b',
    smooth_window=15,  # Size of sliding window (for smoothing)
    smooth_polyorder=3 # Polynomial order for Savitzky-Golay filter
):
    """
    Plot 1×N subplots showing the difference between new metric values and original values 
    (original_accuracy, original_DR, etc.) for Accuracy, DR, DP, EO, and PQP,
    as a function of action_number with mean ± standard deviation.

    Main improvements:
    1) Compute moving average to reduce sharp fluctuations and make curves smoother.
    2) Use Savitzky-Golay filter to reduce noise while preserving trend.
    3) Adjust transparency for better readability.
    4) Automatically adjust ylim to avoid extreme values from affecting visualization.
    """

    num_folds = len(folds)
    if not (len(original_accuracy) == len(original_DR) == len(original_DP) == len(original_EO) == len(original_PQP) == num_folds):
        raise ValueError("original_accuracy, original_DR, original_DP, original_EO, original_PQP must have the same length as folds.")

    measures_info = [
        ("Accuracy", "new_accuracy", original_accuracy),
        ("DR",  "new_DR",  original_DR),
        ("DP",  "new_DP",  original_DP),
        ("EO",  "new_EO",  original_EO),
        ("PQP", "new_PQP", original_PQP),
    ]

    num_subplots = len(measures_info)
    num_rows = 1  
    num_cols = num_subplots  

    if figsize is None:
        figsize = (num_subplots * 5, 5)

    fig = plt.figure(figsize=figsize)

    for i, (measure_name, measure_col, original_list) in enumerate(measures_info, start=1):
        
        for df, orig_val in zip(folds, original_list):
            df['action_number'] = pd.to_numeric(df['action_number'], errors='coerce')
            df[measure_col] = df[measure_col] - orig_val

        # Collect all unique action numbers across all folds
        all_actions = set()
        for df in folds:
            if not df.empty:
                valid_actions = df['action_number'].dropna()
                all_actions.update(valid_actions.astype(int).tolist())

        if len(all_actions) == 0:
            print(f"Warning: All folds are empty or invalid for {measure_col}, skipping this subplot.")
            continue

        # Sort and filter actions >= min_action
        sorted_actions = sorted([a for a in all_actions if a >= min_action])

        measure_values = {}
        for action in sorted_actions:
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
            print(f"Warning: No valid action numbers found for {measure_name} that meet the stop_when_no_data threshold. Skipping.")
            continue

        means = np.array([np.mean(measure_values[action]) for action in action_range])
        stds = np.array([np.std(measure_values[action]) for action in action_range])

        # Smooth the curve using Savitzky-Golay filter
        if len(means) > smooth_window:
            smoothed_means = savgol_filter(means, window_length=smooth_window, polyorder=smooth_polyorder)
        else:
            smoothed_means = means  

        ax = fig.add_subplot(num_rows, num_cols, i)
        ax.set_title(f"{measure_name} Improvement from Original")

        # Baseline reference line
        ax.axhline(y=baseline, color='black', linewidth=2, linestyle='-', label=f'Baseline (y={baseline})')

        # Smoothed mean curve
        ax.plot(action_range, smoothed_means, color=fill_color, label=f'Mean {measure_name} Gap')

        # Shaded area for ±1 standard deviation
        ax.fill_between(
            action_range,
            smoothed_means - stds,
            smoothed_means + stds,
            alpha=fill_alpha,
            color=fill_color,
            label='±1 std dev'
        )

        ax.set_xlabel("Action Number")
        ax.set_ylabel(f"{measure_name} - Original")
        ax.grid(True)
        ax.legend()

        # Limit Y-axis range to avoid extreme values affecting visualization
        ax.set_ylim([min(smoothed_means - stds) * 1.2, max(smoothed_means + stds) * 1.2])

    plt.subplots_adjust(wspace=0.3)
    plt.show()