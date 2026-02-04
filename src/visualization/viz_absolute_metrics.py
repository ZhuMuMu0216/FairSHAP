import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_results(
    folds,
    original_accuracy,  
    original_DR, 
    original_DP, 
    original_EO, 
    original_PQP,
    stop_when_no_data=3,
    min_action=1,
    figsize=None,  # Automatically adjust
    fill_alpha=0.2,
    fill_color='b',
    red_alpha=0.3
):
    """
    Plot Accuracy / DR / DP / EO / PQP in a 1×5 subplot layout to show the mean and standard deviation 
    of each metric across different action_number values. Overlay faint red dashed lines indicating 
    the original values for each of the 5 folds.

    Parameters:
    --------
    1) folds : list of pd.DataFrame
       - Each DataFrame must contain the following columns:
         'action_number', 'new_accuracy', 'new_DR', 'new_DP', 'new_EO', 'new_PQP'.

    2) original_accuracy : list of float
       - Length 5, representing the original accuracy values for each fold.

    3) original_DR, original_DP, original_EO, original_PQP : list of float
       - Length 5, representing the original values for DR, DP, EO, and PQP metrics respectively.

    Other parameters remain unchanged.
    """
    # List of metrics to be plotted
    measures_info = [
        ("Accuracy", "new_accuracy", original_accuracy),
        ("DR",  "new_DR",  original_DR),
        ("DP",  "new_DP",  original_DP),
        ("EO",  "new_EO",  original_EO),
        ("PQP", "new_PQP", original_PQP),
    ]

    num_subplots = len(measures_info)  # Number of subplots
    num_rows = 1  # One row
    num_cols = num_subplots  # Five columns

    # Automatically adjust figure size
    if figsize is None:
        figsize = (num_subplots * 5, 5)

    plt.figure(figsize=figsize)

    for i, (measure_name, measure_col, orig_list) in enumerate(measures_info, start=1):
        # ---- 1) Convert action_number to numeric and check data ----
        for df in folds:
            df['action_number'] = pd.to_numeric(df['action_number'], errors='coerce')

        # ---- 2) Collect all unique action numbers across all folds ----
        all_actions = set()
        for df in folds:
            if measure_col not in df.columns:
                raise ValueError(f"Column {measure_col} does not exist in one of the folds. Please check input data.")
            if not df.empty:
                valid_actions = df['action_number'].dropna()
                all_actions.update(valid_actions.astype(int).tolist())

        if len(all_actions) == 0:
            print(f"Warning: All folds are empty or have invalid action_number for {measure_col}, skipping this subplot.")
            continue

        # Sort and filter actions >= min_action
        sorted_actions = sorted([a for a in all_actions if a >= min_action])

        # ---- 3) Collect measure data; stop if stop_when_no_data folds have no data ----
        measure_values = {}
        for action in sorted_actions:
            current_list = []
            count_no_data = 0

            for df in folds:
                row = df.loc[df['action_number'] == action, measure_col]
                if row.empty:
                    count_no_data += 1
                else:
                    # If multiple rows are found (should not happen), take the first value
                    current_list.append(row.values[0])

            # Stop if too many folds have no data
            if count_no_data >= stop_when_no_data:
                break

            measure_values[action] = current_list

        action_range = sorted(measure_values.keys())
        if len(action_range) == 0:
            print(f"Warning: No valid action numbers found for {measure_name} that meet the criteria. Skipping.")
            continue

        means = []
        stds = []
        for action in action_range:
            vals = measure_values[action]
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        means = np.array(means)
        stds = np.array(stds)

        # ---- 4) Plot on subplot ----
        ax = plt.subplot(num_rows, num_cols, i)  # 1×5 layout, i-th subplot
        ax.set_title(f"{measure_name} vs. Action Number")

        # 4.1) Draw original values as faint red dashed lines
        for idx, val in enumerate(orig_list):
            label = "Original Values" if idx == 0 else None  # Only add legend once
            ax.axhline(
                y=val,
                color='red',
                linestyle='--',
                linewidth=1.5,
                alpha=red_alpha,
                label=label
            )

        # 4.2) Mean line
        ax.plot(action_range, means, color=fill_color, label=f"Mean {measure_name}")

        # 4.3) Fill area for ±1 standard deviation
        ax.fill_between(
            action_range,
            means - stds,
            means + stds,
            alpha=fill_alpha,
            color=fill_color,
            label='±1 std dev'
        )

        ax.set_xlabel("Action Number")
        ax.set_ylabel(f"{measure_name} Value")
        ax.grid(True)
        ax.legend()

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.3)  # Increase horizontal space between subplots
    plt.show()