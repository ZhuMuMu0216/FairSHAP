import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter  # For curve smoothing

'''
This script shows the percentage improvement in DR reduction across different datasets under the effect of FairSHAP.
'''

def plot_multi_dataset_fairness_improvement(
    datasets_info,
    stop_when_no_data=4,
    min_action=1,
    baseline=0.0,
    figsize=None,  
    fill_alpha=0.2,
    color_palette=['b', 'g', 'r', 'c', 'm', 'y'],
    smooth_window=20,
    smooth_polyorder=2
):
    measures_info = [
        ("DR", "new_DR"),
    ]
    num_datasets = len(datasets_info)
    num_metrics = len(measures_info)

    if figsize is None:
        figsize = (12, 6)  # 增加高度以适应两行

    fig, axes = plt.subplots(2, 3, figsize=figsize, squeeze=False)

    for dataset_idx, dataset_info in enumerate(datasets_info):
        dataset_name = dataset_info['name']
        if dataset_name == "COMPAS":
            dataset_name = "COMPAS (Sex)"
        elif "COMPAS" not in dataset_name:
            dataset_name += " (Sex)"

        folds = dataset_info['folds']
        row_idx = dataset_idx // 3
        col_idx = dataset_idx % 3
        ax = axes[row_idx, col_idx]

        for measure_name, measure_col in measures_info:
            original_values = dataset_info[f'original_{measure_name}']

            for df, orig_val in zip(folds, original_values):
                df['modification_num'] = pd.to_numeric(df['action_number'], errors='coerce')
                df[measure_col] = (orig_val - df[measure_col]) / abs(orig_val) * 100

            max_actions = [df['modification_num'].max() for df in folds if not df.empty]
            if len(max_actions) == 0:
                continue
            overall_max_action = int(np.nanmax(max_actions))

            measure_values = {}
            for action in range(min_action, overall_max_action + 1):
                current_list = []
                count_no_data = 0

                for df in folds:
                    row = df.loc[df['modification_num'] == action, measure_col]
                    if row.empty:
                        count_no_data += 1
                    else:
                        current_list.append(row.values[0])

                if count_no_data >= stop_when_no_data:
                    break

                measure_values[action] = current_list

            action_range = sorted(measure_values.keys())
            if len(action_range) == 0:
                continue

            means = np.array([np.mean(measure_values[action]) for action in action_range])
            stds = np.array([np.std(measure_values[action]) for action in action_range])

            if len(means) > smooth_window:
                smoothed_means = savgol_filter(means, window_length=smooth_window, polyorder=smooth_polyorder)
            else:
                smoothed_means = means

            color = color_palette[dataset_idx % len(color_palette)]

            ax.axhline(y=baseline, color='black', linewidth=1, linestyle='--')
            ax.plot(action_range, smoothed_means, color=color, linewidth=2)
            ax.fill_between(action_range, smoothed_means - stds, smoothed_means + stds, alpha=fill_alpha, color=color)

            step = max(1, len(action_range) // 5)
            for i in range(0, len(action_range), step):
                ax.plot(action_range[i], smoothed_means[i], marker='s', markerfacecolor='white', markeredgecolor=color, markersize=5)

            # Set y-label only for left column
            if col_idx == 0:
                ax.set_ylabel("DR Reduction (%)", fontsize=12)

            ax.set_xlabel("Modification Num", fontsize=12)
            ax.set_title(dataset_name, fontsize=12)

            # ax.spines["top"].set_visible(False)
            # ax.spines["right"].set_visible(False)
            ax.grid(True, linestyle='--', alpha=0.7)

            y_min = min(smoothed_means - stds) * 1.2 if min(smoothed_means - stds) < 0 else min(smoothed_means - stds) * 0.8
            y_max = max(smoothed_means + stds) * 1.2 if max(smoothed_means + stds) > 0 else max(smoothed_means + stds) * 0.8
            ax.set_ylim([y_min, y_max])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0.4, wspace=0.2)

    output_filename = "dr_improvement_plot.pdf"
    fig.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_filename

def extract_original_values(fold):
    """Extract original metric values from the first row of a fold's dataframe."""
    original_accuracy = fold.loc[0, 'new_accuracy']
    original_DR = fold.loc[0, 'new_DR']
    original_DP = fold.loc[0, 'new_DP']
    original_EO = fold.loc[0, 'new_EO']
    original_PQP = fold.loc[0, 'new_PQP']
    return original_accuracy, original_DR, original_DP, original_EO, original_PQP


def load_dataset_folds(dataset_path, fold_pattern, num_folds=5):
    """Load all folds for a dataset and prepare data for visualization."""
    folds = []
    original_accuracy = []
    original_DR = []
    original_DP = []
    original_EO = []
    original_PQP = []
    
    for i in range(1, num_folds + 1):
        file_path = fold_pattern.format(i)
        try:
            fold = pd.read_csv(file_path)
            
            # Extract original values
            orig_acc, orig_dr, orig_dp, orig_eo, orig_pqp = extract_original_values(fold)
            original_accuracy.append(orig_acc)
            original_DR.append(orig_dr)
            original_DP.append(orig_dp)
            original_EO.append(orig_eo)
            original_PQP.append(orig_pqp)
            
            # Remove first row (original values)
            fold.drop(fold.index[0], inplace=True)
            folds.append(fold)
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return {
        'folds': folds,
        'original_Accuracy': original_accuracy,
        'original_DR': original_DR,
        'original_DP': original_DP,
        'original_EO': original_EO,
        'original_PQP': original_PQP
    }


if __name__ == "__main__":
    # List of datasets to process
    datasets = [
        {
            'name': 'German Credit',
            'path': 'saved_results/german_credit/',
            'pattern': 'saved_results/german_credit/fairSHAP-DR_NN_{}-fold_results.csv'
        },
        {
            'name': 'COMPAS',
            'path': 'saved_results/compas/',
            'pattern': 'saved_results/compas/fairSHAP-DR_0.05_NN_{}-fold_results.csv'
        },
        {
            'name': 'COMPAS (Race)',
            'path': 'saved_results/compas4race/',
            'pattern': 'saved_results/compas4race/fairSHAP-DR_0.05_NN_{}-fold_results.csv'
        },
        {
            'name': 'Adult',
            'path': 'saved_results/adult/',
            'pattern': 'saved_results/adult/fairSHAP-DR_0.05_NN_{}-fold_results.csv'
        },
        {
            'name': 'Census Income',
            'path': 'saved_results/census_income/',
            'pattern': 'saved_results/census_income/fairSHAP-DR_0.05_NN_{}-fold_results.csv'
        },
        {
            'name': 'Default credit',
            'path': 'saved_results/default_credit/',
            'pattern': 'saved_results/default_credit/fairSHAP-DR_0.05_NN_{}-fold_results.csv'
        }
    ]
    
    # Load and prepare data for each dataset
    datasets_info = []
    for dataset in datasets:
        data = load_dataset_folds(dataset['path'], dataset['pattern'])
        data['name'] = dataset['name']
        datasets_info.append(data)
    
    # Create and display visualization
    fig = plot_multi_dataset_fairness_improvement(
        datasets_info=datasets_info,
        stop_when_no_data=4,
        min_action=1,
        baseline=0.0,
        figsize=(12, 6),  # Width, height
        fill_alpha=0.2,
        color_palette=['b', 'g', 'r', 'c', 'm', 'y'],
        smooth_window=50, 
        smooth_polyorder=1
    )