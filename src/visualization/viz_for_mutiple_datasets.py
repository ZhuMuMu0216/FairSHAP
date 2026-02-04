import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter  # For curve smoothing


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
    """
    Generate fairness improvement line plots in the style of ICLR/ICML.
    Remove top and right spines and save the figure as a PNG file.
    """
    measures_info = [
        ("Accuracy", "new_accuracy"),
        ("DR", "new_DR"),
        ("DP", "new_DP"),
        ("EO", "new_EO"),
        ("PQP", "new_PQP"),
    ]
    
    num_datasets = len(datasets_info)
    num_metrics = len(measures_info)
    
    # Automatically determine figure size
    if figsize is None:
        figsize = (num_metrics * 3.5, num_datasets * 2.5)
    
    fig = plt.figure(figsize=figsize)
    
    for dataset_idx, dataset_info in enumerate(datasets_info):
        dataset_name = dataset_info['name']
        folds = dataset_info['folds']
        
        for metric_idx, (measure_name, measure_col) in enumerate(measures_info):
            original_values = dataset_info[f'original_{measure_name}']
            
            subplot_idx = dataset_idx * num_metrics + metric_idx + 1
            ax = fig.add_subplot(num_datasets, num_metrics, subplot_idx)
            
            for df, orig_val in zip(folds, original_values):
                df['modification_num'] = pd.to_numeric(df['action_number'], errors='coerce')
                df[measure_col] = df[measure_col] - orig_val
            
            max_actions = [df['modification_num'].max() for df in folds if not df.empty]
            if len(max_actions) == 0:
                continue
            overall_max_action = int(np.nanmax(max_actions))

            # 添加这段代码:对特定数据集限制modification number
            if dataset_name == 'COMPAS (Race)':
                overall_max_action = min(overall_max_action, 1500)
            elif dataset_name == 'Adult (Race)':
                overall_max_action = min(overall_max_action, 2500)
            
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
            
            # Smooth the curve
            if len(means) > smooth_window:
                smoothed_means = savgol_filter(means, window_length=smooth_window, polyorder=smooth_polyorder)
            else:
                smoothed_means = means
            
            color = color_palette[dataset_idx % len(color_palette)]
            
            ax.axhline(y=baseline, color='black', linewidth=1.5, linestyle='--')
            ax.plot(action_range, smoothed_means, color=color, linewidth=3)
            ax.fill_between(action_range, smoothed_means - stds, smoothed_means + stds, alpha=fill_alpha, color=color)
            
            # Set title and labels
            if dataset_idx == 0:
                ax.set_title(f"{measure_name}", fontsize=12)
            
            if metric_idx == 0:
                ax.set_ylabel(f"{dataset_name}", fontsize=10)
            
            if dataset_idx == num_datasets - 1:
                ax.set_xlabel("Modification Number", fontsize=10)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            y_min = min(smoothed_means - stds) * 1.2 if min(smoothed_means - stds) < 0 else min(smoothed_means - stds) * 0.8
            y_max = max(smoothed_means + stds) * 1.2 if max(smoothed_means + stds) > 0 else max(smoothed_means + stds) * 0.8
            ax.set_ylim([y_min, y_max])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    
    # Save as PNG file
    output_filename = "fairness_improvement_multirace_plot.png"
    fig.savefig(output_filename, dpi=300, bbox_inches="tight")

    # 添加这行来同时保存为PDF:
    output_pdf_filename = "fairness_improvement_multirace_plot.pdf"
    fig.savefig(output_pdf_filename, format='pdf', bbox_inches="tight")

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


# Example usage
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
            'path': 'saved_results/compas4multirace/',
            'pattern': 'saved_results/compas4multirace/fairSHAP-DR_0.05_NN_{}-fold_results.csv'
        },
        {
            'name': 'Adult',
            'path': 'saved_results/adult/',
            'pattern': 'saved_results/adult/fairSHAP-DR_0.05_NN_{}-fold_results.csv'
        },
        {
            'name': 'Adult (Race)',
            'path': 'saved_results/adult4multirace/',
            'pattern': 'saved_results/adult4multirace/fairSHAP-DR_0.05_NN_{}-fold_results.csv'
        },
        {
            'name': 'Census Income',
            'path': 'saved_results/census_income/',
            'pattern': 'saved_results/census_income/fairSHAP-DR_0.05_NN_{}-fold_results.csv'
        },
        {
            'name': 'Default Credit',
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
        figsize=(20, 15),  # Width, height
        fill_alpha=0.2,
        color_palette=['b', 'g', 'r', 'c', 'm', 'y'],
        smooth_window=50, 
        smooth_polyorder=1
    )