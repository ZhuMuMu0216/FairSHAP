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
    Creates 3x2 subplots for each metric and saves as PDF and PNG files.
    """
    measures_info = [
        ("Accuracy", "new_accuracy"),
        ("DR", "new_DR"),
        ("DP", "new_DP"),
        ("EO", "new_EO"),
        ("PQP", "new_PQP"),
    ]
    
    num_datasets = len(datasets_info)
    
    # Generate one figure per metric, each with 3x2 layout
    output_filenames = []
    
    for metric_idx, (measure_name, measure_col) in enumerate(measures_info):
        if figsize is None:
            figsize = (10, 12)  # Better size for 3x2 layout
        
        fig, axes = plt.subplots(3, 2, figsize=figsize, squeeze=False)
        
        for dataset_idx, dataset_info in enumerate(datasets_info):
            dataset_name = dataset_info['name']
            if dataset_name == "COMPAS":
                dataset_name = "COMPAS (Sex)"
            elif "COMPAS" not in dataset_name:
                dataset_name += " (Sex)"
                
            folds = dataset_info['folds']
            
            # Calculate row and column index for 3x2 layout
            row_idx = dataset_idx // 2
            col_idx = dataset_idx % 2
            
            if row_idx >= 3:  # Skip if we have more than 6 datasets
                continue
                
            ax = axes[row_idx, col_idx]
            
            original_values = dataset_info[f'original_{measure_name}']
            
            # Calculate percentage change for each fold
            for df, orig_val in zip(folds, original_values):
                df['modification_num'] = pd.to_numeric(df['action_number'], errors='coerce')
                # Avoid division by zero
                if abs(orig_val) < 1e-10:
                    df[measure_col] = df[measure_col] - orig_val  # Use absolute difference if original value is close to zero
                else:
                    if measure_name == "Accuracy":
                        # For accuracy, positive percentage is improvement
                        df[measure_col] = (df[measure_col] - orig_val) / abs(orig_val) * 100
                    else:
                        # For fairness metrics, negative percentage is improvement (reduction)
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
                        # The values are already differences from the original value
                        current_list.append(row.values[0])
                
                if count_no_data >= stop_when_no_data:
                    break
                
                measure_values[action] = current_list
            
            action_range = sorted(measure_values.keys())
            if len(action_range) == 0:
                continue
            
            # Compute means and stds with handling for empty lists
            means = []
            stds = []
            for action in action_range:
                values = measure_values[action]
                # Filter out NaN and Inf values
                valid_values = [v for v in values if not np.isnan(v) and not np.isinf(v)]
                if valid_values:
                    means.append(np.mean(valid_values))
                    stds.append(np.std(valid_values) if len(valid_values) > 1 else 0)
                else:
                    # Use the previous value or 0 if no previous value exists
                    means.append(means[-1] if means else 0)
                    stds.append(stds[-1] if stds else 0)
            
            means = np.array(means)
            stds = np.array(stds)
            
            # Smooth the curve - only if we have enough points and no NaNs/Infs
            if len(means) > smooth_window and not np.isnan(means).any() and not np.isinf(means).any():
                try:
                    smoothed_means = savgol_filter(means, window_length=min(smooth_window, len(means) - 1 if len(means) % 2 == 0 else len(means)), 
                                                polyorder=min(smooth_polyorder, min(smooth_window, len(means)) - 1))
                except Exception as e:
                    print(f"Error in smoothing for {dataset_name}, {measure_name}: {e}")
                    smoothed_means = means
            else:
                smoothed_means = means
            
            color = color_palette[dataset_idx % len(color_palette)]
            
            # Draw baseline at 0 (no change from original)
            ax.axhline(y=baseline, color='black', linewidth=1, linestyle='--')
            
            # Plot directly - we've already handled the sign in the calculation
            ax.plot(action_range, smoothed_means, color=color, linewidth=2)
            ax.fill_between(action_range, smoothed_means - stds, smoothed_means + stds, alpha=fill_alpha, color=color)
            
            # Add markers at regular intervals
            step = max(1, len(action_range) // 5)
            for i in range(0, len(action_range), step):
                ax.plot(action_range[i], smoothed_means[i], marker='s', markerfacecolor='white', 
                        markeredgecolor=color, markersize=5)
            
            # Set labels with percentage indicators
            if col_idx == 0:
                if measure_name == "Accuracy":
                    ax.set_ylabel(f"{measure_name} Change (%)", fontsize=12)
                else:
                    ax.set_ylabel(f"{measure_name} Reduction (%)", fontsize=12)
            
            ax.set_xlabel("Modification Num", fontsize=12)
            ax.set_title(dataset_name, fontsize=12)
            
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Handle NaN and Inf values
            lower_bound = smoothed_means - stds
            upper_bound = smoothed_means + stds
            
            # Filter out NaN and Inf values
            lower_valid = lower_bound[~np.isnan(lower_bound) & ~np.isinf(lower_bound)]
            upper_valid = upper_bound[~np.isnan(upper_bound) & ~np.isinf(upper_bound)]
            
            # Only set limits if we have valid values
            if len(lower_valid) > 0 and len(upper_valid) > 0:
                y_min = min(lower_valid) * 1.2 if min(lower_valid) < 0 else min(lower_valid) * 0.8
                y_max = max(upper_valid) * 1.2 if max(upper_valid) > 0 else max(upper_valid) * 0.8
                # Set reasonable defaults if still invalid
                if np.isnan(y_min) or np.isinf(y_min):
                    y_min = -1
                if np.isnan(y_max) or np.isinf(y_max):
                    y_max = 1
                # Ensure y_min < y_max
                if y_min >= y_max:
                    y_min, y_max = -1, 1
                ax.set_ylim([y_min, y_max])
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        
        # Create metric-specific output filenames
        output_filename_pdf = f"{measure_name.lower()}_decrease_plot.pdf"
        output_filename_png = f"{measure_name.lower()}_decrease_plot.png"
        
        # Save as both PDF and PNG
        fig.savefig(output_filename_pdf, dpi=300, bbox_inches="tight")
        fig.savefig(output_filename_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        
        output_filenames.append(output_filename_pdf)
    
    # Also create the original combined plot for backward compatibility
    output_filename = "fairness_decrease_plot.png"
    
    return output_filenames


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
    output_files = plot_multi_dataset_fairness_improvement(
        datasets_info=datasets_info,
        stop_when_no_data=4,
        min_action=1,
        baseline=0.0,
        figsize=(10, 12),  # Size optimized for 3x2 layout
        fill_alpha=0.2,
        color_palette=['b', 'g', 'r', 'c', 'm', 'y'],
        smooth_window=50, 
        smooth_polyorder=1
    )
    
    # Print output filenames
    print("Generated the following files:")
    for file in output_files:
        print(f" - {file}")