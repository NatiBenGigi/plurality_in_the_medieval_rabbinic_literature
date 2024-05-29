import matplotlib.pyplot as plt
import matplotlib.patches as patches
from helper_stuff import regions_to_color

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_authors_ration_X_Y(dict1, dict2, xlabel="Authors’ external out-degree", ylabel="Authors’ resource diversity", title=''):# "Final score combining author's citations variety and references"):
    # Initialize lists for plot values and colors
    values1 = []
    values2 = []
    colors = []

    # Iterate over the regions and nested names in the dictionaries
    for region, inner_dict1 in dict1.items():
        inner_dict2 = dict2[region]
        for name, value1 in inner_dict1.items():
            if name in inner_dict2:
                values1.append(value1)
                values2.append(inner_dict2[name])
                colors.append(regions_to_color.get(region, 'black'))
            else:
                print(f"Skipping {name} as it is not present in both dictionaries.")

    # Determine the limits for the plot
    min_value = min(min(values1), min(values2))
    max_value = max(max(values1), max(values2))
    limit = max(max_value, 3)  # Ensure the upper limit is at least 3

    # Calculate the center point and buffer for the chart
    center_x = (min_value + max_value) / 2
    center_y = center_x  # Assuming a square plot
    range_of_data = max_value - min_value
    buffer = 0.1 * range_of_data
    adjusted_min = min_value - buffer
    adjusted_max = max_value + buffer
    half_range = (adjusted_max - adjusted_min) / 2

    # Create the plot
    fig, ax = plt.subplots()

    # Plot each point with its respective color
    for i in range(len(values1)):
        ax.scatter(values1[i], values2[i], color=colors[i], s=50)

    # Draw the four equal squares
    ax.add_patch(patches.Rectangle((center_x, center_y), half_range, half_range, color='lightgreen', alpha=0.1))  # Top-right
    ax.add_patch(patches.Rectangle((adjusted_min, adjusted_min), half_range, half_range, color='lightcoral', alpha=0.1))  # Bottom-left
    ax.add_patch(patches.Rectangle((adjusted_min, center_y), half_range, half_range, color='none'))  # Bottom-right
    ax.add_patch(patches.Rectangle((center_x, adjusted_min), half_range, half_range, color='none'))  # Top-left

    # Setting the dynamic limits with buffer and labels
    ax.set_xlim([adjusted_min, adjusted_max])
    ax.set_ylim([adjusted_min, adjusted_max])
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)

    plt.show()

def plot_dicts(ref_dict, edge_dict, xlabel= 'Citations quantity', ylabel ='Edges quantity', title =''):
    # Extracting keys and values
    keys = ref_dict.keys()
    values1 = [ref_dict[key] for key in keys]
    values2 = [edge_dict[key] for key in keys]

    # Determine the limits for the plot
    min_value = min(min(values1), min(values2))
    max_value = max(max(values1), max(values2))
    limit = max(max_value, 3)  # Ensuring the upper limit is at least 3

    # Calculate the center point of the chart
    center_x = (min_value + max_value) / 2
    center_y = center_x  # Assuming a square plot

    # Adjust the buffer to be proportional to the range of data
    range_of_data = max_value - min_value
    buffer = 0.1 * range_of_data
    adjusted_min = min_value - buffer
    adjusted_max = max_value + buffer

    # Create the plot
    fig, ax = plt.subplots()

    # Plot each point with its respective color and increased size
    for i, key in enumerate(keys):
        ax.scatter(values1[i], values2[i], color=regions_to_color.get(key, 'black'), s=100)
        ax.annotate(key, (values1[i], values2[i] + 0.2 * buffer), ha='center', fontsize=12)

    # Draw the four equal squares
    half_range = (adjusted_max - adjusted_min) / 2
    ax.add_patch(patches.Rectangle((center_x, center_y), half_range, half_range, color='lightgreen', alpha=0.1))  # Top-right
    ax.add_patch(patches.Rectangle((adjusted_min, adjusted_min), half_range, half_range, color='lightcoral', alpha=0.1))  # Bottom-left
    ax.add_patch(patches.Rectangle((adjusted_min, center_y), half_range, half_range, color='none'))  # Bottom-right
    ax.add_patch(patches.Rectangle((center_x, adjusted_min), half_range, half_range, color='none'))  # Top-left

    # Setting the dynamic limits with buffer and labels with increased font size
    ax.set_xlim([adjusted_min, adjusted_max])
    ax.set_ylim([adjusted_min, adjusted_max])
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)

    plt.show()

def plot_region_bars_single_figure(data, num_of_bars, include_names=True):
    # Determine the layout of the subplots: 3 rows and 2 columns
    num_regions = len(data)
    cols = 3  # 2 columns
    rows = 2  # 3 rows

    # Create a single figure for all subplots
    fig, axes = plt.subplots(rows, cols, figsize=(10, 15))  # Adjusted figure size for new layout
    axes = axes.flatten() if num_regions > 1 else [axes]

    for i, (region, names) in enumerate(data.items()):
        ax = axes[i]

        # Sort the names by value and get the top 5
        if num_of_bars != 1:
            top_names = sorted(names.items(), key=lambda x: x[1], reverse=True)[:num_of_bars]
        else:
            top_names = sorted(names.items(), key=lambda x: x[1], reverse=True)

        # Split names and values
        names, values = zip(*top_names) if top_names else ([], [])

        # Plot on the respective subplot
        ax.bar(names, values, color=regions_to_color.get(region, "black"))

        # Add a horizontal line at y=0.5
        ax.axhline(y=0.5, color='grey', linewidth=1, linestyle='--')

        if num_of_bars != -1:
            ax.set_title(f"Top {num_of_bars} in {region}")
            ax.set_xlabel("Names")
            ax.set_ylabel("Values")
        else:
            ax.set_xlabel(region)

        # Rotate names and align the end of the label with the bar center
        if include_names:
            ax.set_xticklabels(names, rotation=45, ha='right')
        else:
            ax.set_xticklabels([])  # Set empty labels

        # Set uniform y-axis limit
        ax.set_ylim(0, 1.0)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


