import pandas as pd
import sys
import os
import numpy as np
import scipy.stats as stats
import numpy as np
from scipy.stats import chi2

# colors mapping for the plots
global regions_to_color
regions_to_color = {
    "Ashkenaz": "Blue",
    "France": "Red",
    "Spain": "Green",
    "N.Africa": "Yellow",
    "Provence": "Purple",
    "Italy": "Orange",
    "Middle East": "Gray"
}


# Get the current script's directory
script_directory = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_directory = os.path.dirname(script_directory)

# Add the parent directory to the system path
sys.path.insert(0, parent_directory)


def check_chi_test(data, expected_data):

    # Choose a scaling factor
    scaling_factor = 1000

    # Scale the proportions to convert them into counts
    scaled_data = {region: int(proportion * scaling_factor) for region, proportion in data.items()}

    scaled_data_expected = {region: int(proportion * scaling_factor) for region, proportion in expected_data.items()}

    # Converting the values to a list as required by the chi2_contingency function
    values = list(scaled_data.values())
    Values_expected = list(scaled_data_expected.values())

    # Perform the Chi-square test
    chi2, DOF, p = perform_chi_square_test(values, Values_expected)

    # Output the results
    print(f"Chi-square Statistic: {chi2}, p-value: {p}, DOD-value: {DOF}")

    # Interpretation
    if p < 0.05:
        print("Reject the null hypothesis: There is a significant difference in the distribution of values across regions.")
    else:
        print("Fail to reject the null hypothesis: There is no significant difference in the distribution of values across regions.")

def perform_chi_square_test(observed, expected):
    # Convert observed and expected lists to numpy arrays
    observed = np.array(observed)
    expected = np.array(expected)

    # Check if the arrays are 1D or 2D and calculate degrees of freedom accordingly
    if len(observed.shape) == 1:
        degrees_of_freedom = len(observed) - 1
    else:
        degrees_of_freedom = (observed.shape[0] - 1) * (observed.shape[1] - 1)

    # Calculate the chi-square statistic
    chi_square_statistic = np.sum((observed - expected) ** 2 / expected)

    # Calculate the p-value
    p_value = chi2.sf(chi_square_statistic, degrees_of_freedom)

    return chi_square_statistic, degrees_of_freedom, p_value

def calc_expected_values_for_chi_2(filename, df = None):
    # Read the xlsx file
    if filename != "":
        df = pd.read_excel(filename, engine='openpyxl', index_col=0)
    
    # Drop the "total" column and "Middle East" row/column, if present
    df = df.drop('total', axis=1, errors='ignore')
    if "Middle East" in df.index:
        df = df.drop("Middle East")
    if "Middle East" in df.columns:
        df = df.drop("Middle East", axis=1)
    
    # Calculate row totals (total references made by each region)
    row_totals = df.sum(axis=1)
    
    # Calculate column totals (total references received by each region)
    column_totals = df.sum(axis=0)
    
    # Calculate the grand total of all references
    grand_total = row_totals.sum()
    
    # Initialize a DataFrame to store expected values
    expected_df = pd.DataFrame(index=df.index, columns=df.columns)
    
    print(df)

    # Calculate expected counts for each cell and convert to ratios
    for row in df.index:
        for col in df.columns:
            expected_count = (row_totals[row] * column_totals[col]) / grand_total
            expected_df.at[row, col] = expected_count / grand_total  # Divide by grand total


    print(expected_df)
    return expected_df

# for every author in every region check the ratio of :
# total citation vs. external citations
def convert_2_ratio(regions_authors_values_dict):
    regions_dict = {}
    for region, inner_author_dict in regions_authors_values_dict.items():
        # Sort the inner_author_dict by values[0] in descending order
        sorted_authors = sorted(inner_author_dict.items(), key=lambda x: x[1][0], reverse=True)

        authors_dict = {}
        for author, values in sorted_authors:
            if values[0] == 0:
                print("total = 0 for: ", author)
            else:
                authors_dict[author] = values[1] / values[0]

        regions_dict[region] = authors_dict
        
    return regions_dict



