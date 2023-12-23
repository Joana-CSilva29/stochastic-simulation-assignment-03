import pandas as pd
from scipy import stats

# Read in the CSV file
filename = 'Statistical_data_Logarithmic_Quadratic_Boltzmann.csv'
df = pd.read_csv(filename)

# Prepare DataFrame to hold test results
results = pd.DataFrame(columns=['Cooling_Schedule_Pair', 't-Statistic', 'p-Value'])

# Perform pairwise Welch's t-tests
for i, col1 in enumerate(df.columns):
    for j, col2 in enumerate(df.columns[i+1:], i+1):
        t_stat, p_value = stats.ttest_ind(df[col1], df[col2], equal_var=False)
        results = results.append({
            'Cooling_Schedule_Pair': f'{col1} vs {col2}',
            't-Statistic': t_stat,
            'p-Value': p_value
        }, ignore_index=True)

# Append the test results to the bottom of the original DataFrame
df_with_results = pd.concat([df, pd.DataFrame([[""]*len(df.columns)], columns=df.columns), results], ignore_index=True)

# Save to the same CSV file
df_with_results.to_csv(filename, index=False)
