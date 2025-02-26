import pandas as pd

# Load  datasets
salaries = pd.read_csv('salaries.csv')
stats = pd.read_csv('stats.csv', encoding='latin1')

salaries = salaries[['Player Name', '2022/2023']].rename(columns={'2022/2023': 'Salary_2022_2023'})

# Merge the datasets on the player name columns
# Here, 'Player' is the column in stats that contains the player names
merged_data = pd.merge(stats, salaries, left_on='Player', right_on='Player Name', how='left')

# Drop the extra 'Player Name' column if desired
merged_data.drop('Player Name', axis=1, inplace=True)

# Now merged_data contains the stats for the season (assumed to be 2022/2023) 
# and the corresponding salary from the 2022/2023 season.
print(merged_data.head())


#test