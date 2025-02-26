import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load  datasets
salaries = pd.read_csv('salaries.csv')
stats = pd.read_csv('stats.csv', encoding='latin1')

# Read the stats data using the correct delimiter
stats = pd.read_csv('stats.csv', sep=';')

salaries = salaries[['Player Name', '2022/2023']].rename(columns={'2022/2023': 'Salary_2022_2023'})

# Merge the datasets on the player name columns
# Here, 'Player' is the column in stats that contains the player names
merged_data = pd.merge(stats, salaries, left_on='Player', right_on='Player Name', how='left')

# 'Player Name'
merged_data.drop('Player Name', axis=1, inplace=True)



# 1. Load the stats data (using semicolon as the delimiter)
stats = pd.read_csv('stats.csv', sep=';')

# Print out the columns to confirm they are read correctly
print("Columns in stats dataset:")
print(stats.columns)

# 2. Define the features and target variable.
# We'll predict Points (PTS) using a selection of numerical features.
# Feel free to adjust these features as needed.
features = ['Age', 'MP', 'FG%', '3P%', '2P%', 'eFG%', 'FT%', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF']
target = 'PTS'

# Select only the required columns and drop any rows with missing values
data = stats[features + [target]].dropna()

# Display the first few rows to verify our data
print("\nFirst few rows of the prepared data:")
print(data.head())

# 3. Split the data into training and testing sets
X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Build the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make predictions on the test set
y_pred = model.predict(X_test)

# 6. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# 7. (Optional) Examine the model coefficients
coefficients = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_
})
print("\nModel Coefficients:")
print(coefficients)

# 8. (Optional) Visualize the actual vs. predicted points
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Points (PTS)")
plt.ylabel("Predicted Points (PTS)")
plt.title("Actual vs. Predicted Points")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', lw=2)  # 45-degree line
plt.show()

