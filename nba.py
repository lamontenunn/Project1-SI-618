import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# 1. Load and Clean the Data
# -----------------------------

# Load the salary data and keep only the 2022/2023 salary along with player names.
salaries = pd.read_csv('salaries.csv')
salaries = salaries[['Player Name', '2022/2023']].rename(
    columns={'2022/2023': 'Salary_2022_2023'}
)

# Load the stats data using the correct semicolon delimiter.
stats = pd.read_csv('stats.csv', sep=';', encoding='latin1')

# Merge using salaries as the primary dataset.
# This ensures we keep only the players from the salaries CSV.
merged_data = pd.merge(salaries, stats, left_on='Player Name', right_on='Player', how='left')

# Since we want to use the salary CSV player names, drop the duplicate 'Player' column from stats.
merged_data.drop('Player', axis=1, inplace=True)

# Clean the salary column: remove currency symbols and commas, then convert to float.
merged_data['Salary_2022_2023'] = (
    merged_data['Salary_2022_2023']
    .replace({'\$': '', ',': ''}, regex=True)
    .astype(float)
)

# -----------------------------
# 2. Define Efficiency Calculation Function
# -----------------------------
def calc_efficiency(row):
    """
    Calculates an efficiency metric using points, rebounds, and assists per minute.
    Efficiency = (PTS + TRB + AST) / MP.
    Returns 0 if MP is 0.
    """
    if row['MP'] > 0:
        return (row['PTS'] + row['TRB'] + row['AST']) / row['MP']
    else:
        return 0

# Calculate efficiency for each row.
merged_data['Efficiency'] = merged_data.apply(calc_efficiency, axis=1)

# -----------------------------
# 3. Regression Model for Points (PTS)
# -----------------------------
def build_regression_model(data):
    """
    Builds and evaluates a linear regression model to predict Points (PTS)
    using selected features from the stats data.
    Plots actual vs. predicted PTS.
    """
    features = ['Age', 'MP', 'FG%', '3P%', '2P%', 'eFG%', 'FT%', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF']
    target = 'PTS'
    
    # Drop rows with missing feature or target data.
    model_data = data[features + [target]].dropna()
    X = model_data[features]
    y = model_data[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\nRegression Model Evaluation:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-square Score: {r2:.2f}")
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel("Actual Points (PTS)")
    plt.ylabel("Predicted Points (PTS)")
    plt.title("Actual vs. Predicted Points")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', lw=2)
    plt.show()
    
    return model, X_test, y_test, y_pred

# -----------------------------
# 4. Scatter Plot: Salary vs. Efficiency
# -----------------------------
def plot_salary_vs_efficiency(data):
    """
    Creates a scatter plot comparing salary to efficiency (calculated as (PTS+TRB+AST)/MP).
    Uses ScalarFormatter to display full numbers for salary.
    """
    # Filter out rows where MP is zero.
    plot_data = data[data['MP'] > 0]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(plot_data['Salary_2022_2023'], plot_data['Efficiency'], alpha=0.7, color='blue')
    plt.xlabel("Salary 2022/2023 (USD)")
    plt.ylabel("Efficiency ((PTS+TRB+AST)/MP)")
    plt.title("Scatter Plot: Salary vs. Efficiency")
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ScalarFormatter())
    plt.grid(True)
    plt.show()

# -----------------------------
# 5. Bar Chart: Top 10 Players by Efficiency (Salary Comparison)
# -----------------------------
def plot_top10_salary_by_efficiency(data):
    """
    Aggregates data by the 'Player Name' from the salaries file.
    Uses the mean Efficiency (averaged over possible duplicate rows)
    and the salary (assumed constant per player).
    Then plots the top 10 players by efficiency along with their salaries.
    """
    # Group data by 'Player Name' to aggregate efficiency values.
    grouped = data.groupby('Player Name').agg({
        'Efficiency': 'mean',
        'Salary_2022_2023': 'first'
    }).reset_index()
    
    # Sort by efficiency descending and take the top 10.
    top10 = grouped.sort_values(by='Efficiency', ascending=False).head(10)
    
    print("\nTop 10 Players by Aggregated Efficiency:")
    print(top10)
    
    plt.figure(figsize=(10, 6))
    plt.bar(top10['Player Name'], top10['Salary_2022_2023'], color='green', alpha=0.7)
    plt.xlabel("Player Name")
    plt.ylabel("Salary 2022/2023 (USD)")
    plt.title("Top 10 Players by Efficiency: Salary Comparison")
    plt.xticks(rotation=45, ha='right')
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ScalarFormatter())
    plt.tight_layout()
    plt.show()

# -----------------------------
# 6. Interactive Menu to Display Plots
# -----------------------------
def display_menu():
    """
    Provides a menu for the user to select which plot to display.
    Options include: regression model, salary vs. efficiency scatter plot,
    and the top 10 efficiency bar chart.
    """
    while True:
        print("\nSelect a plot to display:")
        print("1 - Regression Model: Actual vs. Predicted Points (PTS)")
        print("2 - Scatter Plot: Salary vs. Efficiency ((PTS+TRB+AST)/MP)")
        print("3 - Bar Chart: Top 10 Players by Efficiency (Salary Comparison)")
        print("4 - Exit")
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            build_regression_model(merged_data)
        elif choice == '2':
            plot_salary_vs_efficiency(merged_data)
        elif choice == '3':
            plot_top10_salary_by_efficiency(merged_data)
        elif choice == '4':
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please try again.")

# Run the Interactive Menu
display_menu()