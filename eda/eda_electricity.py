import pandas as pd
import matplotlib.pyplot as plt
import os

def analyze_electricity():
    """
    Performs an exploratory data analysis on the electricity dataset.
    """
    print("--- Analyzing Electricity Dataset ---")
    
    # Create eda directory if it doesn't exist
    if not os.path.exists('eda'):
        os.makedirs('eda')

    file_path = 'dataset/electricity/electricity.csv'

    try:
        # Load the data
        df = pd.read_csv(file_path, parse_dates=[0], index_col=0)

        # Display basic information
        print("\n--- Data Head ---")
        print(df.head())
        print("\n--- Data Info ---")
        df.info()
        print("\n--- Data Description ---")
        print(df.describe())

        # Check for missing values
        print("\n--- Missing Values ---")
        print(df.isnull().sum())

        # Plotting the first 5 time series
        print("\n--- Plotting first 5 series ---")
        plt.figure(figsize=(15, 10))
        for i in range(min(5, len(df.columns))):
            plt.plot(df.index, df.iloc[:, i], label=df.columns[i])
        
        plt.title('Electricity Consumption for first 5 clients')
        plt.xlabel('Date')
        plt.ylabel('Consumption')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plot_path = 'eda/electricity_plot.png'
        plt.savefig(plot_path)
        print(f"\nPlot saved to {plot_path}")
        plt.close()

    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    analyze_electricity()
