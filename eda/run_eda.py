import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def analyze_dataset(dataset_name, file_path):
    """
    Performs an exploratory data analysis on a given time series dataset.
    """
    print(f"--- Analyzing {dataset_name} Dataset ---")

    # Create eda directory if it doesn't exist
    if not os.path.exists('eda'):
        os.makedirs('eda')

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
        print(f"\n--- Plotting first 5 series for {dataset_name} ---")
        plt.figure(figsize=(15, 10))
        
        num_series_to_plot = min(5, len(df.columns))
        if num_series_to_plot > 0:
            for i in range(num_series_to_plot):
                plt.plot(df.index, df.iloc[:, i], label=df.columns[i])
        
            plt.title(f'{dataset_name} - First {num_series_to_plot} Time Series')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            
            # Save the plot
            plot_path = f'eda/{dataset_name}_plot.png'
            plt.savefig(plot_path)
            print(f"\nPlot saved to {plot_path}")
            plt.close()
        else:
            print("No columns to plot.")


    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EDA for time series datasets.')
    parser.add_argument('dataset_name', type=str, help='Name of the dataset')
    parser.add_argument('file_path', type=str, help='Path to the dataset CSV file')
    args = parser.parse_args()
    
    analyze_dataset(args.dataset_name, args.file_path)
