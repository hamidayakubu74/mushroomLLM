import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.plotting import plot_training_results, plot_phase_space

def main():
    parser = argparse.ArgumentParser(description="Analyze MycoRL Experiment Data")
    parser.add_argument('logfile', type=str, help='Path to the CSV log file')
    args = parser.parse_args()

    if not os.path.exists(args.logfile):
        print(f"Error: File {args.logfile} does not exist.")
        return

    print(f"Analyzing {args.logfile}...")
    
    # Basic Stats
    df = pd.read_csv(args.logfile)
    print("\n--- Statistics ---")
    print(df.describe())

    # Plots
    print("\nGenerating plots...")
    plot_training_results(args.logfile)
    plot_phase_space(args.logfile)

if __name__ == "__main__":
    main()
