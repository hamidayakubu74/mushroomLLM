import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_training_results(log_file):
    """
    Plot rewards and response over time from a CSV log.
    """
    try:
        df = pd.read_csv(log_file)
    except FileNotFoundError:
        print(f"File {log_file} not found.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Plot 1: Stimulus (Frequency & Amplitude)
    sns.lineplot(data=df, x='step', y='frequency', ax=axes[0], label='Frequency (Hz)', color='blue')
    ax2 = axes[0].twinx()
    sns.lineplot(data=df, x='step', y='amplitude', ax=ax2, label='Amplitude (V)', color='orange')
    axes[0].set_title("Stimulation Parameters")
    
    # Plot 2: Response
    sns.lineplot(data=df, x='step', y='response', ax=axes[1], color='green')
    axes[1].set_title("Organism Response")
    axes[1].set_ylabel("Voltage/Resistance")

    # Plot 3: Reward
    sns.lineplot(data=df, x='step', y='reward', ax=axes[2], color='red')
    axes[2].set_title("RL Reward")
    
    plt.tight_layout()
    plt.show()

def plot_phase_space(log_file):
    """
    Plot Phase Space (Response[t] vs Response[t-1]) to see attractors.
    """
    df = pd.read_csv(log_file)
    response = df['response'].values
    
    plt.figure(figsize=(8, 8))
    plt.plot(response[:-1], response[1:], alpha=0.5, lw=1)
    plt.title("Phase Space Reconstruction")
    plt.xlabel("Response[t]")
    plt.ylabel("Response[t+1]")
    plt.grid(True)
    plt.show()
