# MycoRL: Reinforcement Learning Interface for Organic Intelligence

## Overview

MycoRL is an experimental framework designed to interface Reinforcement Learning (RL) agents with organic substrates (specifically mycelial networks) using modulated electrical stimulation. The goal is to train the organic network to perform computational tasks or respond to environmental stimuli by providing feedback via electrical impulses.

This project bridges the gap between biological computing and classical machine learning, treating the organic substrate as a "black box" environment that an RL agent learns to control or collaborate with.

## Architecture

The system consists of three main components:

1.  **The Agent (Digital)**: A Deep Reinforcement Learning model (PPO/SAC) that observes the state of the organic network and decides on stimulation parameters.
2.  **The Interface (Hardware/Driver)**: A bridge that converts digital actions into analog electrical signals (voltage/frequency modulation) and reads back the electrical response (spiking activity/resistance changes) from the substrate.
3.  **The Substrate (Organic)**: The biological neural network or mycelial mat.

## Features

-   **Modulated Stimulation**: Supports Pulse Width Modulation (PWM), Frequency Modulation (FM), and Amplitude Modulation (AM) of stimulation signals.
-   **Real-time Feedback Loop**: Low-latency processing of substrate response to close the RL loop.
-   **Gym-compatible Environment**: Custom OpenAI Gym environment wrapper for easy integration with standard RL libraries (Stable Baselines3, Ray RLLib).
-   **Safety Limits**: Software-defined voltage and current clamps to prevent damage to the organic sample.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Simulation Mode
To test the RL agent without hardware, use the mock environment:

```bash
python main.py --mode simulation
```

### Hardware Mode
To connect to the stimulation controller (e.g., Arduino/DAC):

```bash
python main.py --mode hardware --port /dev/ttyUSB0
```

## Configuration

Edit `config/default_config.yaml` to adjust:
-   Stimulation frequency ranges.
-   Reward function parameters.
-   RL hyperparameters.

## Disclaimer

This software is for research purposes only. Ensure ethical guidelines are followed when working with biological substrates.
