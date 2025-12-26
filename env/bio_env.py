import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging
from src.hardware.stimulator import MockStimulator, SerialStimulator

class BioInterfaceEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This environment connects the RL agent to the biological substrate (or simulation).
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, config, mode='simulation'):
        super(BioInterfaceEnv, self).__init__()
        self.config = config
        self.mode = mode
        self.logger = logging.getLogger(__name__)

        # Define action and observation space
        # Action: [Frequency, Amplitude]
        # Normalized to [-1, 1] for stable baselines
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Observation: History of responses
        # We keep a window of past readings
        self.obs_window = config['environment']['observation_window']
        self.observation_space = spaces.Box(low=-5, high=5, shape=(self.obs_window,), dtype=np.float32)

        # Initialize hardware interface
        if mode == 'hardware':
            self.stimulator = SerialStimulator(config)
        else:
            self.stimulator = MockStimulator(config)

        self.history = np.zeros(self.obs_window)
        self.steps = 0
        self.max_steps = config['experiment']['max_steps']

    def step(self, action):
        self.steps += 1
        
        # Unscale action
        freq_norm, amp_norm = action
        
        min_f = self.config['stimulation']['min_frequency']
        max_f = self.config['stimulation']['max_frequency']
        min_a = self.config['stimulation']['min_amplitude']
        max_a = self.config['stimulation']['max_amplitude']

        # Map [-1, 1] to [min, max]
        frequency = min_f + (max_f - min_f) * ((freq_norm + 1) / 2)
        amplitude = min_a + (max_a - min_a) * ((amp_norm + 1) / 2)

        # Apply action
        self.stimulator.apply_stimulation(frequency, amplitude)

        # Read response
        response = self.stimulator.read_response()

        # Update history
        self.history = np.roll(self.history, -1)
        self.history[-1] = response

        # Calculate reward
        # Goal: Maximize response (just a placeholder objective)
        # Or match a target pattern
        reward = self._calculate_reward(response)

        # Check done
        terminated = False
        truncated = False
        if self.steps >= self.max_steps:
            truncated = True

        info = {
            "frequency": frequency,
            "amplitude": amplitude,
            "response": response
        }

        return self.history, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.history = np.zeros(self.obs_window)
        return self.history, {}

    def render(self, mode='human'):
        pass

    def close(self):
        self.stimulator.close()

    def _calculate_reward(self, current_response):
        # Example reward: Keep response close to 1.0
        target = 1.0
        error = abs(current_response - target)
        return -error
