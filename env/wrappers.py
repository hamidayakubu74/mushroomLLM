import gymnasium as gym
import numpy as np

class NoisyObservationWrapper(gym.ObservationWrapper):
    """
    Adds Gaussian noise to observations to simulate sensor noise
    and make the policy more robust.
    """
    def __init__(self, env, noise_std=0.05):
        super().__init__(env)
        self.noise_std = noise_std

    def observation(self, obs):
        noise = np.random.normal(0, self.noise_std, size=obs.shape)
        return obs + noise

class SafetyClipActionWrapper(gym.ActionWrapper):
    """
    Ensures actions never exceed safety limits, even if the agent tries to.
    """
    def __init__(self, env):
        super().__init__(env)
        # Assuming action space is [-1, 1]
        self.clip_min = -1.0
        self.clip_max = 1.0

    def action(self, action):
        return np.clip(action, self.clip_min, self.clip_max)

class DiscreteActionWrapper(gym.ActionWrapper):
    """
    Converts continuous action space to discrete levels of stimulation.
    Useful for DQN agents.
    """
    def __init__(self, env, bins=5):
        super().__init__(env)
        self.bins = bins
        # Create discrete space
        self.action_space = gym.spaces.MultiDiscrete([bins, bins])
        
    def action(self, action):
        # Convert discrete indices to continuous [-1, 1]
        # action is [freq_idx, amp_idx]
        freq_idx, amp_idx = action
        
        freq_cont = -1 + (2 * freq_idx / (self.bins - 1))
        amp_cont = -1 + (2 * amp_idx / (self.bins - 1))
        
        return np.array([freq_cont, amp_cont], dtype=np.float32)
