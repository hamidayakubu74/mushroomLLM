import numpy as np

class RewardFunctions:
    """
    Collection of reward functions for training the agent.
    """
    
    @staticmethod
    def target_match(response, target=1.0, tolerance=0.1):
        """
        Reward based on how close the response is to a target value.
        """
        error = abs(response - target)
        if error < tolerance:
            return 1.0 - error
        return -error

    @staticmethod
    def maximize_activity(response):
        """
        Reward for maximizing the electrical response (activity).
        """
        return response

    @staticmethod
    def sinusoidal_tracking(response, step, period=100):
        """
        Reward for tracking a generated sine wave.
        """
        target = np.sin(2 * np.pi * step / period)
        error = (response - target) ** 2
        return -error

    @staticmethod
    def stability(history):
        """
        Reward for keeping the signal stable (low variance).
        """
        if len(history) < 2:
            return 0.0
        variance = np.var(history)
        return -variance
