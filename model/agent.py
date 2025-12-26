from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
import os

class RLAgent:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.model_type = config['rl_agent']['algorithm']
        self.model = self._create_model()

    def _create_model(self):
        if self.model_type == "PPO":
            return PPO(
                "MlpPolicy", 
                self.env, 
                verbose=1,
                learning_rate=self.config['rl_agent']['learning_rate'],
                gamma=self.config['rl_agent']['gamma'],
                batch_size=self.config['rl_agent']['batch_size']
            )
        elif self.model_type == "SAC":
            return SAC(
                "MlpPolicy", 
                self.env, 
                verbose=1,
                learning_rate=self.config['rl_agent']['learning_rate'],
                gamma=self.config['rl_agent']['gamma'],
                batch_size=self.config['rl_agent']['batch_size']
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.model_type}")

    def train(self):
        total_timesteps = self.config['experiment']['max_steps'] * self.config['rl_agent']['n_epochs']
        
        # Save checkpoints
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config['experiment']['save_interval'],
            save_path='./logs/',
            name_prefix='rl_model'
        )

        self.model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        if self.model_type == "PPO":
            self.model = PPO.load(path, env=self.env)
        elif self.model_type == "SAC":
            self.model = SAC.load(path, env=self.env)
