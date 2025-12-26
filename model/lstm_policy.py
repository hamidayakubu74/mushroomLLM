import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

class LSTMExtractor(BaseFeaturesExtractor):
    """
    Custom Feature Extractor for Stable Baselines3 that uses an LSTM.
    Useful because the biological substrate has 'memory' and hidden states.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 64):
        super(LSTMExtractor, self).__init__(observation_space, features_dim)
        
        # We assume observation is a 1D array of history [t, t-1, t-2...]
        input_dim = 1 
        hidden_dim = 64
        num_layers = 2
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Observations shape: [batch_size, window_size]
        # LSTM needs: [batch_size, seq_len, input_dim]
        
        # Add the input_dim dimension
        x = observations.unsqueeze(-1) 
        
        # Pass through LSTM
        lstm_out, (hn, cn) = self.lstm(x)
        
        # We take the output of the last time step
        last_time_step = lstm_out[:, -1, :]
        
        return self.linear(last_time_step)

# Note: To use this in SB3, pass policy_kwargs=dict(features_extractor_class=LSTMExtractor)
