import csv
import time
import os
import logging

class ExperimentLogger:
    def __init__(self, log_dir, filename="data_log.csv"):
        self.log_dir = log_dir
        self.filepath = os.path.join(log_dir, filename)
        self.logger = logging.getLogger(__name__)
        
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize CSV with headers if it doesn't exist
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'step', 'frequency', 'amplitude', 'response', 'reward'])
            self.logger.info(f"Created new data log at {self.filepath}")

    def log_step(self, step, action, response, reward):
        """
        Log a single step of the experiment.
        action: tuple or list [frequency, amplitude]
        """
        timestamp = time.time()
        frequency, amplitude = action
        
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, step, frequency, amplitude, response, reward])
