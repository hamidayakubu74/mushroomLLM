import time
import numpy as np
import serial
import logging

class StimulatorInterface:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def apply_stimulation(self, frequency, amplitude):
        raise NotImplementedError

    def read_response(self):
        raise NotImplementedError

    def close(self):
        pass

class SerialStimulator(StimulatorInterface):
    def __init__(self, config):
        super().__init__(config)
        self.port = config['hardware']['port']
        self.baud = config['hardware']['baud_rate']
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=config['hardware']['timeout'])
            self.logger.info(f"Connected to hardware on {self.port}")
        except serial.SerialException as e:
            self.logger.error(f"Failed to connect to hardware: {e}")
            raise

    def apply_stimulation(self, frequency, amplitude):
        # Safety checks
        max_v = self.config['hardware']['safety_limits']['max_voltage']
        amplitude = min(amplitude, max_v)
        
        # Protocol: "STIM:FREQ:AMP\n"
        command = f"STIM:{frequency:.2f}:{amplitude:.2f}\n"
        self.ser.write(command.encode())

    def read_response(self):
        # Expecting a float value representing voltage/resistance response
        if self.ser.in_waiting:
            try:
                line = self.ser.readline().decode().strip()
                return float(line)
            except ValueError:
                return 0.0
        return 0.0

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

class MockStimulator(StimulatorInterface):
    def __init__(self, config):
        super().__init__(config)
        self.state = 0.0
        self.logger.info("Initialized Mock Stimulator")

    def apply_stimulation(self, frequency, amplitude):
        # Simulate a response function: 
        # The "organism" reacts to frequency changes with a delay and noise
        target_response = np.sin(frequency / 10.0) * amplitude
        # Simple low-pass filter to simulate biological inertia
        self.state = 0.9 * self.state + 0.1 * target_response

    def read_response(self):
        # Return state plus some biological noise
        noise = np.random.normal(0, 0.05)
        return self.state + noise
