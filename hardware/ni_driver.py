import logging
import numpy as np
from src.hardware.stimulator import StimulatorInterface

# Mocking nidaqmx to allow code to exist without the heavy driver installed
try:
    import nidaqmx
    from nidaqmx.constants import AcquisitionType
except ImportError:
    nidaqmx = None

class NIDaqDriver(StimulatorInterface):
    """
    Driver for National Instruments DAQ devices (e.g., USB-600x series).
    Requires 'nidaqmx' python package and NI drivers installed.
    """
    def __init__(self, config):
        super().__init__(config)
        self.device_name = config['hardware'].get('ni_device_name', 'Dev1')
        self.ao_channel = f"{self.device_name}/ao0"
        self.ai_channel = f"{self.device_name}/ai0"
        self.sample_rate = 1000
        
        if nidaqmx is None:
            self.logger.warning("nidaqmx module not found. Running in fallback/mock mode.")
            self.mock_mode = True
        else:
            self.mock_mode = False
            self.task_ao = nidaqmx.Task()
            self.task_ai = nidaqmx.Task()
            self._setup_tasks()

    def _setup_tasks(self):
        try:
            self.task_ao.ao_channels.add_ao_voltage_chan(self.ao_channel)
            self.task_ai.ai_channels.add_ai_voltage_chan(self.ai_channel)
        except Exception as e:
            self.logger.error(f"Failed to setup NI Tasks: {e}")
            self.mock_mode = True

    def apply_stimulation(self, frequency, amplitude):
        if self.mock_mode:
            return
            
        # For a simple DAQ, we might just set a DC voltage or generate a waveform
        # Here we assume we are just setting a voltage level for this step
        # In a real scenario, we might write a buffer for a waveform
        try:
            self.task_ao.write(amplitude)
        except Exception as e:
            self.logger.error(f"NI Write Error: {e}")

    def read_response(self):
        if self.mock_mode:
            return np.random.random()
            
        try:
            return self.task_ai.read()
        except Exception as e:
            self.logger.error(f"NI Read Error: {e}")
            return 0.0

    def close(self):
        if not self.mock_mode:
            self.task_ao.close()
            self.task_ai.close()
