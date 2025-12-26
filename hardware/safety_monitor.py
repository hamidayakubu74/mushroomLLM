import logging

class SafetyMonitor:
    """
    Independent safety monitor to ensure the RL agent doesn't fry the mushroom.
    """
    def __init__(self, config):
        self.max_voltage = config['hardware']['safety_limits']['max_voltage']
        self.max_current = config['hardware']['safety_limits']['max_current']
        self.logger = logging.getLogger(__name__)
        self.violation_count = 0

    def check_stimulation(self, voltage, current=None):
        """
        Returns True if stimulation is safe, False otherwise.
        """
        is_safe = True
        
        if abs(voltage) > self.max_voltage:
            self.logger.warning(f"SAFETY VIOLATION: Voltage {voltage:.2f}V exceeds limit {self.max_voltage}V")
            is_safe = False
            
        if current is not None and abs(current) > self.max_current:
            self.logger.warning(f"SAFETY VIOLATION: Current {current:.4f}A exceeds limit {self.max_current}A")
            is_safe = False
            
        if not is_safe:
            self.violation_count += 1
            
        return is_safe

    def get_safe_voltage(self, voltage):
        """
        Clamps the voltage to the safe range.
        """
        return max(-self.max_voltage, min(voltage, self.max_voltage))
