import time
import argparse
import yaml
import sys
import os
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.hardware.stimulator import SerialStimulator, MockStimulator

def calibrate(config_path, mode):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger = logging.getLogger("Calibration")
    logging.basicConfig(level=logging.INFO)

    if mode == 'hardware':
        stim = SerialStimulator(config)
    else:
        stim = MockStimulator(config)

    logger.info("Starting Impedance Check / Calibration...")
    
    # Ramp up voltage to check connectivity
    test_voltages = [0.1, 0.5, 1.0, 2.0, 3.0]
    freq = 10.0 # Hz
    
    results = []

    try:
        for v in test_voltages:
            logger.info(f"Applying {v}V at {freq}Hz...")
            stim.apply_stimulation(freq, v)
            time.sleep(1.0) # Wait for settle
            
            response = stim.read_response()
            logger.info(f"Response: {response:.4f}")
            
            # Simple impedance estimation (V=IR -> R=V/I)
            # Assuming response is current-related or proportional
            if response > 0.001:
                impedance = v / response
                logger.info(f"Estimated Impedance: {impedance:.2f} units")
            else:
                logger.warning("Response too low to estimate impedance.")
            
            results.append((v, response))
            
    except KeyboardInterrupt:
        logger.info("Calibration aborted.")
    finally:
        stim.close()
        logger.info("Calibration complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/default_config.yaml')
    parser.add_argument('--mode', default='simulation')
    args = parser.parse_args()
    
    calibrate(args.config, args.mode)
