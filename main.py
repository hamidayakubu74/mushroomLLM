import argparse
import yaml
import logging
from src.env.bio_env import BioInterfaceEnv
from src.model.agent import RLAgent
from stable_baselines3.common.monitor import Monitor
import os

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("experiment.log"),
            logging.StreamHandler()
        ]
    )

def main():
    parser = argparse.ArgumentParser(description="MycoRL: Organic Intelligence Interface")
    parser.add_argument('--mode', type=str, default='simulation', choices=['simulation', 'hardware'], help='Operation mode')
    parser.add_argument('--config', type=str, default='config/default_config.yaml', help='Path to configuration file')
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting MycoRL in {args.mode} mode")

    # Load configuration
    config = load_config(args.config)

    # Create environment
    env = BioInterfaceEnv(config, mode=args.mode)
    env = Monitor(env, filename=f"./logs/{config['experiment']['name']}")

    # Initialize Agent
    agent = RLAgent(env, config)

    # Train
    logger.info("Starting training...")
    try:
        agent.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    finally:
        # Save final model
        save_path = f"models/{config['experiment']['name']}_final"
        os.makedirs("models", exist_ok=True)
        agent.save(save_path)
        logger.info(f"Model saved to {save_path}")
        
        env.close()
        logger.info("Environment closed.")

if __name__ == "__main__":
    main()
