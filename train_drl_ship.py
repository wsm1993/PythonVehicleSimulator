# train_drl_ship.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from ship_env import ShipClarke83Env  # Use the updated environment

def train_agent():
    # Create environment
    env = ShipClarke83Env()
    check_env(env)  # Validate environment
    
    # Initialize PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./ship_ppo_tensorboard/"
    )
    
    # Train the agent
    model.learn(total_timesteps=1000000)
    
    # Save the model
    model.save("ppo_ship_control")
    print("Training completed. Model saved.")

if __name__ == "__main__":
    train_agent()