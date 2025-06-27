# train_drl_ship.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from ship_env import ShipEnv  # Corrected import
import pygame

def train_agent():
    # Create environment
    env = ShipEnv()  # Create without rendering for training
    
    # Validate environment
    check_env(env)
    
    # Initialize PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./ship_ppo_tensorboard/",
        device="auto"  # Use GPU if available
    )
    
    # Train the agent
    model.learn(total_timesteps=500000, progress_bar=True)  # Increased timesteps
    
    # Save the model
    model.save("ppo_ship_control")
    print("Training completed. Model saved.")
    
    # Close environment
    env.close()

def evaluate_agent():
    # Create environment with rendering for evaluation
    env = ShipEnv()
    
    # Load the trained model
    model = PPO.load("ppo_ship_control")
    
    # Run evaluation episodes
    for episode in range(5):
        obs, _ = env.reset()
        done = False
        truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            env.render()  # Render the environment
            
            # Check if window closed
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
        
        print(f"Episode {episode+1} completed")
    
    env.close()

if __name__ == "__main__":
    train_agent()
    # Uncomment to evaluate after training:
    # evaluate_agent()