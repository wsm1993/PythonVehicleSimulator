# train_drl_ship.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from ship_env import ShipEnv
import pygame

class RenderCallback(BaseCallback):
    """ 
    Callback for rendering the environment during training.
    Renders every N steps to avoid slowing down training too much.
    """
    def __init__(self, render_freq: int, env: gym.Env):
        super(RenderCallback, self).__init__()
        self.render_freq = render_freq
        self.env = env

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            self.env.render()
            
            # Handle pygame events to prevent freezing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False  # Stop training if window closed
        return True

def train_agent(render_during_training=False):
    # Create environment
    env = ShipEnv()  # Create without rendering by default
    
    # Validate environment
    check_env(env)
    
    # Initialize PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./ship_ppo_tensorboard/",
        device="auto",  # Use GPU if available
        n_steps=2048,  # More stable learning
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01
    )
    
    # Create callback if rendering during training
    callbacks = []
    if render_during_training:
        callbacks.append(RenderCallback(render_freq=100, env=env))
    
    # Train the agent
    model.learn(
        total_timesteps=500000, 
        progress_bar=True,
        callback=callbacks if callbacks else None
    )
    
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
            env.render()
            
            # Check if window closed
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
        
        print(f"Episode {episode+1} completed")
    
    env.close()

if __name__ == "__main__":
    # Set to True to see occasional rendering during training
    train_agent(render_during_training=True)
    
    # Evaluate after training
    evaluate_agent()