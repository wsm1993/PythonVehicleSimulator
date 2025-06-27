# train_drl_ship.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from ship_env import ShipClarke83Env  # Use the updated environment

# Register the custom environment
gym.register(
    id="ShipClarke83Env-v0",
    entry_point="ship_env:ShipClarke83Env",
    kwargs={"render_mode": None}
)

def train_agent():
    # Setup vectorized environments
    num_envs = 4  # Reduce number for stability
    env_id = "ShipClarke83Env-v0"
    
    # Create vectorized environment
    vec_env = make_vec_env(
        env_id,
        n_envs=num_envs,
        seed=42,
        vec_env_cls=DummyVecEnv,  # Use DummyVecEnv for better compatibility
        monitor_dir="./logs/",
        env_kwargs={"render_mode": None}
    )
    
    # Evaluation environment
    eval_env = DummyVecEnv([lambda: gym.make(env_id, render_mode=None)])
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./logs/",
        eval_freq=5000,
        n_eval_episodes=3,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Initialize PPO agent
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log="./ship_ppo_tensorboard/",
        device="cuda" if torch.cuda.is_available() else "auto",
        n_steps=2048 // num_envs,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        learning_rate=3e-4
    )
    
    # Train the agent
    try:
        model.learn(
            total_timesteps=200000,
            callback=eval_callback,
            progress_bar=True,
            tb_log_name="PPO"
        )
    except KeyboardInterrupt:
        print("Training interrupted by user")
    
    # Save the model
    model.save("ppo_ship_control")
    print("Training completed. Model saved.")
    
    # Close environments
    vec_env.close()
    eval_env.close()

if __name__ == "__main__":
    import torch
    train_agent()