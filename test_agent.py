import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from ship_env import ShipClarke83Env
import matplotlib.pyplot as plt


def test_agent():
    # Load environment and model
    env = ShipClarke83Env()
    model = PPO.load("ppo_ship_control")

    positions = []
    for episode in range(5):
        obs, _ = env.reset()
        terminated = truncated = False
        positions.append([])
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            positions[-1].append((env.eta[0], env.eta[1]))
        # Plot trajectory after each episode
        x, y = zip(*positions[-1])
        plt.plot(x, y, label=f"Episode {episode+1}")

    plt.plot(env.target[0], env.target[1], 'ro', markersize=10)
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Ship Trajectories')
    plt.legend()
    plt.show()    

    # # Run 5 episodes
    # for episode in range(5):
    #     obs, _ = env.reset()  # Unpack the tuple
    #     terminated = False
    #     truncated = False
    #     while not (terminated or truncated):
    #         action, _ = model.predict(obs, deterministic=True)
    #         obs, reward, terminated, truncated, _ = env.step(action)
    #         # Log ship state
    #         print(
    #             f"Position: ({env.eta[0]:.1f}, {env.eta[1]:.1f}) | "
    #             f"Heading: {np.rad2deg(env.eta[5]):.1f}° | "
    #             f"Speed: {env.ship.nu[0]:.1f} m/s"
    #             f"Action: {action} | "
    #         )
    #     print(f"Episode {episode+1} completed.")

if __name__ == "__main__":
    test_agent()