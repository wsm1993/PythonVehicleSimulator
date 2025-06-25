# ship_env.py
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from python_vehicle_simulator.vehicles.shipClarke83 import shipClarke83  # Using the provided model

class ShipClarke83Env(gym.Env):
    def __init__(self):
        super().__init__()  # Initialize gymnasium.Env
        
        # Ship parameters
        self.L = 15.0  # Length (m)
        self.B = 3.0   # Beam (m)
        self.T = 3.0   # Draft (m)
        self.Cb = 0.7  # Block coefficient
        
        # Initialize ship model
        self.ship = shipClarke83(
            controlSystem='stepInput',
            L=self.L, B=self.B, T=self.T, Cb=self.Cb,
            V_current=0, beta_current=0, tau_X=0
        )
        
        # Maintain position and attitude state
        self.eta = np.zeros(6)  # [x, y, z, φ, θ, ψ]
        self.ship.nu = np.zeros(6)  # Velocity vector
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(6,), 
            dtype=np.float32 
        )
        
        # Control limits
        self.max_tau_X = 1e3  # Max surge force (N)
        self.max_delta = np.deg2rad(30)  # Max rudder angle (rad)
        
        # Target state: [x_target, y_target, ψ_target, u_target]
        self.target = np.array([100, 100, np.deg2rad(45), 5.0])
        
        # Normalization scales
        self.norm_scale = np.array([200, 200, 2*np.pi, 10, 5, np.pi/6])  # More realistic
        
        # Episode parameters
        self.max_steps = 100000
        self.step_count = 0

    def reset(self, seed=None, options=None):
        # Reset ship state
        self.eta = np.zeros(6)
        self.ship.nu = np.zeros(6)
        self.ship.u_actual = np.array([0.0])
        self.ship.tau_X = 0
        self.step_count = 0
        return self._get_obs(), {}  # Return obs and empty info dict

    def _get_obs(self):
        state = np.array([
            self.eta[0], self.eta[1], self.eta[5],
            self.ship.nu[0], self.ship.nu[1], self.ship.nu[5]
        ])
        normalized = state / self.norm_scale
        return normalized.astype(np.float32) 

    def step(self, action):
        # Scale actions to actual values
        tau_X = action[0] * self.max_tau_X
        delta_c = action[1] * self.max_delta
        
        # Update ship's surge force
        self.ship.tau_X = tau_X
        
        # Simulation parameters
        sampleTime = 0.1
        
        # Update ship dynamics
        nu_new, u_actual_new = self.ship.dynamics(
            self.eta, self.ship.nu, self.ship.u_actual, 
            np.array([delta_c]), sampleTime
        )
        
        # Update ship state
        self.ship.nu = nu_new
        self.ship.u_actual = u_actual_new
        
        # Update position and heading (kinematics)
        u, v, r = nu_new[0], nu_new[1], nu_new[5]
        psi = self.eta[5]
        dx = u * np.cos(psi) - v * np.sin(psi)
        dy = u * np.sin(psi) + v * np.cos(psi)
        self.eta += sampleTime * np.array([dx, dy, 0, 0, 0, r])
        
        # Prepare observation
        obs = self._get_obs()
        
        # Calculate reward
        reward, terminated = self._calculate_reward(obs, action)
        truncated = self.step_count >= self.max_steps
        
        # Increment step count
        self.step_count += 1
            
        return obs, reward, terminated, truncated, {}

    def _calculate_reward(self, obs, action):
        state = obs * self.norm_scale
        x, y, psi, u, v, r = state
        x_target, y_target, psi_target, u_target = self.target
        
        # Position reward (inverse distance)
        pos_error = np.sqrt((x - x_target)**2 + (y - y_target)**2)
        position_reward = 1.0 / (1.0 + pos_error)  # [0, 1] range
        
        # Heading reward
        psi_error = min(abs(psi - psi_target), 2*np.pi - abs(psi - psi_target))
        heading_reward = np.cos(psi_error)  # [-1, 1] range
        
        # Speed reward
        speed_error = abs(u - u_target)
        speed_reward = 1.0 / (1.0 + speed_error)
        
        # Action penalties (gentler)
        surge_penalty = 0.001 * (action[0] ** 2)
        rudder_penalty = 0.001 * (action[1] ** 2)
        
        # Composite reward (weighted components)
        reward = (
            2.0 * position_reward +
            1.0 * heading_reward +
            0.5 * speed_reward -
            surge_penalty -
            rudder_penalty
        )
        
        # Success conditions (more achievable)
        terminated = bool(
            pos_error < 20 and  # 20m position tolerance
            psi_error < np.deg2rad(15) and  # 15° heading tolerance
            speed_error < 1.0  # 1m/s speed tolerance
        )
        
        if terminated:
            reward += 50  # Success bonus
            
        return reward, terminated