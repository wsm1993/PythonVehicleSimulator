# ship_env.py
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from python_vehicle_simulator.vehicles.shipClarke83 import shipClarke83
from ship_renderer import ShipRenderer  # <-- NEW IMPORT

class ShipClarke83Env(gym.Env):
    def __init__(self, render_mode=None):
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
        
        # Define DISCRETE action space
        self.action_space = spaces.Discrete(9)  # 3 thrust levels × 3 rudder angles
        
        # Action mapping:
        # 0: thrust_low, rudder_left
        # 1: thrust_low, rudder_center
        # 2: thrust_low, rudder_right
        # 3: thrust_med, rudder_left
        # 4: thrust_med, rudder_center
        # 5: thrust_med, rudder_right
        # 6: thrust_high, rudder_left
        # 7: thrust_high, rudder_center
        # 8: thrust_high, rudder_right
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(9,),  # Changed from (6,) to (9,)
            dtype=np.float32 
        )
        # Control limits
        self.max_tau_X = 1e4  # Max surge force (N)
        self.max_delta = np.deg2rad(30)  # Max rudder angle (rad)
        
        # Define discrete thrust levels (normalized)
        self.thrust_levels = {
            'low': 0.3,    # 30% of max thrust
            'med': 0.7,    # 70% of max thrust
            'high': 1.0    # 100% of max thrust
        }
        
        # Define discrete rudder angles (normalized)
        self.rudder_angles = {
            'left': -1.0,  # Full left
            'center': 0.0, # Center
            'right': 1.0   # Full right
        }
        
        # Initialize targets (will be set in reset)
        self.targets = []  # List to store 5 targets
        self.current_target_index = 0  # Index of current target
        
        # Normalization scales
        self.norm_scale = np.array([200, 200, 2*np.pi, 10, 5, np.pi/6, 200, 200, 2*np.pi])  # More realistic

        # Previous position error for reward calculation
        self.prev_pos_error = None

        # Episode parameters
        self.max_steps = 2000
        self.step_count = 0
        
        # Current control values
        self.current_thrust = 0
        self.current_rudder = 0
        self.current_action = -1
        
        # PyGame rendering setup
        self.render_mode = render_mode

        # PyGame rendering setup
        self.screen = None
        self.clock = None
        self.screen_width = 800
        self.screen_height = 800
        self.scale = 2.0  # Pixels per meter

        # Colors
        self.water_color = (135, 206, 235)  # Sky blue
        self.ship_color = (70, 130, 180)    # Steel blue
        self.target_color = (220, 20, 60)   # Crimson red
        self.trail_color = (30, 144, 255)   # Dodger blue
        self.text_color = (25, 25, 25)      # Dark gray
        self.gauge_color = (50, 50, 50)     # Dark gray for gauges
        self.thrust_color = (0, 128, 0)     # Green for thrust
        self.rudder_color = (128, 0, 0)     # Red for rudder

        # Trail for ship path
        self.trail = []
        self.max_trail_length = 200

        # Renderer instance
        self.renderer = ShipRenderer(
            screen_width=self.screen_width,
            screen_height=self.screen_height,
            scale=self.scale,
            water_color=self.water_color,
            ship_color=self.ship_color,
            target_color=self.target_color,
            trail_color=self.trail_color,
            text_color=self.text_color,
            gauge_color=self.gauge_color,
            thrust_color=self.thrust_color,
            rudder_color=self.rudder_color
        ) if render_mode == "human" else None

    def reset(self, seed=None, options=None):
        # Reset ship state
        self.eta = np.zeros(6)
        self.ship.nu = np.zeros(6)
        self.ship.u_actual = np.array([0.0])
        self.ship.tau_X = 0
        self.step_count = 0
        self.trail = []  # Clear trail
        self.current_target_index = 0  # Reset to first target

        # Reset control values
        self.current_thrust = 0
        self.current_rudder = 0
        self.current_action = -1
        
        # Generate 5 random targets
        self.targets = []
        for _ in range(5):
            r = np.random.uniform(30, 100)  # Random distance (30-50m)
            theta = np.random.uniform(0, 2*np.pi)  # Random angle
            x_target = r * np.cos(theta)
            y_target = r * np.sin(theta)
            psi_target = np.random.uniform(-np.pi, np.pi)  # Random heading
            u_target = np.random.uniform(3.0, 8.0)  # Random target speed (3-8 m/s)
            self.targets.append(np.array([x_target, y_target, psi_target, u_target]))
        
        # Get current target
        current_target = self.targets[0]
        x, y = self.eta[0], self.eta[1]
        x_target, y_target = current_target[0], current_target[1]
        self.prev_pos_error = np.sqrt((x - x_target)**2 + (y - y_target)**2)
        
        # Reset rendering if needed
        if self.render_mode == "human" and self.renderer is not None:
            self.renderer.reset()
            
        return self._get_obs(), {}

    def _get_obs(self):
        # Get current target
        current_target = self.targets[self.current_target_index]
        
        state = np.array([
            self.eta[0], self.eta[1], self.eta[5],
            self.ship.nu[0], self.ship.nu[1], self.ship.nu[5]
        ])
    
        # ADD RELATIVE TARGET INFORMATION
        dx = current_target[0] - self.eta[0]
        dy = current_target[1] - self.eta[1]
        dpsi = current_target[2] - self.eta[5]
        
        # Normalize and add to observation
        relative_state = np.array([dx, dy, dpsi])
        
        full_obs = np.concatenate([state, relative_state])
        normalized = full_obs / self.norm_scale
        return normalized.astype(np.float32)

    def step(self, action):
        # Store current action
        self.current_action = action
        
        # Map discrete action to thrust and rudder commands
        thrust_level = ['low', 'med', 'high'][action // 3]
        rudder_dir = ['left', 'center', 'right'][action % 3]
        
        # Convert to continuous values
        thrust_cont = self.thrust_levels[thrust_level]
        rudder_cont = self.rudder_angles[rudder_dir]
        
        # Scale to actual values
        tau_X = thrust_cont * self.max_tau_X
        delta_c = rudder_cont * self.max_delta

        # Store control values for rendering
        self.current_thrust = tau_X
        self.current_rudder = delta_c

        # Update ship's surge force
        self.ship.tau_X = tau_X

        # Simulation parameters
        sampleTime = 0.1
        substeps = 10  # Number of times to run dynamics per action

        for _ in range(substeps):
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
            
            # Add current position to trail
            self.trail.append((self.eta[0], self.eta[1]))
            if len(self.trail) > self.max_trail_length:
                self.trail.pop(0)

        # Prepare observation
        obs = self._get_obs()

        # Calculate reward
        reward, terminated = self._calculate_reward(obs, action)
        truncated = self.step_count >= self.max_steps

        # Increment step count
        self.step_count += 1
        
        # Render if needed
        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, {}

    def _calculate_reward(self, obs, action):
        terminated = False
        state = obs * self.norm_scale
        x, y, psi, u, v, r, dx, dy, dpsi = state
        current_target = self.targets[self.current_target_index]
        x_target, y_target, psi_target, u_target = current_target
        
        # Calculate position error
        current_pos_error = np.sqrt((x - x_target)**2 + (y - y_target)**2)
        
        # Progressive reward for distance reduction
        if hasattr(self, 'prev_pos_error'):
            distance_reduction = self.prev_pos_error - current_pos_error
        else:
            # First step - no previous error
            distance_reduction = 0
            
        # Position reward (exponential decay)
        position_reward = np.exp(-0.05 * current_pos_error)  # Better gradient
        
        # Heading reward (linear decay)
        psi_error = min(abs(psi - psi_target), 2*np.pi - abs(psi - psi_target))
        heading_reward = 1 - min(psi_error / np.pi, 1)  # Range [0,1]
        
        # Speed reward (Gaussian)
        speed_error = abs(u - u_target)
        speed_reward = np.exp(-0.5 * (speed_error**2))
        
        boundary_size = 200
        # Check if ship is out of bounds
        out_of_bounds = (
            x < -boundary_size or 
            x > boundary_size or 
            y < -boundary_size or 
            y > boundary_size
        )
        reward = 0
        target_reached = False
        # Check if current target reached
        if current_pos_error < 1:  # 1m position tolerance
            target_reached = True
            # Move to next target if available
            if self.current_target_index < len(self.targets) - 1:
                self.current_target_index += 1
                # Update prev_pos_error for new target
                next_target = self.targets[self.current_target_index]
                next_x, next_y = next_target[0], next_target[1]
                self.prev_pos_error = np.sqrt((x - next_x)**2 + (y - next_y)**2)
            else:
                # Last target reached
                terminated = True
            reward += 100  # Target reached bonus

        # Set prev_pos_error for next step if not moving to new target
        if not target_reached:
            self.prev_pos_error = current_pos_error

        # Terminal conditions
        if out_of_bounds:
            terminated = True
            reward -= 100  # Failure penalty
            
        return reward, bool(terminated)
    
    def render(self):
        """Render the environment using ShipRenderer"""
        if self.render_mode == "human" and self.renderer is not None:
            # Get current target for rendering
            current_target = self.targets[self.current_target_index]
            self.renderer.render(
                eta=self.eta,
                nu=self.ship.nu,
                trail=self.trail,
                target=current_target,
                step_count=self.step_count,
                max_steps=self.max_steps,
                current_action=self.current_action,
                current_thrust=self.current_thrust,
                current_rudder=self.current_rudder,
                max_tau_X=self.max_tau_X,
                max_delta=self.max_delta
            )

    def close(self):
        """Close the rendering window"""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None