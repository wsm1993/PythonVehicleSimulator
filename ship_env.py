# ship_env.py
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
from python_vehicle_simulator.vehicles.shipClarke83 import shipClarke83  # Using the provided model

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
            shape=(6,), 
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
        
        # Target state: [x_target, y_target, ψ_target, u_target]
        self.target = np.array([30, 30, np.deg2rad(45), 5.0])
        
        # Normalization scales
        self.norm_scale = np.array([200, 200, 2*np.pi, 10, 5, np.pi/6])  # More realistic
        
        # Episode parameters
        self.max_steps = 1000
        self.step_count = 0
        
        # PyGame rendering setup
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.screen_width = 800
        self.screen_height = 800
        self.scale = 2.0  # Pixels per meter
        
        # Colors
        self.water_color = (135, 206, 235)  # Sky blue
        self.ship_color = (70, 130, 180)    # Steel blue
        self.target_color = (220, 20, 60)    # Crimson red
        self.trail_color = (30, 144, 255)    # Dodger blue
        self.text_color = (25, 25, 25)       # Dark gray
        
        # Trail for ship path
        self.trail = []
        self.max_trail_length = 200

    def reset(self, seed=None, options=None):
        # Reset ship state
        self.eta = np.zeros(6)
        self.ship.nu = np.zeros(6)
        self.ship.u_actual = np.array([0.0])
        self.ship.tau_X = 0
        self.step_count = 0
        self.trail = []  # Clear trail
        
        # Reset rendering if needed
        if self.render_mode == "human":
            self._render_init()
            
        return self._get_obs(), {}  # Return obs and empty info dict

    def _get_obs(self):
        state = np.array([
            self.eta[0], self.eta[1], self.eta[5],
            self.ship.nu[0], self.ship.nu[1], self.ship.nu[5]
        ])
        normalized = state / self.norm_scale
        return normalized.astype(np.float32) 

    def step(self, action):
        # Map discrete action to thrust and rudder commands
        thrust_level = ['low', 'med', 'high'][action // 3]
        rudder_dir = ['left', 'center', 'right'][action % 3]
        
        # Convert to continuous values
        thrust_cont = self.thrust_levels[thrust_level]
        rudder_cont = self.rudder_angles[rudder_dir]
        
        # Scale to actual values
        tau_X = thrust_cont * self.max_tau_X
        delta_c = rudder_cont * self.max_delta

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
        if truncated:
            reward -= 50  # Penalty for truncation

        # Increment step count
        self.step_count += 1
        
        # Render if needed
        if self.render_mode == "human":
            self.render()

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
        # Since actions are discrete, we can add small penalty for using rudder
        rudder_penalty = 0.001 if action % 3 != 1 else 0  # Penalize non-center rudder
        
        # Composite reward (weighted components)
        reward = (
            2.0 * position_reward +
            1.0 * heading_reward +
            0.5 * speed_reward -
            rudder_penalty
        )
        
        # Check if ship is out of bounds
        out_of_bounds = (
            x < -200 or 
            x > 200 or 
            y < -200 or 
            y > 200
        )
        
        # Success conditions (more achievable)
        success = (
            pos_error < 20 and  # 20m position tolerance
            psi_error < np.deg2rad(15) and  # 15° heading tolerance
            speed_error < 1.0  # 1m/s speed tolerance
        )
        
        # Terminate if out of bounds or success condition met
        terminated = out_of_bounds or success
        
        if success:
            reward += 50  # Success bonus
        elif out_of_bounds:
            # Apply penalty for going out of bounds
            reward -= 50
            
        return reward, bool(terminated)
    
    def _render_init(self):
        """Initialize PyGame rendering"""
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Ship Navigation Environment")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)
    
    def render(self):
        """Render the environment using PyGame"""
        if self.screen is None and self.render_mode == "human":
            self._render_init()
        
        # Clear screen
        self.screen.fill(self.water_color)
        
        # Calculate center offset - world coordinates origin at screen center
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2
        
        # Draw grid
        self._draw_grid(center_x, center_y)
        
        # Draw ship trail
        if len(self.trail) > 1:
            pygame.draw.lines(
                self.screen, 
                self.trail_color, 
                False, 
                [(center_x + x * self.scale, center_y - y * self.scale) for x, y in self.trail],
                2
            )
        
        # Draw target
        target_x = center_x + self.target[0] * self.scale
        target_y = center_y - self.target[1] * self.scale
        pygame.draw.circle(self.screen, self.target_color, (int(target_x), int(target_y)), 10)
        
        # Draw target heading indicator
        target_heading = self.target[2]
        end_x = target_x + 20 * np.cos(target_heading)
        end_y = target_y - 20 * np.sin(target_heading)
        pygame.draw.line(
            self.screen, 
            (0, 0, 0), 
            (target_x, target_y), 
            (end_x, end_y), 
            3
        )
        
        # Draw ship
        ship_x = center_x + self.eta[0] * self.scale
        ship_y = center_y - self.eta[1] * self.scale
        heading = self.eta[5]  # Heading in radians
        
        # Create ship polygon (triangle)
        ship_points = []
        for i in range(3):
            angle = heading + i * 2 * np.pi / 3  # Points at 120° intervals
            px = ship_x + 15 * np.cos(angle)
            py = ship_y - 15 * np.sin(angle)
            ship_points.append((px, py))
        
        pygame.draw.polygon(self.screen, self.ship_color, ship_points)
        
        # Draw heading indicator
        end_x = ship_x + 30 * np.cos(heading)
        end_y = ship_y - 30 * np.sin(heading)
        pygame.draw.line(
            self.screen, 
            (0, 0, 0), 
            (ship_x, ship_y), 
            (end_x, end_y), 
            3
        )
        
        # Draw info panel
        self._draw_info_panel()
        
        # Update display
        pygame.display.flip()
        
        # Maintain frame rate
        self.clock.tick(60)
    
    def _draw_grid(self, center_x, center_y):
        """Draw a grid for better spatial reference"""
        # Draw major grid lines every 50 meters
        grid_size = 50 * self.scale
        
        # Vertical lines
        for x in range(-200, 201, 50):
            screen_x = center_x + x * self.scale
            if 0 <= screen_x <= self.screen_width:
                pygame.draw.line(
                    self.screen, 
                    (200, 200, 200), 
                    (screen_x, 0), 
                    (screen_x, self.screen_height), 
                    1
                )
        
        # Horizontal lines
        for y in range(-200, 201, 50):
            screen_y = center_y - y * self.scale
            if 0 <= screen_y <= self.screen_height:
                pygame.draw.line(
                    self.screen, 
                    (200, 200, 200), 
                    (0, screen_y), 
                    (self.screen_width, screen_y), 
                    1
                )
        
        # Draw axes
        pygame.draw.line(
            self.screen, 
            (100, 100, 100), 
            (center_x, 0), 
            (center_x, self.screen_height), 
            2
        )
        pygame.draw.line(
            self.screen, 
            (100, 100, 100), 
            (0, center_y), 
            (self.screen_width, center_y), 
            2
        )
        
        # Draw origin marker
        pygame.draw.circle(self.screen, (0, 0, 0), (center_x, center_y), 5)
    
    def _draw_info_panel(self):
        """Draw information panel at top of screen"""
        # Create info strings
        position = f"Position: ({self.eta[0]:.1f}, {self.eta[1]:.1f}) m"
        heading = f"Heading: {np.rad2deg(self.eta[5]):.1f}°"
        speed = f"Speed: {self.ship.nu[0]:.1f} m/s"
        steps = f"Step: {self.step_count}/{self.max_steps}"
        
        # Render text
        texts = [
            self.font.render(position, True, self.text_color),
            self.font.render(heading, True, self.text_color),
            self.font.render(speed, True, self.text_color),
            self.font.render(steps, True, self.text_color)
        ]
        
        # Draw text panel
        pygame.draw.rect(self.screen, (240, 240, 240), (10, 10, 300, 120))
        
        # Blit texts
        for i, text in enumerate(texts):
            self.screen.blit(text, (20, 20 + i * 24))
    
    def close(self):
        """Close the rendering window"""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None