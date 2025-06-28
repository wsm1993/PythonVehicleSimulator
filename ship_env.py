import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from python_vehicle_simulator.vehicles.shipClarke83 import shipClarke83
from ship_renderer import ShipRenderer
import pygame

class ShipEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'render_fps': 3}
    
    def __init__(self):
        super(ShipEnv, self).__init__()
        
        # Environment parameters
        self.episode_length = 1000  # max steps per episode
        self.success_radius = 10.0  # target reach radius (meters)
        self.heading_tolerance = math.radians(10)  # heading tolerance (radians)
        self.world_boundary = 500.0  # boundary limit (meters)
        self.sim_steps_per_action = 10  # internal simulation steps per action
        self.dt = 0.1  # internal simulation timestep (seconds)
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Updated observation space with heading error
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf,
            shape=(10,),  # [u, v, r, psi, rel_x, rel_y, target_idx, last_thrust, last_rudder, heading_error]
            dtype=np.float32
        )
        
        # Ship control parameters
        self.max_thrust = 1e5  # Newtons
        self.max_rudder = math.radians(30)  # radians
        
        # Target positions (global coordinates) and headings
        self.targets = np.array([
            [100, 0, math.radians(0)],
            [100, 100, math.radians(90)]
        ])
        
        # Initialize state variables
        self.ship = None
        self.eta = None
        self.nu = None
        self.u_actual = None
        self.current_target_idx = 0
        self.step_count = 0
        self.last_thrust = 0.0
        self.last_rudder = 0.0
        self.prev_distance = 0.0
        self.trail = []  # position history for rendering
        
        # Renderer
        self.renderer = None
        self.screen = None

    def reset(self, seed=None, options=None):
        # Set random seed if provided
        super().reset(seed=seed)
        
        # Reset ship to initial state
        self.ship = shipClarke83(controlSystem='stepInput', L=20.0, B=4.0, T=3.0, Cb=0.7)
        
        # Initialize state vectors
        self.eta = np.array([0, 0, 0, 0, 0, 0], dtype=float)  # position/orientation
        self.nu = np.array([2.0, 0, 0, 0, 0, 0], dtype=float)  # velocities
        self.u_actual = np.array([0.0], dtype=float)  # rudder state
        
        # Generate new random targets
        self._generate_random_targets()
        
        # Reset environment state
        self.current_target_idx = 0
        self.step_count = 0
        self.last_thrust = 0.0
        self.last_rudder = 0.0
        self.trail = [self.eta[:2].copy()]  # initial position
        
        # Calculate initial distance to first target
        self.prev_distance = np.linalg.norm(
            self.targets[self.current_target_idx, :2] - self.eta[:2]
        )
        
        # Reset renderer if exists
        if self.renderer:
            self.renderer.reset()
        
        # Return initial observation
        return self._get_obs(), {}

    def _generate_random_targets(self):
        """Generate 2 random targets within 10-100m radius of origin with random headings"""
        rng = np.random.default_rng()
        
        # Generate positions
        angles = rng.uniform(0, 2 * np.pi, size=2)
        radii = rng.uniform(50, 150, size=2)
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        
        # Generate headings
        headings = rng.uniform(-math.pi, math.pi, size=2)
        
        # Create targets array with headings
        self.targets = np.column_stack((x, y, headings))

    def _get_obs(self):
        """Compute current observation vector"""
        target = self.targets[self.current_target_idx]
        target_pos = target[:2]
        target_heading = target[2]

        # Calculate relative position to current target in body frame
        dx = target_pos[0] - self.eta[0]
        dy = target_pos[1] - self.eta[1]
        psi = self.eta[5]
        
        rel_x = dx * math.cos(psi) + dy * math.sin(psi)
        rel_y = -dx * math.sin(psi) + dy * math.cos(psi)
        
        # Calculate heading error (normalized to [-π, π])
        heading_error = (target_heading - psi) % (2 * math.pi)
        if heading_error > math.pi:
            heading_error -= 2 * math.pi
        
        # Normalized target index
        target_idx_norm = self.current_target_idx / 1.0  # Only 2 targets (0 or 1)
        
        # Normalized control inputs
        norm_thrust = self.last_thrust / self.max_thrust
        norm_rudder = self.last_rudder / self.max_rudder
        
        return np.array([
            self.nu[0],              # surge velocity (u)
            self.nu[1],              # sway velocity (v)
            self.nu[5],              # yaw rate (r)
            psi,                    # yaw angle (psi)
            rel_x,                  # relative x (body frame)
            rel_y,                  # relative y (body frame)
            target_idx_norm,        # normalized target index
            norm_thrust,            # normalized thrust [-1, 1]
            norm_rudder,            # normalized rudder [-1, 1]
            heading_error           # heading error to target (rad)
        ], dtype=np.float32)

    def step(self, action):
        # Convert normalized actions to physical values
        thrust = action[0] * self.max_thrust
        rudder_command = action[1] * self.max_rudder
        u_control = np.array([rudder_command], dtype=float)
        
        # Set ship thrust for this action step
        self.ship.tau_X = thrust
        
        # Store initial state for reward calculation
        initial_target_idx = self.current_target_idx
        initial_target = self.targets[initial_target_idx]
        initial_pos = self.eta.copy()
        initial_psi = self.eta[5]
        
        # Run internal simulation steps
        for _ in range(self.sim_steps_per_action):
            # Update ship dynamics
            self.nu, self.u_actual = self.ship.dynamics(
                self.eta, self.nu, self.u_actual, u_control, self.dt
            )
            
            # Update position and orientation
            psi = self.eta[5]
            u = self.nu[0]
            v = self.nu[1]
            r = self.nu[5]
            
            # Kinematic equations (3DOF)
            self.eta[0] += (u * math.cos(psi) - v * math.sin(psi)) * self.dt
            self.eta[1] += (u * math.sin(psi) + v * math.cos(psi)) * self.dt
            self.eta[5] = (self.eta[5] + r * self.dt) % (2 * math.pi)
        
        # Record position for trail
        self.trail.append(self.eta[:2].copy())
        if len(self.trail) > 500:  # limit trail length
            self.trail.pop(0)
        
        # Calculate reward components
        reward = 0.0
        done = False
        truncated = False
        info = {}
        target_reached = False
        
        current_target = self.targets[self.current_target_idx]
        current_pos = self.eta[:2]
        distance = np.linalg.norm(current_target[:2] - current_pos)
        
        # Distance-based reward
        reward += (self.prev_distance - distance) * 10.0
        self.prev_distance = distance
        
        # Heading-based reward
        current_psi = self.eta[5]
        heading_error = (current_target[2] - current_psi) % (2 * math.pi)
        if heading_error > math.pi:
            heading_error -= 2 * math.pi
        
        # Reward for reducing heading error
        reward -= abs(heading_error) * 0.5
        
        # Control efficiency penalties
        thrust_penalty = 0.001 * (thrust / self.max_thrust)**2
        rudder_penalty = 0.1 * (rudder_command / self.max_rudder)**2
        thrust_smooth = 0.001 * ((thrust - self.last_thrust) / self.max_thrust)**2
        rudder_smooth = 0.1 * ((rudder_command - self.last_rudder) / self.max_rudder)**2
        
        reward -= thrust_penalty + rudder_penalty + thrust_smooth + rudder_smooth
        
        # Update control memory
        self.last_thrust = thrust
        self.last_rudder = rudder_command
        
        # Target reached check (position AND heading)
        if (distance < self.success_radius and 
            abs(heading_error) < self.heading_tolerance):
            reward += 100.0
            self.current_target_idx += 1
            target_reached = True
            
            # Only update if there are more targets
            if self.current_target_idx < len(self.targets):
                next_target = self.targets[self.current_target_idx]
                self.prev_distance = np.linalg.norm(
                    next_target[:2] - self.eta[:2]
                )
                reward += 50.0  # bonus for reaching a target
            
            # Check if all targets completed
            if self.current_target_idx >= len(self.targets):
                done = True
                info['episode_success'] = True
                reward += 500.0  # bonus for completing all targets
                # Reset to last target to prevent index error
                self.current_target_idx = len(self.targets) - 1
        
        # Boundary check
        if (abs(self.eta[0]) > self.world_boundary or 
            abs(self.eta[1]) > self.world_boundary):
            reward -= 100.0
            done = True
            truncated = True
            info['episode_truncated'] = True
        
        # Step limit check
        self.step_count += 1
        if self.step_count >= self.episode_length:
            done = True
            truncated = True
            info['episode_truncated'] = True
        
        return self._get_obs(), reward, done, truncated, info

    def render(self, mode='human'):
        if mode == 'human' or mode == 'rgb_array':
            if self.renderer is None:
                # Initialize renderer with parameters
                self.renderer = ShipRenderer(
                    screen_width=800,
                    screen_height=600,
                    scale=0.5,  # pixels per meter
                    water_color=(100, 100, 255),
                    ship_color=(0, 0, 0),
                    target_color=(255, 0, 0),
                    trail_color=(0, 255, 0),
                    text_color=(0, 0, 0),
                    gauge_color=(200, 200, 200),
                    thrust_color=(255, 0, 0),
                    rudder_color=(0, 0, 255)
                )
            
            # Get current target for highlighting
            current_target = self.targets[self.current_target_idx, :2]
            
            # Render all targets with the current one highlighted
            self.renderer.render(
                eta=self.eta,
                nu=self.nu,
                trail=self.trail,
                targets=self.targets,  # Pass only position data
                current_target=current_target,  # Highlight current target
                step_count=self.step_count,
                max_steps=self.episode_length,
                current_thrust=self.last_thrust,
                current_rudder=self.last_rudder,
                max_tau_X=self.max_thrust,
                max_delta=self.max_rudder
            )
            
            if mode == 'rgb_array':
                # Return RGB array for video recording
                return pygame.surfarray.array3d(self.renderer.screen)
        return None

    def close(self):
        if self.renderer:
            self.renderer.close()
            self.renderer = None