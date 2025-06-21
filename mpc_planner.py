# mpc_planner.py
import numpy as np

def create_refined_grid(n_points, sharpness=1.0):
    """
    Create a grid between 0 and 1 with more refinement near 0 using exponential function
    
    Parameters:
    n_points (int): Total number of points in the grid
    sharpness (float): Controls how concentrated points are near 0 (higher = more concentrated)
    
    Returns:
    np.array: Non-uniform grid points
    """
    # Create uniform grid in [0,1] space
    t = np.linspace(0, 1, n_points)
    
    # Apply exponential transformation to concentrate points near 0
    refined = np.exp(-sharpness * t)
    refined = 1 - refined  # Invert to get concentration at start
    
    # Normalize to [0,1] range
    refined = refined / refined[-1]
    
    # Create symmetric negative and positive parts
    positive_side = -refined[:-1][::-1]
    # Combine and sort
    positive_side += 1
    grid = np.concatenate([[0], positive_side])    
    return grid
class MPCPlanner:
    def __init__(self, dt=0.2, horizon=50, max_speed=4, max_turn_rate=5):
        self.dt = dt
        self.horizon = horizon
        self.max_speed = max_speed
        self.max_turn_rate = max_turn_rate  # deg/s
        self.target_position = None
        self.target_heading = None
        self.paths = []
        self.best_index = None
        self.terminal_window = 20
        self.cost_weight = {"heading": 0.3, "distance": 200, 
                            "v_cmd_frame_diff": 200, "yr_cmd_frame_diff": 100,
                            "v_cmd": 1, "yr_cmd": 1}
        self.terminal_cost_weight_scale = {"heading": 20, "distance": 100}
        self.prev_v_cmd = 0
        self.prev_psi_cmd = 0

    def update_MPC_setup(self, horizon=30, dt=0.2):
        self.horizon = horizon
        self.dt = dt
        self.terminal_window = min(self.terminal_window, horizon)
        
    def set_target(self, target_position, target_heading=None):
        self.target_position = target_position
        self.target_heading = target_heading

    def update(self, current_position, current_heading):
        if not self.target_position or None in current_position:
            return 0, 0

        # Calculate position error in meters
        dx = self.target_position[0] - current_position[0]
        dy = self.target_position[1] - current_position[1]
        distance = np.sqrt(dx**2 + dy**2)

        best_cost = float('inf')
        best_v = 0
        best_psi_dot = 0
        self.paths = []
        costs = []

        v_values = self.max_speed * create_refined_grid(50, sharpness = 3)
        psi_dot_values = np.concatenate([[0], np.linspace(-self.max_turn_rate, self.max_turn_rate, 50)])

        for v in v_values:
            for psi_dot in psi_dot_values:
                path, cost = self.simulate(dx, dy, current_heading, v, psi_dot)
                self.paths.append(path)
                costs.append(cost)
                if cost < best_cost:
                    best_cost = cost
                    best_v = v
                    best_psi_dot = psi_dot
                    self.best_index = len(self.paths) - 1

        print(f"Best cost: {best_cost}, v: {best_v}, psi_dot: {best_psi_dot}")

        return best_v, best_psi_dot

    def simulate(self, dx, dy, psi, v, psi_dot):
        cost = 0
        hdg_cost = 0
        dist_cost = 0
        x_sim = 0
        y_sim = 0
        psi_sim = psi
        path = [(x_sim, y_sim)]  # Store path points
        
        for i in range(self.horizon):
            psi_rad = np.radians(psi_sim)
            x_sim += v * np.sin(psi_rad) * self.dt
            y_sim += v * np.cos(psi_rad) * self.dt
            psi_sim = (psi_sim + psi_dot * self.dt) % 360
            path.append((x_sim, y_sim))
            
            hdg_error_cost, distance_cost = self.stage_cost(i,
                x_sim, y_sim, psi_sim, dx, dy, self.target_heading
            )
            
            hdg_cost += hdg_error_cost
            dist_cost += distance_cost
        
        cost = hdg_cost + dist_cost
        # Add cost for smooth control actuation based on frame diff penalty
        cost += (self.prev_v_cmd - v) ** 2 * self.cost_weight['v_cmd_frame_diff']
        + (self.prev_psi_cmd - psi_dot) ** 2 * self.cost_weight['yr_cmd_frame_diff'] # psidot is in degree
        # Add cost for control actuator effort regulators
        cost += self.cost_weight['v_cmd_frame_diff'] * v ** 2 * self.horizon
        cost += self.cost_weight['yr_cmd_frame_diff'] * psi_dot ** 2 * self.horizon
        return path, cost

    def stage_cost(self, i, x_sim, y_sim, psi_sim, dx, dy, target_heading):
        distance_cost = (dx - x_sim)**2 + (dy - y_sim)**2
        distance_cost *= self.cost_weight["distance"]
        hdg_error_cost = 0
            
        if target_heading is not None:
            hdg_error = (psi_sim - target_heading) % 360
            hdg_error_cost = hdg_error**2 * self.cost_weight["heading"]
        
        # Add terminal cost for position and speed condition
        if i > self.horizon - self.terminal_window:
            distance_cost *= self.terminal_cost_weight_scale["distance"]
            hdg_error_cost *= self.terminal_cost_weight_scale["heading"]
        return hdg_error_cost, distance_cost
