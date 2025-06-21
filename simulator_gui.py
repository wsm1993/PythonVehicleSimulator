import numpy as np
import math
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Arrow, Rectangle
from matplotlib.transforms import Affine2D
from python_vehicle_simulator.vehicles.shipClarke83 import shipClarke83  # Using the provided model
from mpc_planner import MPCPlanner

class ShipSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Ship Motion Simulator")
        self.root.geometry("1200x800")
        
        # Ship parameters (default values)
        self.L = 50.0  # ship length (m)
        self.B = 7.0   # ship beam (m)
        self.T = 5.0   # ship draft (m)
        self.Cb = 0.7  # block coefficient
        
        # Initialize ship model
        self.ship = shipClarke83(
            controlSystem="stepInput",
            r=0,
            L=self.L,
            B=self.B,
            T=self.T,
            Cb=self.Cb,
            V_current=0,
            beta_current=0,
            tau_X=1e5
        )
        
        # Simulation state
        self.simulating = False
        self.time = 0.0
        self.sample_time = 0.1  # seconds
        self.eta = np.array([0, 0, 0, 0, 0, 0], float)  # position/orientation
        self.nu = np.array([0, 0, 0, 0, 0, 0], float)   # velocity
        self.path = []  # stores ship's path
        
        # Current settings
        self.current_speed = 0.0
        self.current_direction = 0.0  # degrees
        
        # Control settings
        self.rudder_angle = 0.0  # degrees
        self.surge_force = 100000  # Newtons

        # MPC controller
        self.mpc_planner = MPCPlanner(dt=0.2, horizon=30)
        self.mpc_target = None
        self.mpc_active = False
        self.mpc_update_interval = 1.0  # seconds
        self.last_mpc_update = 0
        
        # Simulation speed
        self.sim_speed = 1  # Simulation speed multiplier (1x to 100x)

        # Create GUI
        self.create_gui()
        
        # Start simulation loop
        self.update_simulation()

    def mps_to_knots(self, mps):
        """Convert meters per second to knots"""
        return mps * 1.94384
    
    def create_gui(self):
        # Create main frames
        control_frame = ttk.LabelFrame(self.root, text="Controls", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create plot
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Ship Motion")
        self.ax.set_xlabel("East (m)")
        self.ax.set_ylabel("North (m)")
        self.ax.grid(True)
        self.ax.set_aspect('equal')
        
        # Initialize ship drawing
        self.ship_patch = Polygon([[0, 0], [0, 0], [0, 0]], closed=True, fc='blue', ec='black')
        self.ax.add_patch(self.ship_patch)
        
        # Initialize rudder drawing
        self.rudder_patch = Rectangle((0, 0), 0, 0, fc='red', ec='black')
        self.ax.add_patch(self.rudder_patch)
        
        # Initialize path line
        self.path_line, = self.ax.plot([], [], 'b-', alpha=0.5)
        
        # Initialize current arrow
        self.current_arrow = Arrow(0, 0, 0, 0, width=2, fc='green', ec='green', alpha=0.7)
        self.ax.add_patch(self.current_arrow)
        
        # Initialize velocity vector
        self.velocity_arrow = Arrow(0, 0, 0, 0, width=1, fc='red', ec='red')
        self.ax.add_patch(self.velocity_arrow)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Control widgets

        # Surge force section
        ttk.Label(control_frame, text="Surge Force (N):").pack(anchor=tk.W, pady=(5, 0))
        self.surge_slider = ttk.Scale(control_frame, from_=-200000, to=200000, 
                                    orient=tk.HORIZONTAL,
                                    command=self.update_surge_force)
        self.surge_slider.pack(fill=tk.X, pady=(0, 10))
        self.surge_label = ttk.Label(control_frame, text=f"{self.surge_force/1000:.1f} kN")
        self.surge_label.pack(anchor=tk.W)
        self.surge_slider.set(self.surge_force)
        
        # Rudder angle section
        ttk.Label(control_frame, text="Rudder Angle (deg):").pack(anchor=tk.W, pady=(5, 0))
        self.rudder_slider = ttk.Scale(control_frame, from_=-30, to=30, 
                                    orient=tk.HORIZONTAL,
                                    command=self.update_rudder_angle)
        self.rudder_slider.pack(fill=tk.X, pady=(0, 10))
        self.rudder_label = ttk.Label(control_frame, text=f"{self.rudder_angle:.1f}°")
        self.rudder_label.pack(anchor=tk.W)
        self.rudder_slider.set(self.rudder_angle)
        
        # Current speed section
        ttk.Label(control_frame, text="Speed (m/s):").pack(anchor=tk.W)
        self.current_speed_slider = ttk.Scale(control_frame, from_=0, to=3, 
                                            orient=tk.HORIZONTAL,
                                            command=self.update_current_speed)
        self.current_speed_slider.pack(fill=tk.X)
        self.current_speed_label = ttk.Label(control_frame, text="0.0 m/s (0.0 kt)")
        self.current_speed_label.pack(anchor=tk.W)
        self.current_speed_slider.set(self.current_speed)

        # Current direction section
        ttk.Label(control_frame, text="Direction (deg):").pack(anchor=tk.W, pady=(5, 0))
        self.current_dir_slider = ttk.Scale(control_frame, from_=0, to=360, 
                                        orient=tk.HORIZONTAL,
                                        command=self.update_current_direction)
        self.current_dir_slider.pack(fill=tk.X)
        self.current_dir_label = ttk.Label(control_frame, text=f"{self.current_direction:.1f}°")
        self.current_dir_label.pack(anchor=tk.W)
        self.current_dir_slider.set(self.current_direction)
                
        # Simulation controls
        sim_frame = ttk.Frame(control_frame)
        sim_frame.pack(fill=tk.X, pady=10)
        
        self.start_button = ttk.Button(sim_frame, text="Start", command=self.start_simulation)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(sim_frame, text="Stop", command=self.stop_simulation)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.reset_button = ttk.Button(sim_frame, text="Reset", command=self.reset_simulation)
        self.reset_button.pack(side=tk.LEFT, padx=5)

        # Simulation speed control
        ttk.Label(control_frame, text="Simulation Speed:").pack(anchor=tk.W, pady=(10, 0))
        self.speed_slider = ttk.Scale(control_frame, from_=1, to=100, 
                                    orient=tk.HORIZONTAL,
                                    command=self.update_sim_speed)
        self.speed_slider.pack(fill=tk.X, pady=(0, 5))
 
        
        # Status display
        status_frame = ttk.LabelFrame(control_frame, text="Ship Status", padding=10)
        status_frame.pack(fill=tk.X, pady=10)
        
        self.position_label = ttk.Label(status_frame, text="Position: (0.0, 0.0)")
        self.position_label.pack(anchor=tk.W)
        
        self.heading_label = ttk.Label(status_frame, text="Heading: 0.0°")
        self.heading_label.pack(anchor=tk.W)
        
        self.speed_label = ttk.Label(status_frame, text="Speed: 0.0 m/s")
        self.speed_label.pack(anchor=tk.W)

        self.sim_speed_label = ttk.Label(control_frame, text=f"Sim Speed: {self.sim_speed}x")
        self.sim_speed_label.pack(anchor=tk.W)

        self.rudder_status_label = ttk.Label(status_frame, text="Rudder: 0.0°")
        self.rudder_status_label.pack(anchor=tk.W)

        # Add MPC controls
        mpc_frame = ttk.LabelFrame(control_frame, text="MPC Controller", padding=10)
        mpc_frame.pack(fill=tk.X, pady=10)
        
        # Target position
        ttk.Label(mpc_frame, text="Target X:").pack(anchor=tk.W)
        self.target_x_entry = ttk.Entry(mpc_frame)
        self.target_x_entry.pack(fill=tk.X)
        
        ttk.Label(mpc_frame, text="Target Y:").pack(anchor=tk.W)
        self.target_y_entry = ttk.Entry(mpc_frame)
        self.target_y_entry.pack(fill=tk.X)
        
        ttk.Label(mpc_frame, text="Target Heading (deg):").pack(anchor=tk.W)
        self.target_heading_entry = ttk.Entry(mpc_frame)
        self.target_heading_entry.pack(fill=tk.X)
        
        # MPC buttons
        btn_frame = ttk.Frame(mpc_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        self.set_target_btn = ttk.Button(
            btn_frame, text="Set Target", command=self.set_mpc_target)
        self.set_target_btn.pack(side=tk.LEFT, padx=2)
        
        self.start_mpc_btn = ttk.Button(
            btn_frame, text="Start MPC", command=self.start_mpc)
        self.start_mpc_btn.pack(side=tk.LEFT, padx=2)
        
        self.stop_mpc_btn = ttk.Button(
            btn_frame, text="Stop MPC", command=self.stop_mpc)
        self.stop_mpc_btn.pack(side=tk.LEFT, padx=2)
        
        # MPC status
        self.mpc_status = ttk.Label(mpc_frame, text="MPC: Inactive")
        self.mpc_status.pack(anchor=tk.W)
        
        # Set initial plot limits
        self.ax.set_xlim(-100, 100)
        self.ax.set_ylim(-100, 100)
    
    def update_surge_force(self, value):
        self.surge_force = float(value)
        self.surge_label.config(text=f"{self.surge_force/1000:.1f} kN")
    
    def update_rudder_angle(self, value):
        self.rudder_angle = float(value)
        self.rudder_label.config(text=f"{self.rudder_angle:.1f}°")
    
    def update_current_speed(self, value):
        self.current_speed = float(value)
        knots = self.mps_to_knots(self.current_speed)
        self.current_speed_label.config(
            text=f"{self.current_speed:.1f} m/s ({knots:.1f} kt)"
        )
    
    
    def update_current_direction(self, value):
        self.current_direction = float(value)
        self.current_dir_label.config(text=f"{self.current_direction:.1f}°")
    
    def update_sim_speed(self, value):
        self.sim_speed = max(1, min(100, float(value)))
        self.sim_speed_label.config(text=f"Sim Speed: {self.sim_speed:.0f}x")

    def start_simulation(self):
        self.simulating = True
    
    def stop_simulation(self):
        self.simulating = False
    
    def reset_simulation(self):
        self.simulating = False
        self.time = 0.0
        self.eta = np.array([0, 0, 0, 0, 0, 0], float)
        self.nu = np.array([0, 0, 0, 0, 0, 0], float)
        self.path = []
        self.update_plot()
    
    def update_simulation(self):
        if self.simulating:
            # Calculate effective time step based on speed multiplier
            effective_sample_time = self.sample_time * self.sim_speed
            steps = max(1, int(self.sim_speed))
            step_time = effective_sample_time / steps

            # MPC control logic
            if self.mpc_active and (self.time - self.last_mpc_update >= self.mpc_update_interval):
                self.last_mpc_update = self.time
                
                # Get current state
                position = (self.eta[0], self.eta[1])
                heading_deg = math.degrees(self.eta[5]) % 360
                
                # Compute MPC commands
                v_command, psi_dot_command = self.mpc_planner.update(
                    position, heading_deg)
                
                # Convert to ship controls
                # (Simple proportional controllers - can be improved)
                surge_error = v_command - self.nu[0]
                self.surge_force = 100000 * max(0, min(1, surge_error))
                
                # Convert turn rate to rudder angle
                self.rudder_angle = psi_dot_command * 2.0  # scaling factor
                
                # Update sliders
                self.rudder_slider.set(self.rudder_angle)
                self.surge_slider.set(self.surge_force)
            
            for _ in range(steps):
                # Update time
                self.time += step_time
                
                # Update current settings
                self.ship.V_c = self.current_speed
                self.ship.beta_c = math.radians(self.current_direction)
                self.ship.tau_X = self.surge_force
                
                # Convert rudder angle to radians
                rudder_rad = math.radians(self.rudder_angle)
                u_control = np.array([rudder_rad], float)
                
                # Update ship dynamics
                self.nu, self.ship.u_actual = self.ship.dynamics(
                    self.eta, self.nu, self.ship.u_actual, u_control, step_time
                )
                
                # Update position
                psi = self.eta[5]
                x_dot = math.cos(psi) * self.nu[0] - math.sin(psi) * self.nu[1]
                y_dot = math.sin(psi) * self.nu[0] + math.cos(psi) * self.nu[1]
                psi_dot = self.nu[5]
                
                self.eta[0] += x_dot * step_time
                self.eta[1] += y_dot * step_time
                self.eta[5] += psi_dot * step_time
                
                # Record path
                self.path.append((self.eta[0], self.eta[1]))
                if len(self.path) > 5000:  # Limit path length
                    self.path.pop(0)
                
            # Update status display
            self.update_status()
        
        # Update plot
        self.update_plot()
        
        # Schedule next update, after() expects milliseconds
        self.root.after(int(self.sample_time * 1000), self.update_simulation)
    
    def update_status(self):
        # Position
        self.position_label.config(text=f"Position: ({self.eta[0]:.1f}, {self.eta[1]:.1f})")
        
        # Heading (convert to degrees)
        heading_deg = math.degrees(self.eta[5]) % 360
        self.heading_label.config(text=f"Heading: {heading_deg:.1f}°")
        
        # Speed (magnitude of surge and sway)
        speed = math.sqrt(self.nu[0]**2 + self.nu[1]**2)
        knots = self.mps_to_knots(speed)
        self.speed_label.config(text=f"Speed: {speed:.1f} m/s ({knots:.1f} kt)")
        
        # Rudder angle (convert to degrees)
        rudder_deg = math.degrees(self.ship.u_actual[0])
        self.rudder_status_label.config(text=f"Rudder: {rudder_deg:.1f}°")
    
    def update_plot(self):
        # Clear previous drawings
        self.ax.clear()
        self.ax.set_title("Ship Motion")
        self.ax.set_xlabel("North (m)")
        self.ax.set_ylabel("East (m)")
        self.ax.grid(True)
        self.ax.set_aspect('equal')
        
        # Update plot limits based on ship position
        margin = 100
        x_min = min([p[0] for p in self.path]) - margin if self.path else -margin
        x_max = max([p[0] for p in self.path]) + margin if self.path else margin
        y_min = min([p[1] for p in self.path]) - margin if self.path else -margin
        y_max = max([p[1] for p in self.path]) + margin if self.path else margin
        
        # Ensure reasonable limits if ship hasn't moved much
        if x_max - x_min < 100:
            center_x = (x_min + x_max) / 2
            x_min = center_x - 100
            x_max = center_x + 100
        
        if y_max - y_min < 100:
            center_y = (y_min + y_max) / 2
            y_min = center_y - 100
            y_max = center_y + 100
        
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        
        # Draw path
        if self.path:
            path_x, path_y = zip(*self.path)
            self.ax.plot(path_x, path_y, 'b-', alpha=0.5)
        
        # Draw ship
        ship_length = self.L
        ship_heading = self.eta[5]
        ship_x = self.eta[0]
        ship_y = self.eta[1]
        
        # Create ship triangle with proper rotation
        # Define points in local ship coordinate system
        front = (ship_length/2, 0)
        back_left = (-ship_length/2, ship_length/4)
        back_right = (-ship_length/2, -ship_length/4)
        
        # Rotate points using rotation matrix
        def rotate_point(x, y, angle):
            x_rot = x * math.cos(angle) - y * math.sin(angle)
            y_rot = x * math.sin(angle) + y * math.cos(angle)
            return x_rot, y_rot
        
        # Rotate ship points
        front_rot = rotate_point(front[0], front[1], ship_heading)
        back_left_rot = rotate_point(back_left[0], back_left[1], ship_heading)
        back_right_rot = rotate_point(back_right[0], back_right[1], ship_heading)
        
        # Translate to global position
        front_global = (ship_x + front_rot[0], ship_y + front_rot[1])
        back_left_global = (ship_x + back_left_rot[0], ship_y + back_left_rot[1])
        back_right_global = (ship_x + back_right_rot[0], ship_y + back_right_rot[1])
        
        ship_poly = Polygon([front_global, back_left_global, back_right_global], 
                           closed=True, fc='blue', ec='black')
        self.ax.add_patch(ship_poly)
        
        # Draw rudder
        rudder_angle = -self.ship.u_actual[0]  # actual rudder angle in radians
        rudder_length = ship_length / 8
        rudder_width = ship_length / 20
        
        # Rudder position (at stern)
        rudder_pos_local = (-ship_length/2 - rudder_length/2, 0)
        rudder_pos_rot = rotate_point(rudder_pos_local[0], rudder_pos_local[1], ship_heading)
        rudder_pos_global = (ship_x + rudder_pos_rot[0], ship_y + rudder_pos_rot[1])
        
        # Create rudder rectangle with rotation
        rudder_rect = Rectangle(
            (rudder_pos_global[0] - rudder_length/2, rudder_pos_global[1] - rudder_width/2),
            rudder_length, rudder_width,
            angle=math.degrees(ship_heading + rudder_angle),
            rotation_point='center',
            fc='red', ec='black'
        )
        self.ax.add_patch(rudder_rect)
        
        # Draw current vector
        current_x = ship_x + 20
        current_y = ship_y + 20
        current_angle = math.radians(self.current_direction)
        current_dx = self.current_speed * 10 * math.cos(current_angle)
        current_dy = self.current_speed * 10 * math.sin(current_angle)
        
        if self.current_speed > 0:
            current_arrow = Arrow(current_x, current_y, current_dx, current_dy, 
                                 width=1, fc='green', ec='green', alpha=0.7)
            self.ax.add_patch(current_arrow)
            current_knots = self.mps_to_knots(self.current_speed)
            self.ax.text(current_x + current_dx/2, current_y + current_dy/2, 
                        f"Current: {self.current_speed:.1f} m/s\n({current_knots:.1f} kt)", 
                        fontsize=9, color='green', alpha=0.7)
        
        # Draw velocity vector
        velocity_scale = 5
        vel_x = ship_x
        vel_y = ship_y
        vel_dx = self.nu[0] * velocity_scale * math.cos(ship_heading) - self.nu[1] * velocity_scale * math.sin(ship_heading)
        vel_dy = self.nu[0] * velocity_scale * math.sin(ship_heading) + self.nu[1] * velocity_scale * math.cos(ship_heading)
        
        velocity_arrow = Arrow(vel_x, vel_y, vel_dx, vel_dy, 
                              width=0.5, fc='red', ec='red')
        self.ax.add_patch(velocity_arrow)
        
        # Redraw canvas
        self.canvas.draw()

    def set_mpc_target(self):
        try:
            x = float(self.target_x_entry.get())
            y = float(self.target_y_entry.get())
            heading = self.target_heading_entry.get()
            
            self.mpc_target = (x, y)
            self.mpc_planner.set_target((x, y), float(heading) if heading else None)
            
            # Draw target on plot
            self.ax.plot(x, y, 'ro', markersize=8)
            if heading:
                self.ax.text(x, y, f"Target\n{heading}°", 
                            ha='center', va='bottom', color='red')
            self.canvas.draw()
            
            self.mpc_status.config(text="MPC: Target Set")
        except ValueError:
            self.mpc_status.config(text="MPC: Invalid Target")

    def start_mpc(self):
        if self.mpc_target:
            self.mpc_active = True
            self.mpc_status.config(text="MPC: Active")
        else:
            self.mpc_status.config(text="MPC: Set Target First")

    def stop_mpc(self):
        self.mpc_active = False
        self.mpc_status.config(text="MPC: Stopped")


# Create and run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = ShipSimulator(root)
    root.mainloop()