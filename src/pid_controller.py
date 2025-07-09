class PIDController:
    def __init__(self, kp, ki, kd, min_output, max_output):
        """
        Initialize PID controller with gains and output limits.
        
        Args:
            kp (float): Proportional gain
            ki (float): Integral gain
            kd (float): Derivative gain
            min_output (float): Minimum output value
            max_output (float): Maximum output value
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_output = min_output
        self.max_output = max_output
        
        self.prev_error = 0
        self.integral = 0
        self.last_time = None
    
    def update(self, error, dt):
        """
        Update the PID controller with new error value and time step.
        
        Args:
            error (float): Current error (setpoint - process_variable)
            dt (float): Time step since last update
            
        Returns:
            float: Control output
        """
        # Proportional term
        proportional = self.kp * error
        
        # Integral term (with anti-windup)
        self.integral += error * dt
        integral = self.ki * self.integral
        
        # Derivative term
        derivative = 0
        if dt > 0:  # Avoid division by zero
            derivative = self.kd * (error - self.prev_error) / dt
        
        # Calculate output
        output = proportional + integral + derivative
        
        # Apply output limits
        output = max(self.min_output, min(self.max_output, output))
        
        # Store error for next update
        self.prev_error = error
        
        return output
    
    def reset(self):
        """Reset the controller's internal state"""
        self.prev_error = 0
        self.integral = 0
        self.last_time = None