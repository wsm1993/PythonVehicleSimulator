import pygame
import numpy as np

class ShipRenderer:
    def __init__(self, screen_width, screen_height, scale,
                 water_color, ship_color, target_color, trail_color,
                 text_color, gauge_color, thrust_color, rudder_color):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.scale = scale
        self.water_color = water_color
        self.ship_color = ship_color
        self.target_color = target_color
        self.trail_color = trail_color
        self.text_color = text_color
        self.gauge_color = gauge_color
        self.thrust_color = thrust_color
        self.rudder_color = rudder_color

        self.screen = None
        self.clock = None
        self.font = None
        self.small_font = None

    def reset(self):
        self._render_init()

    def _render_init(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Ship Navigation Environment")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)
        self.small_font = pygame.font.SysFont(None, 20)

    def render(self, eta, nu, trail, target, step_count, max_steps,
               current_action, current_thrust, current_rudder, max_tau_X, max_delta):
        if self.screen is None:
            self._render_init()

        # Clear screen
        self.screen.fill(self.water_color)

        # Calculate center offset - world coordinates origin at screen center
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2

        # Draw grid
        self._draw_grid(center_x, center_y)

        # Draw ship trail
        if len(trail) > 1:
            pygame.draw.lines(
                self.screen,
                self.trail_color,
                False,
                [(center_x + x * self.scale, center_y - y * self.scale) for x, y in trail],
                2
            )

        # Draw target
        target_x = center_x + target[0] * self.scale
        target_y = center_y - target[1] * self.scale
        pygame.draw.circle(self.screen, self.target_color, (int(target_x), int(target_y)), 10)

        # Draw target heading indicator
        target_heading = target[2]
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
        ship_x = center_x + eta[0] * self.scale
        ship_y = center_y - eta[1] * self.scale
        heading = eta[5]  # Heading in radians

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
        self._draw_info_panel(
            eta, nu, step_count, max_steps,
            current_action, current_thrust, current_rudder, max_tau_X, max_delta
        )

        # Draw control gauges
        self._draw_control_gauges(current_thrust, max_tau_X, current_rudder, max_delta)

        # Update display
        pygame.display.flip()

        # Maintain frame rate
        self.clock.tick(60)

    def _draw_grid(self, center_x, center_y):
        grid_size = 50 * self.scale
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
        pygame.draw.circle(self.screen, (0, 0, 0), (center_x, center_y), 5)

    def _draw_info_panel(self, eta, nu, step_count, max_steps,
                        current_action, current_thrust, current_rudder, max_tau_X, max_delta):
        status = "Status: " + ("SUCCESS" if step_count < max_steps else "RUNNING")
        position = f"Position: ({eta[0]:.1f}, {eta[1]:.1f}) m"
        heading = f"Heading: {np.rad2deg(eta[5]):.1f}°"
        speed = f"Speed: {nu[0]:.1f} m/s"
        steps = f"Step: {step_count}/{max_steps}"

        thrust_level = ['low', 'med', 'high'][current_action // 3] if current_action >= 0 else "N/A"
        rudder_dir = ['left', 'center', 'right'][current_action % 3] if current_action >= 0 else "N/A"
        action_info = f"Action: {current_action} ({thrust_level} thrust, {rudder_dir} rudder)"
        thrust_info = f"Surge Force: {current_thrust:.1f} N"
        rudder_info = f"Rudder Angle: {np.rad2deg(current_rudder):.1f}°"

        texts = [
            self.font.render(status, True, self.text_color),
            self.font.render(position, True, self.text_color),
            self.font.render(heading, True, self.text_color),
            self.font.render(speed, True, self.text_color),
            self.font.render(steps, True, self.text_color),
            self.font.render(action_info, True, self.text_color),
            self.font.render(thrust_info, True, self.text_color),
            self.font.render(rudder_info, True, self.text_color)
        ]

        pygame.draw.rect(self.screen, (240, 240, 240), (10, 10, 400, 180))
        for i, text in enumerate(texts):
            self.screen.blit(text, (20, 20 + i * 24))

    def _draw_control_gauges(self, current_thrust, max_tau_X, current_rudder, max_delta):
        gauge_width = 150
        gauge_height = 20
        gauge_margin = 30

        thrust_x = self.screen_width - gauge_width - 20
        thrust_y = 20
        pygame.draw.rect(self.screen, self.gauge_color,
                         (thrust_x, thrust_y, gauge_width, gauge_height))

        thrust_percent = abs(current_thrust) / max_tau_X
        thrust_fill_width = int(thrust_percent * gauge_width)
        pygame.draw.rect(self.screen, self.thrust_color,
                         (thrust_x, thrust_y, thrust_fill_width, gauge_height))

        thrust_label = self.small_font.render("Thrust", True, self.text_color)
        self.screen.blit(thrust_label, (thrust_x, thrust_y - 20))
        thrust_value = self.small_font.render(f"{current_thrust:.1f} N", True, self.text_color)
        self.screen.blit(thrust_value, (thrust_x + gauge_width + 10, thrust_y))

        rudder_x = self.screen_width - gauge_width - 20
        rudder_y = thrust_y + gauge_height + gauge_margin
        pygame.draw.rect(self.screen, self.gauge_color,
                         (rudder_x, rudder_y, gauge_width, gauge_height))

        pygame.draw.line(self.screen, (200, 200, 200),
                         (rudder_x + gauge_width // 2, rudder_y),
                         (rudder_x + gauge_width // 2, rudder_y + gauge_height), 2)

        rudder_percent = current_rudder / max_delta
        rudder_pos = int(gauge_width // 2 + rudder_percent * (gauge_width // 2))
        rudder_pos = max(0, min(gauge_width, rudder_pos))
        pygame.draw.rect(self.screen, self.rudder_color,
                         (rudder_x + rudder_pos - 2, rudder_y, 4, gauge_height))

        rudder_label = self.small_font.render("Rudder", True, self.text_color)
        self.screen.blit(rudder_label, (rudder_x, rudder_y - 20))
        rudder_value = self.small_font.render(f"{np.rad2deg(current_rudder):.1f}°", True, self.text_color)
        self.screen.blit(rudder_value, (rudder_x + gauge_width + 10, rudder_y))

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None