import pygame
import pymunk
import pymunk.pygame_util
import math
import streamlit as st
from PIL import Image
import numpy as np
from collections import deque

WIDTH, HEIGHT = 800, 600

class CarSimulation:
    def __init__(self):
        pygame.init()
        self.screen = pygame.Surface((WIDTH, HEIGHT))
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        
        # Game state
        self.start_pos = (100, 300)
        self.goal_pos = (700, 300)
        self.goal_radius = 30
        self.total_reward = 0
        self.episode_complete = False
        self.path_history = deque(maxlen=100)  # Track path for visualization
        
        # Path planning
        self.current_target = pymunk.Vec2d(*self.goal_pos)
        self.path = []
        self.path_index = 0
        self.replan_time = 0
        
        # Create elements
        self.car_body, self.car_shape, self.car_w, self.car_h = self.create_car(self.start_pos)
        self.create_track()
        
        # Precompute navigation grid
        self.nav_grid = self.create_navigation_grid()
        self.plan_path(self.start_pos)

    def create_car(self, position):
        mass = 1
        width, height = 40, 20
        moment = pymunk.moment_for_box(mass, (width, height))
        body = pymunk.Body(mass, moment)
        body.position = position
        shape = pymunk.Poly.create_box(body, (width, height))
        shape.elasticity = 0.4
        shape.friction = 0.7
        shape.color = (0, 180, 255, 255)
        shape.collision_type = 1
        self.space.add(body, shape)
        
        # Add collision handler
        handler = self.space.add_collision_handler(1, 2)
        handler.begin = self.handle_collision
        return body, shape, width, height
    
    def handle_collision(self, arbiter, space, data):
        # Trigger path replanning when collision occurs
        current_time = pygame.time.get_ticks()
        if current_time - self.replan_time > 1000:  # Throttle replanning
            self.plan_path((self.car_body.position.x, self.car_body.position.y))
            self.replan_time = current_time
        return True
    
    def create_track(self):
        # Outer boundaries (thick walls)
        thickness = 20
        static_lines = [
            pymunk.Segment(self.space.static_body, (50, 50), (750, 50), thickness),
            pymunk.Segment(self.space.static_body, (50, 550), (750, 550), thickness),
            pymunk.Segment(self.space.static_body, (50, 50), (50, 550), thickness),
            pymunk.Segment(self.space.static_body, (750, 50), (750, 550), thickness),
        ]
        
        # Inner obstacles
        self.obstacles = [
            pymunk.Segment(self.space.static_body, (300, 150), (300, 450), 10),
            pymunk.Segment(self.space.static_body, (500, 100), (500, 500), 10),
        ]
        
        for line in static_lines + self.obstacles:
            line.elasticity = 0.8
            line.friction = 1.0
            line.color = (70, 70, 70)
            line.collision_type = 2
        self.space.add(*static_lines, *self.obstacles)
    
    def create_navigation_grid(self):
        """Create a simple grid representation of the space for path planning"""
        grid_size = 20
        grid_width = WIDTH // grid_size
        grid_height = HEIGHT // grid_size
        grid = np.zeros((grid_height, grid_width), dtype=bool)
        
        # Mark obstacles in the grid
        for obstacle in self.obstacles:
            a, b = obstacle.a, obstacle.b
            x1, y1 = int(a.x/grid_size), int(a.y/grid_size)
            x2, y2 = int(b.x/grid_size), int(b.y/grid_size)
            
            # Draw line in grid
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            x, y = x1, y1
            sx = -1 if x1 > x2 else 1
            sy = -1 if y1 > y2 else 1
            
            if dx > dy:
                err = dx / 2.0
                while x != x2:
                    if 0 <= y < grid_height and 0 <= x < grid_width:
                        grid[y, x] = True
                    err -= dy
                    if err < 0:
                        y += sy
                        err += dx
                    x += sx
            else:
                err = dy / 2.0
                while y != y2:
                    if 0 <= y < grid_height and 0 <= x < grid_width:
                        grid[y, x] = True
                    err -= dx
                    if err < 0:
                        x += sx
                        err += dy
                    y += sy
            
            if 0 <= y2 < grid_height and 0 <= x2 < grid_width:
                grid[y2, x2] = True
        
        # Add boundary walls
        grid[:2, :] = True  # Top wall
        grid[-2:, :] = True  # Bottom wall
        grid[:, :2] = True  # Left wall
        grid[:, -2:] = True  # Right wall
        
        return grid
    
    def plan_path(self, start_pos):
        """A* path planning algorithm to find optimal path"""
        grid_size = 20
        start = (int(start_pos[0]/grid_size), int(start_pos[1]/grid_size))
        goal = (int(self.goal_pos[0]/grid_size), int(self.goal_pos[1]/grid_size))
        
        # Check if start or goal is in obstacle
        if (self.nav_grid[start[1], start[0]] or 
            self.nav_grid[goal[1], goal[0]]):
            return
        
        # A* algorithm implementation
        open_set = {start}
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        while open_set:
            current = min(open_set, key=lambda pos: f_score.get(pos, float('inf')))
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                
                # Convert grid coordinates to world coordinates
                self.path = [(x * grid_size + grid_size//2, 
                            y * grid_size + grid_size//2) for (x, y) in path]
                self.path_index = 0
                return
            
            open_set.remove(current)
            
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), 
                          (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Check bounds and obstacles
                if (neighbor[0] < 0 or neighbor[0] >= self.nav_grid.shape[1] or
                    neighbor[1] < 0 or neighbor[1] >= self.nav_grid.shape[0] or
                    self.nav_grid[neighbor[1], neighbor[0]]):
                    continue
                
                # Diagonal movement cost more
                tentative_g_score = g_score[current] + (1.4 if dx and dy else 1)
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    if neighbor not in open_set:
                        open_set.add(neighbor)
        
        # If no path found, try to go straight to goal
        self.path = [self.goal_pos]
        self.path_index = 0
    
    def heuristic(self, a, b):
        """Euclidean distance heuristic for A*"""
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def get_sensor_data(self):
        rays = [
            self.cast_ray(0),        # Front
            self.cast_ray(math.pi/6),  # Front-left
            self.cast_ray(-math.pi/6), # Front-right
            self.cast_ray(math.pi/2),  # Left
            self.cast_ray(-math.pi/2), # Right
        ]
        return [bool(r[0]) for r in rays], rays
    
    def cast_ray(self, angle_offset, length=150):
        angle = self.car_body.angle + angle_offset
        start = self.car_body.position
        end = (start.x + length * math.cos(angle), start.y + length * math.sin(angle))
        ray = self.space.segment_query_first(start, end, 1, pymunk.ShapeFilter())
        return ray, end
    
    def calculate_reward(self):
        reward = 0
        distance = math.dist((self.car_body.position.x, self.car_body.position.y), self.goal_pos)
        
        if distance < self.goal_radius and not self.episode_complete:
            reward += 100
            self.total_reward += 100
            self.episode_complete = True
            return reward, True
        
        # Reward for making progress toward goal
        if self.path:
            path_progress = self.path_index / len(self.path)
            reward += path_progress * 0.1
        
        reward -= 0.05  # Small time penalty
        self.total_reward += reward
        return reward, False
    
    def autonomous_drive(self):
        if not self.path or self.path_index >= len(self.path):
            self.plan_path((self.car_body.position.x, self.car_body.position.y))
            return
        
        # Get current target from path
        self.current_target = pymunk.Vec2d(*self.path[self.path_index])
        
        # Check if we've reached the current waypoint
        if (self.car_body.position - self.current_target).length < 30:
            self.path_index += 1
            if self.path_index >= len(self.path):
                return
        
        # Calculate direction to current target
        target_vec = self.current_target - self.car_body.position
        target_angle = target_vec.angle
        angle_diff = (target_angle - self.car_body.angle + math.pi) % (2*math.pi) - math.pi
        
        # Apply forces
        force = 300
        fx = force * math.cos(self.car_body.angle)
        fy = force * math.sin(self.car_body.angle)
        self.car_body.apply_force_at_local_point((fx, fy), (0, 0))
        
        # Steering control with PID-like behavior
        steering_gain = 0.8
        damping = 0.3
        angular_velocity = self.car_body.angular_velocity
        steering = angle_diff * steering_gain - angular_velocity * damping
        max_steering = 0.3
        self.car_body.angle += np.clip(steering, -max_steering, max_steering)
        
        # Record path for visualization (using tuple instead of copy)
        self.path_history.append((self.car_body.position.x, self.car_body.position.y))
    
    def draw(self):
        self.screen.fill((30, 30, 40))
        
        # Draw goal
        pygame.draw.circle(self.screen, (100, 255, 100), self.goal_pos, self.goal_radius)
        pygame.draw.circle(self.screen, (0, 255, 0), self.goal_pos, self.goal_radius-5)
        
        # Draw navigation grid (for debugging)
        # self.draw_nav_grid()
        
        # Draw path
        if len(self.path) > 1:
            pygame.draw.lines(self.screen, (255, 0, 255), False, self.path, 2)
        
        # Draw path history
        if len(self.path_history) > 1:
            pygame.draw.lines(self.screen, (0, 200, 255), False, list(self.path_history), 2)
        
        # Draw physics
        self.space.debug_draw(self.draw_options)
        
        # Draw car
        self.draw_car()
        
        # Draw current target
        if self.path and self.path_index < len(self.path):
            target_pos = (int(self.current_target.x), int(self.current_target.y))
            pygame.draw.circle(self.screen, (255, 255, 0), target_pos, 8)
        
        # Draw UI elements
        font = pygame.font.SysFont('Arial', 20)
        reward_text = font.render(f"Total Reward: {self.total_reward:.1f}", True, (255, 255, 255))
        self.screen.blit(reward_text, (10, 10))
        
        if self.episode_complete:
            complete_text = font.render("GOAL REACHED!", True, (0, 255, 0))
            self.screen.blit(complete_text, (WIDTH//2 - 80, 30))
        
        path_text = font.render(f"Path Length: {len(self.path)}", True, (255, 255, 255))
        self.screen.blit(path_text, (10, 40))
    
    def draw_nav_grid(self):
        """Debug function to visualize the navigation grid"""
        grid_size = 20
        for y in range(self.nav_grid.shape[0]):
            for x in range(self.nav_grid.shape[1]):
                if self.nav_grid[y, x]:
                    rect = pygame.Rect(x*grid_size, y*grid_size, grid_size, grid_size)
                    pygame.draw.rect(self.screen, (50, 50, 50, 100), rect)
    
    def draw_car(self):
        angle = self.car_body.angle
        pos = self.car_body.position
        car_surface = pygame.Surface((self.car_w, self.car_h), pygame.SRCALPHA)
        pygame.draw.rect(car_surface, (0, 180, 255), (0, 0, self.car_w, self.car_h), border_radius=3)
        pygame.draw.rect(car_surface, (255, 255, 0), (self.car_w-10, 5, 5, self.car_h-10))
        rotated_surface = pygame.transform.rotate(car_surface, -math.degrees(angle))
        rect = rotated_surface.get_rect(center=(int(pos.x), int(pos.y)))
        self.screen.blit(rotated_surface, rect.topleft)
    
    def reset(self):
        self.space.remove(self.car_body, self.car_shape)
        self.car_body, self.car_shape, self.car_w, self.car_h = self.create_car(self.start_pos)
        self.total_reward = 0
        self.episode_complete = False
        self.path_history.clear()
        self.plan_path(self.start_pos)
    
    def step(self):
        if self.episode_complete:
            return True
        self.autonomous_drive()
        self.space.step(1/60.0)
        self.calculate_reward()
        return False

def main():
    st.title("🚗 Autonomous Car with Optimal Path Planning")
    
    if 'sim' not in st.session_state:
        st.session_state.sim = CarSimulation()
        st.session_state.placeholder = st.empty()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Simulation"):
            st.session_state.running = True
    with col2:
        if st.button("Reset Simulation"):
            st.session_state.sim.reset()
            st.session_state.running = False
    
    st.markdown("""
    ### Enhanced Features:
    - **A* Path Planning**: Finds the shortest optimal path to the goal
    - **Dynamic Replanning**: Adjusts path when obstacles are encountered
    - **Smooth Navigation**: PID-like steering control for smooth movement
    - **Path Visualization**: Shows planned path and actual trajectory
    """)
    
    if st.session_state.get('running', False):
        max_steps = 3000
        for _ in range(max_steps):
            complete = st.session_state.sim.step()
            st.session_state.sim.draw()
            
            img_str = pygame.image.tostring(st.session_state.sim.screen, 'RGB')
            image = Image.frombytes('RGB', (WIDTH, HEIGHT), img_str)
            st.session_state.placeholder.image(image, use_column_width=True)
            
            if complete:
                st.success("Car reached the goal!")
                st.session_state.running = False
                break
            
            pygame.time.delay(20)

if __name__ == "__main__":
    main()