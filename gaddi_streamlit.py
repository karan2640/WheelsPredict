import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.neighbors import NearestNeighbors
import joblib
import pandas as pd
import streamlit.components.v1 as components
import traceback
from streamlit_lottie import st_lottie
import random
import time
import cv2
from PIL import Image
from tensorflow.keras.models import load_model # type: ignore
from stream_app import CarSimulation
import stream_app
import base64
import time
import pygame
import pymunk
import pymunk.pygame_util
import math
import streamlit as st
from   PIL import Image
import numpy as np
from collections import deque
import dent2



WIDTH, HEIGHT = 800, 600
def run(page=" "):


          
        class CarSimulation:
            # if page !=' ' :
            #     st.session_state.autonomous_mode=False
           
            # if page==' ':
            #     st.session_state.autonomous_mode=False
            # else:
           
         
            if page in  [
                    "Home",
                    "Car Price 💰", 
                    "Suggestions for your interest 🛠️", 
                    "Crash Scan", 
                    "About our app ⚡"
                ]:
                 
                 st.session_state.autonomous_mode=False
                 
            else:
                
             st.session_state.autonomous_mode=True 
                #  phone()
            # st.write(page)
            
            #    #select box options switches
            
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
            st.title("Autonomous Car with Optimal Path Planning 🚗")
            
            if 'sim' not in st.session_state:
                st.session_state.sim = CarSimulation()
                st.session_state.placeholder = st.empty()
            
            col1, col2 = st.columns(2)
            with col1:
               
                if st.button("Start Simulation"):
                    st.session_state.autonomous_mode = True
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
                    st.session_state.placeholder.image(image, use_container_width=True)
                    
                    if complete:
                        st.success("Car reached the goal!")

                        st.session_state.running = False
                        st.session_state.autonomous_mode=False
                        break
                    
                    pygame.time.delay(20)
        
            

        if __name__ == "__main__":
            # if st.session_state.running == True:
            main()









with open("car_price_model2.pkl", "rb") as file:
    model = pickle.load(file)

with open("label_encoders2.pkl", "rb") as file:
    label_encoders = pickle.load(file)

file_path = "gaddi.xlsx"  # Update path if needed
df = pd.read_excel(file_path)






#-----------------------------------chatbot----------------------------------------------------------------------






st.sidebar.title("Dashboard 🏁")
if "autonomous_mode" not in st.session_state:
    st.session_state.autonomous_mode = False



with st.sidebar:
    page = st.selectbox("Go to:", [
        "Home 🏠",
        "Car Price 💰", 
        "Suggestions for your interest 🛠️", 
        "Crash Scan", 
        "About our app ⚡",
        " "
    ] ,key="sidebar_page_selector")

    for i in range(5):
        st.write(" ")

    # Lottie animation
    st_lottie("https://lottie.host/58656059-fb59-4c9b-a7c1-1263e88e0e18/hSbRHqJoLr.json")
    st.sidebar.markdown("## 🚀 Autonomous Mode")

    # if st.sidebar.button("🟢 Activate Autonomous Mode"):
    #     page=' '
    #     st.session_state.autonomous_mode = True
    if st.button("🟢 Activate Autonomous Mode"):
        page=' '
        st.session_state.autonomous_mode = True
        
        st.success("✅ Autonomous Mode Activated")
    # if st.button("🟢 Activate Autonomous Mode"):
    #     st.session_state.autonomous_mode = not st.session_state.autonomous_mode
    # if st.session_state.autonomous_mode:
        # st.success("Autonomous mode is ON")
    else:
        if st.session_state.autonomous_mode == False:
            st.info("Normal mode is active")
if page!="":
    st.toast("Welcome! ⭐")
       

        # else:
        #     st.session_state.autonomous_mode = False


# Define your Autonomous Mode view
# def run():
#     st.title("🟢 Autonomous Mode Activated")
#     st.success("Autonomous driving system is now running...")
#     # Add your simulation or logic here


# Main page logic
if st.session_state.autonomous_mode:
    run(page)
    
elif st.session_state.autonomous_mode==False:
     
    # st.session_state.autonomous_mode=False
    # def phone():
        if page=="Home 🏠":
            # if def a():
            #     st.session_state.autonomous_mode=False
            # st.session_state.autonomous_mode=True
            # video=f"""
            # <style>
            # .vid{{

            #     position:fixed;

            #     right:10px;
            #     bottom:10px;
            #     min-width: 100%; 
            #     min-height: 100%;
            
                
            # }}
            # </style>
            

            # <video autoplay loop muted class="vid">
            #     <source src="https://cdn.discordapp.com/attachments/1294905019388395563/1361249473052938292/14989515-hd_1280_720_30fps.mp4?ex=67fe11dd&is=67fcc05d&hm=5a9409c5bdb123860e0b5247840a67721708013693c13f5f210ff7b71e172de8&" type="video/mp4">
            # </video>

            # """

            # st.markdown(video,unsafe_allow_html=True)
            discord_image_url = "https://images.alphacoders.com/872/872897.jpg"

        # Inject CSS to set background
            st.markdown(
                f"""
                <style>
                .stApp {{
                    background-image: url("{discord_image_url}");
                    background-size: cover;
                    background-repeat: no-repeat;
                    background-attachment: fixed;
                }}
                </style>
                """,
                unsafe_allow_html=True
            )
        

            st.markdown("""
                <style>
                    .title-container {
                        text-align: center;
                        margin-top: 30px;
                        margin-bottom: 30px;
                    }

                    .cool-text {
                        display: inline-block;
                        font-family: 'Segoe UI', sans-serif;
                        font-size: 44px;
                        font-weight: bold;
                        color: #005b96;
                        text-shadow: 0 2px 5px rgba(0, 91, 150, 0.3);
                        animation: floatText 3s ease-in-out infinite;
                    }

                    @keyframes floatText {
                        0% {
                            transform: translateY(0);
                            opacity: 0.9;
                        }
                        50% {
                            transform: translateY(-10px);
                            opacity: 1;
                        }
                        100% {
                            transform: translateY(0);
                            opacity: 0.9;
                        }
                    }
                </style>

                <div class="title-container">
                    <div class="cool-text">Autonoma Drive</div>
                </div>
            """, unsafe_allow_html=True)
            import time

            # Glass notification styled bottom-right popup
            st.markdown("""
            <style>
            @keyframes slideFadeInOut {
                0%   {opacity: 0; transform: translateX(100%) scale(0.9);}
                10%  {opacity: 1; transform: translateX(0%) scale(1);}
                90%  {opacity: 1; transform: translateX(0%) scale(1);}
                100% {opacity: 0; transform: translateX(100%) scale(0.9);}
            }

            .toast-container {
                position: fixed;
                bottom: 30px;
                right: 30px;
                backdrop-filter: blur(10px);
                background: rgba(255, 255, 255, 0.15);
                border: 1px solid rgba(255, 255, 255, 0.25);
                border-radius: 16px;
                padding: 25px 30px;
                z-index: 9999;
                text-align: left;
                box-shadow: 0 8px 20px rgba(0, 0, 0, 0.25);
                animation: slideFadeInOut 3s ease-in-out forwards;
                color: #ffffff;
                font-family: 'Segoe UI', sans-serif;
                max-width: 300px;
                width: fit-content;
            }

            .toast-container h3 {
                color: #90caf9;
                margin: 0 0 5px 0;
                font-size: 20px;
            }

            .toast-container p {
                font-size: 15px;
                margin: 1px 0;
                color: #e0f7fa;
            }
            </style>

            <div class="toast-container">
                <h3>Welcome to <strong>Autonoma Drive</strong></h3>
                <p>Your AI-powered car assistant</p>
                <p><strong>Predict</strong> • <strong>Recommend</strong> • <strong>Detect</strong></p>
            </div>
            """, unsafe_allow_html=True)

            # Show for 3 seconds
            time.sleep(2)

            # for i in range(0,30):
            #  st.write(" ")
            
            # st.markdown("<h2 style='text-align: center;'>🚀 Project Features</h2>", unsafe_allow_html=True)
            
            # st.markdown("""
            #     <style>
            #     @keyframes fadeSlideIn {
            #         0% {
            #             opacity: 0;
            #             transform: translateY(-30px);
            #         }
            #         100% {
            #             opacity: 1;
            #             transform: translateY(0);
            #         }
            #     }

            #     @keyframes float {
            #         0% {
            #             transform: translateY(0);
            #         }
            #         50% {
            #             transform: translateY(-10px);
            #         }
            #         100% {
            #             transform: translateY(0);
            #         }
            #     }

            #     .animated-heading {
            #         animation: fadeSlideIn 1s ease-out forwards, float 3s ease-in-out infinite ,scaleUp 4s ease-in-out infinite;;
            #         text-align: center;
            #         font-size: 28px;
            #         margin-top: 20px;
            #     }
            #     </style>
            # """, unsafe_allow_html=True)

            # # Display animated heading
            # st.markdown("<h2 class='animated-heading'>🚀 Project Features</h2>", unsafe_allow_html=True)
            st.markdown("---")

            # Define the card template with inline CSS styles
            

            

    # Custom CSS for wide + tall cards
            st.markdown("""
                <style>
                    .card {
                        background: linear-gradient(135deg, rgba(0,198,255,0.3) 0%, rgba(255,255,255,0.2) 100%);
                        padding: 50px;
                        border-radius: 20px;
                        box-shadow: 0 6px 25px rgba(0,0,0,0.15);
                        text-align: center;
                        transition: all 0.3s ease;
                        color: white;
                        position: relative;
                        overflow: hidden;
                        min-height: 380px;
                        height:300px;
                        width: 300px;
                        display: flex;
                        flex-direction: column;
                        justify-content: top;
                    }

                    .card:hover {
                        transform: scale(1.05);
                        box-shadow: 0 12px 30px rgba(0,0,0,0.25);
                        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 50%, #ffffff 100%);

                    }

                    .card:hover .emoji {
                        transform: scale(1.3) rotate(5deg);
                    }

                    .emoji {
                        font-size: 70px;
                        margin-bottom: 25px;
                        transition: all 0.3s ease;
                    }

                    .title {
                        font-size: 20px;
                        font-weight: bold;
                        margin-bottom: 5px;
                        letter-spacing: 1px;
                    }
                </style>
            """, unsafe_allow_html=True)

            # Layout: 2 columns -> 2 cards each = 4 wide cards
        
            #background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 50%, #ffffff 100%);
            # col1, col2 = st.columns(2)

            # with col1:
            col1a, col1b = st.columns([1, 1])
            with col1a:
                    st.markdown("""
                        <div class="card">
                            <div class="emoji">🚗</div>
                            <div class="title">Car Price</div>
                            <div class="description">
                                Accurately predict car prices using machine learning by analyzing key vehicle features 
                                and understanding current market trends.
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
            with col1b:
                    st.markdown("""
                        <div class="card">
                            <div class="emoji">🎯</div>
                            <div class="title">Car Recommend</div>
                            <div class="description">
                                Build a machine learning-powered car recommendation system that suggests the best vehicles 
                                based on user preferences.
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
            st.write(' ')
            st.write(' ')
            st.write(' ')
            # with col2:
            col2a, col2b = st.columns([1, 1])
            with col2a:
                    st.markdown("""
                        <div class="card">
                            <div class="emoji">🔍</div>
                            <div class="title">Damage Detect</div>
                        
                        <div class="description">
                            Develop an image-based car damage detection system using machine learning to automatically identify dents, scratches, and other damages.
                        </div>
                    """, unsafe_allow_html=True)
            with col2b:
                    st.markdown("""
                        <div class="card">
                            <div class="emoji">🚘</div>
                            <div class="title">Real-Autonomous</div>
                        <div class="description">
                            Create an autonomous car simulation using Pygame where the car navigates and avoids obstacles based on sensor inputs and AI decision-making.
                    
                        </div>
                    """, unsafe_allow_html=True)


        elif page=="Crash Scan":
            

            
                model = load_model('car_dent_model.h5')
                discord_image_url = "https://img-s-msn-com.akamaized.net/tenant/amp/entityid/BB1msG0V?w=0&h=0&q=60&m=6&f=jpg&u=t"

        # Inject CSS to set background
                st.markdown(
                    f"""
                    <style>
                    .stApp {{
                        background-image: url("{discord_image_url}");
                        background-size: cover;
                        background-repeat: no-repeat;
                        background-attachment: fixed;
                    }}
                    </style>
                    """,
                    unsafe_allow_html=True
                
                )
                

                # Your app content
                # st.title("Streamlit App with Discord Background")
                # st.write("This app uses a Discord image as its background!")

                # Prediction function
                def predict_dent(image):
                    img = cv2.resize(image, (128, 128))
                    img = img / 255.0
                    img = np.expand_dims(img, axis=0)
                    prediction = model.predict(img)[0][0]
                    return prediction, "No Damage Found ✅"  if prediction > 0.5 else "Damage Detected! 🚗💥"

                # Custom CSS
                st.markdown("""
                    <style>
                    
                        .title {
                            font-size: 2.5rem;
                            color: #20211f;
                            text-align: center;
                            font-weight: bold;
                        }
                        .subtitle {
                            font-size: 1.2rem;
                            color: #111224;
                            text-align: center;
                            margin-bottom: 20px;
                            font-weight:bold;

                        }
                        .footer {
                            text-align: center;
                            font-size: 0.8rem;
                            color: #aaa;
                            margin-top: 50px;
                        }
                        .result {
                            font-size: 1.5rem;
                            font-weight: bold;
                            padding: 1rem;
                            border-radius: 10px;
                            text-align: center;
                        }
                    </style>
                """, unsafe_allow_html=True)

                # Title
                st.markdown('<div class="title">Car Damage Detection System</div>', unsafe_allow_html=True)
                st.markdown('<div class="subtitle">Upload a car image to check for any dents</div>', unsafe_allow_html=True)
                st.markdown("---")

                # Sidebar info
                with st.sidebar:
                    st.title("🛠️ How it Works")
                    st.markdown("""
                        <ul>
                        <li>Upload a clear image of a car</li>
                        <li>Model processes and analyzes it</li>
                        <li>You’ll get the result instantly</li>
                        </ul>
                        """, unsafe_allow_html=True)
                    st.info("📌 Tip: Upload a side or rear image of the car with visible dents.")

                # File uploader
                uploaded_file = st.file_uploader("📷 Choose a car image...", type=["jpg", "jpeg", "png"])
                

                if uploaded_file is not None:
                    image = Image.open(uploaded_file)
                    # st.image(image, caption="🔍 Uploaded Image", use_column_width=True)
                    # Convert image to display
                    st.image(image, use_container_width=True)

                    # Custom colored caption below the image
                    st.markdown("""
                        <div style="text-align: center; font-size: 1.1rem; color: #7eba4c; margin-top: 8px;">
                            🔍 Uploaded Image
                        </div>
                    """, unsafe_allow_html=True)


                    img_array = np.array(image)
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                    with st.spinner("🔍 Analyzing Image..."):
                        prediction_score, result_text = predict_dent(img_array)

                    # Show result
                    if "Damage Detected! 🚗💥" in result_text:
                        st.markdown(f'<div class="result" style="background-color:#ffdddd; color:#bb0000;">{result_text}</div>', unsafe_allow_html=True)
                        # st.balloons()
                        st.snow()
                    else:
                        st.markdown(f'<div class="result" style="background-color:#ddffdd; color:#228B22;">{result_text}</div>', unsafe_allow_html=True)
                        st.balloons()
                        

                    # Show confidence
                    st.markdown(f"### Confidence Score: `{round(float(prediction_score), 2)}`")
                
        elif page=='Car Price 💰':
            discord_image_url = "https://static.vecteezy.com/system/resources/thumbnails/024/629/229/small_2x/vintage-classic-car-theme-photo.jpg"

        # Inject CSS to set background
            st.markdown(
                    f"""
                    <style>
                    .stApp {{
                        background-image: url("{discord_image_url}");
                        background-size: cover;
                        background-repeat: no-repeat;
                        background-attachment: fixed;
                    }}
                    </style>
                    """,
                    unsafe_allow_html=True
                )
           
              

                # Glass notification styled bottom-right popup
           
            
        
            css1 = f"""
            <style>
                .strip {{
                background: linear-gradient(90deg, #ff8c00, #ffd700, #ff8c00);
                padding: 15px;
                border-radius: 30px;
                text-align: center;
                margin-bottom: 30px;
                color: black;
                font-size: 24px;
                font-weight: bold;
                box-shadow: 0 0 20px 5px rgba(255, 215, 0, 0.6);
                animation: slideIn 1s ease-in-out;
                }}
            </style>
            """
            if css1:
                st.markdown(css1, unsafe_allow_html=True)

                # Display the circular-shaped div with text inside
                st.markdown("<div class='strip'>Car Price Prediction</div>", unsafe_allow_html=True)


            # st.markdown("<div class='title'>Car Recommendation</div>", unsafe_allow_html=True)
            brand_logos = {
                "Maruti-Suzuki": "logo-maruti.jpg", 
                "Chevrolet":"chev.png",
                "Kia":"logo-kia.jpg",
                "Volkswagen":"volks.jpg",
                "Hyundai": "logo-hyundai.jpg",
                "Toyota": "logo-toyota.jpeg",
                "Honda": "honda.jpeg",
                "Ford": "ford.jpg",
                "BMW": "logo-bmw.webp",
                "Audi": "audi.jpg",
                'Mercedes': 'mr.jpg'
                }
            
            feature_names = model.feature_names_in_

        # Streamlit UI
            black_label_style = """
                <style>
                    .black-label {
                        color: black !important;
                        font-weight: 600;
                        font-size: 16px;
                    }
                </style>
                """

            st.markdown(black_label_style, unsafe_allow_html=True)
        
            # brand = st.selectbox("Enter Car Brand", label_encoders["Brand"].classes_)
            st.markdown('<p class="custom-label">Enter Car Brand</p>', unsafe_allow_html=True)
            brand = st.selectbox("", label_encoders["Brand"].classes_)
        
        
            col1,col2=st.columns([2,9])
            with col1:
                    if brand in brand_logos:
                        st.image(brand_logos[brand], width=100)
            with col2:
                    # st.markdown(f"<h3 style='font-size: 23px;'>{brand}</h3>",unsafe_allow_html=True)
                    st.markdown(f"<h3 style='font-size: 23px; color: black;'>{brand}</h3>", unsafe_allow_html=True)


            # model_name = st.selectbox("Model", label_encoders["Model"].classes_)
            # year = st.number_input("Year", min_value=2000, max_value=2025, step=1)
            # engine_size_cc = st.number_input("Engine Size (in cc)", min_value=500, max_value=10000, step=100)
            # mileage = st.number_input("Mileage (in km)", min_value=0, max_value=500000, step=500)
            # doors = st.number_input("Number of Doors", min_value=2, max_value=6, step=1)
            # owner_count = st.number_input("Number of Previous Owners", min_value=0, max_value=5, step=1)
            # fuel_type = st.selectbox("Fuel Type", label_encoders["Fuel_Type"].classes_)
            # transmission = st.selectbox("Transmission", label_encoders["Transmission"].classes_)

        

            # Custom style to enforce black text (works in both light and dark mode)
            

            # Custom style: black, bold, and bigger
            label_style = """
            <style>
                .custom-label {
                    color: black !important;
                    font-weight: bold !important;
                    font-size: 18px !important;
                    margin-bottom: 0.2rem;
                }
            </style>
            """
            st.markdown(label_style, unsafe_allow_html=True)

            st.markdown('<p class="custom-label">Model</p>', unsafe_allow_html=True)
            model_name = st.selectbox("", label_encoders["Model"].classes_)

            st.markdown('<p class="custom-label">Year</p>', unsafe_allow_html=True)
            year = st.number_input("", min_value=2000, max_value=2025, step=1)

            st.markdown('<p class="custom-label">Engine Size (in cc)</p>', unsafe_allow_html=True)
            engine_size_cc = st.number_input("", min_value=500, max_value=10000, step=100)

            st.markdown('<p class="custom-label">Mileage (in km)</p>', unsafe_allow_html=True)
            mileage = st.number_input("", min_value=0, max_value=500000, step=500)

            st.markdown('<p class="custom-label">Number of Doors</p>', unsafe_allow_html=True)
            doors = st.number_input("", min_value=2, max_value=6, step=1)

            st.markdown('<p class="custom-label">Number of Previous Owners</p>', unsafe_allow_html=True)
            owner_count = st.number_input("", min_value=0, max_value=5, step=1)

            st.markdown('<p class="custom-label">Fuel Type</p>', unsafe_allow_html=True)
            fuel_type = st.selectbox("", label_encoders["Fuel_Type"].classes_)

            st.markdown('<p class="custom-label">Transmission</p>', unsafe_allow_html=True)
            transmission = st.selectbox("", label_encoders["Transmission"].classes_)



            # Predict button
            if st.button("Predict Price"):
                # Convert engine size from cc to liters
                engine_size = engine_size_cc / 1000.0
                
                # Encode user inputs
                user_data = {
                    "Year": year,
                    "Engine_Size": engine_size,
                    "Mileage": mileage,
                    "Doors": doors,
                    "Owner_Count": owner_count,
                    "Brand": label_encoders["Brand"].transform([brand])[0],
                    "Model": label_encoders["Model"].transform([model_name])[0],
                    "Fuel_Type": label_encoders["Fuel_Type"].transform([fuel_type])[0],
                    "Transmission": label_encoders["Transmission"].transform([transmission])[0]
                }
                
                # Convert to DataFrame
                data = pd.DataFrame([user_data])
                
                # Ensure correct feature order for XGBoost model
                data = data[feature_names]
                
                # Predict the price
                predicted_price = model.predict(data)[0]
                
                # Display prediction
                usd_price = predicted_price * 87.31
                # st.success(f"Estimated Car Price: ${predicted_price:,.2f}")
                
                st.markdown(f"<div class='strip2'>Estimated Car Price: ${predicted_price:,.2f}</div>", unsafe_allow_html=True)
                # st.success(f"Estimated Car Price in INR: ₹{usd_price:,.2f}")
                st.markdown(f"<div class='strip2'>Estimated Car Price in INR: ₹{usd_price:,.2f}", unsafe_allow_html=True)

        #upto two decimal places
            # st.error("Explore your car by clicking the button below!")
            css3 = f"""
            <style>
                .strip2 {{
                    background: linear-gradient(90deg, #b3f5ee, #f5a6d1); /* Purple to Pink */
                    padding: 10px;
                    border-radius: 0px;  /* Rectangular shape */
                    text-align: center;
                    margin-bottom: 20px;
                    color: black;
                    font-size: 20px;
                    font-weight: bold;
                    box-shadow: 0 0 20px 5px rgba(159, 71, 157, 0.6);  /* Soft purple glow */
                    animation: slideIn 1s ease-in-out;
                }}
            </style>
            """
            st.markdown(css3, unsafe_allow_html=True)
            css2 = f"""
            <style>
                .strip1 {{
                    background: linear-gradient(90deg, #9b4d96, #f5a6d1); /* Purple to Pink */
                    padding: 15px;
                    border-radius: 0px;  /* Rectangular shape */
                    text-align: center;
                    margin-bottom: 30px;
                    color: black;
                    font-size: 24px;
                    font-weight: bold;
                    box-shadow: 0 0 20px 5px rgba(159, 71, 157, 0.6);  /* Soft purple glow */
                    animation: slideIn 1s ease-in-out;
                }}
            </style>
            """
            if css2:
                st.markdown(css2, unsafe_allow_html=True)

                st.markdown("<div class='strip1'>Explore your car by clicking the button below!</div>", unsafe_allow_html=True)

            # st.exception("Explore your car by clicking the button below!")
            # st.markdown("""
            #     <style>
            #     .custom-info > div {
            #         background-color: #FFF8DC !important;  /* Light yellowish-white (Cornsilk) */
            #         color: black !important;               /* Black text for readability */
            #         border-left: 6px solid #FFD700 !important;  /* Gold accent */
            #         box-shadow: 0 0 10px rgba(255, 215, 0, 0.2); /* Soft gold glow */
            #     }
            #     </style>
            # """, unsafe_allow_html=True)

            # # Wrap st.info inside a div with the custom class
            # with st.container():
            #     st.markdown('<div class="custom-info">', unsafe_allow_html=True)
            #     st.info("🚗 Explore your car by clicking the button below!")
            #     st.markdown('</div>', unsafe_allow_html=True)
            
            if st.button("View my Car"):
                    st.markdown(
                        """
                        <style>
                        @keyframes fadeIn {
                            from { opacity: 0; transform: translateY(20px); }
                            to { opacity: 1; transform: translateY(0); }
                        }
                        .image-container {
                            border: 4px solid #008B8B;  /* Medium border, Dark Cyan */
                            border-radius: 13px;
                            padding: 10px;
                            background-color: #ede8d5;  /* Soft Beige */
                            box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.2);
                            text-align: center;
                            width: 100%; 
                            height: 232px; /* Fixed height for uniformity */
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            animation: fadeIn 1.5s ease-in-out;
                        }
                        .image-container img {
                            max-width: 100%;
                            max-height: 100%;
                            border-radius: 10px;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True
                    )
                
                    if model_name=='GLA':
                        model_name = 'GLA'
                        col3,col4=st.columns([1.15,1])
            
                        
                        with col3:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://imgd.aeplcdn.com/370x208/n/cw/ec/169159/gla-facelift-exterior-right-front-three-quarter-3.jpeg?isig=0&q=80">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        with col4:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://imgd-ct.aeplcdn.com/664x415/n/cw/ec/169159/gla-facelift-exterior-left-rear-three-quarter.jpeg?isig=0&q=80">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        st.video('https://www.youtube.com/watch?v=ovIvuiaMIN8')
                    elif model_name=='3 Series':
                        model_name = '3 Series'
                        col3,col4=st.columns([1.15,1])
                        # with col3:
                        #     st.image('https://upload.wikimedia.org/wikipedia/commons/9/91/BMW_G20_%282022%29_IMG_7316_%282%29.jpg')
                        # with col4:
                        #     st.image('https://www.bmw.in/content/dam/bmw/common/all-models/3-series/series-overview/bmw-3er-overview-page-ms-07.jpg')
                        # st.video('https://www.youtube.com/watch?v=g0VWqaYROwQ')
                        with col3:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/BMW_G20_%282022%29_IMG_7316_%282%29.jpg">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        with col4:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://www.bmw.in/content/dam/bmw/common/all-models/3-series/series-overview/bmw-3er-overview-page-ms-07.jpg">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        st.video('https://www.youtube.com/watch?v=g0VWqaYROwQ')

                    elif model_name=='A4':
                        model_name = 'A4'
                        col3,col4=st.columns([1.15,1])
                        # with col3:
                        #     st.image('https://imgd.aeplcdn.com/664x374/n/cw/ec/51909/a4-exterior-left-front-three-quarter-3.jpeg?q=80')
                        # with col4:
                        #     st.image('https://imgd.aeplcdn.com/664x374/n/cw/ec/51909/a4-exterior-left-rear-three-quarter.jpeg?q=80')
                        
                        with col3:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://imgd.aeplcdn.com/664x374/n/cw/ec/51909/a4-exterior-left-front-three-quarter-3.jpeg?q=80">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        with col4:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://imgd.aeplcdn.com/664x374/n/cw/ec/51909/a4-exterior-left-rear-three-quarter.jpeg?q=80">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        st.video('https://www.youtube.com/watch?v=SEbzLLCD1_w')
                    elif model_name=='Q5':
                        model_name = 'Q5'
                        col3,col4=st.columns([1.15,1])
                        # with col3:
                        #     st.image('https://stimg.cardekho.com/images/carexteriorimages/930x620/Audi/Q5/10556/1689594416925/front-left-side-47.jpg')
                        # with col4:
                        #     st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSRzMjtpbimAesPTv9peYcKLRM8pLeAzYk2Fw&s')
                        
                        with col3:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://stimg.cardekho.com/images/carexteriorimages/930x620/Audi/Q5/10556/1689594416925/front-left-side-47.jpg">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        with col4:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSRzMjtpbimAesPTv9peYcKLRM8pLeAzYk2Fw&s">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        st.video('https://www.youtube.com/watch?v=8ffXE6_qXqM')
                    elif model_name=='Golf':
                        model_name = 'Golf'
                        col3,col4=st.columns([1.15,1])
                        # with col3:
                        #     st.image('https://car-images.bauersecure.com/wp-images/12880/1752x1168/50-volkswagen-golf-2024-front-driving.jpg?mode=max&quality=90&scale=down')
                        # with col4:
                        #     st.image('https://i.ytimg.com/vi/3sgnT4Uyfzk/maxresdefault.jpg')
                        with col3:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://car-images.bauersecure.com/wp-images/12880/1752x1168/50-volkswagen-golf-2024-front-driving.jpg?mode=max&quality=90&scale=down">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        with col4:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://i.ytimg.com/vi/3sgnT4Uyfzk/maxresdefault.jpg">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        st.video('https://www.youtube.com/watch?v=sPcbMcTXWRc')
                    elif model_name=='Malibu':
                        model_name = 'Malibu'
                        col3,col4=st.columns([1.15,1])
                        # with col3:
                        #     st.image('https://hips.hearstapps.com/hmg-prod/images/2019-chevrolet-malibu-rs-117-1568289288.jpg?crop=0.830xw:0.678xh;0.0913xw,0.202xh&resize=640:*')
                        # with col4:
                        #     st.image('https://hips.hearstapps.com/hmg-prod/images/2019-chevrolet-malibu-rs-114-1568289287.jpg')
                        with col3:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://hips.hearstapps.com/hmg-prod/images/2019-chevrolet-malibu-rs-117-1568289288.jpg?crop=0.830xw:0.678xh;0.0913xw,0.202xh&resize=640:*">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        with col4:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://hips.hearstapps.com/hmg-prod/images/2019-chevrolet-malibu-rs-114-1568289287.jpg">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        st.video('https://www.youtube.com/watch?v=mbcxRX8D-_k')
                    elif model_name=='Civic':
                        model_name='Civic'
                        col3,col4=st.columns([1.15,1])
                        # with col3:
                        #     st.image('https://imgd.aeplcdn.com/664x374/n/cw/ec/27074/civic-exterior-right-front-three-quarter-148155.jpeg?q=80')
                        # with col4:
                        #     st.image('https://imgd.aeplcdn.com/664x374/n/cw/ec/27074/civic-exterior-rear-view.jpeg?q=80')
                        with col3:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://imgd.aeplcdn.com/664x374/n/cw/ec/27074/civic-exterior-right-front-three-quarter-148155.jpeg?q=80">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        with col4:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://imgd.aeplcdn.com/664x374/n/cw/ec/27074/civic-exterior-rear-view.jpeg?q=80">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        st.video('https://www.youtube.com/watch?v=JQH3qJBYLQg')
                    elif model_name=='Camry':
                        model_name = 'Camry'
                        col3,col4=st.columns([1.15,1])
                        # with col3:
                        #     st.image('https://akm-img-a-in.tosshub.com/indiatoday/images/story/202201/2022_Toyota_Camry_Hybrid-_Exte.jpg')
                        # with col4:
                        #     st.image('https://imgd.aeplcdn.com/1920x1080/n/cw/ec/192443/camry-exterior-right-rear-three-quarter.jpeg?isig=0&q=80&q=80')
                        with col3:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://akm-img-a-in.tosshub.com/indiatoday/images/story/202201/2022_Toyota_Camry_Hybrid-_Exte.jpg">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        with col4:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://imgd.aeplcdn.com/1920x1080/n/cw/ec/192443/camry-exterior-right-rear-three-quarter.jpeg?isig=0&q=80&q=80">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        st.video('https://www.youtube.com/watch?v=j_Jfq0aXkFE')
                    elif model_name=='Optima':
                        model_name = 'Optima'
                        col3,col4=st.columns([1.15,1])
                        # with col3:
                        #     st.image('https://carsguide-res.cloudinary.com/image/upload/f_auto,fl_lossy,q_auto,t_default/v1/editorial/vhs/Kia-Optima-2019-icon.png')
                        # with col4:
                        #     st.image('https://www.autozine.nl/cache/simplescale/728/16825.jpg')
                        with col3:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://carsguide-res.cloudinary.com/image/upload/f_auto,fl_lossy,q_auto,t_default/v1/editorial/vhs/Kia-Optima-2019-icon.png">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        with col4:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://www.autozine.nl/cache/simplescale/728/16825.jpg">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        st.video('https://www.youtube.com/watch?v=i_atZKVaBsU')
                    elif model_name=='Impala':
                        model_name = 'Impala'
                        col3,col4=st.columns([1.15,1])
                        # with col3:
                        #     st.image('https://carsguide-res.cloudinary.com/image/upload/f_auto,fl_lossy,q_auto,t_default/v1/editorial/vhs/Kia-Optima-2019-icon.png')
                        # with col4:
                        #     st.image('https://www.autozine.nl/cache/simplescale/728/16825.jpg')
                        with col3:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://carsguide-res.cloudinary.com/image/upload/f_auto,fl_lossy,q_auto,t_default/v1/editorial/vhs/Kia-Optima-2019-icon.png">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        with col4:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://www.autozine.nl/cache/simplescale/728/16825.jpg">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        # st.video('')
                        st.video('https://www.youtube.com/watch?v=i_atZKVaBsU')
                    elif model_name=='Explorer':
                        model_name='Explorer'
                        col3,col4=st.columns([1.15,1])
                        # with col3:
                        #     st.image('https://car-images.bauersecure.com/wp-images/161878/explorer_058.jpg')
                        # with col4:
                        #     st.image('https://car-images.bauersecure.com/wp-images/161878/explorer_053.jpg')
                        with col3:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://car-images.bauersecure.com/wp-images/161878/explorer_058.jpg">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        with col4:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://car-images.bauersecure.com/wp-images/161878/explorer_053.jpg">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        st.video('https://www.youtube.com/watch?v=RzWR0q-oFeA')
                        
                    elif model_name=='E-Class':
                        model_name='E-Class'
                        col3,col4=st.columns([1.15,1])
                        # with col3:
                        #     st.image('https://imgd.aeplcdn.com/664x374/n/cw/ec/47336/e-class-exterior-right-rear-three-quarter.jpeg?q=80')
                        # with col4:
                        #     st.image('https://upload.wikimedia.org/wikipedia/commons/f/fd/Mercedes-Benz_W214_1X7A1841.jpg')
                        with col3:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://imgd.aeplcdn.com/664x374/n/cw/ec/47336/e-class-exterior-right-rear-three-quarter.jpeg?q=80">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        with col4:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://upload.wikimedia.org/wikipedia/commons/f/fd/Mercedes-Benz_W214_1X7A1841.jpg">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        st.video('https://www.youtube.com/watch?v=UP-FPkiXsXQ')
                    elif model_name=='Accord':
                        model_name='Accord'
                        col3,col4=st.columns([1.15,1])
                        # with col3:
                        #     st.image('https://hips.hearstapps.com/hmg-prod/images/2021-honda-accord-hybrid-109-edit-1604961241.jpg?crop=0.591xw:0.499xh;0.0962xw,0.501xh&resize=2048:*')
                        # with col4:
                        #     st.image('https://cdni.autocarindia.com/Utils/ImageResizer.ashx?n=https://cdni.autocarindia.com/ExtraImages/20170717040703_Honda-Accord-4.jpg&w=700&c=1')
                        with col3:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://hips.hearstapps.com/hmg-prod/images/2021-honda-accord-hybrid-109-edit-1604961241.jpg?crop=0.591xw:0.499xh;0.0962xw,0.501xh&resize=2048:*">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        with col4:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://cdni.autocarindia.com/Utils/ImageResizer.ashx?n=https://cdni.autocarindia.com/ExtraImages/20170717040703_Honda-Accord-4.jpg&w=700&c=1">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        st.video('https://www.youtube.com/watch?v=G7Rkf7cklMo')
                    elif model_name=='5 Series':
                        model_name='5 Series'
                        col3,col4=st.columns([1.15,1])
                        # with col3:
                        #     st.image('https://stimg.cardekho.com/images/carexteriorimages/930x620/BMW/5-Series-2024/10182/1685002609273/front-left-side-47.jpg')
                        # with col4:
                        #     st.image('https://cdni.autocarindia.com/Utils/ImageResizer.ashx?n=https://cdni.autocarindia.com/Reviews/BMW-5-Series-rear.jpg&c=0&w=700')
                        with col3:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://stimg.cardekho.com/images/carexteriorimages/930x620/BMW/5-Series-2024/10182/1685002609273/front-left-side-47.jpg">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        with col4:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://cdni.autocarindia.com/Utils/ImageResizer.ashx?n=https://cdni.autocarindia.com/Reviews/BMW-5-Series-rear.jpg&c=0&w=700">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        st.video('https://www.youtube.com/watch?v=UmTS9m9Voi4')
                    elif model_name=='CR-V':
                        model_name='CR-V'
                        col3,col4=st.columns([1.15,1])
                        with col3:
                            st.image('https://hips.hearstapps.com/hmg-prod/images/2025-honda-cr-v-hybrid-awd-sport-touring-102-679407cb80051.jpg?crop=0.702xw:0.590xh;0.0529xw,0.341xh&resize=2048:*')
                        with col4:
                            st.image('https://images.hindustantimes.com/auto/img/2023/06/13/1600x900/Honda_CR-V_1686624323609_1686624328430.jpg')
                        with col3:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://hips.hearstapps.com/hmg-prod/images/2025-honda-cr-v-hybrid-awd-sport-touring-102-679407cb80051.jpg?crop=0.702xw:0.590xh;0.0529xw,0.341xh&resize=2048:*">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        with col4:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://images.hindustantimes.com/auto/img/2023/06/13/1600x900/Honda_CR-V_1686624323609_1686624328430.jpg">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        st.video('https://www.youtube.com/watch?v=iJ4qEqgC6mQ')
                    elif model_name=='A3':
                        model_name='A3'
                        col3,col4=st.columns([1.15,1])
                        # with col3:
                        #     st.image('https://uploads.audi-mediacenter.com/system/production/media/91178/images/1f201fd6f5fcbd78b452dd0ff4907b1cc4dc0a8c/A202506_web_2880.jpg?1698425163')
                        # with col4:
                        #     st.image('https://www.team-bhp.com/forum/attachments/official-new-car-reviews/1940527d1575165754-audi-a3-official-review-2021audia3sedanrenderingrevealsnormalcompactcarspec_1.jpg')
                        with col3:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://uploads.audi-mediacenter.com/system/production/media/91178/images/1f201fd6f5fcbd78b452dd0ff4907b1cc4dc0a8c/A202506_web_2880.jpg?1698425163">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        with col4:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://www.team-bhp.com/forum/attachments/official-new-car-reviews/1940527d1575165754-audi-a3-official-review-2021audia3sedanrenderingrevealsnormalcompactcarspec_1.jpg">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        st.video('https://www.youtube.com/watch?v=V3BoSR_Rlaw')
                    elif model_name=='C-Class':
                        model_name='C-Class'
                        col3,col4=st.columns([1.15,1])
                        # with col3:
                        #     st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTr15mmbFp4HFgsW45fDCbkfGzHF_g9OQwJMyfhyR-XPjhkDVwljN4yy4CKdu_56Gud5Jo&usqp=CAU')
                        # with col4:
                        #     st.image('https://assets.thehansindia.com/h-upload/2022/04/13/1286728-mercedes-benz-c-class.webp')
                        with col3:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTr15mmbFp4HFgsW45fDCbkfGzHF_g9OQwJMyfhyR-XPjhkDVwljN4yy4CKdu_56Gud5Jo&usqp=CAU">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        with col4:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://assets.thehansindia.com/h-upload/2022/04/13/1286728-mercedes-benz-c-class.webp">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        st.video('https://www.youtube.com/watch?v=J7LueobArWw')
                    elif model_name=='Corolla':
                        model_name='Corolla'
                        col3,col4=st.columns([1.15,1])
                        # with col3:
                        #     st.image('https://media.istockphoto.com/id/1412133515/photo/toyota-corolla.jpg?s=612x612&w=0&k=20&c=lD7qArFDIFMgiTauLrE5yfi0Eof8D0WIwhXJanvzqTQ=')
                        # with col4:
                        #     st.image('https://pictures.dealer.com/a/autonationtoyotascionlasvegas/1246/d1b407923fb1b3c55f8450b1797e3e83x.jpg?impolicy=downsize_bkpt&imdensity=1&w=520')
                        with col3:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://media.istockphoto.com/id/1412133515/photo/toyota-corolla.jpg?s=612x612&w=0&k=20&c=lD7qArFDIFMgiTauLrE5yfi0Eof8D0WIwhXJanvzqTQ=">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        with col4:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://pictures.dealer.com/a/autonationtoyotascionlasvegas/1246/d1b407923fb1b3c55f8450b1797e3e83x.jpg?impolicy=downsize_bkpt&imdensity=1&w=520">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        st.video('https://www.youtube.com/watch?v=dTDQZ7-mI7Y')
                    elif model_name=='Elantra':
                        model_name='Elantra'
                        col3,col4=st.columns([1.15,1])
                        # with col3:
                        #     st.image('https://imgd.aeplcdn.com/664x374/n/cw/ec/41138/elantra-exterior-right-front-three-quarter-3.jpeg?q=40')
                        # with col4:
                        #     st.image('https://imgd-ct.aeplcdn.com/1056x660/n/cw/ec/41138/elantra-exterior-right-rear-three-quarter.jpeg?q=80')
                        with col3:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://imgd-ct.aeplcdn.com/1056x660/n/cw/ec/41138/elantra-exterior-right-rear-three-quarter.jpeg?q=80">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        with col4:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://imgd.aeplcdn.com/664x374/n/cw/ec/41138/elantra-exterior-right-front-three-quarter-3.jpeg?q=40">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        st.video('https://www.youtube.com/watch?v=cGLY-t2f3EI')
                        #https://cdn.jdpower.com/JDPA_2020%20Chevrolet%20Equinox%20Premier%20Dark%20Blue%20Front%20View%20Small.jpg
                    elif model_name=='Equinox':
                        model_name='Equinox'
                        col3,col4=st.columns([1.15,1])
                        # with col3:
                        #     st.image('https://cdn.jdpower.com/JDPA_2020%20Chevrolet%20Equinox%20Premier%20Dark%20Blue%20Front%20View%20Small.jpg')
                        # with col4:
                        #     st.image('https://photos.dealerimagepro.com/lib/g1-freedom-chevrolet-797/01.04.2025/NEW/%28Chevrolet%20-%20Equinox%293GNAXSEG1SL244951%282%29/3GNAXSEG1SL244951--12.jpg?ver=1736051232')
                        with col3:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://cdn.jdpower.com/JDPA_2020%20Chevrolet%20Equinox%20Premier%20Dark%20Blue%20Front%20View%20Small.jpg">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        with col4:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://photos.dealerimagepro.com/lib/g1-freedom-chevrolet-797/01.04.2025/NEW/%28Chevrolet%20-%20Equinox%293GNAXSEG1SL244951%282%29/3GNAXSEG1SL244951--12.jpg?ver=1736051232">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        st.video('https://www.youtube.com/watch?v=cTDj2rrDAMA')
                

                    elif model_name=='X5':
                        model_name='X5'
                        col3,col4=st.columns([1.15,1])
                        # with col3:
                        #     st.image('https://stimg.cardekho.com/images/carexteriorimages/630x420/BMW/X5-2023/10452/1688992642182/front-left-side-47.jpg')
                        # with col4:
                        #     st.image('https://stimg.cardekho.com/images/carexteriorimages/930x620/BMW/X5/10490/1689853299825/rear-left-view-121.jpg')
                        with col3:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://stimg.cardekho.com/images/carexteriorimages/630x420/BMW/X5-2023/10452/1688992642182/front-left-side-47.jpg">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        with col4:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://stimg.cardekho.com/images/carexteriorimages/930x620/BMW/X5/10490/1689853299825/rear-left-view-121.jpg">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        st.video('https://www.youtube.com/watch?v=vnF5i9Gl7Tw')
                    elif model_name=='Tucson':
                        model_name='Tucson'
                        col3,col4=st.columns([1.15,1])
                        # with col3:
                        #     st.image('https://stimg.cardekho.com/images/carexteriorimages/930x620/Hyundai/Tucson/10134/1694668706095/front-left-side-47.jpg')
                        # with col4:
                        #     st.image('https://i.pinimg.com/736x/50/63/2e/50632edc3d61f994f452edc5a1c2d4d4.jpg')
                        with col3:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://stimg.cardekho.com/images/carexteriorimages/930x620/Hyundai/Tucson/10134/1694668706095/front-left-side-47.jpg">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        with col4:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://i.pinimg.com/736x/50/63/2e/50632edc3d61f994f452edc5a1c2d4d4.jpg">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        st.video('https://www.youtube.com/watch?v=E2gswhCWzoA')
                    elif model_name=='Tiguan':
                        model_name='Tiguan'
                        col3,col4=st.columns([1.15,1])
                        # with col3:
                        #     st.image('https://e00-elmundo.uecdn.es/assets/multimedia/imagenes/2024/02/24/17087733566106.jpg')
                        # with col4:
                        #     st.image('https://d1l107ig5zcaf7.cloudfront.net/media/cache/resolve/original/67b39820-a493-4012-b362-98ce9c55de0d/b54bba8e2f23078e7e66ee618ac61a60.jpg')
                        with col3:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://e00-elmundo.uecdn.es/assets/multimedia/imagenes/2024/02/24/17087733566106.jpg">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        with col4:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://d1l107ig5zcaf7.cloudfront.net/media/cache/resolve/original/67b39820-a493-4012-b362-98ce9c55de0d/b54bba8e2f23078e7e66ee618ac61a60.jpg">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        st.video('https://www.youtube.com/watch?v=mdyaDndhR-k')
                    elif model_name=='Sportage':
                        model_name='Sportage'
                        col3,col4=st.columns([1.15,1])
                        # with col3:
                        #     st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSv7V5u0v8g9yNATfUVVMfH7WL_coJg3cs0Vg&s')
                        # with col4:
                        #     st.image('https://static.wixstatic.com/media/6174eb_565481c6f6264c51b206ab56d2e48055~mv2.jpg/v1/fill/w_640,h_400,al_c,q_80,usm_0.66_1.00_0.01,enc_avif,quality_auto/6174eb_565481c6f6264c51b206ab56d2e48055~mv2.jpg')
                        with col3:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSv7V5u0v8g9yNATfUVVMfH7WL_coJg3cs0Vg&s">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        with col4:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://static.wixstatic.com/media/6174eb_565481c6f6264c51b206ab56d2e48055~mv2.jpg/v1/fill/w_640,h_400,al_c,q_80,usm_0.66_1.00_0.01,enc_avif,quality_auto/6174eb_565481c6f6264c51b206ab56d2e48055~mv2.jpg">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        st.video('https://www.youtube.com/watch?v=sw84qVKG6bY')
                    elif model_name=='Sonata':
                        model_name='Sonata'
                        col3,col4=st.columns([1.15,1])
                        # with col3:
                        #     st.image('')
                        # with col4:
                        #     st.image('')
                        with col3:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://content.carlelo.com/source/IMG-20230328-WA0011.jpg">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        with col4:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://pictures.dealer.com/t/townehyundaidenville/0381/64650daf98340d67fb7826d0e2a3fbdfx.jpg?impolicy=downsize_bkpt&w=410">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        st.video('https://www.youtube.com/watch?v=CGo99QVuqjc')
                    elif model_name=='Fiesta':
                        model_name='Fiesta'
                        col3,col4=st.columns([1.15,1])
                        # with col3:
                        #     st.image('')
                        # with col4:
                        #     st.image('')
                        with col3:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://www.topgear.com/sites/default/files/2022/05/51952068138_a79993a8de_k.jpg">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        with col4:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://www.topgear.com/sites/default/files/2022/05/51952307109_b9b8d85293_k.jpg">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        st.video('https://www.youtube.com/watch?v=gQ_0xOTITbQ')
                    elif model_name=='Focus':
                        model_name='Focus'
                        col3,col4=st.columns([1.15,1])
                        # with col3:
                        #     st.image('')
                        # with col4:
                        #     st.image('')
                        with col3:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRA1ZfX1C2Ihk4u4v0ZWw3qQAxYQmuB-kUYMw&s">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        with col4:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://cdni.autocarindia.com/Utils/ImageResizer.ashx?n=https://cdni.autocarindia.com/ExtraImages/20180410053435_Focus8.jpg">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        st.video('https://www.youtube.com/watch?v=CYEsIIP_pDg')
                    elif model_name=='Passat':
                        model_name='Passat'
                        col3,col4=st.columns([1.15,1])
                        # with col3:
                        #     st.image('')
                        # with col4:
                        #     st.image('')
                        with col3:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://gaadiwaadi.com/wp-content/uploads/2019/01/2020-passat-3.jpg">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        with col4:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://i.insider.com/5c3d2aa6bde70f648200e945?width=800&format=jpeg&auto=webp">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        st.video('https://www.youtube.com/watch?v=QlJkrt6orrg')
                    elif model_name=='RAV4':
                        model_name='RAV4'
                        col3,col4=st.columns([1.15,1])
                        # with col3:
                        #     st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTX-Bgepwo6-oTWEaqvdD05cQUYrCrRtD_R4g&s')
                        # with col4:
                        #     st.image('https://preview.redd.it/whats-your-opinion-on-the-toyota-rav4-v0-4up6ltyxtpob1.jpg?width=1080&crop=smart&auto=webp&s=98bd9fa36cbabcf184792f08247b32e7b2aa1b3d')
                        with col3:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTX-Bgepwo6-oTWEaqvdD05cQUYrCrRtD_R4g&s">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        with col4:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://preview.redd.it/whats-your-opinion-on-the-toyota-rav4-v0-4up6ltyxtpob1.jpg?width=1080&crop=smart&auto=webp&s=98bd9fa36cbabcf184792f08247b32e7b2aa1b3d">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        st.video('https://www.youtube.com/watch?v=nb2VgbgnOtA')
                    # elif model_name=='Rio':
                    #     col3,col4=st.columns([1.15,1])
                    #     with col3:
                    #         st.image('https://images.prismic.io/carwow/1993a4c8-cfbd-49ef-a6eb-22e942bfd0e6_Kia+Rio+Front+%C2%BE+static.jpg?auto=format&cs=tinysrgb&fit=crop&q=60&w=750')
                    #     with col4:
                    #         st.image('https://www.topgear.com/sites/default/files/cars-car/carousel/2021/03/2021_kia_rio_rear34_static.jpg')
                    #     st.video('https://www.youtube.com/watch?v=nb2VgbgnOtA')
                    
                    

                    elif model_name == 'Rio':
                        model_name ='Rio'
                        col3, col4 = st.columns([1.15, 1])

                        with col3:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://images.prismic.io/carwow/1993a4c8-cfbd-49ef-a6eb-22e942bfd0e6_Kia+Rio+Front+%C2%BE+static.jpg?auto=format&cs=tinysrgb&fit=crop&q=60&w=750">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        
                        with col4:
                            st.markdown(
                                """
                                <div class="image-container">
                                    <img src="https://www.topgear.com/sites/default/files/cars-car/carousel/2021/03/2021_kia_rio_rear34_static.jpg">
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        st.video('https://www.youtube.com/watch?v=nb2VgbgnOtA')
                                                
                                    


        
        elif page == "About our app ⚡":
            css = f"""
            <style>
            .stApp {{
                background-image: url("https://cdn.discordapp.com/attachments/1294905019388395563/1349646087090208839/contact.jpg?ex=67d3db5e&is=67d289de&hm=db61d57169fdbb6f8618fb79a54da754557d73f574e520336a13692c6727cf8d&");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}

            .strip {{
                background: linear-gradient(90deg, #ff8c00, #ffd700, #ff8c00);
                padding: 15px;
                border-radius: 30px;
                text-align: center;
                margin-bottom: 30px;
                color: black;
                font-size: 24px;
                font-weight: bold;
                box-shadow: 0 0 20px 5px rgba(255, 215, 0, 0.6);
                animation: slideIn 1s ease-in-out;
            }}

            .animated-text {{
                font-family: 'Segoe UI', sans-serif;
                font-size: 23px;
                text-align: justify;
                margin-bottom: 20px;
                color: white;
                opacity: 0;
                animation: slideUp 1s forwards;
            }}

            .animated-text.delay-1 {{ animation-delay: 0.5s; }}
            .animated-text.delay-2 {{ animation-delay: 1s; }}
            .animated-text.delay-3 {{ animation-delay: 1.5s; }}

            @keyframes slideUp {{
                from {{ transform: translateY(40px); opacity: 0; }}
                to {{ transform: translateY(0); opacity: 1; }}
            }}

            @keyframes slideIn {{
                from {{ transform: translateX(-100%); opacity: 0; }}
                to {{ transform: translateX(0); opacity: 1; }}
            }}
            </style>
            """
            st.markdown(css, unsafe_allow_html=True)

            st.title("Our App")

            # Glowing strip with animated intro
            st.markdown("<div class='strip'>Its a perfect Car Journey App</div>", unsafe_allow_html=True)

            # Animated description sections
            st.markdown(
                "<div class='animated-text delay-1'><b>Find the Best Car for You at the Right Price!</b><br>"
                "Exploring cars and discovering their true market value based on parameters and preferences gives users a significant edge in decision-making.</div>",
                unsafe_allow_html=True
            )

            st.markdown(
                "<div class='animated-text delay-2' style='color: yellow;'>This webapp can give the optimum results related to your choices and preferences with advanced data analytics and smart recommendations.</div>",
                unsafe_allow_html=True
            )

            st.markdown(
                "<div class='animated-text delay-3'>✨ <b>About Our Technology</b><br>"
                "Our app uses intelligent machine learning algorithms to analyze vehicle specifications like engine capacity, mileage, year of manufacture, fuel type, and more. "
                "By integrating a smart recommendation engine, we ensure that you get accurate and tailored suggestions.</div>",
                unsafe_allow_html=True
            )

            st.markdown(
                "<div class='animated-text delay-2' style='color: #00ffff;'>💰 <b>How Price Recommendation Works</b><br>"
                "We combine historical pricing data, market trends, and car condition metrics to suggest a fair price. "
                "Whether you're buying or selling, our app empowers you with data-driven confidence.</div>",
                unsafe_allow_html=True
            )

            st.markdown(
                "<div class='animated-text delay-3'>🚀 <b>Why Use Our App?</b><br>"
                "<ul>"
                "<li>✅ Personalized Car Matching Based on Your Preferences</li>"
                "<li>✅ Smart Price Estimator Based on Market & Technical Specs</li>"
                "<li>✅ User-Friendly Interface with Stunning Visual Experience</li>"
                "<li>✅ Seamless Experience from Research to Recommendation</li>"
                "</ul></div>",
                unsafe_allow_html=True
            )

            st.markdown(
                "<div class='animated-text delay-3' style='color: #7CFC00; font-size: 26px; font-weight: bold;'>"
                "Let our AI be your guide in the journey to your perfect car! 🧠🚘"
                "</div>",
                unsafe_allow_html=True
            )
            st.title("📞 Contact Us")

            st.markdown("""
                ### Team Members:
                - **Abhijay Parashar**
                - **Baljinder Singh**
                - **Karanpreet Singh**

                ### 📱 Contact Numbers:
                - +91-XXXXX-67834
                - +91-XXXXX-67245
                - +91-XXXXX-67245

                ### 📧 Email:
                - parasharabhijay@gmail.com **
                - baljindersingh260304@gmail.com **
                - karanpreet30408@gmail.com **

                ---

                🚀 *We're passionate about revolutionizing automotive safety through AI-powered solutions. 
                Whether you're curious, have feedback, or want to collaborate—don't hesitate to reach out!*

                
                """)
            st.markdown("""
                <style>
                .tagline-box {
                    background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
                    color: white;
                    padding: 20px;
                    border-radius: 12px;
                    text-align: center;
                    font-size: 22px;
                    font-weight: bold;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
                    margin-top: 30px;
                }
                </style>

                <div class='tagline-box'>
                    ✨ Driven with <span style='color: #00FFAA;'>Code.</span>, Designed for <span style='color: #FF6B6B;'>Safety.</span> 🛡️
                </div>
                """, unsafe_allow_html=True)




                




        elif page == "Suggestions for your interest 🛠️":
                #https://media.discordapp.net/attachments/1294905019388395563/1349784929461862474/porche.jpg?ex=67d45cac&is=67d30b2c&hm=d11bae4a5bdcee70b371b28cedfe0ecee5ea76d9447729354786c8c8f7610448&=&format=webp&width=1000&height=563
                
                css = f"""
            <style>
            .stApp {{
                background-image: url("https://cdn.discordapp.com/attachments/1294905019388395563/1352918912278466580/pexels-introspectivedsgn-13083182.jpg?ex=67dfc36c&is=67de71ec&hm=624333ca6fea8f6330ec9b5da8b681ed43466b1fa81823e4c51e3a181c227d3e&");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}
            </style>
            """ 
                # a="Honda Civic"
                st.markdown(css, unsafe_allow_html=True)
                # def load_dataset():
                #     data = {
                #         'Car': [a, 'Honda City','Toyota Corolla', 'Ford Focus', 'BMW 3 Series', 'Audi A4', 'Hyundai Elantra'],
                #         'Price': [20000, 20000,22000, 21000, 35000, 37000, 19000],
                #         'Mileage': [30, 29, 28, 27, 25, 24, 31],
                #         'Fuel_Type': ['Petrol', 'Petrol', 'Petrol', 'Diesel', 'Diesel', 'Petrol', 'Petrol'],
                #         'Horsepower': [158, 159, 139, 160, 255, 248, 147]
                #     }
                #     return pd.DataFrame(data)
                # Streamlit Web App
                # st.markdown("<h1 style='text-align: center; color: #00aaff;'>Search</h1>", unsafe_allow_html=True)
                # if a=="Honda Civic":


        # Add CSS animation using Markdown
                st.markdown(
                """
                <style>
                    @keyframes move {
                        0% { transform: translateX(-10px); }
                        50% { transform: translateX(15px); }
                        100% { transform: translateX(-15px); }
                    }

                    .animated-title {
                        text-align: center;
                        color: #00aaff;
                        font-size: 36px;
                        font-weight: bold;
                        animation: move 2s infinite alternate;
                    }
                </style>

                <h1 class="animated-title">Search the best from the best</h1>
                """,
                unsafe_allow_html=True
            )
                # st.info("Enter the entire name of car with company also eg: Tata Nexon")
        

                st.markdown(
                """
                    <style>
                        .custom-info {
                            background-color: #FFE5B4;  /* Change this to any color */
                            color: black;  /* Text color */
                            padding: 10px;
                            border-radius: 5px;
                            font-size: 16px;
                            font-weight: bold;
                        }
                    </style>
                    <div class="custom-info">
                        Enter the entire name of car with company also e.g: <b>Tata Nexon</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


                #ff4444
                # Load and preprocess dataset
                # df = load_dataset()

                # search_query = st.text_input("Search for a car")   
                

        # Search Input Field
                # search_query = st.text_input("Search for a car")
                # search_button = st.button("🔍 Search")
                # if search_query == "Honda Civic" or search_query=='Toyota Fortuner':
                

                
                st.markdown(
                    """
                    <style>
                        /* Center the entire container without extra space */
                        .container {
                            display: flex;
                            justify-content: center;
                            align-items: center;
                            flex-direction: column;
                            margin-top: 0px; /* Removed extra space */
                            opacity: 0;
                            animation: fadeIn 0.8s ease-in-out forwards;
                        }
                        @keyframes fadeIn {
                            0% { opacity: 0; transform: scale(0.9); }
                            100% { opacity: 1; transform: scale(1); }
                        }
                        .title {
                            font-size: 32px;
                            font-weight: bold;
                            margin-bottom: 10px;
                        }
                        .car-card {
                            background: white;
                            padding: 15px;
                            border-radius: 12px;
                            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                            transition: 0.3s ease-in-out;
                            text-align: center;
                            width: 280px;
                        }
                        .car-card:hover {
                            transform: scale(1.05);
                            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
                        }
                        .car-card img {
                            width: 100%;
                            height: 160px;
                            border-radius: 12px;
                            object-fit: cover;
                        }
                        .car-card h3 {
                            font-size: 20px;
                            color: #222;
                            margin-top: 10px;
                        }
                        .car-card p {
                            font-size: 14px;
                            color: #555;
                            margin: 5px 0;
                            
                        }
                        .car-card p1 {
                            font-size: 14px;
                            color: #006400;
                            margin: 5px 0;
                            
                        }
                        .price {
                            font-size: 18px;
                            font-weight: bold;
                            color: #006400;
                            font-family: 'Segoe UI', sans-serif;
                        }
                        /* Align search bar and button properly */
                        .search-container {
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            gap: 10px;
                            width: 100%;
                            margin-top: -25px; /* Adjusted spacing from top */
                        }
                        .search-container {
                            display: flex;
                            align-items: flex-end; /* Align button to the bottom */
                            gap: 10px;
                        }
                        .search-container .stTextInput input {
                            height: 40px !important;
                            font-size: 14px;
                        }
                        .search-container .stButton button {
                            height: 40px !important;
                            width: 50px !important;
                            font-size: 16px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            margin-bottom: 5px; /* Ensures button aligns to bottom */
                        }
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("<div class='search-container'>", unsafe_allow_html=True)
                col1, col2 = st.columns([0.9, 0.05])
                with col1:
                    st.empty()
                    search_query = st.text_input("Search for a car", key="search_bar")
                with col2:
                    st.write(' ')
                    st.write(' ')
                
                    search_button = st.button("🔍", key="search_button")
                # car_list=['Honda Civic',
                    # 'Toyota Fortuner','Tata Nexon','Maruti Suzuki Brezza','Toyota Innova','Hyundai i20',
                    # 'Hyundai Grand i10', 'Tata Punch','Kia Carens','BMW M5','Mahindra XUV 700','Hyundai Creta',
                    # 'Honda City','Mahindra Thar','Volkswagen Virtus',
                    # 'Maruti Suzuki Baleno','Maruti Suzuki Swift',
                    # 'Hyundai Venue','Maruti Suzuki Ertiga','Mahindra XUV400',
                    # 'Mahindra XUV300','Tata Curvv','Mahindra BE 6','MG Hector','Tata Tiago','Mercedes G-Class']
                # col1, col2 = st.columns([0.9, 0.05])
                # with col1:
                #     search_query = st.text_input("Search for a car", key="search_bar")
                # with col2:
                #     st.write(" ")
                #     st.write(" ")
                #     search_button = st.button("🔍", key="search_button")

                # if search_button and search_query:
                #     matched_cars = [car for car in car_list if search_query.lower() in car.lower()]
                #     if matched_cars:
                #         # st.success(f"Results found: {', '.join(matched_cars)}")
                #         search_query
                #     else:
                #         st.warning("No matching car found.")
                st.markdown("</h>", unsafe_allow_html=True)
               

                if search_query =='Honda Civic':
                    st.markdown(
                        """
                        <div class="container">
                            <div class="car-card">
                                <img src="https://imgd.aeplcdn.com/664x374/n/cw/ec/27074/civic-exterior-right-front-three-quarter-148155.jpeg?q=80">
                                <h3>Honda Civic</h3>
                                <p>Fuel Type: Petrol</p>
                                <p>Transmission: Automatic</p>
                                <p1 class="price">₹25L</p1>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                elif search_query == 'Toyota Fortuner':
                    st.markdown(
                        """
                        <div class="container">
                            <div class="car-card">
                                <img src="https://images.hindustantimes.com/auto/img/2021/08/12/600x338/car-vrz_1628592725177_1628767063272.jpg">
                                <h3>Toyota Fortuner</h3>
                                <p>Fuel Type: Diesel</p>
                                <p>Transmission: Manual</p>
                                <p1 class="price">₹50L</p1>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        """
                        <div class="container">
                            <div class="car-card">
                                <img src="https://www.motoroids.com/wp-content/uploads/2020/06/2021-Toyota-Fortuner-1.jpg">
                                <h3>Toyota Fortuner Legender</h3>
                                <p>Fuel Type: Diesel</p>
                                <p>Transmission: Automatic/Manual</p>
                                <p1 class="price">₹13.89L-₹23.79L</p1>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                elif search_query == 'Tata Nexon':
                    st.markdown(
                        """
                        <div class="container">
                            <div class="car-card">
                                <img src="https://www.tatamotors.com/wp-content/uploads/2023/10/Nexon-EV-MAX-DARK-Front-3-4th-1.jpg">
                                <h3>Tata Nexon</h3>
                                <p>Fuel Type: Petrol</p>
                                <p>Transmission: Manual</p>
                                <p1 class="price">₹12L</p1>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                elif search_query == 'Maruti Suzuki Brezza':
                    st.markdown(
                        """
                        <div class="container">
                            <div class="car-card">
                                <img src="https://vehiclecare.in/blaze/wp-content/uploads/2023/10/Grand-Vitara-Breeza-Front-Side-1024x536.jpg">
                                <h3>Maruti Suzuki Brezza</h3>
                                <p>Fuel Type: Petrol</p>
                                <p>Transmission: Manual</p>
                                <p1 class="price">₹12.5L</p1>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                elif search_query == 'Toyota Innova':
                    st.markdown(
                        """
                        <div class="container">
                            <div class="car-card">
                                <img src="https://imgd.aeplcdn.com/664x374/n/cw/ec/20623/innova-crysta-exterior-right-front-three-quarter.jpeg?q=80">
                                <h3>Innova Crysta</h3>
                                <p>Fuel Type: Diesel</p>
                                <p>Transmission: Manual</p>
                                <p1 class="price">₹19.99L</p1>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                # elif search_query == 'Hybrid Innova':
                    st.markdown(
                        """
                        <div class="container">
                            <div class="car-card">
                                <img src="https://www.financialexpress.com/wp-content/uploads/2022/11/Toyota-Innova-HyCross-1.jpg">
                                <h3>Hybrid Innova</h3>
                                <p>Fuel Type: Hybrid</p>
                                <p>Transmission:Automatic</p>
                                <p1 class="price">₹26L</p1>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                elif search_query == 'Hyundai i20':
                    st.markdown(
                        """
                        <div class="container">
                            <div class="car-card">
                                <img src="https://www.motorbeam.com/wp-content/uploads/2024-Hyundai-i20-N-Line-1.jpg">
                                <h3>Hyundai i20</h3>
                                <p>Fuel Type:Petrol</p>
                                <p>Transmission:Automatic</p>
                                <p1 class="price">₹10L</p1>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    
                    )
                elif search_query == 'Hyundai Grand i10':
                    st.markdown(
                        """
                        <div class="container">
                            <div class="car-card">
                                <img src="https://i.ytimg.com/vi/CHL3PJSThfk/sddefault.jpg">
                                <h3>Grand i10</h3>
                                <p>Fuel Type:Petrol</p>
                                <p>Transmission:Automatic/Manual</p>
                                <p1 class="price">₹8.84L</p1>
                            </div>1
                        </div>
                        """,
                        unsafe_allow_html=True
                    
                    )
                elif search_query == 'Tata Punch':
                    st.markdown(
                        """
                        <div class="container">
                            <div class="car-card">
                                <img src="https://www.financialexpress.com/wp-content/uploads/2024/04/Tata-Punch.jpg">
                                <h3>Tata Punch</h3>
                                <p>Fuel Type:Petrol/EV</p>
                                <p>Transmission:Automatic/Manual</p>
                                <p1 class="price">₹6L-₹9L</p1>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    
                    )
                elif search_query == 'Kia Carens':
                    st.markdown(
                        """
                        <div class="container">
                            <div class="car-card">
                                <img src="https://english.cdn.zeenews.com/sites/default/files/2021/12/16/996426-kia-carens-5.jpg">
                                <h3>Kia Carens</h3>
                                <p>Fuel Type:Petrol/Diesel</p>
                                <p>Transmission:Automatic/Manual</p>
                                <p1 class="price">₹10.60L-₹19.70L</p1>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    
                    )
                elif search_query == 'BMW M5':
                    st.markdown(
                        """
                        <div class="container">
                            <div class="car-card">
                                <img src="https://preview.redd.it/facelift-f90-m5-is-the-best-looking-and-last-true-m5-ever-v0-ilac3wec48rd1.jpg?width=1080&crop=smart&auto=webp&s=6b9bab225f94f1926ab8774f9831255ec951d8e4">
                                <h3>BMW M5 f90</h3>
                                <p>Fuel Type:Petrol</p>
                                <p>Transmission:Automatic/Manual</p>
                                <p1 class="price">₹1.60 cr</p1>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    
                    )
                elif search_query == 'Mahindra XUV 700':
                    st.markdown(
                        """
                        <div class="container">
                            <div class="car-card">
                                <img src="https://gaadiwaadi.com/wp-content/uploads/2024/06/Mahindra-XUV700-Deep-Forest_-1068x610.jpg.webp">
                                <h3>XUV 700</h3>
                                <p>Fuel Type:Diesel</p>
                                <p>Transmission:Automatic/Manual</p>
                                <p1 class="price">₹13.99L-₹25.74L</p1>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    
                    )
                elif search_query == 'Hyundai Creta':
                    st.markdown(
                        """
                        <div class="container">
                            <div class="car-card">
                                <img src="https://www.financialexpress.com/wp-content/uploads/2024/01/Hyundai-Creta-reviuew-feature.jpg">
                                <h3>Hyundai Creta</h3>
                                <p>Fuel Type:Petrol/EV</p>
                                <p>Transmission:Automatic</p>
                                <p1 class="price">₹12.62L-₹20L</p1>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    
                    )
                elif search_query == 'Honda City':
                    st.markdown(
                        """
                        <div class="container">
                            <div class="car-card">
                                <img src="https://ackodrive-assets.ackodrive.com/media/test_3ETEJVl.jpeg">
                                <h3>Honda City</h3>
                                <p>Fuel Type:Petrol/Hybrid</p>
                                <p>Transmission:Automatic/Manual</p>
                                <p1 class="price">₹13.89L-₹23.79L</p1>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                elif search_query == 'Mahindra Thar':
                    st.markdown(
                        """
                        <div class="container">
                            <div class="car-card">
                                <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/13/Mahindra_Thar_Photoshoot_At_Perupalem_Beach_%28West_Godavari_District%2CAP%2CIndia_%29_Djdavid.jpg/1200px-Mahindra_Thar_Photoshoot_At_Perupalem_Beach_%28West_Godavari_District%2CAP%2CIndia_%29_Djdavid.jpg">
                                <h3>Mahindra Thar</h3>
                                <p>Fuel Type:Petrol/Diesel</p>
                                <p>Transmission: Automatic/Manual</p>
                                <p1 class="price">₹13.89L-₹20.79L</p1>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                elif search_query == 'Volkswagen Virtus':
                    st.markdown(
                        """
                        <div class="container">
                            <div class="car-card">
                                <img src="https://cdni.autocarindia.com/Utils/ImageResizer.ashx?n=https://cdni.autocarindia.com/Galleries/20220308030907_Virtus_2022%20_1_.jpg&w=736&h=488&q=75&c=1">
                                <h3>Volkswagen Virtus</h3>
                                <p>Fuel Type:Petrol</p>
                                <p>Transmission: Automatic</p>
                                <p1 class="price">₹13L-₹21.79L</p1>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                elif search_query == 'Maruti Suzuki Baleno':
                    st.markdown(
                        """
                        <div class="container">
                            <div class="car-card">
                                <img src="https://imgd.aeplcdn.com/664x374/n/cw/ec/102663/baleno-exterior-right-front-three-quarter-68.jpeg?isig=0&q=80">
                                <h3>Maruti Suzuki Baleno</h3>
                                <p>Fuel Type: Petrol</p>
                                <p>Transmission: Automatic/Manual</p>
                                <p1 class="price">₹7.67L-₹11.28L</p1>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                elif search_query == 'Maruti Suzuki Swift':
                    st.markdown(
                        """
                        <div class="container">
                            <div class="car-card">
                                <img src="https://gaadiwaadi.com/wp-content/uploads/2024/01/maruti-swift-6.jpg">
                                <h3>Swift</h3>
                                <p>Fuel Type:Petrol</p>
                                <p>Transmission: Manual</p>
                                <p1 class="price">₹6.49L</p1>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        """
                        <div class="container">
                            <div class="car-card">
                                <img src="https://cdni.autocarindia.com/Utils/ImageResizer.ashx?n=https://cdni.autocarindia.com/ExtraImages/20241111075122_14%20_2_.jpg&w=700&c=1">
                                <h3>Swift Dzire</h3>
                                <p>Fuel Type: Petrol</p>
                                <p>Transmission: Manual</p>
                                <p1 class="price">₹6.84L</p1>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                elif search_query == 'Hyundai Venue':
                    st.markdown(
                        """
                        <div class="container">
                            <div class="car-card">
                                <img src="https://cdn.cartoq.com/photos/hyundai-venue_colours_denim-blue_74dc5b34.webp">
                                <h3>Hyundai Venue</h3>
                                <p>Fuel Type: Petrol/Diesel</p>
                                <p>Transmission:Automatic/Manual</p>
                                <p1 class="price">₹9.05L-₹15.47L</p1>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                elif search_query == 'Maruti Suzuki Ertiga':
                    st.markdown(
                        """
                        <div class="container">
                            <div class="car-card">
                                <img src="https://imgd.aeplcdn.com/664x374/n/cw/ec/115777/ertiga-exterior-left-rear-three-quarter.jpeg?isig=0&q=80">
                                <h3>Ertiga</h3>
                                <p>Fuel Type:Petrol/CNG</p>
                                <p>Transmission:Automatic/Manual</p>
                                <p1 1class="price">₹8.68L-₹13.02L</p1>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                elif search_query == 'Mahindra XUV400':
                    st.markdown(
                        """
                        <div class="container">
                            <div class="car-card">
                                <img src="https://cdni.autocarindia.com/ExtraImages/20230116045006__AAB9168.jpg">
                                <h3>Mahindra XUV400</h3>
                                <p>Fuel Type:EV</p>
                                <p>Transmission:Automatic/Manual</p>
                                <p1 class="price">₹15.49L-₹17.69L</p1>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                elif search_query=='Mahindra XUV300':
                    st.markdown(
                        """
                        <div class="container">
                            <div class="car-card">
                                <img src="https://financialexpresswpcontent.s3.amazonaws.com/uploads/2019/11/mahindra-xuv300-official.jpg">
                                <h3>Mahindra XUV300</h3>
                                <p>Fuel Type:Petrol/Diesel</p>
                                <p>Transmission:Manual/Automatic</p>
                                <p1 class="price">₹7.99L</p1>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                elif search_query == 'Tata Curvv':
                    st.markdown(
                        """
                        <div class="container">
                            <div class="car-card">
                                <img src="https://stimg.cardekho.com/images/carexteriorimages/930x620/Tata/Curvv/9578/1723033064164/front-left-side-47.jpg">
                                <h3>Tata Curvv</h3>
                                <p>Fuel Type:Petrol/Diesel</p>
                                <p>Transmission:Automatic/Manual</p>
                                <p1 class="price">₹9.99L</p1>
                                <p>(Varies with location and varient)</p>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                elif search_query == 'Mahindra BE 6':
                    st.markdown(
                        """
                        <div class="container">
                            <div class="car-card">
                                <img src="https://imgd.aeplcdn.com/664x374/n/cw/ec/131825/be-6e-exterior-left-front-three-quarter-3.jpeg?isig=0&q=80">
                                <h3> Mahindra BE 6 </h3>
                                <p>Fuel Type:EV (59kWh & 79kWh)</p>
                                <p>Transmission:Automatic/Manual</p>
                                <p1 class="price">₹18.90L-₹26.90L</p1>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                elif search_query == 'MG Hector':
                    st.markdown(
                        """
                        <div class="container">
                            <div class="car-card">
                                <img src="https://imgd.aeplcdn.com/1920x1080/n/cw/ec/130583/hector-exterior-right-front-three-quarter-73.jpeg?isig=0&q=80&q=80">
                                <h3>MG Hector</h3>
                                <p>Fuel Type:Petrol/Diesel</p>
                                <p>Transmission:Manual</p>
                                <p1 class="price">₹14L-₹23.09L</p1>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                elif search_query == 'Tata Tiago':
                    st.markdown(
                        """
                        <div class="container">
                            <div class="car-card">
                                <img src="https://stimg.cardekho.com/images/carexteriorimages/630x420/Tata/Tiago/10655/1738146879386/front-left-side-47.jpg">
                                <h3>Tata Tiago</h3>
                                <p>Fuel Type:Petrol/CNG</p>
                                <p>Transmission:Automatic/Manual</p>
                                <p1 class="price">₹5L-₹8.45L</p1>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                elif search_query == 'Mercedes G-Class':
                    st.markdown(
                        """
                        <div class="container">
                            <div class="car-card">
                                <img src="https://images.overdrive.in/wp-content/uploads/2024/04/24C0076_0041-900x506.jpg">
                                <h3>G-Class (G-Wagen)</h3>
                                <p>Fuel Type:EV (100+kWh)</p>
                                <p>Transmission:Automatic</p>
                                <p1 class="price">₹2.55cr-₹4.00cr</p11>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        """
                        <div class="container">
                            <div class="car-card">
                                <img src="https://imgd.aeplcdn.com/664x374/n/cw/ec/1/versions/mercedes-benz-g-class-amg-g-63-grand-edition1695818429621.jpg?q=80">
                                <h3>G-Class (G-Wagen)</h3>
                                <p>Fuel Type:Petrol/Diesel</p>
                                <p>Transmission:Automatic</p>
                                <p1 class="price">₹3.00cr-above</p1>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                        
                    
                # if search_query:
                #     search_results = df[df["Car"].str.contains(search_query, case=False, na=False)]
                #     if not search_results.empty:
                #         st.subheader("Search Results:")
                #         for car in search_results["Car"].tolist():
                #             st.write(f"- {car}")
                #     else:
                #         st.write("No results found.")
                


                        # Search Logic (Only when button is clicked)
                
                    
        #        
        # Car Data
                st.markdown("<h1 style='text-align: center; color: #00aaff;'> Car Recommendations</h1>", unsafe_allow_html=True)
                cars = [
                    {
                        "name": "Mercedes-Benz A-Class",
                        "image": "https://cdni.autocarindia.com/Utils/ImageResizerV2.ashx?n=https://cms.haymarketindia.net/model/uploads/modelimages/Mercedes-Benz-A-class-Hatchback-210920221745.jpg&w=872&h=578&q=75&c=1",
                        "fuel": "Petrol",
                        "transmission": "Automatic",
                        "price": "₹80L"
                    },
                    {
                        "name": "BMW X5",
                        "image": "https://stimg.cardekho.com/images/carexteriorimages/930x620/BMW/X5-2023/10452/1688992642182/front-left-side-47.jpg",
                        "fuel": "Diesel",
                        "transmission": "Manual",
                        "price": "₹85L"
                    },
                    {
                        "name": "Audi Q3",
                        "image": "https://i.ytimg.com/vi/TipJqbvutiI/hq720.jpg?sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&rs=AOn4CLDoIhJcZi-BYKXWk-bQsnSTQj2UZw",
                        "fuel": "Petrol",
                        "transmission": "Automatic",
                        "price": "₹50L"
                    },
                    {
                        "name": "Volkswagen Golf",
                        "image": "https://www.griffinsautorepair.com/wp-content/uploads/2020/07/2016-Volkswagen-Golf-GTI-Clubsport-001-1080.jpg",
                        "fuel": "Petrol",
                        "transmission": "Automatic",
                        "price": "₹25L"
                    },
                    {
                        "name": "Jeep Wrangler Rubicon",
                        "image": "https://i0.wp.com/moparinsiders.com/wp-content/uploads/2023/02/2023-Jeep%C2%AE-Wrangler-Unlimited-Rubicon-392-20th-Anniversary-Edition.-Jeep.-6-scaled.jpeg?fit=2560%2C1440&ssl=1",
                        "fuel": "Petrol",
                        "transmission": "Automatic, Manual override",
                        "price": "₹71.65L"
                    },
                    {
                        "name": "Kia Seltos",
                        "image": "https://images.carexpert.com.au/resize/3000/-/app/uploads/2023/08/Kia-Seltos-S-Stills-14.jpg",
                        "fuel": "Petrol",
                        "transmission": "Manual",
                        "price": "₹12.88L"
                    },
                    {
                        "name": "Mahindra Thar",
                        "image": "https://www.motorguider.com/wp-content/uploads/2022/06/Mahindra-THAR-2022-Price-in-Pakistan-1024x538.jpg",
                        "fuel": "Petrol/Diesel",
                        "transmission": "Manual/Automatic",
                        "price": "₹13.21L-₹20.31L"
                    },
                    {
                        "name": "Lamborghini Urus SE",
                        "image": "https://st.automobilemag.com/uploads/sites/5/2018/04/2019-Lamborghini-Urus-in-Paris-3.jpg",
                        "fuel": "Petrol",
                        "transmission": "Automatic",
                        "price": "₹4.57cr"
                    },
                    {
                        "name": "Toyota Fortuner",
                        "image": "https://th.bing.com/th/id/OIP.M3ndtRdLD6Ug2abeH67iugHaEa?rs=1&pid=ImgDetMain",
                        "fuel": "Diesel",
                        "transmission": "Manual/Automatic",
                        "price": "₹49.55L"
                    },
                    {
                        "name": "Audi A8",
                        "image": "https://www.financialexpress.com/wp-content/uploads/2022/07/2022-Audi-A8-L.jpg",
                        "fuel": "Petrol",
                        "transmission": "Automatic",
                        "price": "₹1.34cr-₹1.63cr"
                    },
                    {
                        "name": "Porsche 911",
                        "image": "https://imgd.aeplcdn.com/1200x900/n/cw/ec/178277/porsche-911-left-front-three-quarter1.jpeg?isig=0&wm=0",
                        "fuel": "Petrol",
                        "transmission": "Manual/Automatic",
                        "price": "₹1.86cr-₹3.54cr"
                    },
                    {
                        "name": "BMW 3 Series",
                        "image": "https://acko-cms.ackoassets.com/1_7cca4a17d2.jpg",
                        "fuel": "Petrol/Diesel",
                        "transmission": "Manual/Automatic",
                        "price": "₹72.90L-₹73.50L"
                    },
                    {
                        "name": "Mercedes-Benz C-Class",
                        "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTsRJVWqCAicIYhq7FYOTuCrxrgVCAOvZWf5w&s",
                        "fuel": "Petrol/Diesel",
                        "transmission": "9-Speed Automatic",
                        "price": "₹60.00L-₹66.00L"
                    },
                    {
                        "name": "Audi Q5",
                        "image": "https://www.nordiskbil.com/wp-content/uploads/2024/11/Audi-Q5-Sportback-main.jpg",
                        "fuel": "Petrol/Diesel",
                        "transmission": "Automatic",
                        "price": "₹65.18L-₹70.45L"
                    },
                    {
                        "name": "Hyundai kona",
                        "image": "https://i.cdn.newsbytesapp.com/images/l78120230307135733.jpeg",
                        "fuel": "Electric",
                        "transmission": "Single-Speed Automatic",
                        "price": "₹23.84L-₹24.03L"
                    },
                    {
                        "name": "Maruti Suzuki Brezza",
                        "image": "https://stimg.cardekho.com/images/carexteriorimages/930x620/Maruti/Brezza/10388/1694424068944/rear-left-view-121.jpg",
                        "fuel": "Petrol/CNG",
                        "transmission": "5-speed Manual/6-speed Automatic",
                        "price": "₹8.34L-₹14.14L"
                    },
                    {
                        "name": "Skoda Slavia",
                        "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRNuCMIm1fE0EYWakDjMClozWQJIDSk7LvUyA&s",
                        "fuel": "Petrol",
                        "transmission": "6-speed Manual/6-speed Automatic",
                        "price": "₹11.63L-₹19.12L"
                    },
                    {
                        "name": "Honda City",
                        "image": "https://imgd.aeplcdn.com/1056x594/n/7y87bab_1649261.jpg?q=80",
                        "fuel": "Petrol",
                        "transmission": "Manual/Automatic",
                        "price": "₹11.82L-₹16.30L"
                    },
                    {
                        "name": "Honda City",
                        "image": "https://cdni.autocarindia.com/Utils/ImageResizer.ashx?n=https://cdni.autocarindia.com/ExtraImages/20220408071127_Honda_City_Hybrid_front.jpg",
                        "fuel": "Hybrid(Petrol+EV)",
                        "transmission": "e-CVT Automatic",
                        "price": "₹19.00L-₹20.39L"
                    },
                    {
                        "name": "Tata Tigor",
                        "image": "https://images.hindustantimes.com/auto/img/2021/08/25/600x338/Tata_Tigor_EV_1629267387341_1629870198189.jpeg",
                        "fuel": "Electric",
                        "transmission": "Automatic",
                        "price": "₹12.49L-₹13.75L"
                    },
                    {
                        "name": "Tata Tigor",
                        "image": "https://gaadiwaadi.com/wp-content/uploads/2017/08/tata-tigor-petrol-review-3.jpg",
                        "fuel": "Petrol/CNG",
                        "transmission": "Manual",
                        "price": "₹6.30L-₹9.02L"
                    },
                    {
                        "name": "Mahindra BE6",
                        "image": "https://imgd.aeplcdn.com/664x374/n/cw/ec/131825/be-6e-exterior-left-front-three-quarter-3.jpeg?isig=0&q=80",
                        "fuel": "EV(~79kWh)",
                        "transmission": "Automatic",
                        "price": "₹18.90L-₹26.90L"
                    },
                    {
                        "name": "Skoda Kushaq",
                        "image": "https://imgd.aeplcdn.com/1920x1080/n/cw/ec/175993/kushaq-exterior-right-front-three-quarter.jpeg?isig=0&q=80&q=80",
                        "fuel": "Petrol",
                        "transmission": "6-Speed Automatic",
                        "price": "₹11.89L-₹20.49L"
                    },
                    {
                        "name": "Mercedes G-Wagen(AMG G 63)",
                        "image": "https://img-ik.cars.co.za/news-site-za/images/2024/11/Mercedes-AMG-G63-10.jpg?tr=w-800",
                        "fuel": "Petrol",
                        "transmission": "9-speed Automatic",
                        "price": "₹3.30cr"
                    },
                    {
                        "name": "Renault Kwid",
                        "image": "https://content.carlelo.com/uploads/model/kwid-1.webp",
                        "fuel": "Petrol",
                        "transmission": "Manual/Automatic",
                        "price": "₹4.70L-₹6.33L"
                    },
                    {
                        "name": "Hyundai Verna",
                        "image": "https://imgd-ct.aeplcdn.com/664x415/n/ectneab_1652035.jpg?q=80",
                        "fuel": "Petrol",
                        "transmission": "Manual/Automatic",
                        "price": "₹11L-₹17.42L"
                        "(varies by varients and city)"
                    }
                ]

                # Shuffle car order so a different one appears on top
                random.shuffle(cars)

                # Generate HTML dynamically
                html_code = """
                <style>
                    .container {
                        max-width: 1200px;
                        margin: auto;
                        text-align: center;
                        padding: 10px;
                    }
                    .scroll-container {
                        display: flex;
                        flex-wrap: wrap;
                        gap: 15px;
                        max-height: 600px;
                        overflow: auto;
                    }
                    .car-card {
                        background: white;
                        padding: 15px;
                        border-radius: 12px;
                        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                        transition: 0.3s ease-in-out;
                        text-align: center;
                        width: 280px;
                    }
                    .car-card:hover {
                        transform: scale(1.05);
                        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
                    }
                    .car-card img {
                        width: 100%;
                        height: 160px;
                        border-radius: 12px;
                        object-fit: cover;
                    }
                    .car-card h3 {
                        font-size: 20px;
                        color: #222;
                        margin-top: 10px;
                    }
                    .car-card p {
                        font-size: 14px;
                        color: #555;
                        margin: 5px 0;
                    }
                    .price {
                        font-size: 18px;
                        font-weight: bold;
                        color: #007bff;
                    }
                </style>
                <div class="container">
                    <div class="scroll-container">
                """

                for car in cars:
                    html_code += f"""
                        <div class="car-card">
                            <img src="{car['image']}" alt="{car['name']}">
                            <h3>{car['name']}</h3>
                            <p>Fuel Type: {car['fuel']}</p>
                            <p>Transmission: {car['transmission']}</p>
                            <p class="price">{car['price']}</p>
                        </div>
                    """

                html_code += """
                    </div>
                </div>
                """

                # Display HTML component
                components.html(html_code, height=600, scrolling=False)

                # Refresh every 10 seconds
                time.sleep(10)
                # st.experimental_rerun()
        # elif page == "Contact":
               