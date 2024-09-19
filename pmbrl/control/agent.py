# control/agent.py
from copy import deepcopy

import os
import numpy as np
import torch
from PIL import Image, ImageDraw


class Agent(object):
    def __init__(self, env, planner, logger=None):
        self.env = env
        self.planner = planner
        self.logger = logger
        self.current_goal = None

    def get_seed_episodes(self, buffer, n_episodes):
        for _ in range(n_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.env.sample_action()
                next_state, reward, done, _ = self.env.step(action)
                buffer.add(state, action, reward, next_state)
                state = next_state.copy()
                if done:
                    break
        return buffer

    def run_episode(self, buffer=None, action_noise=None, recorder=None):
        total_reward = 0
        total_steps = 0
        done = False

        # Prepare folder for saving frames
        if recorder is not None:
            folder_name = os.path.splitext(recorder.path)[0]  # Remove extension from video path
            os.makedirs(folder_name, exist_ok=True)  # Create folder with the same name as the video file
                
        with torch.no_grad():
            state = self.env.reset()
            self.current_goal = self.planner.global_goal_state.cpu().numpy()  # Initialize current_goal
            while not done:
                action, current_goal = self.planner(state)  # Receive both action and current_goal
                self.current_goal = current_goal.cpu().numpy()  # Update current_goal

                if action_noise is not None:
                    action = self._add_action_noise(action, action_noise)
                action = action.cpu().detach().numpy()

                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                total_steps += 1

                if self.logger is not None and total_steps % 25 == 0:
                    self.logger.log(
                        "> Step {} [reward {:.2f}]".format(total_steps, total_reward)
                    )

                if buffer is not None:
                    buffer.add(state, action, reward, next_state)
                try:
                    if recorder is not None:
                        self.logger.log(f"Recording frame {total_steps}")
                        recorder.capture_frame()
                except AttributeError as e:
                    self.logger.log(f"AttributeError: {e}")

                state = deepcopy(next_state)
                
                # self.render_and_save_frame(state, action, folder_name, total_steps)

                if done:
                    break

        if recorder is not None:
            recorder.close()
            del recorder

        self.env.close()
        stats = self.planner.return_stats()
        return total_reward, total_steps, stats

    def _add_action_noise(self, action, noise):
        if noise is not None:
            action = action + noise * torch.randn_like(action)
        return action

    def render_and_save_frame(self, state, action, folder_name, step):
        """
        Render the current frame, draw the goal and current state, and save the image.

        Args:
            state: Current state of the environment
            action: Action taken
            folder_name: Directory to save the frames
            step: Current step number
        """
        # Render the frame
        frame = self.env.unwrapped.render(mode="rgb_array")
        if frame is None:
            print(f"Warning: Frame {step} could not be rendered.")
            return

        # Create PIL image
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)

        # Get environment name
        env_name = self.env.unwrapped.spec.id if self.env.unwrapped.spec else 'Unknown'

        if env_name == 'Pendulum-v0':
            self._render_pendulum(draw, frame, state)
        elif env_name in ['MountainCarContinuous-v0', 'SparseMountainCar']:
            self._render_mountain_car(draw, frame, state)
        elif env_name in ['HalfCheetahRun', 'HalfCheetahFlip']:
            self._render_half_cheetah(draw, frame, state)
        elif env_name == 'AntMaze':
            self._render_antmaze(draw, frame, state)
        elif env_name in ['DeepMindCatch', 'DeepMindReacher']:  # Adjust as per actual env name
            self._render_dmcatch(draw, frame, state)
        else:
            # For environments without specific rendering, skip
            pass

        # Save the modified frame as an image
        img.save(f"{folder_name}/frame_{step}.png")

    def _render_pendulum(self, draw, frame, state):
        """
        Render the Pendulum environment with the goal and current state.

        Args:
            draw: PIL ImageDraw object
            frame: Frame image array
            state: Current state of the environment
        """
        goal = self.current_goal

        # Get frame dimensions
        height, width = frame.shape[:2]

        # Correct calculation for the goal state using cos(theta) and sin(theta)
        goal_theta = np.arctan2(goal[1], goal[0])

        # Flip the sign of goal_theta to correct the orientation
        goal_theta = -goal_theta

        # Calculate goal position in the image
        goal_x = int(width / 2 + np.sin(goal_theta) * (height / 4))
        goal_y = int(height / 2 - np.cos(goal_theta) * (height / 4))

        # Draw the goal state on the image (green dot)
        draw.ellipse([goal_x - 5, goal_y - 5, goal_x + 5, goal_y + 5],
                    fill=(0, 255, 0), outline=(0, 255, 0))

        # Pendulum state processing
        theta = np.arctan2(state[1], state[0])
        theta = -theta

        # Pendulum position in the image
        pendulum_x = int(width / 2 + np.sin(theta) * (height / 4))
        pendulum_y = int(height / 2 - np.cos(theta) * (height / 4))

        # Draw the current state (orange dot)
        draw.ellipse([pendulum_x - 5, pendulum_y - 5, pendulum_x + 5, pendulum_y + 5],
                    fill=(255, 165, 0), outline=(255, 165, 0))

    def _render_mountain_car(self, draw, frame, state):
        """
        Render the MountainCar environment with the goal and current state.

        Args:
            draw: PIL ImageDraw object
            frame: Frame image array
            state: Current state of the environment
        """
        goal = self.current_goal

        # Get frame dimensions
        height, width = frame.shape[:2]

        # MountainCar state consists of position and velocity
        position = state[0]
        min_position = self.env.unwrapped.min_position
        max_position = self.env.unwrapped.max_position

        # Map position to pixel coordinates
        position_norm = (position - min_position) / (max_position - min_position)
        car_x = int(position_norm * width)
        car_y = int(height * 0.75)  # Approximate y position

        # Draw the car position (orange dot)
        draw.ellipse([car_x - 5, car_y - 5, car_x + 5, car_y + 5],
                    fill=(255, 165, 0), outline=(255, 165, 0))

        # Goal position
        goal_position = self.env.unwrapped.goal_position
        goal_position_norm = (goal_position - min_position) / (max_position - min_position)
        goal_x = int(goal_position_norm * width)
        goal_y = int(height * 0.75)

        # Draw the goal position (green dot)
        draw.ellipse([goal_x - 5, goal_y - 5, goal_x + 5, goal_y + 5],
                    fill=(0, 255, 0), outline=(0, 255, 0))

        # Add red markers at the corners and center for debugging
        # Top-left corner
        draw.ellipse([0 - 5, 0 - 5, 0 + 5, 0 + 5],
                    fill=(255, 0, 0), outline=(255, 0, 0))
        # Top-right corner
        draw.ellipse([width - 5, 0 - 5, width + 5, 0 + 5],
                    fill=(255, 0, 0), outline=(255, 0, 0))
        # Bottom-left corner
        draw.ellipse([0 - 5, height - 5, 0 + 5, height + 5],
                    fill=(255, 0, 0), outline=(255, 0, 0))
        # Bottom-right corner
        draw.ellipse([width - 5, height - 5, width + 5, height + 5],
                    fill=(255, 0, 0), outline=(255, 0, 0))
        # Center of the frame
        center_x = width // 2
        center_y = height // 2
        draw.ellipse([center_x - 5, center_y - 5, center_x + 5, center_y + 5],
                    fill=(255, 0, 0), outline=(255, 0, 0))

    def _render_half_cheetah(self, draw, frame, state):
        """
        Render the HalfCheetah environment with the goal.

        Args:
            draw: PIL ImageDraw object
            frame: Frame image array
            state: Current state of the environment
        """
        # For HalfCheetah, the goal is to move forward.
        # We can draw an arrow or a dot ahead of the cheetah.

        # Get frame dimensions
        height, width = frame.shape[:2]

        # Assuming we can get the x position of the cheetah
        # In the default environment, x position is not part of the state, but can be retrieved from env
        try:
            x_position = self.env.unwrapped.sim.data.qpos[0]
        except AttributeError:
            x_position = 0

        # Current x position mapped to pixel coordinates
        # This mapping is arbitrary, adjust as needed
        x_min = x_position - 5  # Display a window around the current position
        x_max = x_position + 5
        x_norm = (x_position - x_min) / (x_max - x_min)
        cheetah_x = int(x_norm * width)
        cheetah_y = int(height / 2)

        # Draw the cheetah position (orange dot)
        draw.ellipse([cheetah_x - 5, cheetah_y - 5, cheetah_x + 5, cheetah_y + 5],
                    fill=(255, 165, 0), outline=(255, 165, 0))

        # Goal is ahead of the cheetah
        goal_x_position = x_position + 2  # 2 units ahead
        goal_x_norm = (goal_x_position - x_min) / (x_max - x_min)
        goal_x = int(goal_x_norm * width)
        goal_y = cheetah_y

        # Draw the goal position (green dot)
        draw.ellipse([goal_x - 5, goal_y - 5, goal_x + 5, goal_y + 5],
                    fill=(0, 255, 0), outline=(0, 255, 0))

        # Optionally, draw an arrow indicating direction
        draw.line([(cheetah_x, cheetah_y), (goal_x, goal_y)],
                fill=(0, 255, 0), width=2)

    def _render_antmaze(self, draw, frame, state):
        """
        Render the AntMaze environment with the goal.

        Args:
            draw: PIL ImageDraw object
            frame: Frame image array
            state: Current state of the environment
        """
        # For AntMaze, since there's no specific goal, we can skip rendering the goal.
        # Alternatively, if there is a known goal position, we can draw it.

        # Get frame dimensions
        height, width = frame.shape[:2]

        # Ant position
        x = state[2]  # x position
        y = state[3]  # y position

        # Map positions to pixel coordinates
        x_min, x_max = -5, 5
        y_min, y_max = -5, 5
        x_norm = (x - x_min) / (x_max - x_min)
        y_norm = (y - y_min) / (y_max - y_min)
        ant_x = int(x_norm * width)
        ant_y = int((1 - y_norm) * height)

        # Draw the ant position (orange dot)
        draw.ellipse([ant_x - 5, ant_y - 5, ant_x + 5, ant_y + 5],
                    fill=(255, 165, 0), outline=(255, 165, 0))

    def _render_dmcatch(self, draw, frame, state):
        """
        Render the DeepMind Catch environment with the goal.

        Args:
            draw: PIL ImageDraw object
            frame: Frame image array
            state: Current state of the environment
        """
        # For DM Catch, the goal is to catch the ball with the cup.
        # We can draw the target position where the ball should be caught.

        # Get frame dimensions
        height, width = frame.shape[:2]

        # Assuming state includes positions of the ball and cup
        # For illustration purposes, let's assume:
        # state[0]: cup position x
        # state[1]: cup position y
        # state[2]: ball position x
        # state[3]: ball position y

        # Cup position
        cup_x = int(state[0] * width)
        cup_y = int((1 - state[1]) * height)

        # Ball position
        ball_x = int(state[2] * width)
        ball_y = int((1 - state[3]) * height)

        # Draw the cup position (orange dot)
        draw.ellipse([cup_x - 5, cup_y - 5, cup_x + 5, cup_y + 5],
                    fill=(255, 165, 0), outline=(255, 165, 0))

        # Draw the ball position (green dot)
        draw.ellipse([ball_x - 5, ball_y - 5, ball_x + 5, ball_y + 5],
                    fill=(0, 255, 0), outline=(0, 255, 0))

        # Draw a line between cup and ball
        draw.line([(cup_x, cup_y), (ball_x, ball_y)], fill=(255, 0, 0), width=2)
