# Begin file: pmbrl/control/agent.py
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
        self.current_goal = None  # Initialize current_goal

    def get_seed_episodes(self, buffer, n_episodes):
        for _ in range(n_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.env.sample_action()
                next_state, reward, done, _ = self.env.step(action)
                buffer.add(state, action, reward, next_state)
                state = deepcopy(next_state)
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
                        recorder.capture_frame()
                except AttributeError as e:
                    self.logger.log(f"AttributeError: {e}")

                state = deepcopy(next_state)
                
                self.render_and_save_frame(state, action, folder_name, total_steps)

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
        Render the current frame, paint the goal and markers, and save the image.

        Args:
        state: Current state of the environment
        folder_name: Directory to save the frames
        step: Current step number
        """
        # Render the frame
        frame = self.env.unwrapped.render(mode="rgb_array")
        goal = self.current_goal
        # self.logger.log(f"goal : {goal}")

        if frame is not None:
            img = Image.fromarray(frame)
            draw = ImageDraw.Draw(img)

            # Get frame dimensions
            height, width = frame.shape[:2]

            # Correct calculation for the goal state using cos(theta) and sin(theta)
            # We assume the goal is represented in the same way as the pendulum's state: [cos(goal_theta), sin(goal_theta)]
            goal_theta = np.arctan2(goal[1], goal[0])

            # Flip the sign of goal_theta to correct the orientation
            goal_theta = -goal_theta

            # Calculate goal position in the image
            goal_x = int(width / 2 + np.sin(goal_theta) * (height / 4))  # Similar logic to the pendulum's x position
            goal_y = int(height / 2 - np.cos(goal_theta) * (height / 4))  # Similar logic to the pendulum's y position

            # Draw the goal state on the image (green)
            draw.ellipse([goal_x-5, goal_y-5, goal_x+5, goal_y+5], fill=(0, 255, 0), outline=(0, 255, 0))

            # Draw markers at (0,0) and (1000,1000) (red) - Reference points
            draw.ellipse([0-5, 0-5, 0+5, 0+5], fill=(255, 0, 0), outline=(255, 0, 0))
            draw.ellipse([1000-5, 1000-5, 1000+5, 1000+5], fill=(255, 0, 0), outline=(255, 0, 0))

            # Corrected calculation for the pendulum's free end (orange)
            # For Pendulum-v0, state[0] is cos(theta), state[1] is sin(theta)
            theta = np.arctan2(state[1], state[0])

            # Flip the sign of theta to correct the orientation
            theta = -theta

            # Calculate pendulum's free end position, bringing the orange dot closer by reducing the height scaling
            pendulum_x = int(width / 2 + np.sin(theta) * (height / 4))  # Reduced length factor from height / 3 to height / 4
            pendulum_y = int(height / 2 - np.cos(theta) * (height / 4))  # Reduced length factor here as well

            # Debugging print statements to check positions
            # print(f"Step: {step}, Pendulum position: ({pendulum_x}, {pendulum_y}), State: {state}, Action: {action}")
            # print(f'Goal position: ({goal_x}, {goal_y}), Goal state : {self.current_goal}')

            # Draw the current state (pendulum's free end) (orange)
            draw.ellipse([pendulum_x-5, pendulum_y-5, pendulum_x+5, pendulum_y+5], fill=(255, 165, 0), outline=(255, 165, 0))

            # Save the modified frame as an image
            img.save(f"{folder_name}/frame_{step}.png")
        else:
            print(f"Warning: Frame {step} could not be rendered.")
# End file: pmbrl/control/agent.py
