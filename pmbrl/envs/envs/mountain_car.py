# pmbrl/envs/envs/mountain_car.py

import math
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from PIL import Image, ImageDraw

class SparseMountainCarEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, goal_velocity=0):
        super(SparseMountainCarEnv, self).__init__()

        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.45
        self.goal_velocity = goal_velocity

        self.power = 0.0015

        self.low_state = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high_state = np.array([self.max_position, self.max_speed], dtype=np.float32)

        self.viewer = None

        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state, high=self.high_state, dtype=np.float32
        )

        self.seed()
        self.reset()
    
    def get_env_name(self):
        return self.__class__.__name__

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        position, velocity = self.state
        force = np.clip(action[0], self.min_action, self.max_action)

        velocity += force * self.power - 0.0025 * math.cos(3 * position)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)

        if (position == self.min_position and velocity < 0):
            velocity = 0.0

        done = bool(position >= self.goal_position and velocity >= self.goal_velocity)

        reward = 0.0
        if done:
            reward = 1.0  # Sparse reward only upon reaching the goal

        self.state = np.array([position, velocity], dtype=np.float32)
        return self.state, reward, done, {}

    def reset(self):
        position = self.np_random.uniform(low=-0.6, high=-0.4)
        velocity = 0.0
        self.state = np.array([position, velocity], dtype=np.float32)
        return self.state

    def _height(self, xs):
        return np.sin(3 * xs) * 0.45 + 0.55


    def render(self, mode='human'):
        """Renders the environment.

        Args:
            mode (str): The mode to render with. 'human' or 'rgb_array'.

        Returns:
            If mode is 'rgb_array', returns an np.array with shape (X, Y, 3), representing RGB pixel values.
            Otherwise, returns None.
        """
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        carwidth = 40
        carheight = 20

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)

            # Draw the track
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs - self.min_position) * scale, ys * scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            # Draw the car
            clearance = 10

            l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)

            # Front wheel
            frontwheel = rendering.make_circle(carheight / 2.5)
            frontwheel.set_color(0.5, 0.5, 0.5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth / 4, clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)

            # Back wheel
            backwheel = rendering.make_circle(carheight / 2.5)
            backwheel.set_color(0.5, 0.5, 0.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth / 4, clearance)))
            backwheel.add_attr(self.cartrans)
            self.viewer.add_geom(backwheel)

            # Flag at the goal
            flagx = (self.goal_position - self.min_position) * scale
            flagy1 = self._height(self.goal_position) * scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            flagpole.set_color(0, 0, 0)
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon(
                [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
            )
            flag.set_color(0.8, 0.8, 0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation(
            (pos - self.min_position) * scale, self._height(pos) * scale
        )
        self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
