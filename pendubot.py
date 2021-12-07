import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

class Env(object):

    def __init__(self, factor=1):

        self.factor = factor
        self.viewer = None
        self.dt = 0.01 # sampling time

        # maximum control input
        self.control_scale = 1
        self.max_control = 1.0 * self.control_scale * 2

        # Pendubot dynamics from Section V-A in https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9345365
        # x1-theta1, x2-theta1dot, x3-theta2, x4-theta2dot
        # continuous-time dynamics xdot = Ac*x + Bc*u
        Ac = np.array([[0, 1, 0, 0], [67.38, 0, -24.83, 0], [0, 0, 0, 1], [-69.53, 0, 105.32, 0]])
        Bc = np.array([[0], [44.87], [0], [-85.09]]) / self.control_scale

        # discrete-time system
        self.AG = Ac * self.dt + np.eye(4)
        self.BG = Bc * self.dt * factor

        self.CG = None

        self.nx = self.AG.shape[0]
        self.nu = self.BG.shape[1]

        self.time = 0

        self.action_space = spaces.Box(low=-self.max_control, high=self.max_control, shape=(self.nu,))

        ts = 1
        self.x1lim = 1.0 * factor*ts; self.x2lim = 2.0 * factor*ts * 10; self.x3lim = 1.0 * factor*ts; self.x4lim = 4.0 * factor*ts * 5 # xmax limits
        xmax = np.array([self.x1lim, self.x2lim, self.x3lim, self.x4lim])
        self.observation_space = spaces.Box(low=-xmax, high=xmax)
        self.state_space = spaces.Box(low=-xmax, high=xmax)

        self.seed()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):
        x1, x2, x3, x4 = self.state
        u0 = u
        u = np.clip(u, -self.max_control, self.max_control)[0]
        costs = 1/self.factor**2 * (1.0 * x1**2 + 0.05 * x2**2 + 1.0 * x3**2 + 0.05 * x4**2 + 0.2 * (u / self.control_scale * self.factor)**2) - 5.0

        self.state = self.AG @ self.state + self.BG @ [u]

        terminated = False
        if self.time > 200 or not self.state_space.contains(self.state):
            terminated = True

        self.time += 1

        return self.get_obs(), -costs, terminated, {}

    def reset(self):
        high = np.array([0.05, 0.1, 0.05, 0.1]) * self.factor / 1
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        self.time = 0

        return self.get_obs()

    def get_obs(self):
        return  self.state

class Obs_Env(Env):
    def __init__(self, factor=1):
        super().__init__(factor)
        self.obs_scale = 1
        self.CG = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0]
        ]) * self.obs_scale

        # observations are the first state: pos
        xmax = np.array([self.x1lim, self.x3lim]) * self.obs_scale
        self.observation_space = spaces.Box(low=-xmax, high=xmax)

    def get_obs(self):
        return self.CG @ self.state

class Obs_Norm_Env(Obs_Env):
    def __init__(self, factor=1):
        super().__init__(factor)
        self.CG = self.CG / self.observation_space.high[:, np.newaxis]

