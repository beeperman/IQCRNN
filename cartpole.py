import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

class Env(object):

    def __init__(self, factor=1):

        self.factor = factor
        self.viewer = None
        self.dt = 0.02 # sampling time

        # maximum control input
        self.max_control = 2

        # discrete-time model for control synthesis
        self.AG = np.array([
            [1.0, -0.001,  0.02, 0.0],
            [0.0,  1.005,  0.0,  0.02],
            [0.0, -0.079,  1.0, -0.001],
            [0.0,  0.550,  0.0,  1.005]
        ])
        #self.BG = np.array([
        #    [0.0], [0.0], [0.008], [-0.008]
        #]) * factor
        self.BG = np.array([
            [0.0], [0.0], [0.04], [-0.04]
        ]) * factor
        self.CG = None

        self.nx = self.AG.shape[0]
        self.nu = self.BG.shape[1]

        self.time = 0

        self.action_space = spaces.Box(low=-self.max_control, high=self.max_control, shape=(self.nu,))

        ts = 1 # testing scale
        self.x1lim = 1.0 * factor * ts; self.x2lim = np.pi/2 * factor * ts; self.x3lim = 5.0 * factor * ts; self.x4lim = 2.0 * np.pi * factor * ts # xmax limits
        xmax = np.array([self.x1lim, self.x2lim, self.x3lim, self.x4lim])
        self.observation_space = spaces.Box(low=-xmax, high=xmax)
        self.state_space = spaces.Box(low=-xmax, high=xmax)

        self.seed()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):
        x1, x2, x3, x4 = self.state

        u = np.clip(u, -self.max_control, self.max_control)[0]
        costs = 1/self.factor**2 * (1.0 * x1**2 + 1.0 * x2**2 + 0.04 * x3**2 + 0.1 * x4**2 + 0.2 * (u * self.factor)**2) - 5.0

        self.state = self.AG @ self.state + self.BG @ [u]

        terminated = False
        if self.time > 200 or not self.state_space.contains(self.state):
            terminated = True

        self.time += 1

        return self.get_obs(), -costs, terminated, {}

    def reset(self):
        #high = np.array([self.x1lim, self.x2lim, self.x3lim, self.x4lim]) / 2.0 / 10
        # xlim 1, pi/2, 5, pi
        high = np.array([0.05, 0.05, 0.25, 0.15])# / 4 / 4
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        self.time = 0

        return self.get_obs()

    def get_obs(self):
        return  self.state

class Obs_Env(Env):
    def __init__(self, factor=1):
        super().__init__(factor)
        #self.CG = np.array([
        #    [1, 0, 0, 0],
        #    [0, 1, 0, 1]
        #])
        self.CG = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # observations are the first state: pos
        xmax = np.array([self.x1lim, self.x2lim])
        self.observation_space = spaces.Box(low=-xmax, high=xmax)

    def get_obs(self):
        return self.CG @ self.state

class Obs_Norm_Env(Obs_Env):
    def __init__(self, factor=1):
        super().__init__(factor)
        self.CG = self.CG / self.observation_space.high[:, np.newaxis]
