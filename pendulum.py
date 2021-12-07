import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

class Env(object):

    def __init__(self, factor=1):

        self.factor = factor
        self.viewer = None
        self.g = 10.0
        self.m = 0.15
        self.l = 0.5
        self.mu = 0.05
        self.dt = 0.02
        self.max_torque = 2 #
        self.max_speed = 8.0 * factor
        self.max_pos = 1.5 * factor

        self.AG = np.array([
            [1, self.dt],
            [self.g/self.l*self.dt, 1-self.mu/(self.m*self.l**2)*self.dt]
        ])
        self.BG = np.array([[0], [self.dt/(self.m*self.l**2)]]) * factor
        self.CG = None

        # env setup

        self.nx = self.AG.shape[0]
        self.nu = self.BG.shape[1]

        self.time = 0

        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(self.nu,))
        # observations are the two states
        xmax = np.array([self.max_pos, self.max_speed])
        self.observation_space = spaces.Box(low=-xmax, high=xmax)
        self.state_space = spaces.Box(low=-xmax, high=xmax)

        self.seed()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):
        th, thdot = self.state

        g = self.g
        m = self.m
        l = self.l
        mu = self.mu
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        costs = 1/self.factor**2*(th**2 + .1*thdot**2 + 1*((u*self.factor)**2)) - 1

        self.state = self.AG @ self.state + self.BG @ [u]

        terminated = False
        if self.time > 200 or not self.state_space.contains(self.state):
            terminated = True

        self.time += 1

        return self.get_obs(), -costs, terminated, {}

    def reset(self):
        high = np.array([np.pi/30, np.pi/20]) * self.factor
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        self.time = 0

        return self.get_obs()

    def get_obs(self):
        return  self.state / self.factor


class Obs_Env(Env):
    def __init__(self, factor=1):
        super().__init__(factor)
        self.CG = np.array([[1, 0]])

        # observations are the first state: pos
        xmax = np.array([self.max_pos, self.max_speed])
        self.observation_space = spaces.Box(low=-xmax[0:1], high=xmax[0:1])

    def get_obs(self):
        return self.CG @ self.state

class Obs_Norm_Env(Obs_Env):
    def __init__(self, factor=1):
        super().__init__(factor)
        self.CG = self.CG / self.observation_space.high


