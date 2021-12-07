import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

class Env(object):

    def __init__(self, factor=1):

        self.factor = factor
        self.viewer = None
        # Nominal speed of the vehicle travels at.
        self.U = 28.0; # m/s
        # Front cornering stiffness for one wheel.
        self.Ca1 = -61595.0; # unit: Newtons/rad
        # Rear cornering stiffness for one wheel.
        self.Ca3 = -52095.0; # unit: Newtons/rad

        # Front cornering stiffness for two wheels.
        self.Caf = self.Ca1*2.0; # unit: Newtons/rad
        # Rear cornering stiffness for two wheels.
        self.Car = self.Ca3*2.0; # unit: Newtons/rad

        # Vehicle mass
        self.m = 1670.0; # kg
        # Moment of inertia
        self.Iz = 2100.0; # kg/m^2

        # Distance from vehicle CG to front axle
        self.a = 0.99; # m
        # Distance from vehicle CG to rear axle
        self.b = 1.7; # m

        self.g = 10.0
        self.dt = 0.02

        # maximum control input
        self.max_steering = np.pi/6 #* factor

        # continuous-time model
        Ac = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, (self.Caf+self.Car)/(self.m*self.U), -(self.Caf+self.Car)/self.m, (self.a*self.Caf-self.b*self.Car)/(self.m*self.U)],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, (self.a*self.Caf-self.b*self.Car)/(self.Iz*self.U), -(self.a*self.Caf-self.b*self.Car)/self.Iz, (self.a**2*self.Caf+self.b**2*self.Car)/(self.Iz*self.U)]
        ])
        Bc = np.array([
            [0.0], [-self.Caf/self.m], [0.0], [-self.a*self.Caf/self.Iz]
        ]) * factor
        self.nx = Ac.shape[0]
        self.nu = Bc.shape[1]

        # discrete-time model for control synthesis
        self.AG = np.eye(self.nx) + Ac * self.dt
        self.BG = Bc * self.dt
        self.CG = None

        self.time = 0

        self.action_space = spaces.Box(low=-self.max_steering, high=self.max_steering, shape=(self.nu,))

        self.x1lim = 10.0 * factor; self.x2lim = 5.0 * factor; self.x3lim = 1.0 * factor; self.x4lim = 5.0 * factor  # xmax limits
        xmax = np.array([self.x1lim, self.x2lim, self.x3lim, self.x4lim])
        self.observation_space = spaces.Box(low=-xmax, high=xmax)
        self.state_space = spaces.Box(low=-xmax, high=xmax)

        self.seed()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):
        e, edot, etheta, ethetadot = self.state

        u = np.clip(u, -self.max_steering, self.max_steering)[0]
        costs = 0.01 * e**2 + 1/25.0 * edot**2 + etheta**2 + 1/25.0 * ethetadot**2 + 2.0/(np.pi/6.0)**2 * (u*self.factor)**2 - 5.0#2.0/(np.pi/6.0)**2 * (u*self.factor)**2 - 5.0

        self.state = self.AG @ self.state + self.BG @ [u]

        terminated = False
        if self.time > 200 or not self.state_space.contains(self.state):
            terminated = True

        self.time += 1

        return self.get_obs(), -costs, terminated, {}

    def reset(self):
        high = np.array([1.0, 0.5, 0.1, 0.5]) * self.factor
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        self.time = 0

        return self.get_obs()

    def get_obs(self):
        return  self.state


class Obs_Env(Env):
    def __init__(self, factor=1):
        super().__init__(factor)
        # outputs are e and etheta
        #self.CG = np.array([[1, 0, 0, 0], [0, 0, 0, 1]])
        self.CG = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])

        # observations are the first state: pos
        xmax = np.array([self.x1lim, self.x3lim])
        self.observation_space = spaces.Box(low=-xmax, high=xmax)

    def get_obs(self):
        return self.CG @ self.state

class Obs_Norm_Env(Obs_Env):
    def __init__(self, factor=1):
        super().__init__(factor)
        self.CG = self.CG / self.observation_space.high[:, np.newaxis]

