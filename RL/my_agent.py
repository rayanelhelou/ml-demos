# Copyright 2020, Rayan El Helou.
# All rights reserved.

import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from tqdm import trange
import random
from IPython import display


class NotebookRenderer:
    '''
    A context manager for rendering OpenAI gym envs in a notebook cell.
    
    EXAMPLE:
    
        with NotebookRenderer(env) as NR:
            ...         # whatever before
            NR.render() # replaces env.render()
            ...         # whatever after
            
    It's a good idea to create a context manager like this one (and use
    it within the 'with' framework) since you don't want your kernel to crash
    in case an error occurred during the rendering process.
    
    '''
    def __init__(self, env, scale_fig=1.5): 
        self.env = env
        self.img = plt.imshow(self.env.render(mode='rgb_array'))
        
        # Rescale fig
        fig = plt.gcf()
        size = fig.get_size_inches()
        fig.set_size_inches(size[0]*scale_fig, size[1]*scale_fig, forward=True)
        
    def render(self):
        self.img.set_data(self.env.render(mode='rgb_array'))
        plt.axis('off')
        display.display(plt.gcf())
        display.clear_output(wait=True)
      
    def __enter__(self):
        return self
  
    def __exit__(self, exception_type, exception_value, traceback): 
        self.render()
        self.env.close()
        if exception_type: print(exception_type)
        if exception_value: print(exception_value)
        if traceback: print(traceback)
        

class Agent:
    '''
    INPUTS:
    
        "model": uses the StableBaselines3 convention. That is:
                To take actions:
                    action, _ = self.model.predict(obs)
                To update policy/value networks:
                    model.learn(total_timesteps=...)
                    
        "env": uses the OpenAI gym convention. That is:
                To reset:
                    obs = env.reset()
                To sample random actions:
                    action = env.action_space.sample()
                To step (given action):
                    obs, reward, done, info = env.step(action)
                To extract episode length:
                    env._max_episode_steps
                    
    PURPOSES:
    
        - Interfaces between control model and environment
        - Evaluates performance of model
        - Maintain record of average rewards after each training step
        - Renders agent's interaction with environment
    
    '''
    def __init__(self, model, env):
        self.model = model
        self.env = env
        
        # Initialize record of average rewards
        self.R = [self.get_rewards()]
        
    def train(self, N_iter=10, steps_per_iteration=2048, plot_R=True):
        for _ in trange(N_iter):
            self.model.learn(total_timesteps=steps_per_iteration)
            self.R.append(self.get_rewards())
            
        if plot_R:
            plt.plot(self.R)
            plt.plot([200 for _ in self.R], '--k')
            plt.show()
            
    def evaluate(self, show_every=5, random_actions=False):
        obs = self.env.reset()
        with NotebookRenderer(self.env) as NR:
            total_reward = 0
            for t in range(self.env._max_episode_steps):
                if random_actions:
                    action = self.env.action_space.sample()
                else:
                    action, _ = self.model.predict(obs, deterministic=True)
                if t % show_every == 0:
                    NR.render()
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
                if done: break
        print('Total Reward:', total_reward)
        
    def get_rewards(self, num_episodes=10):
        R = 0
        for episode in range(num_episodes):
            obs = self.env.reset()
            for t in range(self.env._max_episode_steps):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                R += reward/num_episodes
                if done: break
        return R