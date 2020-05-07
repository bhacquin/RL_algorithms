import numpy as np


class OUSeries():

    def __init__(self):

        class action_space():
            def __init__(self): 
                self.n = 2

        class observation_space():
            def __init__(self): 
                self.shape = [2]

        self.action_space = action_space()
        self.observation_space = observation_space()
        self.n_iter = 1000 #Total number of timesteps in episode

        self.mean = 0
        self.theta = 0.1
        self.noise_scale = 0.2
        self.current_value = 0
        self.current_step = 0

        self.position = 0 #0 for not holding, 1 for holding a stock

    def step(self,action):
        """
        Explanation here
        """
        change_since_last = self.theta*(self.mean - self.current_value) + np.random.normal(scale = self.noise_scale)
        self.current_value = self.current_value + change_since_last

        if action == 0: #Do nothing
            if self.position == 1:
                rew = change_since_last
            else:
                rew=0
        elif action == 1: #Change position
            if self.position == 1:
                rew=0
                self.position=0
            else:
                rew = change_since_last
                self.position=1
        else:
            raise NotImplementedError

        done=False
        self.current_step += 1
        if self.current_step == self.n_iter:
            done=True

        if self.current_step >self.n_iter:
            print("The environment has already returned done. Stop it!")
            raise NotImplementedError

        return ([self.current_value,self.position],float(rew),done,None)
    
    def reset(self):
        """
        Explanation here
        """
        self.current_step = 0
        self.current_value=0
        self.position=0
        return [self.current_value,self.position]
        
    def seed(self,seed):
        return