
class Triangle():

    def __init__(self):

        class action_space():
            def __init__(self): 
                self.n = 2

        class observation_space():
            def __init__(self): 
                self.shape = [2]

        self.action_space = action_space()
        self.observation_space = observation_space()
        self.n_iter = 100 #Total number of timesteps in episode
        self.reward = 0.1
        self.current_step = 0 #Where we currently are on the triangle
        self.position = 0 #0 for not holding, 1 for holding a stock

    def step(self,action):
        """
        Explanation here
        """
        change_since_last = self.reward if self.current_step<self.n_iter/2 else -self.reward

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

        return ([change_since_last,self.position],float(rew),done,None)
    
    def reset(self):
        """
        Explanation here
        """
        self.current_step=0
        self.position=0
        return [self.reward,self.position]
        
    def seed(self,seed):
        return
