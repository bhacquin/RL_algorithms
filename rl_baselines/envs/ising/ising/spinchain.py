import numpy as np

class SpinChain():

    def __init__(self):

        self.n_spins = 10 #Total number of spins in episode
        self.max_steps = 20 #Number of actions before reset

        class action_space():
            def __init__(self,n_spins): 
                self.n = n_spins

        class observation_space():
            def __init__(self,n_spins): 
                self.shape = [n_spins]

        self.action_space = action_space(self.n_spins+1)
        self.observation_space = observation_space(self.n_spins+1)

        self.current_step = 0

        self.matrix = np.zeros((self.n_spins+1,self.n_spins+1))
        for i in range(self.n_spins-1):
            self.matrix[i,i+1] = -1
            self.matrix[i+1,i] = -1

        self.state = np.ones(self.n_spins+1)
        self.energy = -np.matmul(np.transpose(self.state),np.matmul(self.matrix,self.state))/2

    def step(self,action):
        """
        Explanation here
        """

        done = False
        self.current_step += 1

        if self.current_step >= self.max_steps:
            print("The environment has already returned done. Stop it!")
            raise NotImplementedError

        new_state = np.copy(self.state)
        new_state[action] = -self.state[action]
        rew = 2*new_state[action]*np.matmul(np.transpose(new_state),self.matrix[:,action])
        self.energy -= rew

        if self.current_step == self.max_steps - 1:
            done = True

        self.state = new_state

        return (np.vstack((self.state,self.matrix)),rew,done,None)
    
    def reset(self):
        """
        Explanation here
        """
        self.current_step=0
        self.state = np.ones(self.n_spins+1)
        self.matrix = np.zeros((self.n_spins+1,self.n_spins+1))
        for i in range(self.n_spins-1):
            self.matrix[i,i+1] = -1
            self.matrix[i+1,i] = -1
        self.energy = -np.matmul(np.transpose(self.state),np.matmul(self.matrix,self.state))/2

        return np.vstack((self.state,self.matrix))
        
    def seed(self,seed):
        return

    def calculate_best(self):
        energy = 1e50
        best_state = None

        for i in range(2**self.n_spins):
            list_strings = list(np.binary_repr(i, width=self.n_spins))
            state = 2*np.array([int(x) for x in list_strings]) - 1
            current_energy = -np.matmul(np.transpose(state),np.matmul(self.matrix,state))/2
            if current_energy < energy:
                energy = current_energy
                best_state = state

        return energy, best_state
        
