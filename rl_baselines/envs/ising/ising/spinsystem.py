import numpy as np

class SpinSystem():

    def __init__(self,n_spins=10):

        self.n_spins = n_spins #Total number of spins in episode
        self.max_steps = 2*n_spins #Number of actions before reset
        self.do_nothing_idx = n_spins

        class action_space():
            def __init__(self,n_spins): 
                self.n = n_spins

        class observation_space():
            def __init__(self,n_spins): 
                self.shape = [n_spins,2]

        self.action_space = action_space(self.n_spins+1)
        self.observation_space = observation_space(self.n_spins+1)

        self.current_step = 0

        self.matrix = self.random_matrix()

        self.state = 2*np.random.randint(2,size=(2,self.n_spins+1))-1
        #self.state[0,:] = np.ones(self.n_spins+1)
        self.state[1,:] = np.ones(self.n_spins+1)*self.current_step/self.n_spins
        self.energy = -np.matmul(np.transpose(self.state[0,:]),np.matmul(self.matrix,self.state[0,:]))/2

    def random_matrix(self):

        density = np.random.uniform()
        matrix = np.zeros((self.n_spins+1,self.n_spins+1))
        for i in range(self.n_spins):
            for j in range(i):
                if np.random.uniform() < density:
                    matrix[i,j] = -1
                    matrix[j,i] = -1

        return matrix


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
        new_state[0,action] = -self.state[0,action]
        rew = 2*new_state[0,action]*np.matmul(np.transpose(new_state[0,:]),self.matrix[:,action])
        self.energy -= rew

        if self.current_step == self.max_steps - 1:
            done = True

        self.state = new_state
        self.state[1,:] = np.ones(self.n_spins+1)*self.current_step/self.n_spins

        return (np.vstack((self.state,self.matrix)),rew,done,None)
    
    def reset(self):
        """
        Explanation here
        """
        self.current_step=0
        self.state = 2*np.random.randint(2,size=(2,self.n_spins+1))-1
        #self.state[0,:] = np.ones(self.n_spins+1)
        self.state[1,:] = np.ones(self.n_spins+1)*self.current_step/self.n_spins
        self.matrix = self.random_matrix()
        self.energy = -np.matmul(np.transpose(self.state[0,:]),np.matmul(self.matrix,self.state[0,:]))/2

        return np.vstack((self.state,self.matrix))
        
    def seed(self,seed):
        return

    def calculate_best(self):

        energy = 1e50
        best_state = None

        for i in range(2**(self.n_spins+1)):
            list_strings = list(np.binary_repr(i, width=self.n_spins+1))
            state = 2*np.array([int(x) for x in list_strings]) - 1
            current_energy = -np.matmul(np.transpose(state),np.matmul(self.matrix,state))/2
            if current_energy < energy:
                energy = current_energy
                best_state = state

        return energy, best_state
        
