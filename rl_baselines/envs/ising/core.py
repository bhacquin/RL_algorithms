from rl_baselines.envs.ising.ising.spinchain import SpinChain
from rl_baselines.envs.ising.ising.spinsystem import SpinSystem


def make(id,n_spins=10):

    if id == "SpinChain":
        env = SpinChain()

    if id == "SpinSystem":
        env = SpinSystem(n_spins=n_spins)

    return env