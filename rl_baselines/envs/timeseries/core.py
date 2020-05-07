from rl_baselines.envs.timeseries.series.triangle import Triangle
from rl_baselines.envs.timeseries.series.ouseries import OUSeries

def make(id):

    if id == "Triangle":
        env = Triangle()

    if id == "OUSeries":
        env = OUSeries()

    return env