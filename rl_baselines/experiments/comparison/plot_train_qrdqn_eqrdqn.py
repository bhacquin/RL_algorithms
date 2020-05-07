#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 10:18:48 2019

@author: maxime

"""
import pickle
import time
import random

import matplotlib.pyplot as plt
import numpy as np

alien1 = pickle.load(open("../qrdqn/results/2019-05-07/alien_scores",'rb'))
amidar1 = pickle.load(open("../qrdqn/results/2019-05-07/amidar_scores",'rb'))
assault1 = pickle.load(open("../qrdqn/results/2019-05-07/assault_scores",'rb'))
asterix1 = pickle.load(open("../qrdqn/results/2019-05-07/asterix_scores",'rb'))
breakout1 = pickle.load(open("../qrdqn/results/2019-05-09/breakout_scores",'rb'))

alien2 = pickle.load(open("../eqrdqn/results/2019-05-07/alien_scores",'rb'))
amidar2 = pickle.load(open("../eqrdqn/results/2019-05-07/amidar_scores",'rb'))
assault2 = pickle.load(open("../eqrdqn/results/2019-05-07/assault_scores",'rb'))
asterix2 = pickle.load(open("../eqrdqn/results/2019-05-07/asterix_scores",'rb'))
breakout2 = pickle.load(open("../eqrdqn/results/2019-05-09/breakout_scores",'rb'))

steps = np.arange(50) * 1000000

plt.figure(figsize=(5,4))

plt.title("Breakout", size=20)
plt.xlabel("Frames", size=20)
plt.ylabel("Score", size=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

"""
plt.plot(steps,alien1)
plt.plot(steps,alien2)

plt.plot(steps,amidar1)
plt.plot(steps,amidar2)

plt.plot(steps,assault1)
plt.plot(steps,assault2)

plt.plot(steps,asterix1)
plt.plot(steps,asterix2)
"""
plt.plot(steps,breakout1)
plt.plot(steps,breakout2)


plt.tight_layout()
plt.show()