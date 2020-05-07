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

alien2 = 0.5*(np.array(pickle.load(open("../ide/results/2019-08-20/alien/alien_scores1",'rb'))) + np.array(pickle.load(open("../ide/results/2019-08-20/alien/alien_scores2",'rb'))))
amidar2 = 0.5*(np.array(pickle.load(open("../ide/results/2019-08-20/amidar/amidar_scores1",'rb'))) + np.array(pickle.load(open("../ide/results/2019-08-20/amidar/amidar_scores2",'rb'))))
assault2 = 0.5*(np.array(pickle.load(open("../ide/results/2019-08-20/assault/assault_scores1",'rb'))) + np.array(pickle.load(open("../ide/results/2019-08-20/assault/assault_scores2",'rb'))))
asterix2 = 0.5*(np.array(pickle.load(open("../ide/results/2019-08-20/asterix/asterix_scores1",'rb'))) + np.array(pickle.load(open("../ide/results/2019-08-20/asterix/asterix_scores2",'rb'))))
breakout2 = pickle.load(open("../ide/results/2019-08-20/breakout/breakout_scores1",'rb'))

steps = np.arange(50) * 1000000

plt.figure(figsize=(5,4))

plt.title("Breakout", size=20)
plt.xlabel("Frames", size=20)
plt.ylabel("Score", size=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.plot(steps,alien1)
plt.plot(steps,alien2)
"""
plt.plot(steps,amidar1)
plt.plot(steps,amidar2)

plt.plot(steps,assault1)
plt.plot(steps,assault2)

plt.plot(steps,asterix1)
plt.plot(steps,asterix2)

plt.plot(steps,breakout1)
plt.plot(steps,breakout2)
"""

plt.tight_layout()
plt.show()