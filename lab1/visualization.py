# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

f = open('results.txt', 'r')
fig = plt.figure(figsize=(10, 5))
ax = fig.add_axes([0, 0, 1, 1], frameon=False)
ax.set_xlim(0, 4000), ax.set_xticks([])
ax.set_ylim(4000, 0), ax.set_yticks([])

#Число тел
n_part = 800

bodies = np.zeros(n_part, dtype=[('position', float, 2)])

bodies['position'] = np.zeros((n_part,2))

scat = ax.scatter(bodies['position'][:, 0], bodies['position'][:, 1],s=2)

def update(frame_number):
        line = f.readline()
        if (line!=""):
            bodies['position'] = list(map(lambda x: list(map(lambda y: float(y),x.split('_'))),line.split('|')[:-1]))
        scat.set_offsets(bodies['position'])

animation = FuncAnimation(fig, update, interval=10)
plt.show()

