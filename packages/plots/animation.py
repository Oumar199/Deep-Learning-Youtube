

from typing import Callable

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


def animate(losses):
    
    def animation(i):
        plt.cla()
        plt.xlim(-1, 100)
        plt.ylim(-0.01, 2.5)
        plt.plot(losses[:i], label = "train loss")
    
    ani = FuncAnimation(plt.gcf(), animation, interval = 100)  # type: ignore
    
    plt.show()
    