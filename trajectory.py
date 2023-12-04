import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as ptch

class Trajectory:
    def __init__(self, t, s, u, name) -> None:
        self.t = t
        self.s = s
        self.u = u
        self.name = name
    #:

    def plot(self, must_print_markers: bool = True) -> None:
        _, ax1 = plt.subplots()
        ax1.plot(self.t, self.s[:,0], 'r-')
        ax1.plot(self.t, self.s[:,1], 'b-')
        ax1.set_xlabel(f'time (s)')
        ax1.set_ylim(bottom=np.min(self.s[:,:2]), top=np.max(self.s[:,:2]))
        ax1.set_ylabel('$x, y$ (m)')
        
        ax2 = ax1.twinx()
        ax2.plot(self.t, self.s[:,2] * 180/np.pi, 'g-')
        ax2.set_ylim(bottom=np.min(self.s[:,2]), top=np.max(self.s[:,2]))
        ax2.set_ylabel('$\\theta$ (deg)')
        plt.savefig(f'Results/{self.name}-state.png')

        _, ax = plt.subplots()
        ax.plot(self.s[:,0], self.s[:,1])
        ax.set_title('Trajectory')
        ax.set_xlabel('$x$ (m)')
        ax.set_ylabel('$y$ (m)')
        
        if must_print_markers:
            range_x = np.max(self.s[:,0]) - np.min(self.s[:,0])
            range_y = np.max(self.s[:,1]) - np.min(self.s[:,1])
            d = min(range_x, range_y)
            r_start = 0.015 * d
            r_stop = 0.01 * d
            ax.add_patch(ptch.Circle((self.s[0,0], self.s[0,1]), radius=r_start, fill=True, color='green'))
            ax.add_patch(ptch.RegularPolygon((self.s[-1,0], self.s[-1,1]), 8, radius=r_stop, fill=True, color='red'))
        
        plt.savefig(f'Results/{self.name}-trajectory.png')
    #:plot_trajectory()

#:Trajectory
