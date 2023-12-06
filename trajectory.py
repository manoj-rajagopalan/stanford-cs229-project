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

    def plot(self, results_dir: str) -> None:
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
        plt.savefig(f'{results_dir}/{self.name}-state.png')

        _, ax = plt.subplots()
        xs, ys = self.s[:,0], self.s[:,1]
        x_min, x_max = np.min(xs), np.max(xs)
        y_min, y_max = np.min(ys), np.max(ys)
        x_mid, y_mid = np.average([x_min, x_max]), np.average([y_min, y_max])
        x_range, y_range = (x_max - x_min), (y_max - y_min)
        max_range = np.max([x_range, y_range])

        ax.plot(self.s[:,0], self.s[:,1])
        ax.set_title(f'Trajectory {self.name}')
        ax.set_xlabel('$x$ (m)')
        ax.set_ylabel('$y$ (m)')
        ax.set_xlim(x_mid - 1.2*max_range/2, x_mid + 1.2*max_range/2)
        ax.set_ylim(y_mid - 1.2*max_range/2, y_mid + 1.2*max_range/2)

        # Start/stop markers
        r_start = 0.015 * max_range
        r_stop = 0.01 * max_range
        ax.add_patch(ptch.Circle((self.s[0,0], self.s[0,1]), radius=r_start, fill=True, color='green'))
        ax.add_patch(ptch.RegularPolygon((self.s[-1,0], self.s[-1,1]), 8, radius=r_stop, fill=True, color='red'))
        
        plt.savefig(f'{results_dir}/{self.name}-trajectory.png')
        plt.close()
    #:plot_trajectory()

#:Trajectory

def decimate(traj: Trajectory, aux_data = None) -> (Trajectory, any):
    t = traj.t[range(0, len(traj.t), 10)]
    s = traj.s[range(0, len(traj.s), 10)]
    u = traj.u[range(0, len(traj.u), 10)]
    assert len(t) == len(s)
    if len(t) == len(u):
        t = np.hstack((t, traj.t[-1]))
        s = np.vstack((s, traj.s[[-1]]))
    #:
    assert len(t) == len(u)+1
    assert len(t) == len(s)

    aux_data_decimated = None
    if aux_data is not None:
        assert len(aux_data) == len(traj.u)
        aux_data_decimated = aux_data[range(0, len(aux_data), 10)]
        assert len(aux_data_decimated) == len(u)
    #:if
    return Trajectory(t, s, u, name=f'{traj.name}-decimated'), aux_data_decimated
#:decimate()

def add_noise(traj: Trajectory,
              xy_std_dev: float,
              θ_deg_std_dev: float,
              φ_lr_deg_std_dev: float,
              φdot_lr_deg_per_sec_std_dev) -> Trajectory:
    t = traj.t
    x, y, θ, φ_l, φ_r = traj.s.copy().T
    φdot_l, φdot_r = traj.u.copy().T
    x += np.random.normal(loc=0, scale=xy_std_dev, size=x.shape)
    y += np.random.normal(loc=0, scale=xy_std_dev, size=y.shape)
    θ = np.mod(θ + np.random.normal(loc=0, scale=np.deg2rad(θ_deg_std_dev), size=θ.shape),
               2*np.pi)
    φ_l = np.mod(φ_l + np.random.normal(loc=0, scale=np.deg2rad(φ_lr_deg_std_dev), size=φ_l.shape),
                 2*np.pi)
    φ_r = np.mod(φ_r + np.random.normal(loc=0, scale=np.deg2rad(φ_lr_deg_std_dev), size=φ_r.shape),
                 2*np.pi)

    φdot_l += np.random.normal(loc=0, scale=np.deg2rad(φdot_lr_deg_per_sec_std_dev), size=φdot_l.shape)
    φdot_r += np.random.normal(loc=0, scale=np.deg2rad(φdot_lr_deg_per_sec_std_dev), size=φdot_r.shape)

    # s = np.transpose(np.vstack((x, y, θ, φ_l, φ_r)))
    s = np.vstack((x, y, θ, φ_l, φ_r)).T
    u = np.vstack((φdot_l, φdot_r)).T
    print(f'add_noise:')
    print(f'  |Δs|={np.linalg.norm(s-traj.s)}')
    print(f'  |Δu|={np.linalg.norm(u-traj.u)}')
    return Trajectory(t,s,u, traj.name+'-with-noise')
#:add_noise()
