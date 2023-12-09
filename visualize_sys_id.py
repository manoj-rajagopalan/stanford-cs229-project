import numpy as np
from matplotlib import pyplot as plt

from constants import *
from robot import DDMR
from wheel_model import *

kResultsDir = 'Results/3-SysId_via_SGD'

def φ_profile(wheel: WheelModel,
              N_φ: int) \
    -> np.ndarray:
    pass

#:φ_profile()

def sample_wheel_radii(wheel: WheelModel,
                       φ_bins: int) \
    -> list[float]:

    φ_bins_for_wheel = np.arange(1,361,10,dtype=int)
    wheel_R_samples = [wheel.radius_at(np.deg2rad(φ))
                       for φ in φ_bins]
    return wheel_R_samples
#:sample_wheel_radii()

if __name__ == "__main__":
    for robot_name in kRobotNames:
        real_robot = DDMR(f'Robots/{robot_name}.yaml')
        npz = np.load(f'{kResultsDir}/sysId-{robot_name}.npz')
        R_ls = npz['R_ls-mahalanobis-shuffled']
        R_rs = npz['R_rs-mahalanobis-shuffled']
        L = npz['L-mahalanobis-shuffled']
        N_φ = len(R_ls)
        assert len(R_rs) == N_φ
        φ_bins = np.arange(1, N_φ+1, dtype=int) * 360/N_φ

        # Sample the robot wheels as an approximation.
        φ_bins_for_wheels = np.arange(1,361,10,dtype=int)
        wheel_R_l_samples = [real_robot.left_wheel.radius_at(np.deg2rad(φ))
                             for φ in φ_bins_for_wheels]
        wheel_R_r_samples = [real_robot.right_wheel.radius_at(np.deg2rad(φ))
                             for φ in φ_bins_for_wheels]

        # Plot!
        _, ax = plt.subplots()
        ax.step(φ_bins, R_ls, 'm', label='Learnt $R_l$')
        ax.step(φ_bins_for_wheels, wheel_R_l_samples, 'k', label='Wheel $R_l$')
        ax.set_xlabel('$\\varphi$')
        ax.set_ylabel('$R_l(\\varphi)$')
        ax.set_ylim(0, 0.01+np.max(R_ls))
        ax.legend()
        plt.savefig(f'{kResultsDir}/sysId-{robot_name}-left_wheel.png')
        plt.close()

        _, ax = plt.subplots()
        ax.step(φ_bins, R_rs, 'g', label='Learnt $R_r$')
        ax.step(φ_bins_for_wheels, wheel_R_r_samples, 'k', label='Wheel $R_r$')
        ax.set_xlabel('$\\varphi$')
        ax.set_ylabel('$R_r(\\varphi)$')
        ax.set_ylim(0, 0.01+np.max(R_rs))
        ax.legend()
        plt.savefig(f'{kResultsDir}/sysId-{robot_name}-right_wheel.png')
        plt.close()

        print(f'Learnt L = {L} vs. robot-L = {real_robot.L} for {robot_name}')
    #:for
#:__main__
