import numpy as np
import os

import robot
import ideal
import eval
from constants import *

kResultsDir = 'Results/1-Uncontrolled'

if __name__ == "__main__":
    os.makedirs(kResultsDir, exist_ok=True)
    real_robots = [robot.DDMR(f'Robots/{name}.yaml') for name in kRobotNames]
    ideal_trajectories = ideal.load_trajectories()
    eval.run_robots_with_controls(real_robots,
                                  ideal_trajectories,
                                  kResultsDir)
#:__main__
