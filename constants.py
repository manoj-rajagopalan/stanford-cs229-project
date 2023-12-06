
kSimHz = int(100)
kSimΔt = 1.0 / kSimHz # seconds

kMeasurementHz = int(10)
kMeasurementΔt = 1.0 / kMeasurementHz # seconds

φdot_max_mag_rps = 8.0 # rotations per second, ~ 1 m/s for 20cm radius wheel

kIdealTrajectoryKeys = ['straight',
                        'spin',
                        'circle_ccw',
                        'circle_cw',
                        'figureOf8',
                        'tri_wave_phi']

kRobotNames = ['SmallerLeftWheel',
               'LargerLeftWheel',
               'SmallerBaseline',
               'LargerBaseline',
               'Noisy',
               'Noisier']
