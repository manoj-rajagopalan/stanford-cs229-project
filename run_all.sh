#!/bin/sh

mkdir -p Results

# Ideal/expected behavior
python3 golden.py | tee Results/golden.log

# Prove that these robots are far from ideal
python3 run_test_robots_on_golden_controls.py | tee Results/run_test_robots_on_golden_controls.log

# Training data (figure-of-8 trajectories)
python3 generate_dataset.py | tee Results/generate_dataset.log

# One learning method - SGD
python3 learn_sgd.py | tee Results/learn_sgd.py
