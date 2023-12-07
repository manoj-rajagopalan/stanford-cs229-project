#!/bin/sh

mkdir -p Results

# Ideal/expected behavior
mkdir -p Results/0-Ideal
python3 ideal.py | tee Results/0-Ideal/ideal.log

# Prove that these robots are far from ideal
mkdir -p Results/1-Uncontrolled
python3 uncontrolled_behavior.py | tee Results/1-Uncontrolled/uncontrolled_behavior.log

# Training data (figure-of-8 trajectories)
mkdir -p Results/2-Dataset
python3 generate_dataset.py | tee Results/2-Dataset/generate_dataset.log

# First learning method - System Identification with Stochastic Gradient Descent (SGD)
mkdir -p Results/3-SysId_via_SGD
python3 sys_id_via_sgd.py | tee Results/3-SysId_via_SGD/sys_id_via_sgd.log

# Second learning method - Model-Free control with Fully-Connected Neural Networks (FCNN)
# mkdir -p Results/4-FCNN_Controlled
# python3 learn_fcnn_controller.py | tee Results/4-FCNN_Controlled/learn_fcnn_controller.log
