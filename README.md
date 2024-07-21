# quantumML
A collection of packages applying machine learning to problems in quantum mechanics

## Hamiltonian Learning
Training data can be simulated using data_gen.py in the sdes package (see notebooks/test_data_gen.ipynb for examples). Models can be trained and evaluated using train.py in the models package. See command line help description for details

## TNS Package
TensorFlow implementation of tensor network states (TNS). See networks.py for matrix product state and operator classes, and utility functions. The file tns_solve.py contains implementations of DMRG for finding Hamiltonian ground states and TDVP for simulating Hamiltonian dynamics. The TensorFlow implementation allows for the use of TNS in ML models and optimization routines aimed at finding unknown parameters in TNS modeled quantum systems that maximize the likelihood of observed data

## NQS Package
Neural Quantum State routines in MATLAB for representing Hamiltonian ground states using neural networks. Start with the nqs_run.m script, modifying the package path and save file path as necessary
