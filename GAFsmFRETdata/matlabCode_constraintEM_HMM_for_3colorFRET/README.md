# HMM_3colorFRET
Hidden Markov model (HMM) with the constraint EM algorithm for the analysis of 3color FRET data. 
The algorithm is coded in MATLAB. 
It can read in the original data, analyze it with 6 states (5 on DNA and 1 in solution), and output analyzed results.
To run the code, please follow the below steps:

1. Run `/Applications/MATLAB_R2025b.app/bin/matlab -batch "addpath(pwd); crossval_conditions"`
2. The analyzed results will be saved in a folder with the folder's name the same as the original data file's name.
3. To view the results, one can run aplotFittedFigures2.m file to plot figures of the fitted data. 

This script will divide the timeseries into an 80:20 train:test split according to a random number seed. Then, for each condition, all training timeseries of that condition will be used to train an HMM. The test series is then evaluated using that HMM.

Note: one may need to adjust initial approximations of FRET values, variance values, and transition matrix. 
