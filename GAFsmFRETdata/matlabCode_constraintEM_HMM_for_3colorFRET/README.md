# HMM_3colorFRET
Hidden Markov model (HMM) with the constraint EM algorithm for the analysis of 3color FRET data. 
The algorithm is coded in MATLAB. 
It can read in the original data, analyze it with 6 states (5 on DNA and 1 in solution), and output analyzed results.
To run the code, please follow the below steps:

1. Download all files in one folder.
2. Put the original data in the same folder.
3. Open maincode.m file by MATLAB.
4. At the top of the maincode.m, in the file-read commend line, change the file's name according to your data file.
5. Run the code. The analyzed results will be saved in a folder with the folder's name the same as the original data file's name.
6. To view the results, one can run aplotFittedFigures2.m file to plot figures of the fitted data. 

Note: one may need to adjust initial approximations of FRET values, variance values, and transition matrix. 
