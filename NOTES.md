### April 15, 2026 meeting w/ Maggie
* A trained HMM will give us a transition matrix and emission probabilities.
* The "norm" in NERDSS is a mathematical tool allowing us to define dihedrals in special cases. I think we won't need to set it to anything other than \[0,0,1\].
* The main work to be done in the world of HMMs is:
    1. Write code to automatically scale and clean the experimental trajectories. This will give us labelled experimental states over time.
    2. Using that code, try a Python HMM package to train an HMM using _all_ the trajectories. Be careful to ensure that trajectory start and ends are treated properly. We can't just concatenate them into a long trajectory, because that will artificially mess with the rate with which we go to our initial state.
    3. Ideally, we should check with the MATLAB code to ensure our Python implementation agrees.
* For RDs, we will be performing one big optimization. The optimization wrapper will be trying out different on/off rates. The workflow looks like:
    1. A candidate set of rates is proposed by the optimizer.
    2. Those rates are used to run many NERDSS simulations.
    3. Those NERDSS results are converted to FRET intensity timeseries by code (to be written).
    4. A loss metric (to be defined, probably including length of traj. as well as HMM params) will give a score back to the optimizer.
* How are we including HMM params in the loss function? I'm unsure. I guess this means we need to do the HMM work first.

### April 30, 2026 meeting w/ Maggie
* Free params: on/off rates, h^2, D of prot in soln, D of N in 1D.
* I need to schedule meetings with Yiben and Mankun.

### May 3, 2026 meeting w/ Mankun
* When you do 1D diffusion, use isPromoter. For the specific binding site, translational and rotation diffusion constants should be fully zero.
* 1D diffusion of A binding to N = (1/D_A + 1/D+N)**1 where D_A is the 3D diffusion of A
* TFs should not be able to bypass one another. To prevent this, we should set up a "reference" point with a reaction
