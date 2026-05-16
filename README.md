# YAPSO
=========================================================================
        YAPSO: Yet Another Particle Swarm Optimization

  Author: Maurizio Clemente
          Center for Automotive Research (CAR)
          Department of Mechanical and Aerospace Engineering
          The Ohio State University

  Citation:
     Maurizio Clemente, Marcello Canova, "Quadratic Surrogate Attractor for
     Particle Swarm Optimization", 2026, arXiv:2603.17163.
     https://doi.org/10.48550/arXiv.2603.17163

  Description:

    This class implements an enhanced Particle Swarm Optimizer (PSO) in which
    the classical global-best attractor is replaced by the minimizer of a quadratic
    surrogate model.

    The objective is to mitigate premature convergence and sensitivity to noise
    observed in standard PSO, while incurring only minimal computational overhead.
    Rather than relying on a single global best attractor, the method leverages
    multiple elite particles to construct a surrogate that better reflects first
    the global and later the local landscape to identify the optimum.

    The improvement is particularly pronounced for quasi-convex functions, where the
    surrogate can exploit the underlying convex-like structure of the landscape.
    Provided that a sufficient number of particles is available to construct the surrogate
    (Np ≥ NQ), the approach remains effective even for higher-order functions, as the local
    behavior near the optimum is well approximated by a quadratic model.

<p align="center">
  <img width="640" height="480" alt="Image" src="https://github.com/user-attachments/assets/0b3313b4-4bb3-484a-be91-f79c77071fec" />
</p>

    Inputs:

    obj:        N-dimensional objective function to be optimized (callable)
    bounds:     List of [lower, upper] values defining the search space per dimension
    Ns:         (Optional) number of independent swarms (multi-swarm configuration)
    Np:         (Optional) number of particles per swarm
    w:          (Optional) inertia weight
    c1:         (Optional) cognitive coefficient (attraction to personal best)
    c2:         (Optional) social coefficient (attraction to global/surrogate best)
    Imax:       (Optional) maximum number of PSO iterations
    X0:         (Optional) Initial particle position
    stats:      (Optional) True/False enable stats
    animate:    (Optional) True/False enable (AND SAVE!) runtime animation 
    min:        (Optional) True for minimization, False for maximization
    verbose:    (Optional) enable console logging and diagnostics
    epsilon:    (Optional) false-alarm threshold for convergence P(S* >= kappa | H0) <= epsilon
    tau:        (Optional) exploration gain factor
    vmax:       (Optional) velocity clipping threshold
    ST:         (Optional) self-termination True/False (early stopping logic)
    QS:         (Optional) quadratic surrogate attraction mechanism True/False

    Outputs:

    opt_x:      (ndarray)   Location of the optimal solution
    opt_f:      (float)     Value of the optimal solution

    if stats enabled: (Optional)

    results.swarms:     Final swarm states for all Ns swarms
    results.obj_arr:    Objective function history over iterations
    results.y_mean:     Mean objective value across swarms per iteration
    results.y_min:      Minimum objective value registered per iteration
    results.y_max:      Maximum objective value registered per iteration
    results.t:          Iteration timeline
    results.q25:        25th percentile of objective distribution
    results.q75:        75th percentile of objective distribution

  Methodology:
    Quadratic Surrogate Attractor Particle Swarm Optimization
=========================================================================
