'''
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

  Notes:
    Full set of references at the end of the file
=========================================================================
'''

import warnings
from typing import Callable
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom
import heapq
import matplotlib.animation as animation
from datetime import datetime

class YAPSO:

    def __init__(self, obj, bounds, Ns=None, Np=None, w=None, c1=None, c2=None, Imax=None,
                 X0=None, stats=False, animate=False, min=True, verbose=True, epsilon = None, tau = None, vmax = None, ST = None, QS = None ):

        # Core problem definition
        self.obj = obj              # Objective function f(x) to be optimized
        self.bounds = bounds        # Search space bounds for each dimension [(lb_i, ub_i), ...]

        # Swarm configuration
        self.X0 = X0  # Optional starting particles' position
        self.Ns = Ns                # Number of independent swarms (multi-swarm PSO)
        if self.Ns is None:
            self.Ns = 1             # Default: single swarm
        self.Np = Np                # Number of particles per swarm
        if self.Np is None:
            # Default based on quadratic model size: (n+1)(n+2)/2
            self.Np = int(((len(self.bounds) + 1) * (len(self.bounds) + 2)) / 2)

        # PSO dynamic parameters [1], [2], [3]
        self.w = w                  # Inertia weight
        if self.w is None:
            self.w = 0.72984        # Clerc constriction-based default
        self.c1 = c1                # Cognitive coefficient (particle's own best attraction)
        if self.c1 is None:
            self.c1 = 2.8           # Default self-attraction
        self.c2 = c2                # Social coefficient (global/surrogate best attraction)
        if c2 is None:
            self.c2 = 2.05          # Default social component
        self.vmax = vmax            # Maximum velocity magnitude
        if self.vmax is None:
            self.vmax = 2           # Default velocity clamp
        self.tau = tau              # Exploration safeguard
        if self.tau is None:
            self.tau = 1.2          # Mild exploration bias

        # Execution controls
        self.stats = stats          # Enable/disable statistics after optimization
        self.animate = animate      # Enable/disable visualization after optimization
        self.min = min              # If True → minimization, else maximization
        self.verbose = verbose      # Enable logging / console output
        self.Imax = Imax            # Maximum number of PSO iterations
        if self.Imax is None:
            self.Imax = 300         # Default iterations

        # Algorithmic extensions
        self.QS = QS                # Quadratic-Surrogate attractor
        self.ST = ST                # Self-termination criterion
        self.epsilon = max(abs(epsilon), np.finfo(float).eps)      # False-alarm threshold
        if self.epsilon is None:
            # Default: machine precision
            self.epsilon = np.finfo(float).eps

        # Store initial parameter values for adaptive algorithm
        self.w_init = self.w  # Initial inertia weight
        self.c1_init = self.c1  # Initial cognitive coefficient
        self.c2_init = self.c2  # Initial social coefficient
        self.vmax_init = self.vmax  # Initial velocity bound

    def start(self) -> tuple[list, float]:
        # Particles initialization
        self.initialize()
        if self.verbose:
            print("Starting YAPSO...")
        # Executing steps
        self.step()

        obj_values = [self.obj(g) for g in self.gbest]

        if self.min:                                            # Minimizing
            self.GOPTI_f = np.min(obj_values)                   # Best objective value among swarms
            self.Swarm_Optimal_N = np.argmin(obj_values)        # Swarm that found the best solution
        else:                                                   # Maximizing
            self.GOPTI_f = np.max(obj_values)                   # Best objective value among swarms
            self.Swarm_Optimal_N = np.argmax(obj_values)        # Swarm that found the best solution

        self.GOPTI_x = self.gbest[self.Swarm_Optimal_N]

        if self.animate:
            self.anim.animate()
        if self.verbose:
            print(f'\nOptimal Solution: {np.round(self.GOPTI_x, 64)}')
            print(f'Optimal Objective: {np.round(self.GOPTI_f, 64)}')
            print(f'Found by Swarm {self.Swarm_Optimal_N} at iteration {self.t-1}')
        if self.stats:
            results=self.iterplot()
            return results
        else:
             return self.GOPTI_x.tolist(), self.GOPTI_f.tolist(),

    def initialize(self):
        """
        Initialize the multi-swarm PSO:
            1. Creates Ns independent swarms (populations).
            2. Initializes the global best (gbest) solution for each swarm.
            3. Computes the parameters (S, kappa) used in the sign-test
               described in the paper to control false-alarm probability [3].
            4. Optionally sets up animation utilities.
        """

        # Allocate containers for Ns independent swarms
        # population[i] will store the Particle object representing swarm i
        self.population = [[] for _ in range(self.Ns)]

        # gbest[i] stores the best solution found so far by swarm i
        self.gbest = [[] for _ in range(self.Ns)]

        # Initialize each swarm independently
        for i in range(self.Ns):

            # Create a Particle swarm with:
            # - Np particles
            # - initial position X0
            # - search-space bounds
            # - velocity limit vmax
            # - QS: quadratic surrogate method
            self.population[i] = Particle(
                Np=self.Np,
                X=self.X0,
                bounds=self.bounds,
                vmax=self.vmax,
                QS=self.QS
            )

            # Determine the initial global best (gbest) of this swarm
            for j in range(self.Np):

                if j == 0:
                    # Initialize gbest with the first particle's position
                    # .copy() avoids accidental aliasing
                    self.gbest[i] = self.population[i].X[j].copy()
                else:
                    # Update gbest by comparing the current gbest with particle j
                    self.gbest[i] = self.optimum(
                        self.gbest[i],
                        self.population[i].X[j]
                    )

        # Compute the parameters S and kappa used in the sign-test
        # These control statistical confidence and false-alarm probability
        # as described in the referenced paper [3]
        self.S, self.kappa = self.compute_S_and_kappa()

        # Inform the user of the statistical test configuration
        if self.verbose:
            print(
                f"Using S = {self.S}, κ = {self.kappa}, "
                f"for sign-test (false-alarm ≤ {self.epsilon})"
            )

        # If visualization is enabled, initialize the animation handler
        if self.animate:
            self.anim = ParticleAnimator(
                self.obj,
                self.population,
                bounds=self.bounds,
                iterations=self.Imax
            )

    def compute_S_and_kappa(self, max_S=200):
        """
        Compute (S, kappa) for the non-parametric sign test.
        Finds the smallest S such that there exists a kappa satisfying
        P(S* >= kappa | H0) <= epsilon.

        Reference: [3]
        """
        for S in range(2, max_S + 1):
            # Start from the largest possible kappa and move down
            for kappa in range(S, -1, -1):
                if binom.sf(kappa - 1, S, 0.5) <= self.epsilon:
                    return S, kappa
        return None, None

    def optimum(self, best, particle_x, num_points = None):
        val = self.obj(particle_x)

        if num_points is None:
            # Simple comparison
            if self.min:
                if val < self.obj(best):
                    best = particle_x.copy()
            else:
                if val > self.obj(best):
                    best = particle_x.copy()
            return best

        else:

            # Top-k in self.Q
            if not hasattr(self, "_heap"):
                self._heap = []  # heap stores (value, idx, point)

            if self.min:
                # keep smallest k, max-heap with negatives
                if len(self._heap) < num_points:
                    heapq.heappush(self._heap, (-val, len(self._heap), tuple(particle_x.copy())))
                elif val < -self._heap[0][0]:
                    heapq.heapreplace(self._heap, (-val, len(self._heap), tuple(particle_x.copy())))

                # extract only the points, sorted by value
                self.Q = [px for _, _, px in sorted(self._heap, key=lambda x: -x[0])]
                best = self.Q[0]

            else:
                # keep largest k, min-heap
                if len(self._heap) < num_points:
                    heapq.heappush(self._heap, (val, len(self._heap), particle_x.copy()))
                elif val > self._heap[0][0]:
                    heapq.heapreplace(self._heap, (val, len(self._heap), particle_x.copy()))

                # extract only the points, sorted by value (descending)
                self.Q = [px for _, _, px in sorted(self._heap, key=lambda x: x[0], reverse=True)]
                best = self.Q[0]
            return best

    def step(self):

        self.num_points = (len(self.bounds) + 1) * (len(self.bounds) + 2) // 2 # number of points needed to evaluate QS
        self.t          = 0     # iteration
        self.time       = []    # whole time vector
        self.x_min      = [ [] * len(self.bounds) for i in range(self.Ns)]
        self.f_min      = [ [] for i in range(self.Ns)]
        self.obj_time   = [ [] for i in range(self.Ns)]

        # Initializing flag to exit the cycle in case of self-termination.
        flag = False
        while (self.t <= self.Imax) and (flag == False):

            # coefficient update (w,c1,c2,vmax)
            self.update_coeff()

            for i in range(self.Ns):
                if self.animate:
                    self.anim.run(i,self.t) # saving positions for animation

                # velocity and position update (with clipping)
                self.population[i].update_velocity(self.w, self.c1, self.c2, self.gbest[i], self.num_points, self.t, self.obj, self.f_min[i], self.x_min[i])
                self.population[i].update_position()

                # For each particle, perform Algorithm 1 (from our paper).
                for k in range(self.Np):

                    if self.Np >= self.num_points and self.QS is True:
                        self.population[i].pbest[k] = self.optimum(self.population[i].pbest[k], self.population[i].X[k], self.num_points)
                    else:
                        self.population[i].pbest[k] = self.optimum(self.population[i].pbest[k], self.population[i].X[k])

                    self.gbest[i] = self.optimum(self.gbest[i], self.population[i].X[k])

                    # Exploration safeguard: boost if stagnant.
                    if len(self.obj_time[i]) > self.S:
                        f_current = self.obj(self.population[i].X[k])
                        f_prev = self.obj_time[i][-self.S]
                        gamma = abs(f_current - f_prev) / max(abs(f_prev), 1e-12)
                        if gamma < 0.5:
                            self.w *= self.tau
                        self.w = min(self.w, self.population[i].vmax)

                self.obj_time[i].append(self.obj(self.gbest[i]))

                # if there are enough particles, perform Algorithm 2 (from our paper).
                if self.QS is True:
                    if self.Np >= self.num_points:
                        self.x_min[i], self.f_min[i] = self.QFND()

                if self.verbose:
                        print(f'Iteration: {self.t} | best global cost:{self.obj(self.gbest[i])}')

                #Termination condition: nonparametric signtest.
                if (len(self.obj_time[i]) >= self.S and self.ST == True):
                    improvements = [self.obj_time[i][-k] - self.obj_time[i][-k - 1] for k in range(1, self.S)]
                    if all(delta == 0 for delta in improvements):
                        print(f"Terminating at iteration {self.t}: no improvements in last {self.S} steps (κ = {self.kappa})")
                        flag = True

            self.time.append(self.t)
            self.t += 1

    def update_coeff(self) -> None:
        #Coefficient update as iterations progress (inertia) [4],[5]
        self.w = self.w_init - 0.5 * (self.t / self.Imax)
        self.c1 = self.c1_init - (self.t / self.Imax)
        self.c2 = self.c2_init + (self.t / self.Imax)
        self.vmax = self.vmax_init * np.exp(1 - (self.t / self.Imax))

    def iterplot(self) -> dict:
        """
        Compute statistics of the objective function over time.

        Returns
        -------
        dict
            Dictionary containing:
            - x_opt : np.ndarray
                Global optimal found.
            - f_opt : float
                Objective value at the global optimum.
            - swarms : int
                Number of swarms.
            - obj_arr : np.ndarray
                Array of objective values over time.
            - y_mean : np.ndarray
                Mean objective value per iteration.
            - y_min : np.ndarray
                Minimum objective value per iteration.
            - y_max : np.ndarray
                Maximum objective value per iteration.
            - t : np.ndarray
                Time or iteration steps.
            - q25 : np.ndarray
                25th percentile per iteration.
            - q75 : np.ndarray
                75th percentile per iteration.
        """

        try:
            obj_arr = np.vstack(self.obj_time)
            y_min = np.min(obj_arr, axis=0)
            y_max = np.max(obj_arr, axis=0)
            y_mean = np.mean(obj_arr, axis=0)
            q25 = np.percentile(obj_arr, 25, axis=0)
            q75 = np.percentile(obj_arr, 75, axis=0)
        except:
            # Catch-all error for unexpected failures
            raise RuntimeError(
                "A particle quantum-tunneled where it shouldn't have, please rerun.\n"
            )

        results = {
            "x_opt": self.GOPTI_x,
            "f_opt": self.GOPTI_f,
            "swarms": self.Ns,
            "obj_arr": obj_arr,
            "y_mean": y_mean,
            "y_min": y_min,
            "y_max": y_max,
            "t": self.t,
            "q25": q25,
            "q75": q75,
        }

        return results

    def QFND(self):
        """
         Construct and minimize a quadratic surrogate model in n dimensions
         via interpolation over a given set of sample points.

         This routine builds a full quadratic polynomial surrogate of the form:
             s(x) = c + a^T x + x^T B x
         by solving an interpolation system using function evaluations at NQ points.

         Returns
         -------
         xmin : ndarray of shape (n)
                Optimal point (Argmin / Argmax) of the objective function.
                If the quadratic term matrix is singular, the surrogate value is worse than the global,
                or there are not enough points, defaults to the first point in Q (global best).
         fmin : float
                Objective function value at xmin.

         Method
         ------
         1. Construct the interpolation matrix M ∈ ℝ^{NQ × NQ}
         2. Evaluate the objective
         3. Solve the linear system
         4. Extract coefficients
            - Linear term vector a ∈ ℝ^n
            - Symmetric matrix B ∈ ℝ^{n×n} from quadratic terms
         5. Compute surrogate minimizer
         6. Evaluate
         7. Fallback

         """

        #Initialization
        coeff_mat = np.zeros((self.num_points, self.num_points))
        col = 0

        # Constant Term
        coeff_mat[:, col] = 1
        col += 1
        points = np.array(self.Q)

        # Linear Terms
        for i in range(len(self.bounds)):
            coeff_mat[:, col] = points[:, i]
            col += 1

        # Quadratic Terms
        for i in range(len(self.bounds)):
            for j in range(i, len(self.bounds)):
                coeff_mat[:, col] = points[:, i] * points[:, j]
                col += 1

        values = np.array([self.obj(p) for p in points])

        try:
            # Trying to identify the surrogate (if points not collinear)
            coeffs = np.linalg.solve(coeff_mat, values)
            a = coeffs[1:len(self.bounds) + 1]
            B = np.zeros((len(self.bounds), len(self.bounds)))
            col = len(self.bounds) + 1
            for i in range(len(self.bounds)):
                for j in range(i, len(self.bounds)):
                    B[i, j] = coeffs[col]
                    B[j, i] = coeffs[col]  # symmetry
                    col += 1

            # Minimum occurs at x_min = -0.5 * B^{-1} * a
            x_min = -0.5 * np.linalg.inv(B) @ a
            f_min = self.obj(x_min)

        except:
            # If surrogate not available, fall back on global best
            return self.Q[0], self.obj(self.Q[0])

        return x_min, f_min

class Particle:

    def __init__(self, Np=None, X=None, bounds=None, vmax=None, QS = None):

        self.vmax = vmax
        self.X = X
        self.bounds = bounds
        self.Np = Np
        self.vmax = vmax
        self.QS = QS

        # Randomize initial particle position
        if self.X is None:
            self.X = np.zeros((self.Np, len(self.bounds)))
            for d, (low, high) in enumerate(bounds):
                self.X[:, d] = np.random.uniform(low, high, self.Np)
        else:
            self.X = np.array(X)#(np.array(X, dtype='np.ndarray').reshape(-1, 1))

        # Randomize velocity direction
        rand_dirs = np.random.normal(size=(Np, len(self.bounds)))
        # Normalize to unit length
        rand_dirs /= np.linalg.norm(rand_dirs, axis=1, keepdims=True)
        self.V = self.vmax * rand_dirs
        # Clip velocity
        self.clip_V()

        # Initializing best position with the only known position
        self.pbest = self.X.copy()

    def clip_X(self) -> None:
        # Clip particle position withing the bounds
        if self.bounds is not None:
            for i in range(len(self.bounds)):
                xmin, xmax = self.bounds[i]
                self.X[i] = np.clip(self.X[i], xmin, xmax)

    def clip_V(self) -> None:
        # Clip particle velocity withing the maximum (for convergence)
        if self.bounds is not None:
            self.V = np.clip(self.V, -self.vmax, self.vmax)

    def update_velocity(self,w: float,c1: float,c2: float,gbest: np.ndarray,num_points: int,t: int,
                        obj: Callable[[np.ndarray], float],f_min: float, x_min: np.ndarray) -> None:
        '''
        Updates the particle's velocity using inertia, cognitive,
        and social (or social surrogate) components, then clips it to limits.

        w         : inertia weight
        c1        : cognitive (personal) coefficient
        c2        : social (swarm) coefficient
        gbest     : global best position
        num_points: threshold number of particles required to activate QS update
        t         : current iteration index
        obj       : objective function
        f_min     : value of the surrogate minimum
        x_min     : position associated with the surrogate minimum
        '''

        # Ensure the particle position respects domain bounds before velocity update
        self.clip_X()

        # Inertia component: preserves part of the particle's previous motion
        # Cognitive component: attraction toward the particle's personal best position
        self.V = w * np.array(self.V) + c1 * np.random.rand() * (self.pbest - self.X)

        # Conditional QS-based velocity update:
        # Activated when enough particles exist, optimization has progressed,
        # and the surrogate flag is enabled
        if self.Np >= num_points and t > 0 and self.QS is True:
            # Only apply QS if surrogate minimizer is better than global best position
            if f_min <= obj(gbest):
                # QS velocity: attraction toward surrogate minimizer
                self.V += c2 * np.random.rand() * (x_min - self.X)
        else:
            # Standard social component: attraction toward global best position
            self.V += c2 * np.random.rand() * (gbest - self.X)

        # Clip velocity to remain within predefined maximum limits
        self.V = np.clip(self.V, -self.vmax, self.vmax)

    def update_position(self)-> None:
        # Updates the position of the particle and clips it in case it is outside boundaries.
        self.X += self.V
        self.clip_X()

class ParticleAnimator:
    """
    Collects particle positions during optimization, produces
    a 2D animated visualization of their trajectories, and saves it.
    """

    def __init__(self, obj, swarms, bounds, iterations):

        self.obj = obj                                  # Objective function to be visualized
        self.swarms = swarms                            # List of swarms objects (with particles inside)
        self.bounds = bounds                            # Bounds of the 2D search space
        self.iterations = iterations                    # Total number of optimization iterations
        self.Ns = len(swarms)                           # Number of swarms
        self.iter = []                                  # Iteration indices
        self.history = [[] for _ in range(self.Ns)]     # History of particle positions:
                                                        # history[i][t] = position of particle i at iteration t

    def run(self, i: int, t: int) -> None:
        """
        Creates the history vector appending a copy of the
        current particle position for each t.
        """
        if t not in self.iter:
            self.iter.append(t)

        self.history[i].append(self.swarms[i].X.copy())

    def animate(self) -> None:
        """
        Creates and saves a 2D animation of the particle trajectories
        over a contour plot of the objective function.
        """
        if len(self.bounds) > 2:
            warnings.warn("3D animation not implemented yet!")
        try:
            if len(self.bounds) == 2:
                # Create a grid over the 2D search space
                x = np.linspace(self.bounds[0][0], self.bounds[0][1], 200)
                y = np.linspace(self.bounds[1][0], self.bounds[1][1], 200)
                X, Y = np.meshgrid(x, y)

                # Vectorize the objective function so it can be evaluated over the grid
                # Each (x, y) point is wrapped as expected by the objective function
                vec_obj = np.vectorize(lambda x, y: self.obj(np.array([x, y])))
                Z = vec_obj(X, Y)
                # vec_obj = np.vectorize(lambda x, y: self.obj((np.array([x]), np.array([y]))))
                # Z = vec_obj(X, Y)

                # Colormap used to assign a unique color to each particle
                colors = plt.cm.get_cmap("tab10", self.Ns)

                # Create the figure and axis
                fig, ax = plt.subplots()

                # Draw contour plot of the objective function
                ax.contour(X, Y, Z, levels=10, cmap="viridis", alpha=0.6, zorder=1)

                # Initialize empty scatter plots
                scat = []
                scat = [ax.scatter([], [], c=[colors(i)], s=40, zorder=3) for i in range(self.Ns)]

                # Text box showing the current iteration number
                iter_text = ax.text(
                    0.02, 0.95, f"Iteration: {0}",
                    transform=ax.transAxes,
                    color="black",
                    fontsize=12,
                    ha="left",
                    va="top",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
                )

                def update(frame: int) -> list:
                    """
                    Update function called at each animation frame.
                    Moves each particle to its stored position at the given iteration.
                    """
                    iter_text.set_text(f"Iteration: {self.iter[frame]}/{self.iterations}")
                    for i in range(self.Ns):
                        scat[i].set_offsets(self.history[i][frame])
                    return scat + [iter_text]

                def init() -> None:
                    """
                    Initialization function for the animation.
                    Sets all particles to their initial positions.
                    """
                    iter_text.set_text(f"Iteration: 0")
                    for i in range(self.Ns):
                        scat[i].set_offsets(self.history[i][0])  # each swarm’s initial positions
                    return scat + [iter_text]

                # Create the animation object
                self.anim = animation.FuncAnimation(
                    fig,
                    update,
                    frames=len(self.history[0]),
                    init_func=init,
                    blit=True,
                    interval=150,
                    repeat=True
                )

                # Save the animation as a GIF with a timestamped filename
                self.anim.save(
                    f'pso_animation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.gif',
                    writer="pillow",
                    fps=10
                )

                # Display the animation
                plt.show()
            elif len(self.bounds) == 1:

                # Create a grid over the 1D search space
                x = np.linspace(self.bounds[0][0], self.bounds[0][1], 500)

                # Vectorize the objective function so it can be evaluated over the grid
                # Each (x, y) point is wrapped as expected by the objective function
                vec_obj = np.vectorize(lambda x: self.obj(np.array([x])))
                y = vec_obj(x)
                # Colormap used to assign a unique color to each particle
                colors = plt.cm.get_cmap("tab10", self.Ns)
                # Create the figure and axis
                fig, ax = plt.subplots()

                # Draw contour plot of the objective function
                ax.plot(x, y, color="black", linewidth=2, alpha=0.7)

                # Initialize empty scatter plots
                scat = []
                scat = [ax.scatter([], [], c=[colors(i)], s=40, zorder=3) for i in range(self.Ns)]

                # Text box showing the current iteration number
                iter_text = ax.text(
                    0.02, 0.95, f"Iteration: {0}",
                    transform=ax.transAxes,
                    color="black",
                    fontsize=12,
                    ha="left",
                    va="top",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
                )

                def update(frame: int) -> list:
                    """
                    Update function called at each animation frame.
                    Moves each particle to its stored position at the given iteration.
                    """
                    iter_text.set_text(f"Iteration: {self.iter[frame]}/{self.iterations}")
                    for i in range(self.Ns):
                        x_vals = self.history[i][frame]  # shape (Np, 1)
                        y_vals = np.array([self.obj(pos) for pos in x_vals])
                        offsets = np.column_stack((x_vals.flatten(), y_vals))
                        scat[i].set_offsets(offsets)

                    return scat + [iter_text]

                def init() -> None:
                    """
                    Initialization function for the animation.
                    Sets all particles to their initial positions.
                    """
                    iter_text.set_text(f"Iteration: 0")
                    for i in range(self.Ns):
                        x_vals = self.history[i][0]  # shape (Np, 1)
                        y_vals = np.array([self.obj(pos) for pos in x_vals])
                        offsets = np.column_stack((x_vals.flatten(), y_vals))
                        scat[i].set_offsets(offsets)

                    return scat + [iter_text]

                # Create the animation object
                self.anim = animation.FuncAnimation(
                    fig,
                    update,
                    frames=len(self.history[0]),
                    init_func=init,
                    blit=True,
                    interval=150,
                    repeat=True
                )

                # Save the animation as a GIF with a timestamped filename
                self.anim.save(
                    f'pso_animation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.gif',
                    writer="pillow",
                    fps=10
                )

                # Display the animation
                plt.show()
        except:
            # Catch-all error for unexpected failures
            if len(self.bounds) != 2 and len(self.bounds) != 1:
                raise NotImplementedError(
                    "Visualization is only implemented for 1D and 2D objective functions.\n"
                )
            else:
                raise RuntimeError(
                    "Unknown Error.\n"
                )

"""
        ______________________________________________________________________________________________________________
        
        If you use this software in academic work, please cite:

        M. Clemente and M. Canova, “Quadratic Surrogate Attractor for Particle Swarm Optimization”,
        American Control Conference (ACC), New Orleans, United States, 2026, Available on arXiv:
        https://doi.org/10.48550/arXiv.2603.17163
        
        Bibtex        
        @misc{clemente2026quadraticsurrogateattractorparticle,
              title={Quadratic Surrogate Attractor for Particle Swarm Optimization}, 
              author={Maurizio Clemente and Marcello Canova},
              year={2026},
              eprint={2603.17163},
              archivePrefix={arXiv},
              primaryClass={cs.NE},
              url={https://arxiv.org/abs/2603.17163}, 
        }
        
        APA (7th ed.)
        Maurizio Clemente, & Marcello Canova. (2026). Quadratic surrogate attractor for particle swarm optimization.
        arXiv. https://arxiv.org/abs/2603.17163
        
        Author–Date
        Clemente, M., and Canova, M. 2026. Quadratic Surrogate Attractor for Particle Swarm Optimization. 
        arXiv preprint arXiv:2603.17163. https://arxiv.org/abs/2603.17163
        
        Harvard
        Clemente, M. and Canova, M., 2026. Quadratic Surrogate Attractor for Particle Swarm Optimization. 
        arXiv preprint arXiv:2603.17163. Available at: https://arxiv.org/abs/2603.17163 [Accessed 13 May 2026].
        
        Standard
        CLEMENTE, Maurizio; CANOVA, Marcello. Quadratic Surrogate Attractor for Particle Swarm Optimization. 
        arXiv preprint arXiv:2603.17163, 2026. Available from: https://arxiv.org/abs/2603.17163
        
        RIS
        TY  - PREPRINT
        TI  - Quadratic Surrogate Attractor for Particle Swarm Optimization
        AU  - Clemente, Maurizio
        AU  - Canova, Marcello
        PY  - 2026
        ET  - arXiv:2603.17163
        UR  - https://arxiv.org/abs/2603.17163
        ER  -
        
        Vancouver
        Clemente M, Canova M. Quadratic Surrogate Attractor for Particle Swarm Optimization [preprint].
        arXiv; 2026. Available from: https://arxiv.org/abs/2603.17163

        ______________________________________________________________________________________________________________
        
        Other references:

        [1] M. Clerc and J. Kennedy, “The particle swarm - explosion, stability, and convergence in a 
        multidimensional complex space,” IEEE Transactions on Evolutionary Computation, vol. 6,
        no. 1, pp. 58–73, 2002. 
        
        [2] He, Yan, Ma, Wei Jin, and Zhang, Ji Ping, “The parameters selection of pso algorithm 
        influencing on performance of fault diagnosis,” MATEC Web Conf., vol. 63, p. 02019, 2016.
        
        [3] N. M. Kwok, Q. P. Ha, D. K. Liu, G. Fang, and K. C. Tan, “Efficient particle swarm 
        optimization: a termination condition based on the decision-making approach,” in 2007 
        IEEE Congress on Evolutionary Computation, pp. 3353–3360, 2007
        
        [4] Y. Shi and R. Eberhart, “A modified particle swarm optimizer,” in
        1998 IEEE International Conference on Evolutionary Computation
        Proceedings. IEEE World Congress on Computational Intelligence
        (Cat. No.98TH8360), pp. 69–73, 1998.
        
        [5] G. Sermpinis, K. Theofilatos, A. Karathanasopoulos, E. F. Georgopoulos, and C. Dunis, “Forecasting foreign exchange rates with
        adaptive neural networks using radial-basis functions and particle
        swarm optimization,” European Journal of Operational Research,
        vol. 225, no. 3, pp. 528–540, 2013.

"""