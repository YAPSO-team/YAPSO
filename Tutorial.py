from Yapso import YAPSO
import matplotlib.pyplot as plt
import numpy as np

# This piece of code just makes picture using Latex font (which I prefer), feel free to remove it.
# <editor-fold desc="Latex Figures">
J = 'Objective Function'
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": "Helvetica",
})
# plt.rcParams['figure.dpi']=200

SMALL_SIZE = 16
MEDIUM_SIZE = 22
BIGGER_SIZE = 26

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# </editor-fold>

# The only two required inputs are the OBJECTIVE FUNCTION and the OPTIMIZATION BOUNDS.
# The function obj is N-dimensional (callable) and I include here the examples used in the paper.
#
#      Maurizio Clemente, Marcello Canova, "Quadratic Surrogate Attractor for
#      Particle Swarm Optimization", 2026, arXiv:2603.17163.
#      https://doi.org/10.48550/arXiv.2603.17163

def obj(X):
    X = np.asarray(X)

    # Sphere function ND:
    # f = np.sum(X ** 2)

    # Flower function ND:
    # f = np.sum(np.log(np.abs(X) + 1))

    # Griewank function ND:
    # f = np.sum(X ** 2 / 4000.0) - np.prod(np.cos(X / np.sqrt(np.arange(1, len(X) + 1)))) + 1

    # Ackley function ND:
    f = (-20 * np.exp(-0.2 * np.sqrt(np.sum(X**2, axis=0) / X.shape[0])) - np.exp(np.sum(np.cos(2 * np.pi * X), axis=0) / X.shape[0]) + np.e + 20)
    return f

# The bounds are passed as a list of [lower, upper] float values defining the search space per dimension.
Nd = 2
bounds = [[-5.12, 5.12] for _ in range(Nd)]

# The rest of the function arguments are OPTIONAL.
# It is possible to modify the inertia weight w, cognitive coefficient c1 (attraction to personal best), social
# coefficient c2 (attraction to global/surrogate best), and maximum speed vmax of the particles. If these are not specified
# (or set to None), the solver will automatically use values that guarantee convergence as for one of the papers
# in the references.

w = None        # (Optional) Inertia weight - float.
c1 = None       # (Optional) Cognitive coefficient - float.
c2 = None       # (Optional) Social coefficient (attraction to global/surrogate best) - float.
vmax = None     # (Optional) Maximum speed of the particles - float.

# If the number of swarms Ns is None or not specified it is automatically set to 1.
# If the number of particles per swarm Np is None or not specified, it is automatically set to the minimum
# to employ QS algorithm.

Ns = 2          # (Optional) Number of independent swarms (multi-swarm configuration) - int.
Np = 6          # (Optional) Number of particles per swarm - int.
QS = True       # (Optional) Quadratic surrogate attraction mechanism - True/False
stats = False   # (Optional) Enables a larger output of the function, more information after YAPSO call - True/False
verbose = True  # (Optional) Enables console logging and diagnostics - True/False.
animate = True  # (Optional) Enables (AND SAVE!) runtime animations - True/False.
min = True      # (Optional) True for minimization, False for maximization - True/False.

# In case the maximum number of iteration is None or not specified, the default is set to 300.
# When self-termination is enabled, it is possible to specify a different false-alarm threshold for convergence
# P(S* >= kappa | H0) <= epsilon. Its enforced lower bound is the machine precision.

Imax = 50       # (Optional) Maximum number of PSO iterations - int.
ST = True       # (Optional) Self-termination (early stopping logic) - True/False.
epsilon = 0     # (Optional) False-alarm threshold - float.

# To counteract the risk that the decay of the inertia parameter may cause particles to become trapped in local minima,
# the exploration safeguard described in Section III-C of Kwok et al. 2007 has been implemented. If tau is None or not
# specified, it is set to 1.2.

tau = None      # (Optional) exploration gain factor - float.

# In case the user wants to test convergence from a specific set of points instead of randomly generated ones, it is
# possible to specify the list of points. Each of the Np lines represents a different particle as a list of coordinates.

# e.g., [[ 3.50205784,  2.94871913], [ 0.34583032, -3.05420814],[-2.66068532, -2.49320437],[ 0.40450498,  2.67170664],
# [ 4.83401638, -2.60894226],[-1.8257552,  -4.26186218]].

X0 = None       # (Optional) Initial particle position - ndarray.

# If specified, X0 will initialize EVERY SWARM to start from the set of points.


# The command to run the optimizer is: YAPSO(obj, bounds, optionals).start()

run_data = YAPSO(obj, bounds, Ns=Ns, Np=Np, w=w, c1=c1, c2=c2, Imax=Imax,
                 X0=X0, stats=stats, animate=animate, min=min, verbose=verbose, epsilon = epsilon,
                 tau = tau, vmax = vmax, ST = ST, QS = QS ).start()

# Depending on the value of stats, the user will get different outputs.
# If stats is False, the function will return

# opt_x:      Location of the optimal solution - ndarray.
# opt_f:      Value of the optimal solution - float.

# If stats is True, the function will return

# results.opt_x:      Location of the optimal solution - ndarray.
# results.opt_f:      Value of the optimal solution - float.
# results.swarms:     Final swarm states for all Ns swarms - int.
# results.obj_arr:    Best objective function over iterations - ndarray.
# results.y_mean:     Mean objective value across swarms per iteration - ndarray.
# results.y_min:      Minimum objective value registered per iteration - ndarray.
# results.y_max:      Maximum objective value registered per iteration - ndarray.
# results.t:          Iteration performed - int.
# results.q25:        25th percentile of objective distribution - ndarray.
# results.q75:        75th percentile of objective distribution - ndarray.

if stats:

    print(f'Optimal x: {run_data["x_opt"]}')
    print(f'Optimal f: {run_data["f_opt"]}')

    print(f'Swarm that found the optimal solution: {run_data["swarms"]}')
    print(f'Average objective value among swarms per iteration: {run_data["y_mean"]}')
    print(f'Minimum objective value among swarms per iteration: {run_data["y_min"]}')
    print(f'Iterations executed: {run_data["t"]}')
    print(f'Lower interquartile range value among swarms per iteration: {run_data["q25"]}')
    print(f'Upper interquartile range value among swarms per iteration: {run_data["q75"]}')
    print(f'Array with the global best per iteration: {run_data["obj_arr"]}')

else:

    print(run_data)

