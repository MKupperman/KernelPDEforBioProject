from functools import partial

import math

import numpy as np
from clawpack import riemann
import matplotlib.pyplot as plt

# source/sink function should be of this form
# def f(x, q):
#     """
#     Source term function ψ given by f(x, q) = 1 + x/2.
#     """
#     return 1 + x/2

def step_source(solver, state, dt, f):
    """
    Step source function to integrate the source term over a time step.
    """
    q = state.q[0, :]
    x = state.grid.x.centers
    source_term = f(x, q)
    state.q[0, :] += dt * source_term

def auxinit(state, ux):
    # Initialize petsc Structures for aux
    # ux must be vectorized.
    xc=state.grid.x.centers
    state.aux[0,:] = ux(xc)

def setup(u0, ux, forcing, x_min=0.0, x_max=1.0, nx=100, kernel_language='Python', use_petsc=False,
          solver_type='classic', weno_order=5, T_final=1.0,
          time_integrator='SSP104', outdir='./_output'):


    # First, setup the Riemann solver
    if use_petsc:
        import clawpack.petclaw as pyclaw
    else:
        from clawpack import pyclaw

    if solver_type=='classic':
        if kernel_language == 'Fortran':
            solver = pyclaw.ClawSolver1D(riemann.advection_color_1D)
        elif kernel_language=='Python':
            solver = pyclaw.ClawSolver1D(riemann.vc_advection_1D_py.vc_advection_1D)
    elif solver_type=='sharpclaw':
        if kernel_language == 'Fortran':
            solver = pyclaw.SharpClawSolver1D(riemann.advection_color_1D)
        elif kernel_language=='Python':
            solver = pyclaw.SharpClawSolver1D(riemann.vc_advection_1D_py.vc_advection_1D)
        solver.weno_order=weno_order
    else: raise Exception('Unrecognized value of solver_type.')


    solver.kernel_language = kernel_language
    solver.step_source = partial(lambda solver, state, dt: step_source(solver, state, dt, forcing))
    # Set the boundary conditions
    solver.bc_lower[0] = pyclaw.BC.periodic
    solver.bc_upper[0] = pyclaw.BC.periodic
    solver.aux_bc_lower[0] = pyclaw.BC.periodic
    solver.aux_bc_upper[0] = pyclaw.BC.periodic

    # Setup the domain
    x = pyclaw.Dimension(x_min, x_max, nx, name='x')
    domain = pyclaw.Domain(x)
    state = pyclaw.State(domain, solver.num_eqn, num_aux=1)

    # Set the advection velocity - we'll change this later
    # state.problem_data['u'] = 1  # Advection velocity

    auxinit(state, ux=ux)

    # Initial data
    state.q[0, :] = u0(domain.grid.x.centers)
    # np.exp(-beta * (xc-x0)**2) * np.cos(gamma * (xc - x0))

    claw = pyclaw.Controller()
    claw.keep_copy = True
    claw.solution = pyclaw.Solution(state, domain)
    claw.solver = solver

    if outdir is not None:
        claw.outdir = outdir
    else:
        claw.output_format = None

    claw.tfinal = T_final
    # claw.setplot = setplot

    return claw


def solve_1D_advection(x_min, x_max, dt, u0, ux, T_final, forcing, nx=100, pyvis=False):
    from clawpack.pyclaw.util import run_app_from_main
    if pyvis == False:
        outdir = None
    else:
        outdir = './_output'
    claw = setup(u0=u0, x_min=x_min, x_max=x_max, nx=nx, outdir=outdir,
                 ux=ux, T_final=T_final, forcing=forcing)
    claw.tfinal = T_final

    # if forcing is not None:
        # claw.solution.state.problem_data['forcing'] = forcing

    # claw.solution.state.q[0, :] = u0
    claw.solver.dt_initial = dt
    claw.num_output_times = math.ceil(T_final / dt)  # Change the number of frames here
    claw.run()


    t_nodes = np.linspace(0, T_final, len(claw.frames))
    u_nodes = np.array([frame.q[0, :] for frame in claw.frames])

    return t_nodes, u_nodes


# Plotting code and checks/main

def setplot(plotdata):
    """
    Plot solution using VisClaw.
    """
    plotdata.clearfigures()  # clear any old figures, axes, items data

    plotfigure = plotdata.new_plotfigure(name='q', figno=1)

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.ylimits = [-.2, 1.0]
    plotaxes.title = 'q'

    # Set up for item on these axes:
    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = 0
    plotitem.plotstyle = '-o'
    plotitem.color = 'b'
    plotitem.kwargs = {'linewidth':2, 'markersize':5}

    return plotdata

if __name__ == "__main__":
    from clawpack.pyclaw.util import run_app_from_main
    output = run_app_from_main(setup, setplot)
    claw = setup()
    claw.run()
    for frame in claw.frames:
        plt.figure()
        plt.plot(frame.state.grid.x.centers, frame.state.q[0, :], '-o', linewidth=2, markersize=5)
        plt.ylim(-0.1, 1.1)
        plt.title('q at time = {}'.format(frame.t))
        plt.xlabel('x')
        plt.ylabel('q')
        plt.show()
