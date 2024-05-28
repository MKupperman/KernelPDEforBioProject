import numpy as np
from clawpack import riemann
import matplotlib.pyplot as plt

def setup(nx=100, kernel_language='Python', use_petsc=False, solver_type='classic', weno_order=5, 
          time_integrator='SSP104', outdir='./_output', beta=100, gamma=0, x0=0.75):

    if use_petsc:
        import clawpack.petclaw as pyclaw
    else:
        from clawpack import pyclaw

    if kernel_language == 'Fortran':
        riemann_solver = riemann.advection_1D
    elif kernel_language == 'Python':
        riemann_solver = riemann.advection_1D_py.advection_1D
            
    if solver_type=='classic':
        solver = pyclaw.ClawSolver1D(riemann_solver)
    elif solver_type=='sharpclaw':
        solver = pyclaw.SharpClawSolver1D(riemann_solver)
        solver.weno_order = weno_order
        solver.time_integrator = time_integrator
        if time_integrator == 'SSPLMMk3':
            solver.lmm_steps = 5
            solver.check_lmm_cond = True
    else:
        raise Exception('Unrecognized value of solver_type.')

    solver.kernel_language = kernel_language

    solver.bc_lower[0] = pyclaw.BC.periodic
    solver.bc_upper[0] = pyclaw.BC.periodic

    x = pyclaw.Dimension(0.0, 1.0, nx, name='x')
    domain = pyclaw.Domain(x)
    state = pyclaw.State(domain, solver.num_eqn)

    state.problem_data['u'] = 1.  # Advection velocity

    # Initial data
    xc = state.grid.x.centers
    state.q[0, :] = np.exp(-beta * (xc-x0)**2) * np.cos(gamma * (xc - x0))

    claw = pyclaw.Controller()
    claw.keep_copy = True
    claw.solution = pyclaw.Solution(state, domain)
    claw.solver = solver

    if outdir is not None:
        claw.outdir = outdir
    else:
        claw.output_format = None

    claw.tfinal = 1.0
    claw.setplot = setplot

    return claw

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

def solve_1D_advection(dt, u0, c, T_final, forcing=None, nx=100):
    from clawpack.pyclaw.util import run_app_from_main
    claw = setup(nx=nx, outdir=None)
    claw.tfinal = T_final

    if forcing is not None:
        claw.solution.state.problem_data['forcing'] = forcing

    claw.solution.state.q[0, :] = u0
    claw.solver.dt_initial = dt
    claw.run()

    t_nodes = np.linspace(0, T_final, len(claw.frames))
    u_nodes = np.array([frame.q[0, :] for frame in claw.frames])
    
    return t_nodes, u_nodes

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
