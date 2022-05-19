from module.dvfn_engine import DVFN

import numpy as np

from problems.production_optimization import make_problem_definition

# obj_list = []
# forward_sol_list = []
# par_change_list = []
# time_list = []
for num in range(50):
    problem_definition = make_problem_definition(n_stages=7)
    solver = DVFN(problem_definition)
    solver.solve()
    # obj = solver.simulate_solution(sol=solver.forward_sol[solver.iteration_count - 1])
    # obj_list.append(obj)

    np.save("trained/po/new_3/forward_sol_{}".format(num), solver.forward_sol)
    np.save("trained/po/new_3/par_change_{}".format(num), solver.par_change)
    np.save("trained/po/new_3/time_{}".format(num), solver.iter_time)

    for stage in range(1, solver.n_stages):
        solver.ICNNs[stage].save("icnn/po/new_3/vf_stage{}_{}.h5".format(stage, num))

    import matplotlib.pyplot as plt

    plt.plot(solver.iter_time)
    plt.ylim(0, 25)
    plt.savefig("figs/po/new_3/time_plot_{}".format(num))
    plt.show()

    x = np.arange(0, solver.iteration_count)
    y1 = []
    y2 = []
    y3 = []
    for i in range(solver.iteration_count):
        y1.append(solver.forward_sol[i][0, 0])
        y2.append(solver.forward_sol[i][1, 0])
        y3.append(solver.forward_sol[i][2, 0])
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.plot(x, y3)
    plt.legend(['production1', 'production2', 'production3'])
    plt.savefig("figs/po/new_3/first_sol_plot_{}".format(num))
    plt.show()