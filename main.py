import numpy as np

def solve_defined_problem(algorithm,
                          problem,
                          n_stages,
                          n_batches,
                          hyper_dic):

    if problem.upper() == "PO":
        from problems.production_optimization import make_problem_definition, MSP_problem

    else:
        from problems.energy_planning import make_problem_definition, MSP_problem

    problem_definition = make_problem_definition(n_stages, n_batches, algorithm)

    if algorithm.upper() == "DVFN":
        from solver.dvfn_solver import DVFN
        solver = DVFN(problem_definition,
                      hyper_dic)

        sol, obj = solver.solve()
        time = solver.iter_time

        return np.array(sol), obj, np.array(time)

    elif algorithm.upper() == "MSP":

        MSP_sol, MSP_obj, MSP_time = MSP_problem(n_stages, scenario_node=n_batches)

        return MSP_sol, MSP_obj, MSP_time

    elif algorithm.upper() == "SDDP":
        from solver.vfgl_sddp_solver import solver

        solver = solver(problem_definition, paramdict={})
        solver.algorithm = algorithm
        solver.burnIn = 0
        solver.minIter = hyper_dic['min_iter']
        solver.maxIter = hyper_dic['max_iter']
        solver.roundSolution = -1
        solver.solve()
        obj, _ = solver.find_SDDP_upper_bound(iter=100)
        sol_list = []
        for i in range(len(solver.forwardSol)):
            sol_list.append(solver.forwardSol[i][0])

        sol_list = np.array(sol_list)
        sol = sol_list
        sol = sol.reshape((sol.shape[0], sol.shape[1], 1))

        return sol, obj, np.array(solver.iterTime)

    else:
        from solver.vfgl_sddp_solver import solver

        solver = solver(problem_definition, paramdict={})
        solver.algorithm = "VFGL"
        solver.form = algorithm[4:]
        solver.initialize_VF()
        solver.burnIn = 0
        solver.minIter = hyper_dic['min_iter']
        solver.maxIter = hyper_dic['max_iter']
        solver.roundSolution = -1
        solver.solve()
        KD_mean, KD_std, obj_mean, obj_std = solver.calc_ave_KKT_deviation(100)
        sol_list = []
        for i in range(len(solver.forwardSol)):
            sol_list.append(solver.forwardSol[i][0])

        sol_list = np.array(sol_list)
        sol = sol_list
        sol = sol.reshape((sol.shape[0], sol.shape[1], 1))

        return sol, obj_mean, np.array(solver.iterTime)

# User defined setting
algorithm = "DVFN" # You have 6 options 1) MSP 2) SDDP 3) VFGLexp 4) VFGLquad 5) VFGLlinear 6) DVFN

problem = "PO" # You have 2 options 1) PO 2) EP

n_stages = 11

n_batches = 3

hyper_dic = {}
hyper_dic['min_iter'] = 75
hyper_dic['max_iter'] = 75
hyper_dic['n_epochs'] = 5 # for only DVFN
hyper_dic['n_nodes'] = 64 # for only DVFN
hyper_dic['n_hlayers'] = 1 # for only DVFN
hyper_dic['optimizer'] = "Adam" # for only DVFN
hyper_dic['lr'] = 0.0015 # for only DVFN
hyper_dic['activation'] = 'softplus' # for only DVFN

sol, obj, time = solve_defined_problem(algorithm,
                                       problem,
                                       n_stages,
                                       n_batches,
                                       hyper_dic)

print("First Stage Solution: ")
print(sol[-1])
print("Objective Value: ", obj)
print("Total Elapsed Time: ", sum(time))
