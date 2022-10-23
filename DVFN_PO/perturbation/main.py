from module.dvfn_engine import DVFN
import numpy as np
from problems.production_optimization import make_problem_definition

def solve_PO(n_stages=11,
             n_nodes=64,
             n_hlayers=1,
             n_epochs=5,
             ICNN_optimizer="Adam",
             lr=0.0015,
             min_iter=75,
             max_iter=75,
             maximum_resource=10):

    problem_definition = make_problem_definition(n_stages=n_stages, maximum_resource=maximum_resource)
    solver = DVFN(problem_definition,
                  n_nodes,
                  n_hlayers,
                  n_epochs,
                  ICNN_optimizer,
                  lr,
                  min_iter,
                  max_iter,
                  maximum_resource)

    sol, obj = solver.solve()

    return sol, obj

# User defined setting
n_stages = 11
n_nodes = 64
n_hlayers = 1
n_epochs = 5
ICNN_optimizer = "Adam"
lr = 0.0015
min_iter = 75
max_iter = 75

perturbation_list = np.arange(8, 15, 15)
for perturbation in perturbation_list:
    sol, obj = solve_PO(n_stages, n_nodes, n_hlayers, n_epochs, ICNN_optimizer, lr, min_iter, max_iter, perturbation)
    print("Production: ", np.reshape(sol, sol.shape[0])[0:3])
    print("Outsource: ", np.reshape(sol, sol.shape[0])[3:6])
    print("Storage: ", np.reshape(sol, sol.shape[0])[6:9])
    print("Objective value: ", obj)
