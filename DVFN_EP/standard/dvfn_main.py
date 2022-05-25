from module.dvfn_engine import DVFN
import numpy as np
from problems.energy_planning import make_problem_definition

def solve_EP(n_stages=7,
             n_nodes=128,
             n_hlayers=1,
             n_epochs=20,
             ICNN_optimizer="Adam",
             lr=0.001,
             min_iter=200,
             max_iter=200):

    problem_definition = make_problem_definition(n_stages=n_stages)
    solver = DVFN(problem_definition,
                  n_nodes,
                  n_hlayers,
                  n_epochs,
                  ICNN_optimizer,
                  lr,
                  min_iter,
                  max_iter)

    sol, obj = solver.solve()

    return sol, obj

# User defined setting
n_stages = 7
n_nodes = 128
n_hlayers = 1
n_epochs = 20
ICNN_optimizer = "Adam"
lr = 0.001
min_iter = 200
max_iter = 200

sol, obj = solve_EP(n_stages, n_nodes, n_hlayers, n_epochs, ICNN_optimizer, lr, min_iter, max_iter)
print("Hydro: ", np.reshape(sol, sol.shape[0])[2])
print("Thermal: ", np.reshape(sol, sol.shape[0])[3])
print("Objective value: ", obj)