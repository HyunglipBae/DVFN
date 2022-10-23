from module.dvfn_engine import DVFN
import numpy as np
from problems.energy_planning import make_problem_definition

def solve_EP(n_stages=15,
             n_nodes=64,
             n_hlayers=1,
             n_epochs=5,
             ICNN_optimizer="Adam",
             lr=0.001,
             min_iter=150,
             max_iter=150,
             h_cost=2,
             t_cost=7):

    problem_definition = make_problem_definition(n_stages=n_stages, h_cost=h_cost, t_cost=t_cost)
    solver = DVFN(problem_definition,
                  n_nodes,
                  n_hlayers,
                  n_epochs,
                  ICNN_optimizer,
                  lr,
                  min_iter,
                  max_iter,
                  h_cost,
                  t_cost)

    sol, obj = solver.solve()

    return sol, obj

# User defined setting
n_stages = 15
n_nodes = 64
n_hlayers = 1
n_epochs = 5
ICNN_optimizer = "Adam"
lr = 0.001
min_iter = 150
max_iter = 150

perturbation_list = [[2, 5], [2, 5.5], [2, 6], [2, 6.5], [2, 7], [2, 7.5], [2, 8], [3, 7], [3.5, 7], [4, 7], [4.5, 7], [5, 7]]
for perturbation in perturbation_list:
    h_cost = perturbation[0]
    t_cost = perturbation[1]
    sol, obj = solve_EP(n_stages, n_nodes, n_hlayers, n_epochs, ICNN_optimizer, lr, min_iter, max_iter, h_cost, t_cost)
    print("Hydro: ", np.reshape(sol, sol.shape[0])[2])
    print("Thermal: ", np.reshape(sol, sol.shape[0])[3])
    print("Objective value: ", obj)
