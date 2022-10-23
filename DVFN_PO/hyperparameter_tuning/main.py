from module.dvfn_engine import DVFN
import numpy as np
from problems.production_optimization import make_problem_definition
import tensorflow as tf

def solve_PO(n_stages=11,
             n_nodes=64,
             n_hlayers=1,
             n_epochs=5,
             ICNN_optimizer="Adam",
             lr=0.0015,
             min_iter=75,
             max_iter=75):

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
    loss = solver.ICNN0_loss

    return sol, obj, loss

# Hyperparameter tuning
n_simulations = 10
n_stages = 11
# n_nodes = 64
# n_hlayers = 1
# n_epochs = 30
# ICNN_optimizer = "Adam"
# lr = 0.0015

hyper_dic = {'epoch': [5, 10, 15, 20],
             'node': [32, 64, 128, 256],
             'hlayer': [1, 2, 3],
             'optimizer': ['Adam', 'Adagrad', 'SGD', 'RMSprop'],
             'lr': [0.0005, 0.001, 0.0015, 0.002]}

min_median_idx = {}
for i, hyperparameter in enumerate(['epoch', 'node', 'hlayer', 'optimizer', 'lr']):
    median_loss_list = []
    for parameter in hyper_dic[hyperparameter]:
        loss_list = []
        for j in range(n_simulations):
            if i == 0:
                sol, obj, loss = solve_PO(n_epochs=parameter)
            elif i == 1:
                sol, obj, loss = solve_PO(n_epochs=hyper_dic['epoch'][min_median_idx['epoch']],
                                          n_nodes=parameter)
            elif i == 2:
                sol, obj, loss = solve_PO(n_epochs=hyper_dic['epoch'][min_median_idx['epoch']],
                                          n_nodes=hyper_dic['node'][min_median_idx['node']],
                                          n_hlayers=parameter)
            elif i == 3:
                sol, obj, loss = solve_PO(n_epochs=hyper_dic['epoch'][min_median_idx['epoch']],
                                          n_nodes=hyper_dic['node'][min_median_idx['node']],
                                          n_hlayers=hyper_dic['hlayer'][min_median_idx['hlayer']],
                                          ICNN_optimizer=parameter)
            elif i == 4:
                sol, obj, loss = solve_PO(n_epochs=hyper_dic['epoch'][min_median_idx['epoch']],
                                          n_nodes=hyper_dic['node'][min_median_idx['node']],
                                          n_hlayers=hyper_dic['hlayer'][min_median_idx['hlayer']],
                                          ICNN_optimizer=hyper_dic['optimizer'][min_median_idx['optimizer']],
                                          lr=parameter)
            loss_list.append(loss[-1])
        median_loss_list.append(np.median(loss_list))
    min_median_idx[hyperparameter] = np.where(median_loss_list == min(median_loss_list))[0][0]

print('{:=^80}'.format('Hyperparameter Result'))
print('{object: <30}: {value}'.format(object='Number of epoch per iteration', value=str(hyper_dic['epoch'][min_median_idx['epoch']])))
print('{object: <30}: {value}'.format(object='Number of Node', value=str(hyper_dic['node'][min_median_idx['node']])))
print('{object: <30}: {value}'.format(object='Number of hidden layer', value=str(hyper_dic['hlayer'][min_median_idx['hlayer']])))
print('{object: <30}: {value}'.format(object='ICNN optimizer', value=str(hyper_dic['optimizer'][min_median_idx['optimizer']])))
print('{object: <30}: {value}'.format(object='Learning rate', value=str(hyper_dic['lr'][min_median_idx['lr']])))
print('{:=^80}'.format(''))
