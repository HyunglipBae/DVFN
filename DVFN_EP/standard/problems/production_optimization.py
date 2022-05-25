import numpy as np
import sympy as sp

# parameter ====================================================================
storage_c = [3.0, 7.0, 10.0]
production_c = [1.0, 2.0, 5.0]
outsourcing_c = [6.0, 12.0, 20.0]
maximum_resource = 10.0
pdim = 3

random_demand = [(5.0, 3.0, 1.0), (6.0, 2.0, 1.0), (1.0, 2.0, 2.0)]
# ==============================================================================


def ScenarioGenerator():
    """
    Generate random variable for stochastic process
    :return: sample: random demand for next stage
    """
    while True:
        for sample in random_demand:
            yield sample


def decomposed_problems(n_stages):
    """
    Generate stagewise decomposed problem
    :param n_stages: The number of stages
    :returns: problem: List of dictionary for decomposed problems
              stage_to_prob: Mapping from stage to stagewise problem (t=0 -> 0, 0<t<T -> 1, t=T -> 2)
    """
    problem = []

    # stage 0 problem ============================================================
    x = sp.Matrix(sp.symbols(' '.join(['x_' + str(i) for i in range(pdim)])))
    y = sp.Matrix(sp.symbols(' '.join(['y_' + str(i) for i in range(pdim)])))
    s = sp.Matrix(sp.symbols(' '.join(['s_' + str(i) for i in range(pdim)])))

    decision_var = sp.Matrix([x, y, s])
    VF_var_idx = [6, 7, 8]

    objective = s.dot(storage_c) + y.dot(outsourcing_c)
    ineq_constraints = []
    eq_constraints = []

    # initial condition
    ineq_constraints += [x.dot(production_c) - maximum_resource]
    eq_constraints += [s[i] - x[i] - y[i] for i in range(pdim)]

    # Non-negativity
    for var in decision_var:
        ineq_constraints += [(-1) * var]

    slack = sp.Matrix([sp.symbols('slack')])
    decision_var_ff = sp.Matrix([decision_var, slack])
    ineq_constraints_ff = ineq_constraints.copy()
    for i in range(len(ineq_constraints_ff)):
        ineq_constraints_ff[i] -= decision_var_ff[-1]

    stage_0 = {'objective': objective,
               'objective_ff': slack[0],
               'ineq_constraints': ineq_constraints,
               'ineq_constraints_ff': ineq_constraints_ff,
               'eq_constraints': eq_constraints,
               'VF_var_idx': VF_var_idx,
               'decision_var': decision_var,
               'decision_var_ff': decision_var_ff,
               'previous_stage_var': [],
               'random_var': []
               }

    problem.append(stage_0)

    # 0 < stage < T ==============================================================
    x = sp.Matrix(sp.symbols(' '.join(['x_' + str(i) for i in range(pdim)])))
    y = sp.Matrix(sp.symbols(' '.join(['y_' + str(i) for i in range(pdim)])))
    s = sp.Matrix(sp.symbols(' '.join(['s_' + str(i) for i in range(pdim)])))
    decision_var = sp.Matrix([x, y, s])
    VF_var_idx = [6, 7, 8]

    prevs = sp.Matrix(sp.symbols(' '.join(['prevs_' + str(i) for i in range(pdim)])))
    previous_stage_var = sp.Matrix([prevs])

    d = sp.Matrix(sp.symbols(' '.join(['d_' + str(i) for i in range(pdim)])))
    random_var = sp.Matrix([d])

    objective = s.dot(storage_c) + y.dot(outsourcing_c)

    ineq_constraints = []
    eq_constraints = []

    # initial condition
    ineq_constraints += [x.dot(production_c) - maximum_resource]
    eq_constraints += [s[i] - prevs[i] - x[i] - y[i] + d[i] for i in range(pdim)]

    # Non-negativity
    for var in decision_var:
        ineq_constraints += [(-1) * var]

    slack = sp.Matrix([sp.symbols('slack')])
    decision_var_ff = sp.Matrix([decision_var, slack])
    ineq_constraints_ff = ineq_constraints.copy()
    for i in range(len(ineq_constraints_ff)):
        ineq_constraints_ff[i] -= decision_var_ff[-1]

    stage_t = {'objective': objective,
               'objective_ff': slack[0],
               'ineq_constraints': ineq_constraints,
               'ineq_constraints_ff': ineq_constraints_ff,
               'eq_constraints': eq_constraints,
               'VF_var_idx': VF_var_idx,
               'decision_var': decision_var,
               'decision_var_ff': decision_var_ff,
               'previous_stage_var': previous_stage_var,
               'random_var': random_var
               }
    problem.append(stage_t)

    # stage T problem ============================================================
    x = sp.Matrix(sp.symbols(' '.join(['x_' + str(i) for i in range(pdim)])))
    y = sp.Matrix(sp.symbols(' '.join(['y_' + str(i) for i in range(pdim)])))
    s = sp.Matrix(sp.symbols(' '.join(['s_' + str(i) for i in range(pdim)])))
    decision_var = sp.Matrix([x, y, s])
    VF_var_idx = [6, 7, 8]

    prevs = sp.Matrix(sp.symbols(' '.join(['prevs_' + str(i) for i in range(pdim)])))
    previous_stage_var = sp.Matrix([prevs])

    d = sp.Matrix(sp.symbols(' '.join(['d_' + str(i) for i in range(pdim)])))
    random_var = sp.Matrix([d])

    objective = y.dot(outsourcing_c)

    ineq_constraints = []
    eq_constraints = []

    # initial condition
    ineq_constraints += [x.dot(production_c) - maximum_resource]
    eq_constraints += [s[i] - prevs[i] - x[i] - y[i] + d[i] for i in range(pdim)]

    # Non-negativity
    for var in decision_var:
        ineq_constraints += [(-1) * var]

    slack = sp.Matrix([sp.symbols('slack')])
    decision_var_ff = sp.Matrix([decision_var, slack])
    ineq_constraints_ff = ineq_constraints.copy()
    for i in range(len(ineq_constraints_ff)):
        ineq_constraints_ff[i] -= decision_var_ff[-1]

    stage_T = {'objective': objective,
               'objective_ff': slack[0],
               'ineq_constraints': ineq_constraints,
               'ineq_constraints_ff': ineq_constraints_ff,
               'eq_constraints': eq_constraints,
               'VF_var_idx': VF_var_idx,
               'decision_var': decision_var,
               'decision_var_ff': decision_var_ff,
               'previous_stage_var': previous_stage_var,
               'random_var': random_var
               }

    problem.append(stage_T)

    # Index set definition
    stage_to_prob = []
    for i in range(n_stages):
        if i == 0:
            stage_to_prob.append(0)
        elif i == n_stages - 1:
            stage_to_prob.append(2)
        else:
            stage_to_prob.append(1)

    return problem, stage_to_prob


def make_problem_definition(n_stages):
    """
    Generate information of desired problem
    :param n_stages: The number of stages
    :return: problem_definition: Dictionary for information of problem
    """
    sg = ScenarioGenerator()
    problem, stage_to_prob = decomposed_problems(n_stages)
    problem_definition = {'n_stages': n_stages,
                          'ScenarioGenerator': sg,
                          'problem': problem,
                          'stage_to_prob': stage_to_prob}

    return problem_definition