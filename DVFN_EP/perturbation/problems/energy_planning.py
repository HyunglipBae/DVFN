import numpy as np
import sympy as sp

# parameter ====================================================================
# h_cost = [2.0]
# t_cost = [7.0]
utility_coeff = [0.1]
utility_scale = 5.0
initial_reservoir = 40.0
demand = 20.0
mean = 20.0
scale = 5.0
# ==============================================================================

def ScenarioGenerator(batchNum=3):
    """
    Generate random variable for stochastic process
    :param batchNum: The number of sample in one batch
    :return: sample: random water inflow for next stage
    """
    while True:
        batch_sample = np.random.normal(loc=mean, scale=scale, size=batchNum)
        normalized_batch_sample = (batch_sample - np.mean(batch_sample))
        normalized_batch_sample = normalized_batch_sample/np.std(batch_sample)
        rescaled = normalized_batch_sample*scale
        rescaled = rescaled + mean
        for sample in rescaled:
            yield [sample]


def decomposed_problems(n_stages, h_cost, t_cost):
    """
    Generate stagewise decomposed problem
    :param n_stages: The number of stages
    :returns: problem: List of dictionary for decomposed problems
              stage_to_prob: Mapping from stage to stagewise problem (t=0 -> 0, 0<t<T -> 1, t=T -> 2)
    """
    problem = []

    # stage 0 problem ============================================================
    water_init_0 = sp.Matrix(sp.symbols(['water_init_0']))
    water_final_0 = sp.Matrix(sp.symbols(['water_final_0']))
    hydro_0 = sp.Matrix(sp.symbols(['hydro_0']))
    thermal_0 = sp.Matrix(sp.symbols(['thermal_0']))
    decision_var = sp.Matrix([water_init_0, water_final_0, hydro_0, thermal_0])

    VF_var_idx = [1]

    objective = hydro_0.dot(h_cost) + thermal_0.dot(t_cost) + sp.exp(-water_final_0.dot(utility_coeff) + utility_scale)
    ineq_constraints = []
    eq_constraints = []

    # initial condition
    eq_constraints += [water_init_0[0] - initial_reservoir]
    eq_constraints += [water_final_0[0] - water_init_0[0] + hydro_0[0]]
    ineq_constraints += [demand - hydro_0[0] - thermal_0[0]]

    # Non-negativity
    for i, var in enumerate(decision_var):
        if i != 0:
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
    water_init_t = sp.Matrix(sp.symbols(['water_init_t']))
    water_final_t = sp.Matrix(sp.symbols(['water_final_t']))
    hydro_t = sp.Matrix(sp.symbols(['hydro_t']))
    thermal_t = sp.Matrix(sp.symbols(['thermal_t']))
    decision_var = sp.Matrix([water_init_t, water_final_t, hydro_t, thermal_t])

    VF_var_idx = [1]

    X_prev = sp.Matrix(sp.symbols(['X_prev']))
    previous_stage_var = sp.Matrix([X_prev])
    R = sp.Matrix(sp.symbols(['R']))
    random_var = sp.Matrix([R])

    objective = hydro_t.dot(h_cost) + thermal_t.dot(t_cost) + sp.exp(-water_final_t.dot(utility_coeff) + utility_scale)
    ineq_constraints = []
    eq_constraints = []

    # initial condition
    eq_constraints += [water_init_t[0] - X_prev[0] - R[0]]
    eq_constraints += [water_final_t[0] - water_init_t[0] + hydro_t[0]]
    ineq_constraints += [demand - hydro_t[0] - thermal_t[0]]

    # Non-negativity
    for i, var in enumerate(decision_var):
        if i != 0:
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

    # Index set definition
    stage_to_prob = []
    for i in range(n_stages):
        if i == 0:
            stage_to_prob.append(0)
        else:
            stage_to_prob.append(1)

    return problem, stage_to_prob


def make_problem_definition(n_stages, h_cost, t_cost):
    """
    Generate information of desired problem
    :param n_stages: The number of stages
    :return: problem_definition: Dictionary for information of problem
    """
    sg = ScenarioGenerator()
    problem, stage_to_prob = decomposed_problems(n_stages, [h_cost], [t_cost])
    problem_definition = {'n_stages': n_stages,
                          'ScenarioGenerator': sg,
                          'problem': problem,
                          'stage_to_prob': stage_to_prob}

    return problem_definition