import numpy as np
import sympy as sp
import cvxpy as cp
import time

# parameter ====================================================================
h_cost = 2.0
t_cost = 7.0
utility_coeff = 0.1
utility_scale = 5.0
initial_reservoir = 40.0
demand = 20.0
mean = 20
scale = 5
# ==============================================================================

def dot(var1, var2):
    prod = [a*b for a, b in zip(var1, var2)]
    return sum(prod)

def ScenarioGenerator(batchNum, algorithm):
    """
    Generate random variable for stochastic process
    :param batchNum: The number of sample in one batch
    :return: sample: random water inflow for next stage
    """
    mean = 20
    scale = 5
    if algorithm.upper() == "DVFN":
        while True:
            batch_sample = np.random.normal(loc=mean, scale=scale, size=batchNum)
            normalized_batch_sample = (batch_sample - np.mean(batch_sample))
            normalized_batch_sample = normalized_batch_sample/np.std(batch_sample)
            rescaled = normalized_batch_sample*scale
            rescaled = rescaled + mean
            for sample in rescaled:
                yield [sample]
    else:
        regime = 0
        while True:
            batch_sample = np.random.normal(loc=mean, scale=scale, size=batchNum)
            normalized_batch_sample = (batch_sample - np.mean(batch_sample))
            normalized_batch_sample = normalized_batch_sample / np.std(batch_sample)
            rescaled = normalized_batch_sample * scale
            rescaled = rescaled + mean
            for sample in rescaled:
                yield regime, sample


def create_scenarioTree(n_stages, node):
    mean = 20
    scale = 5
    scenarioTree = [['stage0']] + [[] for _ in range(n_stages - 1)]
    for idx in range(n_stages - 1):
        batch_sample = np.random.normal(loc=mean, scale=scale, size=node)
        normalized_batch_sample = (batch_sample - np.mean(batch_sample))
        normalized_batch_sample = normalized_batch_sample/np.std(batch_sample)
        rescaled = normalized_batch_sample*scale
        rescaled = rescaled + mean
        rescaled = np.sort(rescaled)
        scenarioTree[idx + 1].append(rescaled.tolist())

    return scenarioTree

def decomposed_problems(n_stages, algorithm):
    """
    Generate stagewise decomposed problem
    :param n_stages: The number of stages
    :returns: problem: List of dictionary for decomposed problems
              stage_to_prob: Mapping from stage to stagewise problem (t=0 -> 0, 0<t<T -> 1, t=T -> 2)
    """

    problem = []
    if algorithm.upper() == "DVFN":
        # stage 0 problem ============================================================
        water_init_0 = sp.Matrix(sp.symbols(['water_init_0']))
        water_final_0 = sp.Matrix(sp.symbols(['water_final_0']))
        hydro_0 = sp.Matrix(sp.symbols(['hydro_0']))
        thermal_0 = sp.Matrix(sp.symbols(['thermal_0']))
        decision_var = sp.Matrix([water_init_0, water_final_0, hydro_0, thermal_0])

        VF_var_idx = [1]

        objective = hydro_0.dot([h_cost]) + thermal_0.dot([t_cost]) + sp.exp(-water_final_0.dot([utility_coeff]) + utility_scale)
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

        objective = hydro_t.dot([h_cost]) + thermal_t.dot([t_cost]) + sp.exp(-water_final_t.dot([utility_coeff]) + utility_scale)
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
    else:
        water_init_0 = cp.Variable(pos=True)
        water_final_0 = cp.Variable(pos=True)
        hydro_0 = cp.Variable(pos=True)
        thermal_0 = cp.Variable(pos=True)
        var_VF = [water_final_0]
        X = [water_init_0, water_final_0, hydro_0, thermal_0]

        # water_utility_0 => negative of water level piecewise linear utility.
        objective = hydro_0 * h_cost + thermal_0 * t_cost + cp.exp(-utility_coeff * water_final_0 + utility_scale)
        constraints = []
        constraints += [water_init_0 == initial_reservoir]  # Initial condition
        constraints += [water_final_0 == water_init_0 - hydro_0]  # water level balance
        constraints += [hydro_0 + thermal_0 >= demand]  # stage 0 demand
        for idx, var in enumerate([water_init_0, water_final_0, hydro_0, thermal_0]):
            constraints += [var >= 0]  # Non-negativity

        stage_0 = {'objective': objective, 'constraints': constraints, 'var_VF': var_VF, 'var_decision': X,
                   'var_previous_stage': [], 'var_random': [], 'var_parameter': []}
        problem.append(stage_0)

        # 0 < Stage < T ==========================================================
        water_init_t = cp.Variable(pos=True)
        water_final_t = cp.Variable(pos=True)
        hydro_t = cp.Variable(pos=True)
        thermal_t = cp.Variable(pos=True)
        var_VF = [water_final_t]
        X = [water_init_t, water_final_t, hydro_t, thermal_t]
        X_prev = cp.Parameter(shape=1)  # previous stage decision variable x_t
        R = cp.Parameter(shape=1)  # Random variable. water inflow

        objective = hydro_t * h_cost + thermal_t * t_cost + cp.exp(-utility_coeff * water_final_t + utility_scale)
        constraints = []
        constraints += [water_init_t == X_prev + R]  # Initial water level
        constraints += [water_final_t == water_init_t - hydro_t]  # water level balance
        constraints += [hydro_t + thermal_t >= demand]  # stage 0 demand
        for idx, var in enumerate([water_init_t, water_final_t, hydro_t, thermal_t]):
            constraints += [var >= 0]  # Non-negativity

        stage_t = {'objective': objective, 'constraints': constraints, 'var_VF': var_VF, 'var_decision': X,
                   'var_previous_stage': [X_prev], 'var_random': [R], 'var_parameter': []}
        problem.append(stage_t)

    # Index set definition
    stage_to_prob = []
    for i in range(n_stages):
        if i == 0:
            stage_to_prob.append(0)
        else:
            stage_to_prob.append(1)

    return problem, stage_to_prob


def MSP_problem(stageNum, scenario_node, mm=True):

    start = time.time()
    scenarios = [[]]
    for _ in range(stageNum - 1):
        batch_sample = np.random.normal(loc=mean, scale=scale, size=scenario_node)
        if mm:
            normalized_batch_sample = (batch_sample - np.mean(batch_sample))
            normalized_batch_sample = normalized_batch_sample / np.std(batch_sample)
            rescaled = normalized_batch_sample * scale
            rescaled = rescaled + mean
            scenarios.append(rescaled)
        else:
            scenarios.append(batch_sample)
    number_of_current_node = 1
    curIdx = 0
    totIndice = []
    stagewise_Indice = []
    for stage in range(stageNum):
        ind = list(range(curIdx, curIdx + number_of_current_node))
        totIndice += ind
        stagewise_Indice.append(ind)
        curIdx += number_of_current_node
        number_of_current_node = number_of_current_node * scenario_node
    stagewise_ind_length = [len(item) for item in stagewise_Indice]

    def find_stage(node: int):
        if node == 0:
            return 0
        stage = 1
        while True:
            if node <= sum([item for stagei, item in enumerate(stagewise_ind_length) if stagei <= stage]) - 1:
                return stage
            stage += 1

    def retrieve_parent_index(node: int):
        cur_Stage = find_stage(node)
        if cur_Stage < 1:
            raise Exception('0 stage has no parent node')
        modulo = stagewise_Indice[cur_Stage].index(node)
        quotient = modulo // scenario_node
        remainder = modulo % scenario_node
        return stagewise_Indice[cur_Stage - 1][quotient], remainder

    # STAGE 0
    water_init_0 = cp.Variable()
    water_final_0 = cp.Variable()
    hydro_0 = cp.Variable()
    thermal_0 = cp.Variable()
    totVars = [(water_init_0, water_final_0, hydro_0, thermal_0)]

    objective = hydro_0 * h_cost + thermal_0 * t_cost + cp.exp(-utility_coeff * water_final_0 + utility_scale)
    constraints = []
    constraints += [water_init_0 == initial_reservoir]  # Initial condition
    constraints += [water_final_0 == water_init_0 - hydro_0]  # water level balance
    constraints += [hydro_0 + thermal_0 >= demand]  # stage 0 demand
    # constraints += [hydro_0 == 17.47]
    # constraints += [thermal_0 == 2.53]
    for idx, var in enumerate([water_init_0, water_final_0, hydro_0, thermal_0]):
        constraints += [var >= 0]  # Non-negativity

    for stage, indSet in enumerate(stagewise_Indice):
        node_probability = 1 / len(indSet)
        # Stage t problem
        if stage > 0:
            for nodeIdx in indSet:
                water_init_t = cp.Variable()
                water_final_t = cp.Variable()
                hydro_t = cp.Variable()
                thermal_t = cp.Variable()
                totVars.append((water_init_t, water_final_t, hydro_t, thermal_t))

                objective += node_probability * (hydro_t * h_cost + thermal_t * t_cost + cp.exp(
                    -utility_coeff * water_final_t + utility_scale))
                parentIdx, scenIdx = retrieve_parent_index(nodeIdx)
                prevVar = totVars[parentIdx]
                R = scenarios[stage][scenIdx]
                # Initial condition
                constraints += [water_init_t == prevVar[1] + R]  # Initial water level
                constraints += [water_final_t == water_init_t - hydro_t]  # water level balance
                constraints += [hydro_t + thermal_t >= demand]  # stage 0 demand
                for idx, var in enumerate([water_init_t, water_final_t, hydro_t, thermal_t]):
                    constraints += [var >= 0]  # Non-negativity

    problem = cp.Problem(cp.Minimize(objective), constraints)
    print('Problem Defined. Now Solving...')
    try:
        problem.solve(verbose=False, solver=cp.MOSEK)
    except:
        problem.solve(verbose=False, solver=cp.ECOS)
    optStage0 = [float(item.value) for item in totVars[0]]
    optStage0_n = [str(item) for item in totVars[0]]
    pt = [item + ': ' + str(optStage0[idx]) + ',' for idx, item in enumerate(optStage0_n)]
    MSP_time = time.time() - start
    optStage0 = [item.value for item in totVars[0]]
    optStage0 = np.array(optStage0).reshape((1, 4, 1))

    return optStage0, problem.value, np.array([MSP_time])

def make_problem_definition(n_stages, n_batches, algorithm):
    """
    Generate information of desired problem
    :param n_stages: The number of stages
    :return: problem_definition: Dictionary for information of problem
    """
    sg = ScenarioGenerator(n_batches, algorithm)
    st = create_scenarioTree(n_stages, node=n_batches)
    problem, stage_to_prob = decomposed_problems(n_stages, algorithm)
    problem_definition = {'name': "EP",
                          'n_stages': n_stages,
                          'n_batches': n_batches,
                          'regime': 1,
                          'ScenarioGenerator': sg,
                          'ScenarioTree': st,
                          'problem': problem,
                          'stage_to_prob': stage_to_prob,
                          'paramVal': [()] * n_stages}

    return problem_definition