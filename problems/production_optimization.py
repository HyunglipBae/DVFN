import numpy as np
import sympy as sp
import cvxpy as cp
import time

# parameter ====================================================================
storage_c = [3.0, 7.0, 10.0]
production_c = [1.0, 2.0, 5.0]
outsourcing_c = [6.0, 12.0, 20.0]
maximum_resource = 10.0
pdim = 3

random_demand = [(5.0, 3.0, 1.0), (6.0, 2.0, 1.0), (1.0, 2.0, 2.0)]
# ==============================================================================

def dot(var1, var2):

    prod = [a*b for a, b in zip(var1, var2)]

    return sum(prod)

def ScenarioGenerator(algorithm):
    """
    Generate random variable for stochastic process
    :return: sample: random demand for next stage
    """
    if algorithm.upper() == "DVFN":
        while True:
            for sample in random_demand:
                yield sample

    else:
        regime = 0
        while True:
            for sample in random_demand:
                yield regime, sample


def create_scenarioTree(n_stages):
    scenarioTree = [['stage0']] + [[random_demand] for _ in range(n_stages - 1)]
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

    else:
        # Stage 0 problem  ==========================================================
        production = cp.Variable(shape=pdim)
        outsource = cp.Variable(shape=pdim)
        storage = cp.Variable(shape=pdim)

        var_VF = [storage]
        X = [production, outsource, storage]

        # objective = -1*((1 - gamma)*cp.power(C_0, 1 - gamma))
        objective = dot(storage_c, storage) + dot(outsourcing_c, outsource)
        constraints = []
        # Initial condition
        constraints += [dot(production_c, production) <= maximum_resource]
        constraints += [storage[i] - p - outsource[i] == 0 for i, p in enumerate(production)]
        # Non-negativity
        for idx, var in enumerate(X):
            if var.size > 0:
                constraints += [elevar >= 0 for elevar in var]
            else:
                constraints += [var >= 0]

        stage_0 = {'objective': objective, 'constraints': constraints, 'var_VF': var_VF, 'var_decision': X,
                   'var_previous_stage': [], 'var_random': [], 'var_parameter': []}
        problem.append(stage_0)

        # 0 < Stage < T ==========================================================
        production = cp.Variable(shape=pdim)
        outsource = cp.Variable(shape=pdim)
        storage = cp.Variable(shape=pdim)

        var_VF = [storage]
        X = [production, outsource, storage]

        # prev decision variable
        storage_prev = cp.Parameter(shape=pdim)  # x_t

        # random variables
        d = cp.Parameter(shape=pdim)  # Random Demand

        objective = dot(storage_c, storage) + dot(outsourcing_c, outsource)
        # objective = 0
        constraints = []

        constraints += [dot(production_c, production) <= maximum_resource]
        constraints += [-storage[i] - d[i] + p + outsource[i] == -storage_prev[i] for i, p in enumerate(production)]
        # Non-negativity
        for idx, var in enumerate(X):
            if var.size > 0:
                constraints += [elevar >= 0 for elevar in var]
            else:
                constraints += [var >= 0]

        stage_t = {'objective': objective, 'constraints': constraints, 'var_VF': var_VF, 'var_decision': X,
                   'var_previous_stage': [storage_prev], 'var_random': [d], 'var_parameter': []}
        problem.append(stage_t)

        # Stage == T ==========================================================
        production = cp.Variable(shape=pdim)
        outsource = cp.Variable(shape=pdim)


        X = [production, outsource]

        # prev decision variable
        storage_prev = cp.Parameter(shape=pdim)  # x_t

        # random variables
        d = cp.Parameter(shape=pdim)  # Random Demand

        objective = dot(outsourcing_c, outsource)
        # objective = 0
        constraints = []
        constraints += [dot(production_c, production) <= maximum_resource]
        constraints += [d[i] - p - outsource[i] - storage_prev[i] <= 0 for i, p in enumerate(production)]
        # Non-negativity
        for idx, var in enumerate(outsource):
            constraints += [var >= 0]

        stage_T = {'objective': objective, 'constraints': constraints, 'var_VF': [], 'var_decision': X,
                   'var_previous_stage': [storage_prev], 'var_random': [d], 'var_parameter': []}
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

def MSP_problem(n_stages, scenario_node):

    start = time.time()
    scenarios = [[]]
    for _ in range(n_stages - 1):
        scenarios.append([(5, 3, 1), (6, 2, 1), (1, 2, 2)])

    number_of_current_node = 1
    curIdx = 0
    totIndice = []
    stagewise_Indice = []
    for stage in range(n_stages):
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
    production = cp.Variable(shape=pdim)
    outsource = cp.Variable(shape=pdim)
    storage = cp.Variable(shape=pdim)
    objective = dot(storage_c, storage) + dot(outsourcing_c, outsource)
    constraints = []
    # Initial condition
    constraints += [dot(production_c, production) <= maximum_resource]
    constraints += [storage[i] - p - outsource[i] == 0 for i, p in enumerate(production)]
    # Non-negativity
    for idx, var in enumerate([production, outsource, storage]):
        constraints += [var >= 0]

    totVars = [(production, outsource, storage)]
    for stage, indSet in enumerate(stagewise_Indice):
        node_probability = 1 / len(indSet)
        # Stage t problem
        if stage > 0 and stage < n_stages - 1:
            for nodeIdx in indSet:
                production = cp.Variable(shape=pdim)
                outsource = cp.Variable(shape=pdim)
                storage = cp.Variable(shape=pdim)

                totVars.append((production, outsource, storage))
                objective += node_probability * (dot(storage_c, storage) + dot(outsourcing_c, outsource))

                parentIdx, scenIdx = retrieve_parent_index(nodeIdx)
                prevVar = totVars[parentIdx]
                d = scenarios[stage][scenIdx]

                # Initial condition
                constraints += [dot(production_c, production) <= maximum_resource]
                constraints += [storage[i] - p - outsource[i] + d[i] == prevVar[2][i] for i, p in enumerate(production)]
                # Non-negativity
                for idx, var in enumerate([production, outsource, storage]):
                    constraints += [var >= 0]
        # Stage T problem
        if stage == n_stages - 1:
            for nodeIdx in indSet:
                production = cp.Variable(shape=pdim)
                outsource = cp.Variable(shape=pdim)
                totVars.append((production, outsource))
                objective += node_probability * dot(outsourcing_c, outsource)

                parentIdx, scenIdx = retrieve_parent_index(nodeIdx)
                prevVar = totVars[parentIdx]
                d = scenarios[stage][scenIdx]

                # Initial condition
                constraints += [dot(production_c, production) <= maximum_resource]
                constraints += [d[i] - p - outsource[i] == prevVar[2][i] for i, p in enumerate(production)]
                # Non-negativity
                for idx, var in enumerate([production, outsource]):
                    constraints += [var >= 0]

    problem = cp.Problem(cp.Minimize(objective), constraints)
    print('Problem Defined. Now Solving...')
    try:
        problem.solve(verbose=False, solver=cp.CPLEX)
    except:
        problem.solve(verbose=False, solver=cp.MOSEK)
    optStage0 = [item.value for item in totVars[0]]
    optStage0 = np.array(optStage0).reshape((1, 9, 1))
    MSP_time = time.time() - start

    return optStage0, problem.value, np.array([MSP_time])

def make_problem_definition(n_stages, n_batches, algorithm):
    """
    Generate information of desired problem
    :param n_stages: The number of stages
    :return: problem_definition: Dictionary for information of problem
    """
    sg = ScenarioGenerator(algorithm)
    st = create_scenarioTree(n_stages)
    problem, stage_to_prob = decomposed_problems(n_stages, algorithm)
    problem_definition = {'name': "PO",
                          'n_stages': n_stages,
                          'n_batches': n_batches,
                          'regime': 1,
                          'ScenarioGenerator': sg,
                          'ScenarioTree': st,
                          'problem': problem,
                          'stage_to_prob': stage_to_prob,
                          'paramVal': [()]*n_stages}

    return problem_definition