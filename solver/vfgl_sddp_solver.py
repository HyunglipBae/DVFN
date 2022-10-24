from tqdm import tqdm
from sympy.matrices import matrix_multiply_elementwise
from sympy import DiracDelta, Heaviside
from math import exp, log
import scipy.optimize as opt
import numpy as np
import cvxpy as cp
import sympy as sp
import pickle
import time
import os


def dot(var1, var2):
    prod = [a*b for a, b in zip(var1, var2)]
    return sum(prod)


class solver:
    def __init__(self,
                 probDef,
                 paramdict={}):
        self.p_key = probDef.keys()
        # Environment Setting
        self.name = probDef['name']
        self.stageNum = probDef['n_stages']
        self.regimeNum = probDef['regime']
        self.scenarioTree = None
        if 'ScenarioTree' in self.p_key:
            self.scenarioTree = probDef['ScenarioTree']
        if 'ScenarioGenerator' in self.p_key:
            self.scenarioGenerator = probDef['ScenarioGenerator']

        # Problem Definition and Value Function Approximation
        self.objective = []
        self.constraints = []
        self.stage2prob = []
        self.var_VF = ['not defined']*max(0, self.stageNum - 1)                                                 # Value function cvxpy variable
        self.var_decision = ['not defined']*max(0, self.stageNum - 1)                                           # cvxpy stagewise decision variable
        self.var_previous_stage = ['not defined']*max(0, self.stageNum - 1)                                     # cvxpy previous stage decision variable
        self.var_parameter = ['not defined']*max(0, self.stageNum - 1)                                          # cvxpy parameter variable
        self.var_random = ['not defined']*max(0, self.stageNum - 1)                                    # cvxpy random variable
        self.val_parameter = []                                                                                 # values to be fed to self.var_parameter
        self.variable_dimension = {'VF': [], 'decision': [], 'previous_stage': [], 'parameter': [], 'random_variable': []}  # Dictionary of variable dimensions
        self.insert_problem(probDef['problem'], probDef['stage_to_prob'], probDef['paramVal'])

        # Parametric Value Function Approximation
        self.VF = [['not defined']*self.regimeNum for _ in range(max(0, self.stageNum - 1))]        # Value function cvxpy expression
        self.param_VF = [['not defined']*self.regimeNum for _ in range(max(0, self.stageNum - 1))]  # cvxpy parameters to be learned
        self.VFGL_funs = [dict()]*(self.stageNum - 1)  # Dictionaries of VFGL functions including gradient, loss, GD methods
        self.param_nonneg_idx = [[] for _ in range(self.stageNum - 1)]
        self.gd_method = ['ISGD', 'ADAM', 'RMSProp', 'momentum'][0]
        self.ave_sample_grad = [['not defined']*self.regimeNum for _ in range(self.stageNum - 1)]      # Average sampled target grad
        self.ave_VF_grad = [['not defined']*self.regimeNum for _ in range(self.stageNum - 1)]          # Average parametric VF update grad
        self.ave_VF_grad_squared = [['not defined']*self.regimeNum for _ in range(self.stageNum - 1)]  # Average parametric VF update grad squared
        self.ADAM_beta1 = 0.2                                                                          # Adam beta 1. Momentum weight
        self.ADAM_beta2 = 0.9                                                                        # Adam beta 2. Squared Momentum weight.
        self.paramdict = paramdict
        self.form = 'exp'

        # SDDP Value Function Approximation
        self.Z = [cp.Variable() for _ in range(self.stageNum - 1)]                                                 # Piecewise linear cut auxiliary variable
        self.benders_cut = [[[self.Z[stage] >= -100000000]]*self.regimeNum for stage in range(self.stageNum - 1)]  # Value function cuts. Give lower bound to prevent unboundedness
        self.benders_cut_values = [[[]]*self.regimeNum for _ in range(self.stageNum - 1)]                          # Tuples that store gradients/coefficients of benders cuts
        if 'transitionProbability' in self.p_key:
            self.transitionP = probDef['transitionProbability']                                    # Regime transition probability
        else:
            self.transitionP = (1/self.regimeNum)*np.ones(shape=(self.regimeNum, self.regimeNum))
        self.stationaryP = np.round(self.transitionP**50, decimals=6).reshape((self.regimeNum))    # Stationary probability for the initial transition
        self.SDDP_lower_bound = []
        self.SDDP_upper_bound = []
        self.SDDP_upper_bound_std = []

        # Iteration Related
        self.iterationCount = 0
        self.visitCount = [[0]*self.regimeNum for _ in range(self.stageNum - 1)]
        self.visitCount[0] = [0]
        self.minIter = self.stageNum + 1
        self.maxIter = 1000
        self.curStage = 0
        self.curRegime = 0
        self.curSolver = 'MOSEK'
        self.curAlgorithm = ''
        self.burnIn = self.stageNum + 20
        self.iterTime = []
        self.VF_loaded = False
        self.preTraining = False
        self.preTraining_iteration = 20
        self.initialize_linear_param = False

        self.sddp_cut_selection_iter = 20  # How many iterations for the cut selection?
        self.sddp_cut_selection_num = -1  # How many Benders cuts to keep for the filtering work. negative if keep all that has non-zero count.


        # Forward Pass Related
        self.batch = probDef['n_batches']  # Number of batch per sample gradient. Must be a positive integer (1~10 recommended).
        self.stage_to_be_updated = 0
        self.regime_to_be_updated = 0
        self.theta_update = 0
        self.theta_update_history = []
        self.forward_cumulative_loss = 0
        self.forward_cumulative_loss_history = []
        self.forwardSol = []

        # Algorithm Parameters
        self.algorithm = ['SDDP/VFGL MIX', 'VFGL', 'SDDP'][0]
        self.solverSequence = ['MOSEK', 'CPLEX', 'ECOS', 'SCS']  # Order of solving attempt. Moves on if an error encountered
        self.roundSolution = -1  # Round solution to this decimal point. -1 to disable this function.
        self.roundTheta = -1     # Round VF parameter value to this decimal point. -1 to disable this function
        self.theta_update_threshold = -1

    def insert_problem(self, problems, stage_map_idx, parameter_value):
        """
        Inserts user defined problem to the solver
        :param problem: problem definition. List of dictionary that defines stagewise problem. Dictionary with the keys:
                        {'objective', 'constraints', 'var_VF', 'var_decision', 'var_previous_stage', 'var_random', 'var_parameter'}
        :param stage_map_idx: stage to problem index map
        :param parameter_value: list of parameter values
        :return: None
        """
        self.objective = [swProb['objective'] for swProb in problems]
        self.constraints = [swProb['constraints'] for swProb in problems]
        for c_idx, constraint in enumerate(self.constraints):  # This orders constraints by equalities -> inequalities
            eq = []
            ineq = []
            for indCon in constraint:
                if isinstance(indCon, cp.constraints.Equality):
                    eq.append(indCon)
                elif isinstance(indCon, cp.constraints.Inequality):
                    ineq.append(indCon)
                else:
                    raise Exception('Invalid constraint')
            self.constraints[c_idx] = eq + ineq
        self.var_VF = [swProb['var_VF'] for swProb in problems]
        self.var_decision = [swProb['var_decision'] for swProb in problems]
        self.var_previous_stage = [swProb['var_previous_stage'] for swProb in problems]
        self.var_random = [swProb['var_random'] for swProb in problems]
        self.var_parameter = [swProb['var_parameter'] for swProb in problems]
        self.stage2prob = stage_map_idx
        self.val_parameter = parameter_value

        self.variable_dimension['VF'] = [[self.varDim(var) for var in varList] for varList in self.var_VF if isinstance(varList, list)]
        self.variable_dimension['decision'] = [[self.varDim(var) for var in varList] for varList in self.var_decision if isinstance(varList, list)]
        self.variable_dimension['previous_stage'] = [[self.varDim(var) for var in varList] for varList in self.var_previous_stage if isinstance(varList, list)]
        self.variable_dimension['parameter'] = [[self.varDim(var) for var in varList] for varList in self.var_parameter if isinstance(varList, list)]
        self.variable_dimension['random_variable'] = [[self.varDim(var) for var in varList] for varList in self.var_random if isinstance(varList, list)]
        print('Problem defined...')

    def initialize_VF(self):
        """
        This function creates Value Function in both cvxpy and sympy context and further creates Loss, Loss gradient
        computing functions.
        :return: None
        """
        print('Initializing the Parametric Value Function')
        for stage in tqdm(range(self.stageNum - 1), total=self.stageNum-1):
            for regime in range(self.regimeNum):
                if stage == 0 and regime > 0:  # stage 0 has only 1 regime.
                    continue
                probIdx = self.stage2prob[stage]
                x = []
                for var in self.var_VF[probIdx]:
                    x += self.decomposeVar(var)
                xDim = sum(self.variable_dimension['VF'][probIdx])

                # =============================== cvxpy context VF definition. =========================================
                # ==================== Change here if you wish to use different parametric function ====================
                # Production Optimization
                if self.name.upper() == "PO":
                    ec1 = cp.Parameter(xDim)
                    ec1.value = [-1]*ec1.size

                    if self.form == 'quad':
                        ec2 = cp.Parameter(xDim, pos=True)
                    else:
                        ec2 = cp.Parameter(xDim)

                    ec2.value = [1]*ec2.size
                    param = [ec1, ec2]
                    if self.form == 'exp':
                        expression = dot(x, ec1) + sum([cp.exp(-x[i] + ec2[i]) for i in range(xDim)])
                    elif self.form == 'quad':
                        expression = dot(x, ec1) + x[0] ** 2 * ec2[0] + x[1] ** 2 * ec2[1] + x[2] ** 2 * ec2[2]
                    else:
                        expression = dot(x, ec1)

                # ENERGY PLANNING
                if self.name.upper() == "EP":
                    ec1 = cp.Parameter(xDim)
                    ec1.value = [-1]*ec1.size
                    ec2 = cp.Parameter(xDim, pos=True)
                    ec2.value = [100]*ec2.size

                    param = [ec1, ec2]
                    if self.form == 'exp':
                        expression = dot(x, ec1) + sum([cp.exp(-ec2[i]*x[i] + 5) for i in range(xDim)])
                    elif self.form == 'quad':
                        expression = dot(x, ec1) + x[0] ** 2 * ec2[0]
                    else:
                        expression = dot(x, ec1)
                # ======================================================================================================
                self.param_VF[stage][regime] = param
                self.VF[stage][regime] = expression
                if stage == 0:
                    break  # Regardless to the number of regimes, we only have 1 VF at stage 0.

            # From here, we define symbolic gradient computations of VF. This expression is shared among VFs within the same stage.
            if xDim > 1:
                x = sp.Matrix(sp.symbols(' '.join(['x_' + str(i) for i in range(xDim)])))                 # "Sampled" X-value (previous stage solution)
                y = sp.Matrix(sp.symbols(' '.join(['y_' + str(i) for i in range(xDim)])))                 # Target gradient
                ave_y = sp.Matrix(sp.symbols(' '.join(['ave_y_' + str(i) for i in range(y.shape[0])])))   # Average target gradient
            elif xDim == 1:
                x = sp.symbols('x')
                x = sp.Matrix([x])
                y = sp.symbols('y')
                y = sp.Matrix([y])
                ave_y = sp.symbols('ave_y')
                ave_y = sp.Matrix([ave_y])
            else:
                raise Exception('input dimension error: {xDim}'.format(xDim=xDim))

            # ==================== Change here if you wish to use different parametric function ====================
            # Sympy VF Definition ===============================================
            alpha = sp.Matrix([sp.symbols('alpha')])

            # PRODUCTION OPTIMIZATION
            if self.name.upper() == "PO":
                ec1 = sp.Matrix(sp.symbols(' '.join(['ec1_' + str(i) for i in range(xDim)])))
                ec2 = sp.Matrix(sp.symbols(' '.join(['ec2_' + str(i) for i in range(xDim)])))
                theta = sp.Matrix([ec1, ec2])
                if self.form == 'exp':
                    g = x.dot(ec1) + sum([sp.exp(-x[i] + ec2[i]) for i in range(xDim)])
                elif self.form == 'quad':
                    g = x.dot(ec1) + x[0] ** 2 * ec2[0] + x[1] ** 2 * ec2[1] + x[2] ** 2 * ec2[2]
                else:
                    g = x.dot(ec1)

            # ENERGY PLANNING
            if self.name.upper() == 'EP':
                ec1 = sp.Matrix([sp.symbols('ec1')])
                ec2 = sp.Matrix([sp.symbols('ec2')])
                theta = sp.Matrix([ec1, ec2])
                if self.form == "exp":
                    g = x.dot(ec1) + sp.exp(-ec2[0]*x[0] + 5)
                elif self.form == 'quad':
                    g = x.dot(ec1) + x[0] ** 2 * ec2[0]
                else:
                    g = x.dot(ec1)
            # ==============================================================

            if theta.shape[0] > 1:
                old_theta = sp.Matrix(sp.symbols(' '.join(['theta_old_' + str(i) for i in range(theta.shape[0])])))
                momentum = sp.Matrix(sp.symbols(' '.join(['momentum_' + str(i) for i in range(theta.shape[0])])))
                squared_momentum = sp.Matrix(sp.symbols(' '.join(['squared_momentum_' + str(i) for i in range(theta.shape[0])])))
            elif theta.shape[0] == 1:
                old_theta = sp.Matrix([sp.symbols('theta_old')])
                momentum = sp.Matrix([sp.symbols('momentum')])
                squared_momentum = sp.Matrix([sp.symbols('squared_momentum')])

            g_dx = sp.diff(g, x)  # This is a vector with a class of sympy.ImmutableDenseMatrix (VF gradient w.r.t. prev_decision variable)
            lambdified_g_dx = sp.lambdify(sp.Matrix([x, theta]), g_dx, ['numpy', {'DiracDelta': DiracDelta, 'Heaviside': Heaviside}])
            def eval_g_dx(x: list, theta: list):
                arg = x + theta
                return lambdified_g_dx(*arg)

            g_dxdt = []
            for gdx in g_dx:
                g_dxdt.append(sp.diff(gdx, theta))
            loss = 0
            idx = 0
            for g_dx_i, y_i in zip(g_dx, y):
                # loss += (y_i - g_dx_i)**2  # Without the normalizer. This creates biased towards x-dimension with the huge gradient error.
                loss += (1/(ave_y[idx]**2))*(y_i - g_dx_i)**2
                idx += 1
            lambdified_loss = sp.lambdify(sp.Matrix([x, y, ave_y, theta]), loss, ['numpy', {'DiracDelta': DiracDelta, 'Heaviside': Heaviside}])
            def eval_loss(x: list, y: list, ave_y: list, theta: list):
                arg = x + y + ave_y + theta
                return lambdified_loss(*arg)
            d = []
            for idx, gdx in enumerate(g_dx):
                # d.append(-2*(y[idx] - gdx))
                d.append(-2*(y[idx] - gdx)/(ave_y[idx]**2))  # Divided by ave_y squared
            loss_dt = []
            for d_i, g_dxdt_i in zip(d, g_dxdt):
                loss_dt.append(d_i*g_dxdt_i)
            theta_gradient = 0
            for Q_i in loss_dt:
                if isinstance(theta_gradient, int):
                    theta_gradient = Q_i
                else:
                    theta_gradient += Q_i

            lambdified_theta_gradient = sp.lambdify(sp.Matrix([x, y, ave_y, theta]), theta_gradient, ['numpy', {'DiracDelta': DiracDelta, 'Heaviside': Heaviside}])
            def eval_theta_gradient(x: list, y: list, ave_y: list, theta: list):
                arg = x + y + ave_y + theta
                return lambdified_theta_gradient(*arg)
            ISGD_Expression = -theta + old_theta -alpha[0]*theta_gradient
            ISGD_sympy_lambdification = sp.lambdify(sp.Matrix([x, y, ave_y, old_theta, alpha]), ISGD_Expression, ['sympy'])
            def ISGD(cur_x: list, cur_y: list, ave_y: list, cur_theta: list, cur_alpha: list or float):
                if not isinstance(cur_alpha, list):
                    cur_alpha = [cur_alpha]
                arg = cur_x + cur_y + ave_y + cur_theta + cur_alpha
                theta_function = ISGD_sympy_lambdification(*arg)
                theta_function = sp.lambdify(theta, theta_function, ['numpy', {'DiracDelta': DiracDelta, 'Heaviside': Heaviside}])
                def fwrap(t):
                    return list(theta_function(*t).squeeze())
                new_theta = opt.root(fwrap, cur_theta)
                return new_theta.x.tolist()

            mterm = self.ADAM_beta1*momentum + (1 - self.ADAM_beta1)*theta_gradient
            Ex_ISGD_M_Expression = -theta + old_theta -alpha[0]*mterm
            Ex_ISGD_M_sympy_lambdification = sp.lambdify(sp.Matrix([x, y, ave_y, old_theta, alpha, momentum]), Ex_ISGD_M_Expression, ['sympy'])

            squared_theta_gradient = sp.Matrix([tg**2 for tg in theta_gradient])
            squared_mterm = self.ADAM_beta2*squared_momentum + (1 - self.ADAM_beta2)*squared_theta_gradient
            inverse_root_squared_mterm = sp.Matrix([1/sp.Max(sp.sqrt(item), 1.0e-6) for item in squared_mterm])
            Ex_ISGD_Expression = -theta + old_theta -alpha[0]*matrix_multiply_elementwise(mterm, inverse_root_squared_mterm)
            Ex_ISGD_sympy_lambdification = sp.lambdify(sp.Matrix([x, y, ave_y, old_theta, alpha, momentum, squared_momentum]), Ex_ISGD_Expression, ['sympy'])
            def Extended_ISGD(cur_x: list, cur_y: list, ave_y: list, cur_theta: list, cur_alpha: list or float, momentum=[], squared_momentum=[]):
                if not isinstance(cur_alpha, list):
                    cur_alpha = [cur_alpha]
                if not momentum:
                    momentum = [0]*len(theta)
                if not squared_momentum:
                    arg = cur_x + cur_y + ave_y + cur_theta + cur_alpha + momentum
                    theta_function = Ex_ISGD_M_sympy_lambdification(*arg)
                    theta_function = sp.lambdify(theta, theta_function, ['numpy', {'DiracDelta': DiracDelta, 'Heaviside': Heaviside}])
                if momentum and squared_momentum:
                    arg = cur_x + cur_y + ave_y + cur_theta + cur_alpha + momentum + squared_momentum
                    theta_function = Ex_ISGD_sympy_lambdification(*arg)
                    theta_function = sp.lambdify(theta, theta_function, ['numpy', {'DiracDelta': DiracDelta, 'Heaviside': Heaviside}])
                def fwrap(t):
                    return list(theta_function(*t).squeeze())
                new_theta = opt.root(fwrap, cur_theta)
                return new_theta.x.tolist()
            self.VFGL_funs[stage] = {'g_dx': eval_g_dx, 'loss': eval_loss, 'loss_dt': eval_theta_gradient, 'ISGD': ISGD, 'Extended_ISGD': Extended_ISGD}
        time.sleep(0.1)


    def solve(self):
        ts_time = time.time()
        self.print_problem_summary()
        if 'SDDP' in self.algorithm.upper() or self.preTraining:  # SDDP Algorithm
            self.curAlgorithm = 'SDDP'
            self.iterationCount = 0
            self.forwardSol = []
            stop = False
            pbar = tqdm(total=self.iterationCount)
            pbar.set_description("SDDP Iteration: ")
            while not stop:
                start = time.time()
                forward_solution, forward_vf_solution, _, _ = self.forward_pass()
                self.forwardSol.append(forward_solution)
                self.backward_pass(forward_vf_solution)
                ub, ubstd = self.find_SDDP_upper_bound(iter=10)
                self.SDDP_upper_bound_std.append(ubstd)
                self.SDDP_upper_bound.append(ub)
                self.iterTime.append(time.time() - start)
                pbar.update(1)
                pbar.set_description('Stage0Sol: {stage0Sol}, lower_bound: {lb}, Stage 0 # cuts: [{cutN}]'
                                     .format(stage0Sol=self.forwardSol[-1][0], lb=self.SDDP_lower_bound[-1], cutN=len(self.benders_cut[0][0])))
                stop = self.check_stopping_criterion('SDDP')
            pbar.close()

        if ('VFGL' in self.algorithm.upper()) or ('MIX' in self.algorithm.upper()):
            self.curAlgorithm = 'VFGL'
            self.iterationCount = 0
            self.forwardSol = []
            stop = False
            pbar = tqdm(total=self.iterationCount)
            pbar.set_description("VFGL Episode Sample: ")
            while not stop:
                start = time.time()
                self.sample_episode_and_update_parameter()
                self.iterTime.append(time.time() - start)
                stop = self.check_stopping_criterion('VFGL')
                pbar.update(1)
                if not self.initialize_linear_param and not self.preTraining and self.iterationCount > 0:
                    tv = np.round(self.varValues(self.param_VF[0][0]), decimals=5).tolist()
                    pbar.set_description('Forward Loss: {loss:.5f}, Theta Update: {theta:.5f}, Stage0Sol: {stage0Sol}, Theta: {thetaVal}'
                                         .format(loss=self.forward_cumulative_loss, theta=self.theta_update,
                                                 stage0Sol=np.around(self.forwardSol[-1][0], decimals=4), thetaVal=tv[:min(6, len(tv))]))
            pbar.close()
            print('stage 0 VF {}'.format(self.VF[0][0]))
            print('stage 0 VF coefficients: {}'.format(self.varValues(self.decomposeVar(self.param_VF[0][0]))))
            print('{:=^70}'.format(' VF node visit Counts '))
            for reg in range(self.regimeNum):
                regCount = self.visitCount[0]
                for stage in range(1, self.stageNum - 1):
                    regCount.append(self.visitCount[stage][reg])
                print('{}'.format('Regime ' + str(reg) + ':'))
                print(regCount)
            print('{:=^70}'.format(''))
            print('')

        ttime = sum(self.iterTime)
        print('Total Time Elapsed: ', ttime)
        print('Past {} solutions'.format(self.stageNum))
        np.set_printoptions(suppress=True)
        for solIdx in range(min(self.minIter, 3)):
            print('{:=^70}'.format('Problem ' + str(solIdx)))
            sol = self.forwardSol[-solIdx-1]
            for sIdx in range(self.stageNum):
                val = ['{0:.4f}'.format(np.round(item, decimals=4)) for item in sol[sIdx]]
                maxVlen = max([len(item) for item in val])
                val = [('{0: >' + str(maxVlen) + '}').format(item) for item in val]
                val = '[' + ', '.join(val) + ']'
                print('stage {stage}: {solution}'.format(stage=sIdx, solution=val))
        print('')

    def print_problem_summary(self):
        """
        Prints out the current problem summary
        """
        print('{:=^80}'.format('Algorithm Summary'))
        print('{object: <30}: {value}'.format(object='Problem', value=str(self.name)))
        print('{object: <30}: {value}'.format(object='Number of Stage', value=str(self.stageNum)))
        print('{object: <30}: {value}'.format(object='Number of Branches', value=str(self.batch)))
        print('{object: <30}: {value}'.format(object='Number of Regime', value=str(self.regimeNum)))
        print('{object: <30}: {value}'.format(object='SGD Method', value=str(self.gd_method)))
        print('{object: <30}: {value}'.format(object='Algorithm', value=str(self.algorithm)))
        print('{object: <30}: {value}'.format(object='Regime transition probability', value=str(self.transitionP)))
        print('{object: <30}: {value}'.format(object='Regime stationary probability', value=str(self.stationaryP)))
        print('{object: <30}: {value}'.format(object='Maximum iteration', value=str(self.maxIter)))
        print('{object: <30}: {value}'.format(object='Theta Change Threshold', value=str(self.theta_update_threshold)))
        print('{object: <30}: {value}'.format(object='Solver sequence', value=str(self.solverSequence)))
        print('{:=^80}'.format(''))
        time.sleep(0.1)

    def check_stopping_criterion(self, algorithm=['SDDP', 'VFGL'][0]):
        """
        Check the stopping criterion
        :return:
        """
        self.iterationCount += 1
        if self.preTraining:
            if algorithm == 'SDDP' and (self.iterationCount >= self.preTraining_iteration):
                return True
            elif algorithm == 'VFGL' and self.iterationCount >= 10:
                self.preTraining = False
                self.iterationCount = 0
        elif algorithm == 'SDDP':
            if self.iterationCount < self.minIter:
                return False
            if self.iterationCount >= self.maxIter:
                print('')
                print('Maximum iteration reached. Terminating the SDDP algorithm.')
                print('Iteration Count: ', self.iterationCount)
                return True
        else:  # VFGL Algorithm
            if self.initialize_linear_param and self.iterationCount >= self.stageNum:
                self.iterationCount = 0
                self.initialize_linear_param = False
            if self.iterationCount < self.minIter:
                return False
            if self.iterationCount >= self.maxIter:
                print('')
                print('Maximum iteration reached. Terminating the algorithm.')
                print('Iteration Count: ', self.iterationCount)
                print('Theta update level: {:.4f}'.format(self.theta_update))
                return True
            if self.theta_update < self.theta_update_threshold:
                print('Terminating the algorithm. Theta converged. Theta update level: {:.4f}'.format(self.theta_update))
                return True
        return False

    def sample_episode_and_update_parameter(self):
        """
        Samples a scenario path and solves the episode sequentially. VF parameters are updated accordingly.
        Step1) Assign parameter values
        Step2) Assign previous stage decision variables if stage > 0
        Step3) Sample rv and assign rv for self.batch many times if stage > 0
        Step4) Combine the results from above to find reference gradient (y).
        Step5) Update the value function with the information from step 4)
        Note: Select the last batch scenario to be the forward path.
        :return: None
        """
        self.stage_to_be_updated = 0
        self.regime_to_be_updated = 0
        self.theta_update = 0
        self.forward_cumulative_loss = 0
        episode_solution = []
        for stage in range(self.stageNum):
            stageIdx = self.stage2prob[stage]
            batch = self.batch
            batchwise_episode_solution = []
            batchwise_sample_gradient = []
            batchwise_regimes = []
            self.assign(self.var_parameter[stageIdx], self.val_parameter[stage])  # Step 1
            if stage == 0:
                batch = 1  # If stage 0, we do not need batch sampling because there is no r.v. involved.
            else:
                self.assign(self.var_previous_stage[stageIdx], prevSol)  # Step 2
            # ============================================= Batch Loop =============================================
            batchwise_prevSol = []
            for batch_idx in range(batch):  # Step 3
                if stage == 0:
                    regime, rv = (0, '')
                else:
                    regime, rv = next(self.scenarioGenerator)  # Sample random variable
                    self.assign(self.var_random[stageIdx], rv)
                batchwise_regimes.append(regime)

                if (stage == self.stageNum - 1):  # or (self.iterationCount < self.stageNum - stage):
                    objective = self.objective[stageIdx]
                else:
                    objective = self.objective[stageIdx] + self.VF[stage][regime]
                constraints = self.constraints[stageIdx]
                if self.preTraining and stage < self.stageNum - 1:
                    objective = self.objective[stageIdx] + self.Z[stage]
                    constraints = constraints + self.benders_cut[stage][regime]
                problem = cp.Problem(cp.Minimize(objective), constraints)
                solution = self.solve_subproblem(problem, stage, regime)
                batchwise_episode_solution.append(solution)
                batch_prevSol = []
                for var in self.var_VF[stageIdx]:
                    try:
                        batch_prevSol += list(var.value.squeeze())
                    except (AttributeError, TypeError):
                        batch_prevSol += [float(var.value)]
                batchwise_prevSol.append(batch_prevSol)
                if stage > 0:
                    # a = problem.get_problem_data(solver=cp.ECOS)
                    gradient = self.find_gradient(stage=self.stage_to_be_updated + 1,
                                                  dual=list(problem.solution.dual_vars.values()),
                                                  constraints=constraints)
                    batchwise_sample_gradient.append(np.array(gradient))
            # ============================================= Batch Loop =============================================
            if stage > 0:
                sample_ave_gradient = np.average(batchwise_sample_gradient, axis=0).squeeze().tolist()
                if not isinstance(sample_ave_gradient, list):
                    sample_ave_gradient = [sample_ave_gradient]
                if not self.initialize_linear_param:
                    theta_update, loss_amount = self.update_VF(prevSol, sample_ave_gradient)
                    self.theta_update += theta_update
                    self.forward_cumulative_loss += loss_amount
                else:
                    self.update_only_linear(sample_ave_gradient)
            self.stage_to_be_updated = stage
            self.regime_to_be_updated = batchwise_regimes[-1]
            episode_solution.append(batchwise_episode_solution[-1])
            prevSol = batchwise_prevSol[np.random.randint(batch)]
        if not self.initialize_linear_param:
            self.forwardSol.append(episode_solution)
            self.theta_update_history.append(self.theta_update)
            self.forward_cumulative_loss_history.append(self.forward_cumulative_loss)

    def calc_ave_KKT_deviation(self, sampleNum=30):
        sample_objs = []
        sample_devs = [[] for _ in range(self.stageNum-1)]
        prevSol = []
        for _ in range(sampleNum):
            stagewise_regimes = []
            obj = 0
            for stage in range(self.stageNum):
                stageIdx = self.stage2prob[stage]
                batch = 1
                batchwise_episode_solution = []
                batchwise_sample_gradient = []
                batchwise_regimes = []
                self.assign(self.var_parameter[stageIdx], self.val_parameter[stage])  # Step 1
                if stage == 0:
                    batch = 1  # If stage 0, we do not need batch sampling because there is no r.v. involved.
                else:
                    self.assign(self.var_previous_stage[stageIdx], prevSol)  # Step 2
                # ============================================= Batch Loop =============================================
                batchwise_sols = []
                for batch_idx in range(batch):  # Step 3
                    if stage == 0:
                        regime, rv = (0, '')
                    else:
                        if self.name == "EP":
                            regime = 0
                            rv = np.random.normal(loc=20, scale=5)
                        else:
                            regime, rv = next(self.scenarioGenerator)  # Sample random variable
                        self.assign(self.var_random[stageIdx], rv)
                    batchwise_regimes.append(regime)

                    if (stage == self.stageNum - 1):  # or (self.iterationCount < self.stageNum - stage):
                        objective = self.objective[stageIdx]
                    else:
                        objective = self.objective[stageIdx] + self.VF[stage][regime]
                    constraints = self.constraints[stageIdx]
                    problem = cp.Problem(cp.Minimize(objective), constraints)
                    solution = self.solve_subproblem(problem, stage, regime)
                    batchwise_episode_solution.append(solution)
                    batch_prevSol = []
                    for var in self.var_VF[stageIdx]:
                        try:
                            batch_prevSol += list(var.value.squeeze())
                        except (AttributeError, TypeError):
                            batch_prevSol += [float(var.value)]
                    batchwise_sols.append(batch_prevSol)
                    if stage > 0:
                        gradient = self.find_gradient(stage=stage,
                                                      dual=list(problem.solution.dual_vars.values()),
                                                      constraints=constraints)
                        batchwise_sample_gradient.append(np.array(gradient))

                    obj += self.objective[stageIdx].value

                next_idx = np.random.randint(batch)
                  # Randomly select objective
                # ============================================= Batch Loop =============================================
                stagewise_regimes.append(batchwise_regimes[-1])
                if stage > 0:
                    sample_ave_gradient = np.average(batchwise_sample_gradient, axis=0).squeeze().tolist()
                    if not isinstance(sample_ave_gradient, list):
                        sample_ave_gradient = [sample_ave_gradient]
                    theta = self.varValues(self.param_VF[stage-1][stagewise_regimes[-2]])
                    approx = self.VFGL_funs[stage-1]['g_dx'](x=prevSol, theta=theta).squeeze()
                    diff = np.array(sample_ave_gradient) - np.array(approx)
                    diff = np.linalg.norm(diff)
                    sample_devs[stage-1].append(diff)
                prevSol = batchwise_sols[next_idx]  # Randomly select prevSol
            sample_objs.append(obj)
        sample_devs = np.array(sample_devs)
        sample_devs = np.sum(sample_devs, axis=0)
        mean_devs = np.mean(sample_devs)
        std_devs = np.std(sample_devs, ddof=1)
        mean_obj = np.mean(sample_objs)
        std_obj = np.std(sample_objs, ddof=1)
        return mean_devs, std_devs, mean_obj, std_obj

    def forward_pass(self, ub_ind=False):
        """
        Forward pass of SDDP algorithm
        """
        forward_sols = []     # Total forward solution
        forward_VF_sols = []  # Forward solutions only for the value function input variables. Needed for the backward pass.
        ub = []               # Upper bound calculation
        bendersCut_activity = []  # Tracks which Benders Cuts are active (non-zero dual variable) during the forward pass
        for stage in range(self.stageNum):
            self.curStage = stage
            stageIdx = self.stage2prob[stage]
            if stage == 0:
                regime, rv = (np.random.choice(self.regimeNum, 1, p=self.stationaryP)[0], '')
            else:
                regime = np.random.choice(self.regimeNum, 1, p=self.transitionP[self.curRegime, :])[0]
                rv_idx = np.random.choice(len(self.scenarioTree[stage][regime]), 1)[0]
                try:
                    rv = list(self.scenarioTree[stage][regime][rv_idx])
                except TypeError:
                    rv = [self.scenarioTree[stage][regime][rv_idx]]
            self.curRegime = regime
            # Assign rv, previous decision variable, and parameter
            if stage > 0:
                prevStageIdx = self.stage2prob[stage - 1]
                self.assign(self.var_random[stageIdx], rv)                      # Assign rv
                prevSol = []
                for var in self.var_VF[prevStageIdx]:
                    try:
                        prevSol += list(var.value.squeeze())
                    except (AttributeError, TypeError):
                        prevSol += [float(var.value)]
                self.assign(self.var_previous_stage[stageIdx], prevSol)      # Assign previous decision variable
            self.assign(self.var_parameter[stageIdx], self.val_parameter[stage])  # Assign parameter values

            constraints = self.constraints[stageIdx]
            if stage == self.stageNum - 1:
                objective = self.objective[stageIdx]
            else:
                objective = self.objective[stageIdx] + self.Z[stage]
                constraints = constraints + self.benders_cut[stage][regime]
            problem = cp.Problem(cp.Minimize(objective), constraints)
            solution = self.solve_subproblem(problem, stage, regime)
            forward_sols.append(solution)
            forward_VF_sol = []
            for var in self.var_VF[stageIdx]:
                if var.size == 1:
                    forward_VF_sol += [float(var.value)]
                else:
                    forward_VF_sol += var.value.tolist()
            forward_VF_sols.append(forward_VF_sol)
            if stage < self.stageNum - 1:
                bc_activity = []
                for idx, con in enumerate(self.benders_cut[stage][regime]):
                    if np.around(con.dual_value, decimals=3) != 0:
                        bc_activity.append(idx)
                bendersCut_activity.append((regime, bc_activity))
                ub.append(problem.solution.opt_val - float(self.Z[stage].value))
            else:
                ub.append(problem.solution.opt_val)
            if stage == 0 and not ub_ind:
                self.SDDP_lower_bound.append(problem.solution.opt_val)
        return forward_sols, forward_VF_sols, bendersCut_activity, sum(ub)

    def backward_pass(self, forward_solution):
        forward_solution = forward_solution[:-1]  # Exclude final stage solution because it is not used for the benders cut calculation.
        for stage, prevSol in enumerate(forward_solution[::-1]):  # Iterates from T-1 -> 0 stage.
            stage = self.stageNum - 1 - stage
            if stage == 0:
                break
            self.curStage = stage
            stageIdx = self.stage2prob[stage]

            self.assign(self.var_previous_stage[stageIdx], prevSol)      # Assign previous decision variable
            self.assign(self.var_parameter[stageIdx], self.val_parameter[stage])  # Assign parameter values
            grads = []
            cons = []
            for regime in range(self.regimeNum):
                self.curRegime = regime
                regimewise_grad = []
                regimewise_constant = []
                constraints = self.constraints[stageIdx]
                if stage == self.stageNum - 1:
                    objectiveFunction = self.objective[stageIdx]
                else:
                    objectiveFunction = self.objective[stageIdx] + self.Z[stage]
                    constraints = constraints + self.benders_cut[stage][regime]
                for rv in self.scenarioTree[stage][regime]:
                    try:
                        rv = list(rv)
                    except TypeError:
                        rv = [rv]
                    self.assign(self.var_random[stageIdx], rv)                      # Assign rv

                    problem = cp.Problem(cp.Minimize(objectiveFunction), constraints)
                    _ = self.solve_subproblem(problem, stage, regime)
                    grad, constant = self.calculate_benders_cut(problem.solution, constraints, prevSol)
                    regimewise_grad.append(grad)
                    regimewise_constant.append(constant)
                grads.append(np.mean(regimewise_grad, axis=0).squeeze())
                cons.append(np.average(regimewise_constant))
            prevStageIdx = self.stage2prob[stage - 1]
            for prevRegime in range(self.regimeNum):
                bendersCut_grad = np.average(grads, axis=0, weights=self.transitionP[prevRegime, :])
                bendersCut_const = np.average(cons, weights=self.transitionP[prevRegime, :])
                if isinstance(bendersCut_grad, float):
                    bendersCut_grad = [bendersCut_grad]

                bendersCut = bendersCut_const
                if isinstance(bendersCut_grad, list):
                    bendersCut_values = (bendersCut_grad, bendersCut_const)
                else:
                    bendersCut_values = (bendersCut_grad.tolist(), bendersCut_const)

                g_idx = 0
                for VF_var in self.var_VF[prevStageIdx]:
                    if VF_var.size > 1:
                        for VF_var_ele in VF_var:
                            bendersCut += VF_var_ele*bendersCut_grad[g_idx]
                            g_idx += 1
                    else:
                        bendersCut += VF_var*bendersCut_grad[g_idx]
                        g_idx += 1
                bendersCut = bendersCut <= self.Z[stage - 1]
                valid, redundant_idx = self.check_benders_cut_similarity(self.benders_cut_values[stage - 1][prevRegime], bendersCut_values)
                if valid:
                    self.benders_cut[stage - 1][prevRegime].append(bendersCut)
                    self.benders_cut_values[stage - 1][prevRegime].append(bendersCut_values)
                    if self.iterationCount == 0:
                        self.benders_cut[stage - 1][prevRegime] = self.benders_cut[stage - 1][prevRegime][1:]
                elif (not valid) and (redundant_idx is not None):
                    self.benders_cut[stage - 1][prevRegime][redundant_idx] = bendersCut
                    self.benders_cut_values[stage - 1][prevRegime][redundant_idx] = bendersCut_values

    def calculate_benders_cut(self, solution, constraints, prevSol):
        """
        Updates value function with stochastic gradient descent method.
        :param solution:
        :param constraints:
        :param prevSol:
        :return: Theta update amount in 2-norm.
        """
        grad = self.find_gradient(stage=self.curStage, dual=list(solution.dual_vars.values()), constraints=constraints)
        constant = solution.opt_val
        for grad_elementwise, prev_sol_elementwise in zip(grad, prevSol):
            constant += -grad_elementwise*prev_sol_elementwise
        return grad, constant

    def find_SDDP_upper_bound(self, iter=20):
        sampleObj = []
        for i in range(iter):
            _, _, _, ub = self.forward_pass(ub_ind=True)
            sampleObj.append(ub)
        return np.average(sampleObj), np.std(sampleObj, ddof=1)

    def check_benders_cut_similarity(self, bc_coeffs, candidate):
        # Check if the new cut is too close to the existing cut
        grad_dist = [np.linalg.norm(np.array(item[0]) - np.array(candidate[0])) > 1.0e-10 for item in bc_coeffs]
        if not grad_dist or all(grad_dist):
            return True, None  # No cut is too close
        too_close_idx = [i for i, isClose in enumerate(grad_dist) if not isClose][0]
        existing_cut_constant = bc_coeffs[too_close_idx][1]
        if candidate[1] < existing_cut_constant:  # New cut is better
            return False, too_close_idx           # Found a redundant cut. Return its index
        return False, None                        # Better cut already exists

    def solve_subproblem(self, problem, stage=0, regime=0):
        """
        Solves the subproblem. First tries to solve in CPLEX and then solves in ECOS if error occurs.
        :param problem: cvx problem
        :return: solution
        """
        stageIdx = self.stage2prob[stage]
        status = 'not_solved'
        for solverType in self.solverSequence:
            try:
                if solverType == 'CPLEX':
                    problem.solve(verbose=False, warm_start=True, solver=cp.CPLEX)
                    self.curSolver = 'CPLEX'
                elif solverType == 'MOSEK':
                    problem.solve(verbose=False, warm_start=True, solver=cp.MOSEK)
                    self.curSolver = 'MOSEK'
                elif solverType == 'ECOS':
                    problem.solve(verbose=False, warm_start=True, solver=cp.ECOS)
                    self.curSolver = 'ECOS'
                elif solverType == 'SCS':
                    problem.solve(verbose=False, warm_start=True, solver=cp.SCS)
                    self.curSolver = 'SCS'
            except:
                status = problem.status
                continue
            status = problem.status
            if status == 'optimal':
                break
            else:
                warning = "{solver}: {status}".format(solver=solverType, status=status)
                print(warning)

        if status == 'optimal':
            solution = self.varValues(self.var_decision[stageIdx])
            if self.roundSolution >= 0:
                solution = np.round(solution, decimals=self.roundSolution).tolist()
                self.assign(self.var_decision[stageIdx], solution)
            return solution
        else:
            print('stage: {}, regime: {}'.format(stage, regime))
            if stage < self.stageNum - 1:
                theta = self.varValues(self.VF_param[stage][regime])
                print('Current theta value: {}'.format(theta))
            if stage > 0:
                prevStageIdx = self.stage2prob[stage - 1]
                prevSol = []
                for var in self.VF_var[prevStageIdx]:
                    prevSol += list(var.value.squeeze())
                print('PrevSol: {}'.format(prevSol))
            print('Current cumulative theta update: {}'.format(self.theta_update))
            raise Exception(status)

    def find_active_benders_cut(self):
        print('Finding the most active Benders cuts for the VFGL initialization...')
        time.sleep(0.1)
        bendersCut_activity_distribution = []
        for stage in range(self.stageNum - 1):
            temp = []
            for regime in range(self.regimeNum):
                temp.append(np.zeros(len(self.benders_cut_values[stage][regime])))
            bendersCut_activity_distribution.append(temp)

        for _ in tqdm(range(self.sddp_cut_selection_iter), total=self.sddp_cut_selection_iter):
            _, _, bendersCut_activity, _ = self.forward_pass()
            for bc_stage, bc_activity in enumerate(bendersCut_activity):
                for active_index in bc_activity[1]:
                    bendersCut_activity_distribution[bc_stage][regime][active_index] += 1

        active_benders_cuts = []
        active_benders_cut_vals = []
        for stage in range(self.stageNum - 1):
            regimewise_active_benders_cut = []
            regimewise_active_benders_cut_vals = []
            for regime in range(self.regimeNum):
                non_zero_index = np.where(bendersCut_activity_distribution[stage][regime] > 0)[0]
                if self.sddp_cut_selection_num > 0:
                    cut_num = min(len(non_zero_index), self.sddp_cut_selection_num)
                else:
                    cut_num = len(non_zero_index)
                cut_idx = np.argsort(bendersCut_activity_distribution[stage][regime])[-cut_num:]
                regimewise_active_benders_cut.append([cut for idx, cut in enumerate(self.benders_cut[stage][regime]) if idx in cut_idx])
                regimewise_active_benders_cut_vals.append([cut for idx, cut in enumerate(self.benders_cut_values[stage][regime]) if idx in cut_idx])
            active_benders_cuts.append(regimewise_active_benders_cut)
            active_benders_cut_vals.append(regimewise_active_benders_cut_vals)
        print('Active Benders Cuts filtered')
        time.sleep(0.1)
        return active_benders_cuts, active_benders_cut_vals

    def assign(self, variable, value):
        """
        This function assigns values to cvxpy parameters
        :param variable: cvxpy parameters to be assigned.
        :param value: values to be assigned to cvxpy parameters
        :return: None
        """
        if self.isVar(variable):
            variable = [variable]
        elif not isinstance(variable, list):
            variable = list(variable)
        if isinstance(value, np.ndarray):
            value = value.squeeze()
        if not isinstance(value, list):
            try:
                value = list(value)
            except TypeError:
                value = [float(value)]
        valueIdx = 0
        try:
            for var in variable:
                varDim = int(var.size)
                try:
                    var.value = value[valueIdx:valueIdx + varDim]
                except ValueError:
                    var.value = value[valueIdx:valueIdx + varDim][0]
                valueIdx += varDim
        except:
            raise Exception('Stage: {}, Invalid parameter value: {}'.format(self.stage_to_be_updated, value))

    def update_VF(self, x, y):
        """
        Updates value function with stochastic gradient descent method.
        :param solution:
        :param constraints:
        :param prevSol:
        :return: Theta update amount in 2-norm.
        """
        updateStage = self.stage_to_be_updated
        updateRegime = self.regime_to_be_updated
        if not self.preTraining:
            self.visitCount[updateStage][updateRegime] += 1

        # learning_rate = 1/max(self.visitCount[updateStage][updateRegime] - self.burnIn, 1)
        learning_rate = 1/max((self.visitCount[updateStage][updateRegime] - self.burnIn)/(self.stageNum - 1 -updateStage), 1)


        if isinstance(self.ave_VF_grad[updateStage][updateRegime], str):
            self.ave_sample_grad[updateStage][updateRegime] = y
        else:
            self.ave_sample_grad[updateStage][updateRegime] = y = list((1-learning_rate)*np.array(self.ave_sample_grad[updateStage][updateRegime]) + learning_rate*np.array(y))
        ave_y = self.ave_sample_grad[updateStage][updateRegime]
        theta = self.varValues(self.param_VF[updateStage][updateRegime])

        if 'SGD' == self.gd_method and not self.preTraining:
            direction = self.VFGL_funs[updateStage]['loss_dt'](x, y, theta, self.ave_sample_grad[updateStage][updateRegime]).squeeze()
            theta_new = np.array(theta) - learning_rate*direction
            theta_new = theta_new.tolist()
        elif not any([item in self.gd_method.upper() for item in ['ADAM', 'RMSPROP', 'MOMENTUM']]) and not self.preTraining:  # Regular ISGD
            theta_new = self.VFGL_funs[updateStage]['ISGD'](cur_x=x, cur_y=y, ave_y=ave_y, cur_theta=theta,
                                                            cur_alpha=learning_rate)
        else:
            loss_dt = self.VFGL_funs[updateStage]['loss_dt'](x=x, y=y, ave_y=ave_y, theta=theta).squeeze()
            if self.iterationCount == 0:
                self.ave_VF_grad[updateStage][updateRegime] = loss_dt.tolist()
                self.ave_VF_grad_squared[updateStage][updateRegime] = (np.power(loss_dt, 2)).tolist()
            ave_VF_grad = []
            ave_VF_grad_squared = []
            if any([item in self.gd_method.upper() for item in ['ADAM', 'MOMENTUM']]):
                ave_VF_grad = self.ave_VF_grad[updateStage][updateRegime]
            if any([item in self.gd_method.upper() for item in ['ADAM', 'RMSPROP']]):
                ave_VF_grad_squared = self.ave_VF_grad_squared[updateStage][updateRegime]
            theta_new = self.VFGL_funs[updateStage]['Extended_ISGD'](cur_x=x, cur_y=y, ave_y=ave_y, cur_theta=theta,
                                                                     cur_alpha=learning_rate,
                                                                     momentum=ave_VF_grad,
                                                                     squared_momentum=ave_VF_grad_squared)
            ave_VF_grad = self.ADAM_beta1*np.array(self.ave_VF_grad[updateStage][updateRegime]) + (1 - self.ADAM_beta1)*loss_dt
            ave_VF_grad_squared = self.ADAM_beta2*np.array(self.ave_VF_grad_squared[updateStage][updateRegime]) + (1 - self.ADAM_beta2)*np.power(loss_dt, 2)
            self.ave_VF_grad[updateStage][updateRegime] = ave_VF_grad.tolist()
            self.ave_VF_grad_squared[updateStage][updateRegime] = ave_VF_grad_squared.tolist()

        rawGrad = self.VFGL_funs[updateStage]['loss_dt'](x, y, theta, self.ave_sample_grad[updateStage][updateRegime])
        y_expected_gradient_with_old_theta = self.VFGL_funs[updateStage]['g_dx'](x, theta)
        y_expected_gradient_with_new_theta = self.VFGL_funs[updateStage]['g_dx'](x, theta_new)  # Not necessary. For check only.

        if self.roundTheta >= 0:
            theta_new = np.round(theta_new, decimals=self.roundTheta).tolist()
        self.assign(self.param_VF[updateStage][updateRegime], theta_new)
        updateAmount = np.linalg.norm(np.array(theta) - np.array(theta_new))  # Theta update amount.
        lossAmount = self.VFGL_funs[updateStage]['loss'](x, y, self.ave_sample_grad[updateStage][updateRegime], theta)
        return updateAmount, lossAmount

    def update_sample_ave_grad(self, stage, regime, grad, learning_rate=1):
        sample_ave_grad = self.sample_ave_grad[stage][regime]
        grad_squared = np.array([item**2 for item in grad])
        if self.gd_method.upper() == 'ISGD':
            return 0
        elif 'ISGD_ADAM' in self.gd_method.upper():
            if isinstance(sample_ave_grad, str):
                self.ADAM_Momentum[stage][regime] = np.array(grad)
                return grad_squared
            else:
                self.ADAM_Momentum[stage][regime] = self.ADAM_beta1*self.ADAM_Momentum[stage][regime] + (1-self.ADAM_beta1)*np.array(grad)
                return self.ADAM_beta2*sample_ave_grad + (1-self.ADAM_beta2)*grad_squared
        elif 'ADAGRAD' in self.gd_method.upper():
            if isinstance(sample_ave_grad, str):
                return grad_squared
            else:
                return sample_ave_grad + grad_squared
        elif 'RMSPROP' in self.gd_method.upper():
            if isinstance(sample_ave_grad, str):
                return grad_squared
            else:
                return (1-learning_rate)*sample_ave_grad + learning_rate*grad_squared

    def find_gradient(self, stage, dual, constraints):
        """
        Finds the gradient of previous stage decision variables.
        :param stage: Current stage
        :param dual: Current subproblem optimal dual variable
        :param constraints: Current constraints
        :return:
        """
        stageIdx = self.stage2prob[stage]
        prevX = self.var_previous_stage[stageIdx]
        prevX = self.decomposeVar(prevX)
        grad = [0]*len(prevX)

        paramVal = self.var_parameter[stageIdx]
        rv = self.var_random[stageIdx]
        dv = self.var_decision[stageIdx]

        sep_dual = []
        for d in dual:
            if isinstance(d, float) or isinstance(d, int):
                sep_dual.append(d)
            elif isinstance(d, np.ndarray):
                sep_dual += d.tolist()
            elif isinstance(d, list):
                sep_dual += d
            else:
                raise Exception('Unknown dual type' + str(type(d)))

        for constraint, dualVal in zip(constraints, sep_dual):
            constraint = str(constraint)
            ctype = constraint[::-1][constraint[::-1].index('=') + 1]
            if ctype == '=':
                lhs = constraint[:constraint.index('=')]
            else:
                lhs = constraint[:constraint.index('=') - 1]
            lhs = self.replaceVar(lhs, paramVal)
            lhs = self.replaceVar(lhs, rv)
            lhs = self.replaceVar(lhs, dv, zero=True)
            if ctype == '=':
                rhs = constraint[constraint.index('=') + 2:]
            else:
                rhs = constraint[constraint.index('=') + 1:]
            rhs = self.replaceVar(rhs, paramVal)
            rhs = self.replaceVar(rhs, rv)
            rhs = self.replaceVar(rhs, dv, zero=True)
            for idx, px in enumerate(prevX):
                lc = self.find_coefficient(lhs, px)
                if ctype in {'>'}:
                    lc = -1*lc
                rc = self.find_coefficient(rhs, px)
                if ctype in {'<'}:
                    rc = -1*rc
                if self.curSolver == 'CPLEX' and ctype in {'='}:
                    lc = -1*lc
                    rc = -1*rc
                grad[idx] -= lc*float(dualVal)
                grad[idx] -= rc*float(dualVal)
        return grad

    def update_only_linear(self, y):
        updateStage = self.stage_to_be_updated
        updateRegime = self.regime_to_be_updated
        theta = self.varValues(self.param_VF[updateStage][updateRegime])
        theta_new = y + theta[len(y):]
        if self.roundTheta >= 0:
            theta_new = np.round(theta_new, decimals=self.roundTheta).tolist()
        self.assign(self.param_VF[updateStage][updateRegime], theta_new)
        return np.inf, np.inf

    def isVar(self, variable):
        """
        Identifies if the given input is cvx variable
        :param variable: variable candidate
        :return: Boolean
        """
        try:
            variable.value
            return True
        except AttributeError:
            return False

    def varDim(self, variable):
        """
        Returns the cvx variable dimension
        :param variable: cvx variable
        :return: cvx variable dimension
        """
        if self.isVar(variable):
            variable = [variable]
        elif not isinstance(variable, list):
            variable = list(variable)
        output = 0
        for var in variable:
            output += var.size
        return output

    def varValues(self, variable):
        """
        Returns the current value of cvx variables
        :param variable: cvx variables
        :return: assigned values of cvx variables
        """
        if self.isVar(variable):
            variable = [variable]
        elif not isinstance(variable, list):
            variable = list(variable)
        output = []
        for var in variable:
            try:
                output += list(var.value)
            except TypeError:
                output += [float(var.value)]
        return output

    def decomposeVar(self, variable):
        """
        Decomposes the multidimensional variable into elementwise index variables
        :param variable: cvx variable
        :return: Decomposed variable
        """
        if self.isVar(variable):
            variable = [variable]
            if variable[0].size == 1:
                return variable
        if len(variable) == 1:
            if variable[0].size == 1:
                return variable
        output = []
        for var in variable:
            for varElement in var:
                output.append(varElement)
        return output

    def replaceVar(self, strConstraint, variable, zero=False):
        """
        Replaces variable in the string constraints to its value
        :param strConstraint: problem constraint in string value
        :param variable: variable to replace
        :param zero: True if 0 should be assigned to the variable to be replaced
        :return: replaced string constraint
        """
        if self.isVar(variable):
            variable = [variable]
        else:
            variable = list(variable)
        for var in variable:
            if var.size == 1:
                if not zero:
                    strConstraint = strConstraint.replace(str(var), str(float(var.value)))
                else:
                    strConstraint = strConstraint.replace(str(var), str(0))
            else:
                for varEle in var:
                    if not zero:
                        strConstraint = strConstraint.replace(str(varEle), str(varEle.value))
                    else:
                        strConstraint = strConstraint.replace(str(varEle), str(0))
        return strConstraint

    def find_coefficient(self, strConstraint, var):
        """
        This function finds the coefficient of the variable from string constraint. It is assumed that all the parameter
        values are assigned in float value.
        :param strConstraint: string constraint
        :param var: cvx variable
        :return: coefficient of var
        """
        output = [0]
        if strConstraint.find('exp') != -1:
            if strConstraint[strConstraint.find('exp') + 3] != '(':
                strConstraint = strConstraint.replace('exp', 'exp(')
                for i in range(strConstraint.find('exp'), len(strConstraint)):
                    if strConstraint[i] in {'+', '-', '*'}:
                        strConstraint = strConstraint[:i] + ')' + strConstraint[i:]
                        break
        if not isinstance(var, str):
            var = str(var)
        # Remove all parenthesis by evaluating them
        while strConstraint.find('(') != -1:
            start = strConstraint.find('exp(')
            if start == -1:
                start = strConstraint.find('(')
            parenthesis = strConstraint[start:strConstraint.find(')') + 1]
            parenthesis = parenthesis.replace("@", "*")
            parenthesis = eval(parenthesis)
            strConstraint = strConstraint[:start] + str(parenthesis) + strConstraint[strConstraint.find(')') + 1:]
        terms = []
        term = ''
        for idx, ch in enumerate(strConstraint):
            if ch not in {'+', '-', ' '}:
                term += ch
            elif ch in {'+', '-'}:
                term = term.strip()
                if len(term) > 0:
                    terms.append(term)
                term = ch
        terms.append(term)
        for term in terms:
            if var in term:
                term = term.replace('+' + var, '1')
                term = term.replace('-' + var, '-1')
                term = term.replace(var, '1')
                if self.name.upper() == "MPO":
                    term = term.replace('@', '*')
                output.append(eval(term))
        return sum(output)

    def save_VF(self, filename: str):
        if 'VFGL' not in self.algorithm:
            print('Not a VFGL algorithm. No parameter to save.')
            return
        root = './'
        flist = os.listdir(root)
        while 'problems' not in flist and 'trained' not in flist:
            root += '../'
        root += 'trained/'
        filename = root + filename
        if '.npy' not in filename:
            filename += '.npy'

        output = []
        for stage_param in self.param_VF:
            temp_output = []
            for regime in range(self.regimeNum):
                val = self.varValues(stage_param[regime])
                temp_output.append(val)
            output.append(temp_output)

        with open(filename, 'wb') as f:
            np.save(f, output)
        print('Value function parameters saved at {}'.format(filename))

    def load_VF(self, filename, stageIdx=[]):
        if 'VFGL' not in self.algorithm:
            print('Not a VFGL algorithm. No parameter to load.')
            return
        root = './'
        flist = os.listdir(root)
        while 'problems' not in flist and 'trained' not in flist:
            root += '../'
        root += 'trained/'
        filename = root + filename
        if '.npy' not in filename:
            filename += '.npy'
        with open(filename, 'rb') as f:
            params = np.load(f)
            if isinstance(params, np.ndarray):
                params = params.tolist()
        # if len(self.VF_param) > len(params):
        #     lenDiff = len(self.VF_param) - len(params)
        #     params += params[-1]*lenDiff
        if not stageIdx:
            for stage_param, stage_param_loaded in zip(self.param_VF, params):
                for regime in range(self.regimeNum):
                    self.assign(stage_param[regime], stage_param_loaded[regime])
        else:
            for stage, stage_param_loaded in zip(stageIdx, params):
                stage_param = self.param_VF[stage]
                for regime in range(self.regimeNum):
                    self.assign(stage_param[regime], stage_param_loaded[regime])
                    if stage == 0:
                        break
        self.VF_loaded = True
        print('Pretrained value function parameter values successfully loaded')

    def save_BC(self, fn=''):
        if 'SDDP' not in self.algorithm:
            print('Not a SDDP algorithm. No parameter to save.')
            return
        root = './'
        flist = os.listdir(root)
        while 'problems' not in flist and 'trained' not in flist:
            root += '../'
        root += 'trained/'
        filename = root + fn
        if '.pkl' not in filename:
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump((self.benders_cut, self.benders_cut_values), f, pickle.HIGHEST_PROTOCOL)
        print('Benders cuts successfully saved')

    def read_BC(self, fn=''):
        root = './'
        flist = os.listdir(root)
        while 'problems' not in flist and 'trained' not in flist:
            root += '../'
        root += 'trained/'
        filename = root + fn
        if '.pkl' not in filename:
            filename += '.pkl'
        with open(filename, 'rb') as f:
            self.benders_cut, self.benders_cut_values = pickle.load(f)
        print('Benders cuts successfully loaded')
