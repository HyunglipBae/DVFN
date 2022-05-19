import numpy as np
import sympy as sp
import math
from tqdm import tqdm_notebook
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import time
import warnings

warnings.filterwarnings(action='ignore')

class DVFN():
    def __init__(self, problem_definition):

        # Problem Related
        self.n_stages = problem_definition['n_stages']
        self.scenario_generator = problem_definition['ScenarioGenerator']
        self.problem = problem_definition['problem']
        self.stage_to_prob = problem_definition['stage_to_prob']
        self.objective = []
        self.objective_ff = []
        self.ineq_constraints = []
        self.ineq_constraints_ff = []
        self.eq_constraints = []
        self.VF_var_idx = []
        self.decision_var = []
        self.decision_var_ff = []
        self.previous_stage_var = []
        self.random_var = []
        self.insert_problem()

        # Forward Pass Related
        self.batch = 3
        self.iteration_count = 0
        self.iter_time = []
        self.min_iter = 50
        self.max_iter = 50
        self.forward_sol = []
        # self.test_rvs = np.random.standard_normal(10000000)
        # self.rvs = np.random.standard_normal(100000)
        # self.true_obj = self.work(self.test_rvs, 0.05, 0.55, 0.4)

        # Primal-Dual Algorithm Related
        self.alpha = 0.01
        self.beta = 0.5
        self.epsilon0 = 0.01
        self.epsilon = 0.01
        self.quasi = 0.00000000001
        self.repeat = 17
        self.pda_round = 2

        # ICNN Related
        self.ICNNs = [None]
        self.nodes = 64
        self.h_layers = 1
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0015)
        self.n_epochs = 30
        self.n_cuts = 20
        self.ICNN_parameter = []
        for t in range(self.n_stages):
            self.ICNN_parameter.append([])

        self.par_change = []

        self.weight = np.zeros((self.n_stages, 3))
        self.prev_sol = []
        self.vf_grad = []
        for t in range(self.n_stages):
            self.prev_sol.append([])
            self.vf_grad.append([])

    def insert_problem(self):
        """
        Inserts user defined problem to the solver
        :param problem: problem definition. List of dictionary that defines stagewise problem. Dictionary with the keys:
                       {'objective', 'constraints', 'var_VF', 'var_decision', 'var_previous_stage', 'var_random', 'var_parameter'}
        :param stage_map_idx: stage to problem index map
        :param parameter_value: list of parameter values
        :return: None
        """
        self.objective = [swprob['objective'] for swprob in self.problem]
        self.objective_ff = [swprob['objective_ff'] for swprob in self.problem]
        self.ineq_constraints = [swprob['ineq_constraints'] for swprob in self.problem]
        self.ineq_constraints_ff = [swprob['ineq_constraints_ff'] for swprob in self.problem]
        self.eq_constraints = [swprob['eq_constraints'] for swprob in self.problem]
        self.VF_var_idx = [swprob['VF_var_idx'] for swprob in self.problem]
        self.decision_var = [swprob['decision_var'] for swprob in self.problem]
        self.decision_var_ff = [swprob['decision_var_ff'] for swprob in self.problem]
        self.previous_stage_var = [swprob['previous_stage_var'] for swprob in self.problem]
        self.random_var = [swprob['random_var'] for swprob in self.problem]
        self.VF_var_dim = len(self.VF_var_idx[0])
        print('Problem is defined...')

    def initialize_ICNN(self):

        print('Initializing the Parametric Value Function')
        # kernel_initializer = tf.keras.initializers.RandomNormal()
        # bias_initializer = tf.keras.initializers.Ones()
        for t in range(1, self.n_stages):
            y = keras.Input(shape=(self.VF_var_dim,))
            z = layers.Dense(units=self.nodes)(y)
            # z = layers.BatchNormalization()(z)
            z = keras.activations.elu(z)
            for h in range(self.h_layers):
                zh_1 = layers.Dense(units=self.nodes,
                                    kernel_constraint=keras.constraints.NonNeg())(z)
                zh_2 = layers.Dense(units=self.nodes,
                                    use_bias=False)(y)
                z = layers.add([zh_1, zh_2])
                # z = layers.BatchNormalization()(z)
                z = keras.activations.elu(z)
            zo_1 = layers.Dense(units=1,
                                kernel_constraint=keras.constraints.NonNeg(),
                                use_bias=False)(z)
            zo_2 = layers.Dense(units=1,
                                use_bias=False)(y)
            z = layers.add([zo_1, zo_2])
            model = keras.Model(inputs=y, outputs=z, name="V_{}".format(t + 1))
            self.ICNNs.append(model)
        self.ICNNs.append(0.0)
        # kernel_initializer = keras.initializers.RandomUniform(0.0, 0.01),

        for stage in range(1, self.n_stages):
            self.ICNN_parameter[stage].append(self.ICNNs[stage].get_weights())

    def ICNN_loss(self, stage):

        x_true = self.prev_sol[stage]
        y_grad_true = self.vf_grad[stage]

        n_samples = len(y_grad_true)

        x_true = np.array(x_true).reshape((n_samples, self.VF_var_dim))
        y_grad_true = np.array(y_grad_true).reshape((n_samples, self.VF_var_dim))

        # for epoch in range(self.n_epochs):
        #
        #     with tf.GradientTape(persistent=True) as tape:
        #
        #         x_tensor_true = tf.convert_to_tensor(x_true, dtype=tf.float32)
        #         tape.watch(x_tensor_true)
        #         y_tensor_pred = self.ICNNs[stage](x_tensor_true, training=True)
        #         y_grad_pred = tape.gradient(y_tensor_pred, x_tensor_true)
        #         ICNN_loss = tf.reduce_sum(tf.pow(y_grad_true - y_grad_pred, 2))

        x_train_tensor = tf.Variable(x_true)

        with tf.GradientTape() as tape:
            y_tensor = self.ICNNs[stage](x_train_tensor)
        y_grad_pred = tape.gradient(y_tensor, x_train_tensor)

        y_grad_pred = y_grad_pred / self.weight[stage]
        y_grad_true = y_grad_true / self.weight[stage]
        ICNN_loss = sum(keras.losses.mse(y_grad_pred, y_grad_true)) * self.VF_var_dim / (n_samples)

        # ICNN_loss = tf.reduce_sum(tf.pow(y_grad_true - y_grad_pred, 2))

            # grads = tape.gradient(ICNN_loss, self.ICNNs[stage].trainable_variables)
            # self.optimizer.apply_gradients(zip(grads, self.ICNNs[stage].trainable_variables))
            #
            # if epoch == self.n_epochs - 1:
            #     K.print_tensor(ICNN_loss, message='ICNN_loss = ')

        return ICNN_loss

    def ICNN_loss_gradient(self, stage):
        with tf.GradientTape() as tape:
            tape.watch(self.ICNNs[stage].trainable_variables)
            loss_value = self.ICNN_loss(stage)
        loss_gradient = tape.gradient(loss_value, self.ICNNs[stage].trainable_variables)

        return loss_value, loss_gradient

    def update_ICNN(self, stage):

        optimizer = self.optimizer
        for epoch in range(self.n_epochs):
        # while loss_value > self.required_loss:
            # epoch_loss_avg = tf.keras.metrics.Mean()

            loss_value, loss_gradient = self.ICNN_loss_gradient(stage)
            optimizer.apply_gradients(zip(loss_gradient, self.ICNNs[stage].trainable_variables))

            if epoch == self.n_epochs - 1:
                K.print_tensor(loss_value, message='ICNN_loss = ')

        # print("{} Loss: {:.7f}".format(self.ICNNs[stage].name, loss_value))
        # if epoch == self.n_epochs - 1:
        #   print("{} Epoch {:03d}: Loss: {:.7f}".format(self.ICNNs[stage].name, epoch, epoch_loss_avg.result()))
        self.ICNN_parameter[stage].append(self.ICNNs[stage].get_weights())

    def caculate_parameter_change(self):

        total_change = []
        for stage in range(1, self.n_stages):
            parameter_before = self.ICNN_parameter[stage][self.iteration_count]
            parameter_after = self.ICNN_parameter[stage][self.iteration_count + 1]
            change_list = []
            par_length = len(parameter_before)
            for par in range(par_length):
                norm = np.linalg.norm(parameter_after[par] - parameter_before[par])
                change_list.append(norm)
            total_change.append(np.mean(change_list))

        self.par_change.append(sum(total_change))

    def assign(self, ineq_con_ff, ineq_con, eq_con, previous_stage_var, random_var, prev_sol, realization):

        ineq_con_ff_result = []
        ineq_con_result = []
        eq_con_result = []

        for j in range(len(ineq_con_ff)):
            a = ineq_con_ff[j]
            for i in range(len(prev_sol)):
                for r in range(len(realization)):
                    a = a.subs([(previous_stage_var[i], prev_sol[i, 0]), (random_var[r], realization[r])])
            ineq_con_ff_result.append(a)

        for j in range(len(ineq_con)):
            a = ineq_con[j]
            for i in range(len(prev_sol)):
                for r in range(len(realization)):
                    a = a.subs([(previous_stage_var[i], prev_sol[i, 0]), (random_var[r], realization[r])])
            ineq_con_result.append(a)

        for h in range(len(eq_con)):
            b = eq_con[h]
            for i in range(len(prev_sol)):
                for r in range(len(realization)):
                    b = b.subs([(previous_stage_var[i], prev_sol[i, 0]), (random_var[r], realization[r])])
            eq_con_result.append(b)

        return ineq_con_ff_result, ineq_con_result, eq_con_result

    def find_gradient(self, f_x_grad_prev, ineq_for_grad, eq_for_grad, previous_stage_var, random_var, u, v, prev_sol, realization):
        ineq_con_term = 0
        for i in range(u.shape[0]):
            ineq_i_dx = sp.diff(ineq_for_grad[i], previous_stage_var)
            lambdified_ineq_i_dx = sp.lambdify(previous_stage_var, ineq_i_dx, "numpy")
            gradient = lambdified_ineq_i_dx(*prev_sol).astype(np.float32)
            ineq_con_term += u[i, 0] * gradient

        eq_con_term = 0
        for i in range(v.shape[0]):
            eq_i_dx = sp.diff(eq_for_grad[i], previous_stage_var)
            tar_var = sp.Matrix([previous_stage_var, random_var])
            lambdified_eq_i_dx = sp.lambdify(tar_var, eq_i_dx, "numpy")
            if len(realization) > 1:
              realization = np.array(realization).reshape(len(realization), 1)
              tar_input = np.concatenate((prev_sol, realization))
            else:
              tar_input = np.concatenate((prev_sol, [realization]))
            gradient = lambdified_eq_i_dx(*tar_input).astype(np.float32)
            gradient = gradient.reshape(prev_sol.shape[0], 1)
            eq_con_term += v[i, 0] * gradient

        obj_term = f_x_grad_prev(*prev_sol).astype(np.float32)

        vf_grad = ineq_con_term + eq_con_term + obj_term

        return vf_grad

    # def find_objective(self, f_x, ICNN, x):
    #
    #     f_value = f_x(*x).astype(np.float32)
    #     if self.n_stages == 2:
    #         ICNN_value = 0.0
    #     else:
    #         ICNN_value = ICNN(np.array([[x[0, 0], x[1, 0]]])).numpy()[0, 0]
    #
    #     obj = f_value + ICNN_value
    #
    #     return obj

    def solve_stagewise_decomposed_problem(self, stage, realization, prev_sol):

        stage_idx = self.stage_to_prob[stage]

        if self.iteration_count == 0:
            ICNN = 0.0
        else:
            ICNN = self.ICNNs[stage + 1]
        objective = self.objective[stage_idx]
        objective_ff = self.objective_ff[stage_idx]
        ineq_con = self.ineq_constraints[stage_idx]
        ineq_con_ff = self.ineq_constraints_ff[stage_idx]
        eq_con = self.eq_constraints[stage_idx]
        decision_var = self.decision_var[stage_idx]
        decision_var_ff = self.decision_var_ff[stage_idx]
        VF_var_idx = self.VF_var_idx[stage_idx]
        previous_stage_var = self.previous_stage_var[stage_idx]
        random_var = self.random_var[stage_idx]

        if stage > 0:
            ineq_con_ff, ineq_con, eq_con = self.assign(ineq_con_ff,
                                           ineq_con,
                                           eq_con,
                                           previous_stage_var,
                                           random_var,
                                           prev_sol,
                                           realization
                                           )

        f_dx = sp.diff(objective, decision_var)
        f_x_grad = sp.lambdify(decision_var, f_dx, "numpy")

        f_dxdx = sp.diff(f_dx, decision_var)
        f_dxdx = f_dxdx.reshape(len(decision_var), len(decision_var))
        f_x_hess = sp.lambdify(decision_var, f_dxdx, "numpy")

        f_dx_ff = sp.diff(objective_ff, decision_var_ff)
        f_x_grad_ff = sp.lambdify(decision_var_ff, f_dx_ff, "numpy")

        f_dxdx_ff = sp.diff(f_dx_ff, decision_var_ff)
        f_dxdx_ff = f_dxdx_ff.reshape(len(decision_var_ff), len(decision_var_ff))
        f_x_hess_ff = sp.lambdify(decision_var_ff, f_dxdx_ff, "numpy")

        h_x = []
        h_x_grad = []
        # h_x_hess = []
        for i in range(len(ineq_con)):
            lambdified_h_i_x = sp.lambdify(decision_var, ineq_con[i], "numpy")
            h_x.append(lambdified_h_i_x)

            h_i_dx = sp.diff(ineq_con[i], decision_var)
            lambdified_h_i_dx = sp.lambdify(decision_var, h_i_dx, "numpy")
            h_x_grad.append(lambdified_h_i_dx)

        h_x_ff = []
        h_x_grad_ff = []
        # h_x_hess = []
        for i in range(len(ineq_con_ff)):
            lambdified_h_i_x_ff = sp.lambdify(decision_var_ff, ineq_con_ff[i], "numpy")
            h_x_ff.append(lambdified_h_i_x_ff)

            h_i_dx_ff = sp.diff(ineq_con_ff[i], decision_var_ff)
            lambdified_h_i_dx_ff = sp.lambdify(decision_var_ff, h_i_dx_ff, "numpy")
            h_x_grad_ff.append(lambdified_h_i_dx_ff)

        sw_problem = {'ICNN': ICNN,
                      'f_x_grad': f_x_grad,
                      'f_x_hess': f_x_hess,
                      'h_x': h_x,
                      'h_x_grad': h_x_grad,
                      'eq_con': eq_con,
                      'decision_var': decision_var,
                      'VF_var_idx': VF_var_idx,
                      'previous_stage_var': previous_stage_var,
                      'random_var': random_var,
                      'stage_idx': stage_idx}

        sw_problem_ff = {'ICNN': 0.0,
                      'f_x_grad': f_x_grad_ff,
                      'f_x_hess': f_x_hess_ff,
                      'h_x': h_x_ff,
                      'h_x_grad': h_x_grad_ff,
                      'eq_con': eq_con,
                      'decision_var': decision_var_ff,
                      'VF_var_idx': VF_var_idx,
                      'previous_stage_var': previous_stage_var,
                      'random_var': random_var,
                      'stage_idx': stage_idx}

        x0, u0, v0 = self.InitialSolution(sw_problem_ff, realization, stage, prev_sol)
        # print("Initial solutions are ready")
        x, u, v = self.PDA(sw_problem, x0=x0, u0=u0, v0=v0)

        if stage > 0:
            ineq_for_grad = self.ineq_constraints[stage_idx]
            eq_for_grad = self.eq_constraints[stage_idx]
            f_dx_prev = sp.diff(objective, previous_stage_var)
            f_x_grad_prev = sp.lambdify(previous_stage_var, f_dx_prev, "numpy")
            vf_grad = self.find_gradient(f_x_grad_prev, ineq_for_grad, eq_for_grad, previous_stage_var, random_var, u, v, prev_sol, realization)
            # vf_grad = np.round(vf_grad, self.pda_round)
            # vf_obj = self.find_objective(f_x, sw_problem["ICNN"], x)
        else:
            vf_grad = None

        vf_obj = None

        return x, u, v, vf_obj, vf_grad

    def DVFN_forward(self):

        sc_update = 0
        prev_sol = []
        realization = []

        for stage in range(self.n_stages):
            stage_idx = self.stage_to_prob[stage]
            if stage == 0:
                x, u, v, vf_obj, vf_grad = self.solve_stagewise_decomposed_problem(stage, realization, prev_sol)
                print("stage: ", stage)
                print("x: ", np.round(x, self.pda_round))
                # print("u: ", u)
                # print("v: ", v)
                self.forward_sol.append(x)
                prev_sol = x[self.VF_var_idx[stage_idx], :]
                # print("prev_sol: ", prev_sol)

                self.prev_sol[stage + 1].append(prev_sol)
                # print("stage {} Solution Ready".format(stage))
                if self.n_stages != 2:
                    if len(self.prev_sol[stage + 1]) > self.n_cuts:
                        self.prev_sol[stage + 1] = self.prev_sol[stage + 1][-self.n_cuts:]
            else:
                sample_gradient = []
                batchwise_sol = []
                for s in range(self.batch):
                    realization = next(self.scenario_generator)
                    x, u, v, vf_obj, vf_grad = self.solve_stagewise_decomposed_problem(stage, realization, prev_sol)
                    # if stage == 1:
                    #     print("stage: ", stage)
                    #     print("batch: ", s)
                    #     print("realization: ", realization)
                    #     print("x: ", x)
                    #     print("u: ", u)
                    #     print("v: ", v)
                    sample_gradient.append(vf_grad)
                    # print("vf_grad", vf_grad)
                    batchwise_sol.append(x)
                gradient = sum(sample_gradient) / self.batch
                # print("gradient: ", gradient)
                # print("objective value: ", value)
                self.update_weight(self.iteration_count, stage, gradient)
                self.vf_grad[stage].append(gradient)
                # print("gradient: ", gradient)
                # print("weight: ", self.weight)
                if stage != self.n_stages - 1:
                    if len(self.vf_grad[stage]) > self.n_cuts:
                        self.vf_grad[stage] = self.vf_grad[stage][-self.n_cuts:]

                self.update_ICNN(stage)

                if stage < self.n_stages - 1:
                    path = np.random.randint(self.batch)
                    next_prev_sol = batchwise_sol[path][self.VF_var_idx[stage_idx], :]
                    prev_sol = next_prev_sol
                    # print("prev_sol: ", prev_sol)

                    self.prev_sol[stage + 1].append(prev_sol)
                    # print("stage {} Solution Ready".format(stage))

                    if stage != (self.n_stages-2):
                        if len(self.prev_sol[stage + 1]) > self.n_cuts:
                            self.prev_sol[stage + 1] = self.prev_sol[stage + 1][-self.n_cuts:]

                # d_sc += changed_sc

    def update_weight(self, iteration, stage, gradient):

        gradient = gradient.reshape(gradient.shape[0])
        for i in range(gradient.shape[0]):
            self.weight[stage][i] = (iteration / (iteration + 1)) * self.weight[stage][i] + (1 / (iteration + 1)) * \
                                    gradient[i]


    # def DVFN_backward(self):
    #
    #     for stage in reversed(range(1, self.n_stages)):
    #         # print("stage {} Update Start".format(stage - 1))
    #         self.update_ICNN(stage)

    def eq_con_to_matrix(self, sw_problem):

        eq_con = sw_problem['eq_con']
        decision_var = sw_problem['decision_var']

        Ab = sp.linear_eq_to_matrix(eq_con, *decision_var)

        A = Ab[0]
        A = np.array(A).astype(np.float32)
        b = Ab[1]
        b = np.array(b).astype(np.float32)

        return A, b

    def f_grad_value(self, sw_problem, x):

        f_x_grad = sw_problem['f_x_grad']
        ICNN = sw_problem['ICNN']
        VF_var_idx = sw_problem['VF_var_idx']

        # ICNN gradient
        if (ICNN == None) or (ICNN == 0.0):
            ICNN_gradient = 0.0
        else:
            VF_var = x[VF_var_idx, :]
            VF_var_tensor = tf.Variable(VF_var.reshape((1, VF_var.shape[0])), dtype=tf.float32)
            with tf.GradientTape() as tape:
                tape.watch(VF_var_tensor)
                y_tensor = ICNN(VF_var_tensor)

            ICNN_partial_gradient = tape.gradient(y_tensor, VF_var_tensor)
            ICNN_partial_gradient = tf.reshape(ICNN_partial_gradient, [VF_var_tensor.shape[1], 1])
            ICNN_gradient = np.zeros((x.shape[0], 1))
            for i, idx in enumerate(VF_var_idx):
                ICNN_gradient[idx, 0] = ICNN_partial_gradient[i, 0]

        # original obj gradient
        obj_gradient = f_x_grad(*x).astype(np.float32)

        gradient = np.array(ICNN_gradient + obj_gradient)
        if gradient.ndim != 2:
            gradient = gradient.reshape((x.shape[0], 1))

        return np.array(gradient)
        # if sw_problem['stage_idx'] != 2:
        #     return gradient + 0.0000001 * x
        # else:
        #     return gradient

    def f_hess_value(self, sw_problem, x):

        f_x_hess = sw_problem['f_x_hess']
        ICNN = sw_problem['ICNN']
        VF_var_idx = sw_problem['VF_var_idx']

        # ICNN hessian
        if (ICNN == None) or (ICNN == 0.0):
            ICNN_hessian = np.zeros((x.shape[0], x.shape[0]))
        else:
            VF_var = x[VF_var_idx, :]
            VF_var = VF_var.reshape((1, VF_var.shape[0]))
            VF_var_tensor = tf.Variable(VF_var, dtype=tf.float32)
            with tf.GradientTape() as tape2:
                with tf.GradientTape() as tape1:
                    y_tensor = ICNN(VF_var_tensor)
                gradient = tape1.gradient(y_tensor, VF_var_tensor)
            ICNN_partial_hessian = tape2.jacobian(gradient, VF_var_tensor)
            ICNN_partial_hessian = tf.reshape(ICNN_partial_hessian, [VF_var_tensor.shape[1], VF_var_tensor.shape[1]])
            ICNN_hessian = np.zeros((x.shape[0], x.shape[0]))
            for i, idx in enumerate(VF_var_idx):
                for j, jdx in enumerate(VF_var_idx):
                    ICNN_hessian[idx, jdx] = ICNN_partial_hessian[i, j]

        obj_hessian = np.array(f_x_hess(*x)).astype(np.float32)

        hessian = ICNN_hessian + obj_hessian

        if hessian.ndim != 2:
            hessian = hessian.reshape((x.shape[0], x.shape[0]))

        return hessian

    def h_x_value(self, sw_problem, x):

        h_x = sw_problem['h_x']
        result = []
        for i in range(len(h_x)):
            result.append(h_x[i](*x).astype(np.float32))

        result = np.array(result)

        return result

    def h_x_grad_value(self, sw_problem, x, ith):

        h_x_grad = sw_problem['h_x_grad']

        gradient = h_x_grad[ith](*x).astype(np.float32)

        return np.array(gradient)

    def h_x_hess_value(self, sw_problem, x, ith):

        # h_i_dx = sp.diff(ineq_con[ith], decision_var)
        # h_i_dxdx = sp.diff(h_i_dx, decision_var)
        # h_i_dxdx = h_i_dxdx.reshape(x.shape[0], x.shape[0])
        # lambdified_h_i_dxdx = sp.lambdify(decision_var, h_i_dxdx, "numpy")
        # hessian = lambdified_h_i_dxdx(*x)

        hessian = np.zeros((x.shape[0], x.shape[0]))
        return np.array(hessian)

    def residual(self, sw_problem, A, b, x, u, v, tau, f_grad):

        h_grad = self.h_x_grad_value(sw_problem, x, 0)
        if u.shape[0] > 1:
            for idx in range(1, u.shape[0]):
                h_grad = np.hstack((h_grad, self.h_x_grad_value(sw_problem, x, idx)))
        U = np.diag(u.reshape(u.shape[0]))

        # make r1
        r1 = f_grad + np.matmul(h_grad, u) + np.matmul(A.T, v)

        # make r2
        r2 = np.matmul(U, self.h_x_value(sw_problem, x)) + tau * np.ones((u.shape[0], 1))

        # make r3
        r3 = np.matmul(A, x) - b

        r = np.vstack((r1, r2, r3))

        return r

    def residual_jacobian(self, sw_problem, A, b, x, u, v, B):

        if sw_problem['stage_idx'] != 2:
            f_hess = B
        else:
            f_hess = self.f_hess_value(sw_problem, x)

        # f_hess = self.f_hessian(sw_problem, x)

        h_grad = self.h_x_grad_value(sw_problem, x, 0)
        if u.shape[0] > 1:
            for idx in range(1, u.shape[0]):
                h_grad = np.hstack((h_grad, self.h_x_grad_value(sw_problem, x, idx)))
        U = np.diag(u.reshape(u.shape[0]))
        H = np.diag(self.h_x_value(sw_problem, x).reshape(u.shape[0]))

        # make rj1
        u_hi_hess_sum = 0
        for idx in range(u.shape[0]):
            u_hi_hess_sum += u[idx, 0] * self.h_x_hess_value(sw_problem, x, idx)

        rj11 = f_hess + u_hi_hess_sum
        rj12 = h_grad
        rj13 = A.T

        rj1 = np.hstack((rj11, rj12, rj13))

        # make rj2
        rj21 = np.matmul(U, h_grad.T)
        rj22 = H
        rj23 = np.zeros((u.shape[0], v.shape[0]))

        rj2 = np.hstack((rj21, rj22, rj23))

        # make rj3
        rj31 = A
        rj32 = np.zeros((v.shape[0], u.shape[0]))
        rj33 = np.zeros((v.shape[0], v.shape[0]))

        rj3 = np.hstack((rj31, rj32, rj33))

        rj = np.vstack((rj1, rj2, rj3))

        return rj

    def NewtonStep(self, r, rj, x, u, v):

        r.reshape(x.shape[0] + u.shape[0] + v.shape[0])
        try:
            linsys_sol = np.linalg.solve(rj, -r)
        except:
            linsys_sol = np.linalg.lstsq(rj, -r, rcond=None)[0]

        # linsys_sol.reshape((x.shape[0] + u.shape[0] + v.shape[0], 1))

        del_x = linsys_sol[0:x.shape[0], :]
        del_u = linsys_sol[x.shape[0]:x.shape[0] + u.shape[0], :]
        del_v = linsys_sol[x.shape[0] + u.shape[0]:x.shape[0] + u.shape[0] + v.shape[0], :]

        return del_x, del_u, del_v

    def Is_r_reduced(self, sw_problem, A, b, x_update, u_update, v_update, rhs_r, tau, theta, f_grad_update):

        alpha = self.alpha
        lhs = np.linalg.norm(self.residual(sw_problem, A, b, x_update, u_update, v_update, tau, f_grad_update))
        rhs = (1 - alpha * theta) * rhs_r

        if lhs > rhs:
            return True

        else:
            return False

    def Is_h_negative(self, sw_problem, x_update):
        # print(self.h_x_value(sw_problem, x_update, initial))
        # print((self.h_x_value(sw_problem, x_update, initial) < 0).all())
        if (self.h_x_value(sw_problem, x_update) < 0).all():
            return False
        else:
            return True

    def Backtracking(self, sw_problem, A, b, x, u, v, del_x, del_u, del_v, tau, r):

        beta = self.beta

        # dual feasibility
        ratio = []
        for i in range(u.shape[0]):
            if del_u[i, 0] < 0:
                ratio.append(-u[i, 0] / del_u[i, 0])
        if len(ratio) == 0:
            theta_max = 1.0
        else:
            theta_max = min(1.0, min(ratio))

        # primal feasibility
        theta = 0.99 * theta_max
        x_update = x + theta * del_x
        while self.Is_h_negative(sw_problem, x_update):
            theta = beta * theta
            x_update = x + theta * del_x
        # reduce norm(r(x, u, v))
        u_update = u + theta * del_u
        v_update = v + theta * del_v

        rhs_r = np.linalg.norm(r)
        f_grad_update = self.f_grad_value(sw_problem, x_update)
        while self.Is_r_reduced(sw_problem, A, b, x_update, u_update, v_update, rhs_r, tau, theta, f_grad_update):
            theta = beta * theta
            x_update = x + theta * del_x
            u_update = u + theta * del_u
            v_update = v + theta * del_v
            f_grad_update = self.f_grad_value(sw_problem, x_update)

        return x_update, u_update, v_update, f_grad_update

    def PDA_StoppingCriteria(self, sw_problem, x, u, v, r, repeat):
        if sw_problem['stage_idx'] == 0:
            epsilon = self.epsilon0
        elif sw_problem['stage_idx'] == 2:
            epsilon = 0.0001
        else:
            epsilon = self.epsilon

        surrogate = (-1) * np.matmul(self.h_x_value(sw_problem, x).T, u)[0, 0]

        prim_residual = r[x.shape[0] + u.shape[0]:x.shape[0] + u.shape[0] + v.shape[0], 0]
        dual_residual = r[0:x.shape[0], 0]
        sum_residual = np.sqrt(np.linalg.norm(prim_residual) ** 2 + np.linalg.norm(dual_residual) ** 2)

        # print("surrogate: ",  surrogate)
        # print("sum_residual: ", sum_residual)

        if repeat <= self.repeat or sw_problem['stage_idx'] == 2:
            if surrogate > epsilon or sum_residual > epsilon:
                return True

            else:
                return False
        else:
            print("maximum repeat, quit PDA")
            return False

    def InitialSolution(self, sw_problem_ff, realization, stage, prev_sol):

        decision_var = sw_problem_ff['decision_var']
        eq_con = sw_problem_ff['eq_con']
        h_x = sw_problem_ff['h_x']


        # var_dim = decision_var.shape[0]
        # x00 = np.zeros((var_dim, 1))
        # value = max([self.h_x_value(sw_problem_ff, x00[0:var_dim, :])[i, 0] for i in range(len(h_x))]) + 0.1
        # x00[var_dim-1, 0] = value
        # u00 = np.ones((len(h_x), 1))
        # v00 = np.ones((len(eq_con), 1))
        #
        # x0, u0, v0 = self.PDA(sw_problem_ff, x00, u00, v00)
        # x0 = x0[0:var_dim-1, :]

        # Production Optimization

        x0 = np.array([[0.01], [0.01], [0.01], [0.01], [0.01], [0.01], [0.02], [0.02], [0.02]])
        u0 = 10 * np.ones((len(h_x), 1))
        # v0 = np.array([[6.0], [12.0], [20.0]])
        v0 = 10 * np.ones((len(eq_con), 1))

        # # Energy Planning
        # x0 = np.array([[40.0], [29.0], [11.0], [11.0]])
        # u0 = np.ones((len(h_x), 1))
        # v0 = np.ones((len(eq_con), 1))

        # Merton Portfolio Optimization
        # best?

        # x0 = 0.3 * np.ones((len(decision_var)-1, 1))
        # u0 = np.ones((len(h_x), 1))
        # v0 = np.ones((len(eq_con), 1))

        # slight = np.array([[0.0], [0.0], [0.1], [0.1]])

        return x0, u0, v0

    def PDA(self, sw_problem, x0, u0, v0):

        A, b = self.eq_con_to_matrix(sw_problem)
        B = np.eye(x0.shape[0], dtype=np.float32)

        x, u, v = x0, u0, v0

        sigma = 0.1
        tau = -sigma * (np.matmul(self.h_x_value(sw_problem, x).T, u)[0, 0] / u.shape[0])

        f_grad = self.f_grad_value(sw_problem, x)
        r = self.residual(sw_problem, A, b, x, u, v, tau, f_grad)
        rj = self.residual_jacobian(sw_problem, A, b, x, u, v, B)
        repeat = 1
        while self.PDA_StoppingCriteria(sw_problem, x, u, v, r, repeat):
            del_x, del_u, del_v = self.NewtonStep(r, rj, x, u, v)
            x_update, u_update, v_update, f_grad_update = self.Backtracking(sw_problem, A, b, x, u, v, del_x, del_u, del_v, tau, r)

            if sw_problem['stage_idx'] != 2:
                B = self.Update_B_matrix(sw_problem, B, x_update, x, f_grad_update, f_grad)

            f_grad = f_grad_update
            x, u, v = x_update, u_update, v_update
            tau = -sigma * (np.matmul(self.h_x_value(sw_problem, x).T, u)[0, 0] / u.shape[0])
            r = self.residual(sw_problem, A, b, x, u, v, tau, f_grad)
            rj = self.residual_jacobian(sw_problem, A, b, x, u, v, B)
            repeat += 1
        # return np.round(x, self.pda_round), np.round(u, self.pda_round), np.round(v, self.pda_round)
        return x, u, v
        # return x+0.0001, u, v

    def Update_B_matrix(self, sw_problem, B, x_update, x, f_grad_update, f_grad):
        s = x_update - x
        y = f_grad_update - f_grad

        t1 = np.matmul(np.matmul(B, s), np.matmul(s.T, B))
        t2 = np.matmul(s.T, np.matmul(B, s))
        t3 = np.matmul(y, y.T)
        t4 = np.matmul(y.T, s)
        if t2[0][0] < self.quasi or t4[0][0] < self.quasi:
            B_update = self.f_hess_value(sw_problem, x_update)
        else:
            B_update = B - t1/t2 + t3/t4

        return B_update

    def check_stopping_criteria(self):

        par_diff = self.par_change[self.iteration_count]
        print("Norm of parameters changed by ", par_diff)
        self.iteration_count += 1
        if par_diff < 0.001:
            if self.iteration_count < self.min_iter:
                return False
            else:
                return True
        else:
            if self.iteration_count >= self.max_iter:
                return True
            else:
                return False

    def solve(self):

        pbar = tqdm_notebook(total=self.iteration_count)

        self.initialize_ICNN()
        self.iteration_count = 0
        self.forward_sol = []
        stop = False
        pbar.set_description('DVFN Itertaion: ')
        while not stop:
            print("Iteration {}".format(self.iteration_count))
            start = time.time()
            self.DVFN_forward()
            # self.DVFN_backward()
            self.caculate_parameter_change()
            print("iteration time(s): ", time.time() - start)
            self.iter_time.append(time.time() - start)
            stop = self.check_stopping_criteria()
            pbar.update(1)
            # print(self.forward_sol)

        pbar.close()
        ttime = sum(self.iter_time)

        print('Algorithm Terminated. Time Elapsed: ', int(np.floor(ttime)))

    def po_simulate_solution(self, sol):
        print("Calculate Optimal Value")
        c1 = np.array([[1, 2, 5, 6, 12, 20, 3, 7, 10]])
        c2 = np.array([[0, 0, 0, 6, 12, 20, 0, 0, 0]])
        random_demand = [(5.0, 3.0, 1.0), (6.0, 2.0, 1.0), (1.0, 2.0, 2.0)]

        total_obj = []
        for path in tqdm(range(50)):
            prev_sol = sol[self.VF_var_idx[0], :]
            stagewise_obj = []
            value = np.dot(c1, sol)[0, 0]
            stagewise_obj.append(value)
            for stage in range(1, self.n_stages):
                stage_idx = self.stage_to_prob[stage]
                realization = random_demand[np.random.randint(3)]
                x, u, v, vf_obj, vf_grad = self.solve_stagewise_decomposed_problem(stage, realization, prev_sol)
                if stage == self.n_stages - 1:
                    value = np.dot(c2, x)[0, 0]
                    stagewise_obj.append(value)
                else:
                    value = np.dot(c1, x)[0, 0]
                    stagewise_obj.append(value)
                prev_sol = x[self.VF_var_idx[stage_idx], :]
            total_obj.append(sum(stagewise_obj))

        obj = np.mean(total_obj)
        print("Objective = ", obj)

        return obj
