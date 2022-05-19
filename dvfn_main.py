from module.dvfn_engine import DVFN

# import module.visualizer as vs
import numpy as np

from problems.production_optimization import make_problem_definition

# obj_list = []
# forward_sol_list = []
# par_change_list = []
# time_list = []
for num in range(50):
    problem_definition = make_problem_definition(n_stages=7)
    solver = DVFN(problem_definition)
    solver.solve()
    # obj = solver.simulate_solution(sol=solver.forward_sol[solver.iteration_count - 1])
    # obj_list.append(obj)

    np.save("trained/po/new_3/forward_sol_{}".format(num), solver.forward_sol)
    np.save("trained/po/new_3/par_change_{}".format(num), solver.par_change)
    np.save("trained/po/new_3/time_{}".format(num), solver.iter_time)

    for stage in range(1, solver.n_stages):
        solver.ICNNs[stage].save("icnn/po/new_3/vf_stage{}_{}.h5".format(stage, num))

    import matplotlib.pyplot as plt

    plt.plot(solver.iter_time)
    plt.ylim(0, 25)
    plt.savefig("figs/po/new_3/time_plot_{}".format(num))
    plt.show()

    x = np.arange(0, solver.iteration_count)
    y1 = []
    y2 = []
    y3 = []
    for i in range(solver.iteration_count):
        y1.append(solver.forward_sol[i][0, 0])
        y2.append(solver.forward_sol[i][1, 0])
        y3.append(solver.forward_sol[i][2, 0])
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.plot(x, y3)
    plt.legend(['production1', 'production2', 'production3'])
    plt.savefig("figs/po/new_3/first_sol_plot_{}".format(num))
    plt.show()

    # import matplotlib.pyplot as plt
    #
    # x = np.arange(0, solver.iteration_count)
    # y1 = []
    # y2 = []
    # for i in range(solver.iteration_count):
    #     y1.append(solver.forward_sol[i][2, 0])
    #     y2.append(solver.forward_sol[i][3, 0])
    # plt.plot(x, y1)
    # plt.plot(x, y2)
    # plt.legend(['Hydro', 'Thermal'])
    # plt.savefig("figs/ep/first_sol_plot_{}".format(num))
    # plt.show()

# np.save("result/obj_list1", obj_list)
#
# np.save("trained/po/forward_sol_list1", forward_sol_list)
# np.save("trained/po/par_change_list1", par_change_list)
# np.save("trained/po/time_list1", time_list)

# import matplotlib.pyplot as plt
# x  = np.arange(0, solver.iteration_count)
# y1 = []
# y2 = []
# y3 = []
# for i in range(solver.iteration_count):
#     y1.append(solver.forward_sol[i][0, 0])
#     y2.append(solver.forward_sol[i][1, 0])
#     y3.append(solver.forward_sol[i][2, 0])
# plt.plot(x, y1)
# plt.plot(x, y2)
# plt.plot(x, y3)
# plt.legend(['production1', 'production2', 'production3'])
# plt.show()

# import matplotlib.pyplot as plt
# x = np.arange(0, solver.iteration_count)
# y1 = []
# y2 = []
# y3 = []
# for i in range(solver.iteration_count):
#     y1.append(solver.forward_sol[i][0, 0])
#     y2.append(solver.forward_sol[i][1, 0])
#     y3.append(solver.forward_sol[i][2, 0])
# plt.plot(x, y1)
# plt.plot(x, y2)
# plt.plot(x, y3)
# plt.legend(['Bond', 'Stock', 'Consume'])
# plt.show()

# import matplotlib.pyplot as plt
# x  = np.arange(0, solver.iteration_count)
# y1 = []
# y2 = []
# for i in range(solver.iteration_count):
#     y1.append(solver.forward_sol[i][2, 0])
#     y2.append(solver.forward_sol[i][3, 0])
# plt.plot(x, y1)
# plt.plot(x, y2)
# plt.legend(['Hydro', 'Thermal'])
# plt.show()


# range_01 = np.arange(0, 1, 0.01)

# stock = []
# bond = []
# for i in range(100):
#     for j in range(100):
#         bond.append(range_01[i])
#         stock.append(range_01[j])
#
# z = []
# for i in range(10000):
#     value = solver.ICNNs[1](np.array([[bond[i], stock[i]]])).numpy()[0, 0]
#     z.append(value)
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.scatter(bond, stock, z, s=1)
# # plt.savefig("test")
# plt.show()

# range_01 = np.arange(10, 40, 0.01)
#
# final = []
# for i in range(range_01.shape[0]):
#     final.append(range_01[i])
#
# z = []
# for i in range(range_01.shape[0]):
#     value = solver.ICNNs[1](np.array([[final[i]]])).numpy()[0, 0]
#     z.append(value)
#
# plt.scatter(final, z, s=1)
# # plt.savefig("test")
# plt.show()

#
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
#
# folder = "back3"
# idx_list = [1, 4, 9, 10, 11, 12, 13, 16, 25]
# for num in range(2, 11):
#     idx = idx_list[num-2]
#
#     a = np.load("trained/po/{}/forward_sol_list1.npy".format(folder))
#     b = np.load("trained/po/{}/time_list1.npy".format(folder))
#     c = np.load("trained/po/{}/par_change_list1.npy".format(folder))
#     a = a[idx]
#     b = b[idx]
#     c = c[idx]
#
#     np.save("D:/KAIST/MS_Ph.D/Research/Deep Value Function Networks/neurips/po/{}/forward_sol".format(num), a)
#     np.save("D:/KAIST/MS_Ph.D/Research/Deep Value Function Networks/neurips/po/{}/time_2".format(num), b)
#     np.save("D:/KAIST/MS_Ph.D/Research/Deep Value Function Networks/neurips/po/{}/par_change".format(num), c)

    # vf = []
    # for stage in range(1, 7):
    #     vf.append(keras.models.load_model("icnn/po/{}/vf_stage{}_{}.h5".format(folder, stage, idx)))
    #     vf[-1].save_weights("D:/KAIST/MS_Ph.D/Research/Deep Value Function Networks/neurips/po/{}/vf_stage{}_weights.h5".format(num, stage))
    #     vf[-1].save("D:/KAIST/MS_Ph.D/Research/Deep Value Function Networks/neurips/po/{}/vf_stage{}.h5".format(num, stage))

# np.load("D:/KAIST/MS_Ph.D/Research/Deep Value Function Networks/neurips/po/1/time.npy")