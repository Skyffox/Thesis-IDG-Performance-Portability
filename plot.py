"""Plot all the data"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 22})

import matplotlib.pyplot as plt
import numpy as np

from os import listdir
from os.path import isfile, join


path = "/home/julius/Downloads/idg-fpga-master/thesis/"

data_size = "small_param"
# data_size = "big_param"

path_buffer = path + "buffer/output/" + data_size + "/"
path_ref = path + "reference/output/" + data_size + "/"
path_implicit = path + "implicit/output/" + data_size + "/"

# NR_BASELINES * NR_TIMESTEPS * NR_CHANNELS or ((NR_STATIONS * (NR_STATIONS - 1)) / 2) * (NR_TIMESTEPS_SUBGRID * NR_TIMESLOTS) * NR_CHANNELS
# small params: (10 * 9) / 2) * (128 * 2) * 16 = 184320
# big params:   (48 * 47) / 2) * (128 * 4) * 16 = 9240576
visibilities = 184320
# visibilities = 9240576

# the width of the bars of the plots
width = 0.1

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

# NOTE: Order of input data!
# The input files are in the following order:
# Reference:    object creation | object initialisation | kernel duration (empty) | kernel duration
# Implicit:     object creation | object initialisation | kernel duration (empty) | kernel duration
# Buffer:       object creation | object initialisation | buffer creation/initialisation | kernel duration (empty) | kernel duration

# NOTE: The output files are in the following order:
# Reference:    object creation | object initialisation | kernel (empty) | kernel duration | kernel-empty
# Implicit:     object creation | object initialisation | kernel (empty) | kernel duration | kernel-empty
# Buffer:       object creation | object initialisation | buffer init | kernel (empty) | kernel duration | init+buffer | kernel-empty

def read_data(path, type=None):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    data = []
    tmp = []

    for f in files:
        file_data = []
        it = 1 if type == "ref" else 0

        for line in open(path + f):
            line = list(line.strip().split(" "))
            line = [i for i in line if i != '']
            if line == []: continue
            if it > 0: file_data.append(float(line[-1]) / 1000000)
            it += 1

        data.append(file_data)

    # NOTE: to quickly check irregularities
    # print(np.transpose(data))

    if type == "buf":
        kernel_minus_empty = [x[4] - x[3] for x in data]
        init_plus_init_buf = [x[1] + x[2] for x in data]
        tmp = [[np.mean(init_plus_init_buf), np.std(init_plus_init_buf)]]
    else:
        kernel_minus_empty = [x[3] - x[2] for x in data]

    tmp2 = [[np.mean(kernel_minus_empty), np.std(kernel_minus_empty)]]
    data = np.transpose(data)

    return [[np.mean(x), np.std(x)] for x in data] + tmp + tmp2

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

def plot_template(ref_i5, ref_gold, ref_plat, ref_gen9, ref_gen9cpu, ref_iris, ref_iriscpu,
                  impl_i5, impl_gold, impl_plat, impl_gen9, impl_gen9cpu, impl_iris, impl_iriscpu,
                  buf_i5, buf_gold, buf_plat, buf_gen9, buf_gen9cpu, buf_iris, buf_iriscpu, type=None):

    labels = ['Reference', 'Implicit', 'Buffer']
    x = np.arange(len(labels)) # the label locations

    # The first index are the means the second are the standard deviations
    i5_data   = [[ref_i5[0], impl_i5[0], buf_i5[0]], [ref_i5[1], impl_i5[1], buf_i5[1]]]
    gold_data = [[ref_gold[0], impl_gold[0], buf_gold[0]], [ref_gold[1], impl_gold[1], buf_gold[1]]]
    plat_data = [[ref_plat[0], impl_plat[0], buf_plat[0]], [ref_plat[1], impl_plat[1], buf_plat[1]]]
    gen9cpu_data = [[ref_gen9cpu[0], impl_gen9cpu[0], buf_gen9cpu[0]], [ref_gen9cpu[1], impl_gen9cpu[1], buf_gen9cpu[1]]]
    iriscpu_data = [[ref_gen9cpu[0], impl_gen9cpu[0], buf_gen9cpu[0]], [ref_gen9cpu[1], impl_gen9cpu[1], buf_gen9cpu[1]]]
    gen9_data = [[ref_gen9[0], impl_gen9[0], buf_gen9[0]], [ref_gen9[1], impl_gen9[1], buf_gen9[1]]]
    iris_data = [[ref_iris[0], impl_iris[0], buf_iris[0]], [ref_iris[1], impl_iris[1], buf_iris[1]]]

    fig, ax = plt.subplots()
    fig.set_figwidth(15)
    fig.set_figheight(8)

    if plot == "kernel_vis" or plot == "empty_kernel_vis":
        ax.bar(x - (width * 3), np.divide(visibilities, i5_data[0]), width, label='Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz', edgecolor='k')
        ax.bar(x - (width * 2), np.divide(visibilities, gold_data[0]), width, label='Intel(R) Xeon(R) Gold 6128 CPU @ 3.40GHz', edgecolor='k')
        ax.bar(x - width, np.divide(visibilities, plat_data[0]), width, label='Intel(R) Xeon(R) Platinum 8153 CPU @ 2.00GHz', edgecolor='k')
        ax.bar(x, np.divide(visibilities, gen9cpu_data[0]), width, label='Intel(R) Xeon(R) E-2176G CPU @ 3.70GHz', edgecolor='k')
        ax.bar(x + width, np.divide(visibilities, iriscpu_data[0]), width, label='Intel(R) Core(TM) i9-10920X CPU @ 3.50GHz', edgecolor='k')
        ax.bar(x + (width * 2), np.divide(visibilities, gen9_data[0]), width, label='Intel(R) UHD Graphics P630', edgecolor='k')
        ax.bar(x + (width * 3), np.divide(visibilities, iris_data[0]), width, label='Intel(R) Iris(R) Xe MAX Graphics', edgecolor='k')

        ax.set_ylabel('Visibilities per second')
    else:
        ax.bar(x - (width * 3), i5_data[0], width, yerr=i5_data[1], label='Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz', edgecolor='k')
        ax.bar(x - (width * 2), gold_data[0], width, yerr=gold_data[1], label='Intel(R) Xeon(R) Gold 6128 CPU @ 3.40GHz', edgecolor='k')
        ax.bar(x - width, plat_data[0], width, yerr=plat_data[1], label='Intel(R) Xeon(R) Platinum 8153 CPU @ 2.00GHz', edgecolor='k')
        ax.bar(x, gen9cpu_data[0], width, yerr=gen9cpu_data[1], label='Intel(R) Xeon(R) E-2176G CPU @ 3.70GHz', edgecolor='k')
        ax.bar(x + width, iriscpu_data[0], width, yerr=iriscpu_data[1], label='Intel(R) Core(TM) i9-10920X CPU @ 3.50GHz', edgecolor='k')
        ax.bar(x + (width * 2), gen9_data[0], width, yerr=gen9_data[1], label='Intel(R) UHD Graphics P630', edgecolor='k')
        ax.bar(x + (width * 3), iris_data[0], width, yerr=iris_data[1], label='Intel(R) Iris(R) Xe MAX Graphics', edgecolor='k')

        ax.set_ylabel('Execution time (in milliseconds)')

    ax.set_xlabel('Different implementations and hardware')

    # if type == "creation":
    #     ax.set_title('Execution time of the object creation averaged over 20 runs expressed in milliseconds')
    # if type == "init":
    #     ax.set_title('Execution time of the object initialisation averaged over 20 runs expressed in milliseconds')
    # if type == "empty":
    #     ax.set_title('Execution time of an empty kernel averaged over 20 runs expressed in milliseconds')
    # if type == "kernel":
    #     ax.set_title('Execution time of a kernel iteration averaged over 20 runs expressed in milliseconds')
    # if type == "empty_kernel":
    #     ax.set_title('Execution time of a kernel iteration minus the empty kernel duration averaged over 20 runs expressed in milliseconds')
    # if type == "kernel_vis":
    #     ax.set_title('Execution time of a kernel iteration averaged over 20 runs expressed as visibilities per second (higher is better)')
    # if type == "empty_kernel_vis":
    #     ax.set_title('Execution time of a kernel iteration minus the empty kernel duration averaged over 20 runs expressed as visibilities per second (higher is better)')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.grid(True)
    fig.tight_layout()

    if type == "creation":
        plt.savefig(path + "plots/" + data_size + "/" + data_size + "_create.png")
    if type == "init":
        plt.savefig(path + "plots/" + data_size + "/" + data_size + "_init.png")
    if type == "empty":
        plt.savefig(path + "plots/" + data_size + "/" + data_size + "_empty_kernel.png")
    if type == "kernel":
        plt.savefig(path + "plots/" + data_size + "/" + data_size + "_kernel_duration.png")
    if type == "empty_kernel":
        plt.savefig(path + "plots/" + data_size + "/" + data_size + "_kernel_minus_empty.png")
    if type == "kernel_vis":
        plt.savefig(path + "plots/" + data_size + "/" + data_size + "_kernel_duration_visibilities.png")
    if type == "empty_kernel_vis":
        plt.savefig(path + "plots/" + data_size + "/" + data_size + "_kernel_minus_empty_visibilities.png")

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

# Raw data, Plot performance data using 1 cluster per version, and 1 colored-bar per machine
i5_ref      = read_data(path_ref + "i5/", type="ref")
gold_ref    = read_data(path_ref + "gold/", type="ref")
plat_ref    = read_data(path_ref + "platinum/", type="ref")
gen9_ref    = read_data(path_ref + "gen9/", type="ref")
gen9cpu_ref = read_data(path_ref + "gen9cpu/", type="ref")
iris_ref    = read_data(path_ref + "iris/", type="ref")
iriscpu_ref = read_data(path_ref + "iriscpu/", type="ref")

i5_impl      = read_data(path_implicit + "i5/")
gold_impl    = read_data(path_implicit + "gold/")
plat_impl    = read_data(path_implicit + "platinum/")
gen9_impl    = read_data(path_implicit + "gen9/")
gen9cpu_impl = read_data(path_implicit + "gen9cpu/")
iris_impl    = read_data(path_implicit + "iris/")
iriscpu_impl = read_data(path_implicit + "iriscpu/")

i5_buf      = read_data(path_buffer + "i5/", type="buf")
gold_buf    = read_data(path_buffer + "gold/", type="buf")
plat_buf    = read_data(path_buffer + "platinum/", type="buf")
gen9_buf    = read_data(path_buffer + "gen9/", type="buf")
gen9cpu_buf = read_data(path_buffer + "gen9cpu/", type="buf")
iris_buf    = read_data(path_buffer + "iris/", type="buf")
iriscpu_buf = read_data(path_buffer + "iriscpu/", type="buf")

# Plot the object creation
plot_template(i5_ref[0], gold_ref[0], plat_ref[0], gen9cpu_ref[0], iriscpu_ref[0], gen9_ref[0], iris_ref[0], \
              i5_impl[0], gold_impl[0], plat_impl[0], gen9cpu_impl[0], iriscpu_impl[0], gen9_impl[0], iris_impl[0], \
              i5_buf[0], gold_buf[0], plat_buf[0], gen9cpu_buf[0], iriscpu_buf[0], gen9_buf[0], iris_buf[0], type="creation")

# Plot the object initialisation
plot_template(i5_ref[1], gold_ref[1], plat_ref[1], gen9cpu_ref[1], iriscpu_ref[1], gen9_ref[1], iris_ref[1], \
              i5_impl[1], gold_impl[1], plat_impl[1], gen9cpu_impl[1], iriscpu_impl[1], gen9_impl[1], iris_impl[1], \
              i5_buf[5], gold_buf[5], plat_buf[5], gen9cpu_buf[5], iriscpu_buf[5], gen9_buf[5], iris_buf[5], type="init")

# Plot the empty kernels
plot_template(i5_ref[2], gold_ref[2], plat_ref[2], gen9cpu_ref[2], iriscpu_ref[2], gen9_ref[2], iris_ref[2], \
              i5_impl[2], gold_impl[2], plat_impl[2], gen9cpu_impl[2], iriscpu_impl[2], gen9_impl[2], iris_impl[2], \
              i5_buf[3], gold_buf[3], plat_buf[3], gen9cpu_buf[3], iriscpu_buf[3], gen9_buf[3], iris_buf[3], type="empty")

# Plot the kernel duration
plot_template(i5_ref[3], gold_ref[3], plat_ref[3], gen9cpu_ref[3], iriscpu_ref[3], gen9_ref[3], iris_ref[3], \
              i5_impl[3], gold_impl[3], plat_impl[3], gen9cpu_impl[3], iriscpu_impl[3], gen9_impl[3], iris_impl[3], \
              i5_buf[4], gold_buf[4], plat_buf[4], gen9cpu_buf[4], iriscpu_buf[4], gen9_buf[4], iris_buf[4], type="kernel")

# Plot kernel duration minus the empty kernel
plot_template(i5_ref[4], gold_ref[4], plat_ref[4], gen9cpu_ref[4], iriscpu_ref[4], gen9_ref[4], iris_ref[4], \
              i5_impl[4], gold_impl[4], plat_impl[4], gen9cpu_impl[4], iriscpu_impl[4], gen9_impl[4], iris_impl[4], \
              i5_buf[6], gold_buf[6], plat_buf[6], gen9cpu_buf[6], iriscpu_buf[6], gen9_buf[6], iris_buf[6], type="empty_kernel")

# Plot the kernel duration expressed as visibilities per second
plot_template(i5_ref[3], gold_ref[3], plat_ref[3], gen9cpu_ref[3], iriscpu_ref[3], gen9_ref[3], iris_ref[3], \
              i5_impl[3], gold_impl[3], plat_impl[3], gen9cpu_impl[3], iriscpu_impl[3], gen9_impl[3], iris_impl[3], \
              i5_buf[4], gold_buf[4], plat_buf[4], gen9cpu_buf[4], iriscpu_buf[4], gen9_buf[4], iris_buf[4], type="kernel_vis")

# Plot the kernel duration minus the empty kernel expressed as visibilities per second
plot_template(i5_ref[4], gold_ref[4], plat_ref[4], gen9cpu_ref[4], iriscpu_ref[4], gen9_ref[4], iris_ref[4], \
              i5_impl[4], gold_impl[4], plat_impl[4], gen9cpu_impl[4], iriscpu_impl[4], gen9_impl[4], iris_impl[4], \
              i5_buf[6], gold_buf[6], plat_buf[6], gen9cpu_buf[6], iriscpu_buf[6], gen9_buf[6], iris_buf[6], type="empty_kernel_vis")
