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

# data_size = "small_param"
data_size = "big_param"

path_buffer = path + "buffer/output/" + data_size + "/"
path_ref = path + "reference/output/" + data_size + "/"
path_implicit = path + "implicit/output/" + data_size + "/"

# NR_BASELINES * NR_TIMESTEPS * NR_CHANNELS or ((NR_STATIONS * (NR_STATIONS - 1)) / 2) * (NR_TIMESTEPS_SUBGRID * NR_TIMESLOTS) * NR_CHANNELS
# small params: (10 * 9) / 2) * (128 * 2) * 16 = 184320
# big params:   (48 * 47) / 2) * (128 * 4) * 16 = 9240576
# visibilities = 184320
visibilities = 9240576

# the width of the bars of the plots
width = 0.12

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

# NOTE: Order of input data!
# The input files are in the following order:
# Reference:    object creation | object initialisation | kernel duration (empty) | kernel duration - empty
# Implicit:     object creation | object initialisation | kernel duration (empty) | kernel duration - empty
# Buffer:       object creation | object initialisation | buffer creation/initialisation | kernel duration (empty) | kernel duration - empty

# NOTE: The output files are in the following order:
# Reference:    object creation | object initialisation | kernel (empty) | kernel-empty | kernel duration
# Implicit:     object creation | object initialisation | kernel (empty) | kernel-empty | kernel duration
# Buffer:       object creation | object initialisation | buffer init | kernel (empty) | kernel-empty | init+buffer | kernel duration

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
        kernel_duration = [x[4] + x[3] for x in data]
        init_plus_init_buf = [x[1] + x[2] for x in data]
        tmp = [[np.mean(init_plus_init_buf), np.std(init_plus_init_buf)]]
    else:
        kernel_duration = [x[3] + x[2] for x in data]

    tmp2 = [[np.mean(kernel_duration), np.std(kernel_duration)]]
    data = np.transpose(data)

    return [[np.mean(x), np.std(x)] for x in data] + tmp + tmp2


def set_to_zero(x_values, y_values):
    for i, x in enumerate(x_values):
        if x == 0:
            y_values[i] = 0

    return y_values

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

def plot_template(ref_i5, ref_gold, ref_plat, ref_gen9cpu, ref_iriscpu, ref_gen9, ref_iris,
                  impl_i5, impl_gold, impl_plat, impl_gen9cpu, impl_iriscpu, impl_gen9, impl_iris,
                  buf_i5, buf_gold, buf_plat, buf_gen9cpu, buf_iriscpu, buf_gen9, buf_iris, type=None):

    labels = ['Reference', 'Implicit', 'Buffer']
    x = np.arange(len(labels)) # the label locations

    # The first index are the means the second are the standard deviations
    i5_data   = [[ref_i5[0], impl_i5[0], buf_i5[0]], [ref_i5[1], impl_i5[1], buf_i5[1]]]
    gold_data = [[ref_gold[0], impl_gold[0], buf_gold[0]], [ref_gold[1], impl_gold[1], buf_gold[1]]]
    plat_data = [[ref_plat[0], impl_plat[0], buf_plat[0]], [ref_plat[1], impl_plat[1], buf_plat[1]]]
    gen9cpu_data = [[ref_gen9cpu[0], impl_gen9cpu[0], buf_gen9cpu[0]], [ref_gen9cpu[1], impl_gen9cpu[1], buf_gen9cpu[1]]]
    iriscpu_data = [[ref_iriscpu[0], impl_iriscpu[0], buf_iriscpu[0]], [ref_iriscpu[1], impl_iriscpu[1], buf_iriscpu[1]]]
    gen9_data = [[ref_gen9[0], impl_gen9[0], buf_gen9[0]], [ref_gen9[1], impl_gen9[1], buf_gen9[1]]]
    iris_data = [[ref_iris[0], impl_iris[0], buf_iris[0]], [ref_iris[1], impl_iris[1], buf_iris[1]]]

    fig, ax = plt.subplots()
    fig.set_figwidth(15)
    fig.set_figheight(10)

    if type == "kernel_vis" or type == "empty_kernel_vis":
        bar1 = ax.bar(x - (width * 3), np.around(np.divide(visibilities, i5_data[0])), width, label='Core i5-6200U CPU @ 2.30GHz', edgecolor='k')
        bar2 = ax.bar(x - (width * 2), np.around(np.divide(visibilities, gold_data[0])), width, label='Xeon Gold 6128 CPU @ 3.40GHz', edgecolor='k')
        bar3 = ax.bar(x - width, np.around(np.divide(visibilities, plat_data[0])), width, label='Xeon Platinum 8153 CPU @ 2.00GHz', edgecolor='k')
        bar4 = ax.bar(x, np.around(np.divide(visibilities, gen9cpu_data[0])), width, label='Xeon E-2176G CPU @ 3.70GHz', edgecolor='k')
        bar5 = ax.bar(x + width, np.around(np.divide(visibilities, iriscpu_data[0])), width, label='Core i9-10920X CPU @ 3.50GHz', edgecolor='k')
        bar6 = ax.bar(x + (width * 2), np.around(np.divide(visibilities, gen9_data[0])), width, label='UHD Graphics P630', edgecolor='k')

        if data_size == "small_param":
            bar7 = ax.bar(x + (width * 3), np.around(np.divide(visibilities, iris_data[0])), width, label='Iris Xe MAX Graphics', edgecolor='k')

        ax.set_ylabel('Visibilities per second')
    else:
        bar1_values = np.around(i5_data[0])
        bar2_values = np.around(gold_data[0])
        bar3_values = np.around(plat_data[0])
        bar4_values = np.around(gen9cpu_data[0])
        bar5_values = np.around(iriscpu_data[0])
        bar6_values = np.around(gen9_data[0])

        set_to_zero(bar1_values, i5_data[1])
        set_to_zero(bar2_values, gold_data[1])
        set_to_zero(bar3_values, plat_data[1])
        set_to_zero(bar4_values, gen9cpu_data[1])
        set_to_zero(bar5_values, iriscpu_data[1])
        set_to_zero(bar6_values, gen9_data[1])

        bar1 = ax.bar(x - (width * 3), bar1_values, width, yerr=i5_data[1], label='Core i5-6200U CPU @ 2.30GHz', edgecolor='k')
        bar2 = ax.bar(x - (width * 2), bar2_values, width, yerr=gold_data[1], label='Xeon Gold 6128 CPU @ 3.40GHz', edgecolor='k')
        bar3 = ax.bar(x - width, bar3_values, width, yerr=plat_data[1], label='Xeon Platinum 8153 CPU @ 2.00GHz', edgecolor='k')
        bar4 = ax.bar(x, bar4_values, width, yerr=gen9cpu_data[1], label='Xeon E-2176G CPU @ 3.70GHz', edgecolor='k')
        bar5 = ax.bar(x + width, bar5_values, width, yerr=iriscpu_data[1], label='Core i9-10920X CPU @ 3.50GHz', edgecolor='k')
        bar6 = ax.bar(x + (width * 2), bar6_values, width, yerr=gen9_data[1], label='UHD Graphics P630', edgecolor='k')

        if data_size == "small_param":
            bar7_values = np.around(iris_data[0])

            set_to_zero(bar7_values, iris_data[1])

            bar7 = ax.bar(x + (width * 3), bar7_values, width, yerr=iris_data[1], label='Iris Xe MAX Graphics', edgecolor='k')

        ax.set_ylabel('Execution time (in milliseconds)')

    ax.set_xlabel('Different implementations and hardware')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=14, bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)

    ax.bar_label(bar1, fontsize=12)
    ax.bar_label(bar2, fontsize=12)
    ax.bar_label(bar3, fontsize=12)
    ax.bar_label(bar4, fontsize=12)
    ax.bar_label(bar5, fontsize=12)
    ax.bar_label(bar6, fontsize=12)

    if data_size == "small_param":
        ax.bar_label(bar7, fontsize=12)

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
iriscpu_ref = read_data(path_ref + "iriscpu/", type="ref")

i5_impl      = read_data(path_implicit + "i5/")
gold_impl    = read_data(path_implicit + "gold/")
plat_impl    = read_data(path_implicit + "platinum/")
gen9_impl    = read_data(path_implicit + "gen9/")
gen9cpu_impl = read_data(path_implicit + "gen9cpu/")
iriscpu_impl = read_data(path_implicit + "iriscpu/")

i5_buf      = read_data(path_buffer + "i5/", type="buf")
gold_buf    = read_data(path_buffer + "gold/", type="buf")
plat_buf    = read_data(path_buffer + "platinum/", type="buf")
gen9_buf    = read_data(path_buffer + "gen9/", type="buf")
gen9cpu_buf = read_data(path_buffer + "gen9cpu/", type="buf")
iriscpu_buf = read_data(path_buffer + "iriscpu/", type="buf")

if data_size == "small_param":
    iris_ref    = read_data(path_ref + "iris/", type="ref")
    iris_impl   = read_data(path_implicit + "iris/")
    iris_buf    = read_data(path_buffer + "iris/", type="buf")
else:
    # NOTE: since no iris data exists for the big_param
    iris_ref  = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    iris_impl = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    iris_buf  = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]

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
plot_template(i5_ref[4], gold_ref[4], plat_ref[4], gen9cpu_ref[4], iriscpu_ref[4], gen9_ref[4], iris_ref[4], \
              i5_impl[4], gold_impl[4], plat_impl[4], gen9cpu_impl[4], iriscpu_impl[4], gen9_impl[4], iris_impl[4], \
              i5_buf[6], gold_buf[6], plat_buf[6], gen9cpu_buf[6], iriscpu_buf[6], gen9_buf[6], iris_buf[6], type="kernel")

# Plot kernel duration minus the empty kernel
plot_template(i5_ref[3], gold_ref[3], plat_ref[3], gen9cpu_ref[3], iriscpu_ref[3], gen9_ref[3], iris_ref[3], \
              i5_impl[3], gold_impl[3], plat_impl[3], gen9cpu_impl[3], iriscpu_impl[3], gen9_impl[3], iris_impl[3], \
              i5_buf[4], gold_buf[4], plat_buf[4], gen9cpu_buf[4], iriscpu_buf[4], gen9_buf[4], iris_buf[4], type="empty_kernel")

# Plot the kernel duration expressed as visibilities per second
plot_template(i5_ref[4], gold_ref[4], plat_ref[4], gen9cpu_ref[4], iriscpu_ref[4], gen9_ref[4], iris_ref[4], \
              i5_impl[4], gold_impl[4], plat_impl[4], gen9cpu_impl[4], iriscpu_impl[4], gen9_impl[4], iris_impl[4], \
              i5_buf[6], gold_buf[6], plat_buf[6], gen9cpu_buf[6], iriscpu_buf[6], gen9_buf[6], iris_buf[6], type="kernel_vis")

# Plot the kernel duration minus the empty kernel expressed as visibilities per second
plot_template(i5_ref[3], gold_ref[3], plat_ref[3], gen9cpu_ref[3], iriscpu_ref[3], gen9_ref[3], iris_ref[3], \
              i5_impl[3], gold_impl[3], plat_impl[3], gen9cpu_impl[3], iriscpu_impl[3], gen9_impl[3], iris_impl[3], \
              i5_buf[4], gold_buf[4], plat_buf[4], gen9cpu_buf[4], iriscpu_buf[4], gen9_buf[4], iris_buf[4], type="empty_kernel_vis")
