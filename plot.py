"""Plot all the data"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from os import listdir
from os.path import isfile, join


path = "/home/julius/Downloads/idg-fpga-master/thesis/"

data_size = "small_param/"
# data_size = "big_param/"

path_buffer = path + "buffer/output/" + data_size
path_ref = path + "reference/output/" + data_size
path_explicit = path + "explicit/output/" + data_size
path_implicit = path + "implicit/output/" + data_size

# NR_BASELINES * NR_TIMESTEPS * NR_CHANNELS or ((NR_STATIONS * (NR_STATIONS - 1)) / 2) * (NR_TIMESTEPS_SUBGRID * NR_TIMESLOTS) * NR_CHANNELS
# small params: (10 * 9) / 2) * (128 * 2) * 16 = 184320
# big params:   (48 * 47) / 2) * (128 * 4) * 16 = 9240576
visibilities = 184320
# visibilities = 9240576

# the width of the bars of the plots
width = 0.1

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

# NOTE: order of data!
# First we have a list for the device information that has the CPU/GPU it is run on then the max work group size then
# max compute units on the device.
# The second and third list contain a list of data we obtained per kernel, this can differ. The order are:
# Reference / Implicit: object creation | object initialisation | kernel duration
# Buffer:               object creation | object initialisation | buffer creation/initialisation | kernel duration
# Explicit:             creation (host) | creation (device) | object initialisation | H2D | kernel duration | D2H

def read_data(path, type=None):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    data = []

    for f in files:
        file_data = []
        it = 1 if type == "ref" else 0

        for line in open(path + f):
            devcloud_test = list(line)
            line = list(line.strip().split(" "))

            if line == [""] or devcloud_test[0] == "#": continue
            if it > 0: file_data.append(float(line[-1]))
            it += 1

        data.append(file_data)

    data = np.transpose(data)
    return [[np.mean(x), np.std(x)] for x in data]

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

def plot_object_create(buf_i5, buf_gold, buf_plat, buf_gen9, buf_iris,
                       ref_i5, ref_gold, ref_plat, ref_gen9, ref_iris,
                       impl_i5, impl_gold, impl_plat, impl_gen9, impl_iris):

    labels = ['Buffer', 'Reference', 'Implicit']
    x = np.arange(len(labels)) # the label locations

    # The first index are the means the second are the standard deviations
    i5_data   = [[buf_i5[0], ref_i5[0], impl_i5[0]], [buf_i5[1], ref_i5[1], impl_i5[1]]]
    gold_data = [[buf_gold[0], ref_gold[0], impl_gold[0]], [buf_gold[1], ref_gold[1], impl_gold[1]]]
    plat_data = [[buf_plat[0], ref_plat[0], impl_plat[0]], [buf_plat[1], ref_plat[1], impl_plat[1]]]
    gen9_data = [[buf_gen9[0], ref_gen9[0], impl_gen9[0]], [buf_gen9[1], ref_gen9[1], impl_gen9[1]]]
    iris_data = [[buf_iris[0], ref_iris[0], impl_iris[0]], [buf_iris[1], ref_iris[1], impl_iris[1]]]

    fig, ax = plt.subplots()
    fig.set_figwidth(15)
    fig.set_figheight(8)

    ax.bar(x - (width * 2), i5_data[0], width, yerr=i5_data[1], label='i5', color='blue', edgecolor='k')
    ax.bar(x - width, gold_data[0], width, yerr=gold_data[1], label='gold', color='green', edgecolor='k')
    ax.bar(x, plat_data[0], width, yerr=plat_data[1], label='platinum', color='red', edgecolor='k')
    ax.bar(x + width, gen9_data[0], width, yerr=gen9_data[1], label='gen9', color='orange', edgecolor='k')
    ax.bar(x + (width * 2), iris_data[0], width, yerr=iris_data[1], label='iris', color='magenta', edgecolor='k')

    ax.set_ylabel('Execution time (in nanoseconds)')
    ax.set_xlabel('Different implementations and hardware')
    ax.set_title('Execution time of the object creation averaged over 20 runs')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.grid(True)

    fig.tight_layout()
    plt.savefig(path + "plots/" + data_size + "creation_time.png")

    # plt.yscale('log')
    # ax.set_title('Execution time of the object creation averaged over 20 runs in log scale')
    # plt.tight_layout()
    # plt.savefig(path + "plots/" + data_size + "creation_time_log.png")

# ---------------------------------------------------------------------------------------------------------------------

def plot_object_init(buf_i5, buf_gold, buf_plat, buf_gen9, buf_iris,
                     buf_i5_2, buf_gold_2, buf_plat_2, buf_gen9_2, buf_iris_2,
                     ref_i5, ref_gold, ref_plat, ref_gen9, ref_iris,
                     impl_i5, impl_gold, impl_plat, impl_gen9, impl_iris):

    labels = ['Buffer', 'Reference', 'Implicit']
    x = np.arange(len(labels)) # the label locations

    # The first index are the means the second are the standard deviations
    i5_data   = [[buf_i5[0], ref_i5[0], impl_i5[0]], [buf_i5[1], ref_i5[1], impl_i5[1]]]
    gold_data = [[buf_gold[0], ref_gold[0], impl_gold[0]], [buf_gold[1], ref_gold[1], impl_gold[1]]]
    plat_data = [[buf_plat[0], ref_plat[0], impl_plat[0]], [buf_plat[1], ref_plat[1], impl_plat[1]]]
    gen9_data = [[buf_gen9[0], ref_gen9[0], impl_gen9[0]], [buf_gen9[1], ref_gen9[1], impl_gen9[1]]]
    iris_data = [[buf_iris[0], ref_iris[0], impl_iris[0]], [buf_iris[1], ref_iris[1], impl_iris[1]]]

    # This is the stacked data for the buffer initialisation
    i5_data_stacked   = [[buf_i5_2[0], 0, 0], [buf_i5_2[1], 0, 0]]
    gold_data_stacked = [[buf_gold_2[0], 0, 0], [buf_gold_2[1], 0, 0]]
    plat_data_stacked = [[buf_plat_2[0], 0, 0], [buf_plat_2[1], 0, 0]]
    gen9_data_stacked = [[buf_gen9_2[0], 0, 0], [buf_gen9_2[1], 0, 0]]
    iris_data_stacked = [[buf_iris_2[0], 0, 0], [buf_iris_2[1], 0, 0]]

    fig, ax = plt.subplots()
    fig.set_figwidth(15)
    fig.set_figheight(8)

    ax.bar(x - (width * 2), i5_data[0], width, yerr=i5_data[1], label='i5', color='blue', edgecolor='k')
    ax.bar(x - width, gold_data[0], width, yerr=gold_data[1], label='gold', color='green', edgecolor='k')
    ax.bar(x, plat_data[0], width, yerr=plat_data[1], label='platinum', color='red', edgecolor='k')
    ax.bar(x + width, gen9_data[0], width, yerr=gen9_data[1], label='gen9', color='orange', edgecolor='k')
    ax.bar(x + (width * 2), iris_data[0], width, yerr=iris_data[1], label='iris', color='magenta', edgecolor='k')

    ax.bar(x - (width * 2), i5_data_stacked[0], width, yerr=i5_data_stacked[1], color='cyan', edgecolor='k', bottom=i5_data[0])
    ax.bar(x - width, gold_data_stacked[0], width, yerr=gold_data_stacked[1], color='lime', edgecolor='k', bottom=gold_data[0])
    ax.bar(x, plat_data_stacked[0], width, yerr=plat_data_stacked[1], color='coral', edgecolor='k', bottom=plat_data[0])
    ax.bar(x + width, gen9_data_stacked[0], width, yerr=gen9_data_stacked[1], color='gold', edgecolor='k', bottom=gen9_data[0])
    ax.bar(x + (width * 2), iris_data_stacked[0], width, yerr=iris_data_stacked[1], color='pink', edgecolor='k', bottom=iris_data[0])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Execution time (in nanoseconds)')
    ax.set_xlabel('Different implementations and hardware')
    ax.set_title('Execution time of the object initialisation averaged over 20 runs')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.grid(True)

    fig.tight_layout()
    plt.savefig(path + "plots/" + data_size + "init_time.png")

    # plt.yscale('log')
    # ax.set_title('Execution time of the object initialisation averaged over 20 runs in log scale')
    # plt.tight_layout()
    # plt.savefig(path + "plots/" + data_size + "init_time_log.png")

# ---------------------------------------------------------------------------------------------------------------------

# Return the visibilities per second, this is done times the amount of times we ran the kernel in total.
def misc(data):
    return 1000 * np.divide(visibilities, data)

def plot_kernel_duration(buf_i5, buf_gold, buf_plat, buf_gen9, buf_iris,
                         ref_i5, ref_gold, ref_plat, ref_gen9, ref_iris,
                         impl_i5, impl_gold, impl_plat, impl_gen9, impl_iris):

    labels = ['Buffer', 'Reference', 'Implicit']
    x = np.arange(len(labels)) # the label locations

    # The first index are the means the second are the standard deviations
    i5_data   = [[buf_i5[0], ref_i5[0], impl_i5[0]], [buf_i5[1], ref_i5[1], impl_i5[1]]]
    gold_data = [[buf_gold[0], ref_gold[0], impl_gold[0]], [buf_gold[1], ref_gold[1], impl_gold[1]]]
    plat_data = [[buf_plat[0], ref_plat[0], impl_plat[0]], [buf_plat[1], ref_plat[1], impl_plat[1]]]
    gen9_data = [[buf_gen9[0], ref_gen9[0], impl_gen9[0]], [buf_gen9[1], ref_gen9[1], impl_gen9[1]]]
    iris_data = [[buf_iris[0], ref_iris[0], impl_iris[0]], [buf_iris[1], ref_iris[1], impl_iris[1]]]

    fig, ax = plt.subplots()
    fig.set_figwidth(15)
    fig.set_figheight(8)

    ax.bar(x - (width * 2), misc(i5_data[0]), width, label='i5', color='blue', edgecolor='k')
    ax.bar(x - width, misc(gold_data[0]), width, label='gold', color='green', edgecolor='k')
    ax.bar(x, misc(plat_data[0]), width, label='platinum', color='red', edgecolor='k')
    ax.bar(x + width, misc(gen9_data[0]), width, label='gen9', color='orange', edgecolor='k')
    ax.bar(x + (width * 2), misc(iris_data[0]), width, label='iris', color='magenta', edgecolor='k')

    ax.set_ylabel('Visibilities per second')
    ax.set_xlabel('Different implementations and hardware')
    ax.set_title('A kernel iteration averaged over 20 runs expressed as visibilities per second (higher is better)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.grid(True)

    fig.tight_layout()
    plt.savefig(path + "plots/" + data_size + "kernel_duration.png")

    # plt.yscale('log')
    # ax.set_title('A kernel iteration averaged over 20 runs in log scale expressed as visibilities per second (higher is better)')
    # plt.tight_layout()
    # plt.savefig(path + "plots/" + data_size + "kernel_duration_log.png")

# ---------------------------------------------------------------------------------------------------------------------

def plot_total(buf_i5, buf_gold, buf_plat, buf_gen9, buf_iris,
               ref_i5, ref_gold, ref_plat, ref_gen9, ref_iris,
               impl_i5, impl_gold, impl_plat, impl_gen9, impl_iris):

    labels = ['Buffer', 'Reference', 'Implicit']
    x = np.arange(len(labels)) # the label locations

    # The first index are the means the second are the standard deviations
    i5_data   = [[buf_i5[0][0], ref_i5[0][0], impl_i5[0][0]], [buf_i5[0][1], ref_i5[0][1], impl_i5[0][1]]]
    gold_data = [[buf_gold[0][0], ref_gold[0][0], impl_gold[0][0]], [buf_gold[0][1], ref_gold[0][1], impl_gold[0][1]]]
    plat_data = [[buf_plat[0][0], ref_plat[0][0], impl_plat[0][0]], [buf_plat[0][1], ref_plat[0][1], impl_plat[0][1]]]
    gen9_data = [[buf_gen9[0][0], ref_gen9[0][0], impl_gen9[0][0]], [buf_gen9[0][1], ref_gen9[0][1], impl_gen9[0][1]]]
    iris_data = [[buf_iris[0][0], ref_iris[0][0], impl_iris[0][0]], [buf_iris[0][1], ref_iris[0][1], impl_iris[0][1]]]

    # This is the stacked data for the object initialisation
    i5_data_st1   = [[buf_i5[1][0], ref_i5[1][0], impl_i5[1][0]], [buf_i5[1][1], ref_i5[1][1], impl_i5[1][1]]]
    gold_data_st1 = [[buf_gold[1][0], ref_gold[1][0], impl_gold[1][0]], [buf_gold[1][1], ref_gold[1][1], impl_gold[1][1]]]
    plat_data_st1 = [[buf_plat[1][0], ref_plat[1][0], impl_plat[1][0]], [buf_plat[1][1], ref_plat[1][1], impl_plat[1][1]]]
    gen9_data_st1 = [[buf_gen9[1][0], ref_gen9[1][0], impl_gen9[1][0]], [buf_gen9[1][1], ref_gen9[1][1], impl_gen9[1][1]]]
    iris_data_st1 = [[buf_iris[1][0], ref_iris[1][0], impl_iris[1][0]], [buf_iris[1][1], ref_iris[1][1], impl_iris[1][1]]]

    # This is the stacked data for the buffer initialisation
    i5_data_st2   = [[buf_i5[2][0], 0, 0], [buf_i5[2][1], 0, 0]]
    gold_data_st2 = [[buf_gold[2][0], 0, 0], [buf_gold[2][1], 0, 0]]
    plat_data_st2 = [[buf_plat[2][0], 0, 0], [buf_plat[2][1], 0, 0]]
    gen9_data_st2 = [[buf_gen9[2][0], 0, 0], [buf_gen9[2][1], 0, 0]]
    iris_data_st2 = [[buf_iris[2][0], 0, 0], [buf_iris[2][1], 0, 0]]

    # This is the stacked data for the kernel duration
    i5_data_st3   = [[buf_i5[3][0], ref_i5[2][0], impl_i5[2][0]], [buf_i5[3][1], ref_i5[2][1], impl_i5[2][1]]]
    gold_data_st3 = [[buf_gold[3][0], ref_gold[2][0], impl_gold[2][0]], [buf_gold[3][1], ref_gold[2][1], impl_gold[2][1]]]
    plat_data_st3 = [[buf_plat[3][0], ref_plat[2][0], impl_plat[2][0]], [buf_plat[3][1], ref_plat[2][1], impl_plat[2][1]]]
    gen9_data_st3 = [[buf_gen9[3][0], ref_gen9[2][0], impl_gen9[2][0]], [buf_gen9[3][1], ref_gen9[2][1], impl_gen9[2][1]]]
    iris_data_st3 = [[buf_iris[3][0], ref_iris[2][0], impl_iris[2][0]], [buf_iris[3][1], ref_iris[2][1], impl_iris[2][1]]]

    i5_bottom1   = np.sum([i5_data[0], i5_data_st1[0]], axis=0)
    gold_bottom1 = np.sum([gold_data[0], gold_data_st1[0]], axis=0)
    plat_bottom1 = np.sum([plat_data[0], plat_data_st1[0]], axis=0)
    gen9_bottom1 = np.sum([gen9_data[0], gen9_data_st1[0]], axis=0)
    iris_bottom1 = np.sum([iris_data[0], iris_data_st1[0]], axis=0)

    i5_bottom2   = np.sum([i5_data[0], i5_data_st1[0], i5_data_st2[0]], axis=0)
    gold_bottom2 = np.sum([gold_data[0], gold_data_st1[0], gold_data_st2[0]], axis=0)
    plat_bottom2 = np.sum([plat_data[0], plat_data_st1[0], plat_data_st2[0]], axis=0)
    gen9_bottom2 = np.sum([gen9_data[0], gen9_data_st1[0], gen9_data_st2[0]], axis=0)
    iris_bottom2 = np.sum([iris_data[0], iris_data_st1[0], iris_data_st2[0]], axis=0)

    fig, ax = plt.subplots()
    fig.set_figwidth(15)
    fig.set_figheight(8)

    ax.bar(x - (width * 2), i5_data[0], width, yerr=i5_data[1], label='i5', color='blue', edgecolor='k')
    ax.bar(x - width, gold_data[0], width, yerr=gold_data[1], label='gold', color='darkgreen', edgecolor='k')
    ax.bar(x, plat_data[0], width, yerr=plat_data[1], label='platinum', color='maroon', edgecolor='k')
    ax.bar(x + width, gen9_data[0], width, yerr=gen9_data[1], label='gen9', color='orange', edgecolor='k')
    ax.bar(x + (width * 2), iris_data[0], width, yerr=iris_data[1], label='iris', color='purple', edgecolor='k')

    ax.bar(x - (width * 2), i5_data_st1[0], width, yerr=i5_data_st1[1], color='deepskyblue', edgecolor='k', bottom=i5_data[0])
    ax.bar(x - width, gold_data_st1[0], width, yerr=gold_data_st1[1], color='limegreen', edgecolor='k', bottom=gold_data[0])
    ax.bar(x, plat_data_st1[0], width, yerr=plat_data_st1[1], color='red', edgecolor='k', bottom=plat_data[0])
    ax.bar(x + width, gen9_data_st1[0], width, yerr=gen9_data_st1[1], color='darkgoldenrod', edgecolor='k', bottom=gen9_data[0])
    ax.bar(x + (width * 2), iris_data_st1[0], width, yerr=iris_data_st1[1], color='magenta', edgecolor='k', bottom=iris_data[0])

    ax.bar(x - (width * 2), i5_data_st2[0], width, yerr=i5_data_st2[1], color='cyan', edgecolor='k', bottom=i5_bottom1)
    ax.bar(x - width, gold_data_st2[0], width, yerr=gold_data_st2[1], color='lime', edgecolor='k', bottom=gold_bottom1)
    ax.bar(x, plat_data_st2[0], width, yerr=plat_data_st2[1], color='tomato', edgecolor='k', bottom=plat_bottom1)
    ax.bar(x + width, gen9_data_st2[0], width, yerr=gen9_data_st2[1], color='gold', edgecolor='k', bottom=gen9_bottom1)
    ax.bar(x + (width * 2), iris_data_st2[0], width, yerr=iris_data_st2[1], color='deeppink', edgecolor='k', bottom=iris_bottom1)

    ax.set_ylabel('Execution time (in nanoseconds)')
    ax.set_xlabel('Different implementations and hardware')
    ax.set_title('Object creation and initialisation averaged over 20 runs expressed in nanoseconds')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.grid(True)

    fig.tight_layout()
    plt.savefig(path + "plots/" + data_size + "total_minus_kernel.png")

    # plt.yscale('log')
    # ax.set_title('Object creation and initialisation averaged over 20 runs expressed in nanoseconds in log scale')
    # plt.tight_layout()
    # plt.savefig(path + "plots/" + data_size + "total_minus_kernel_log.png")

    # NOTE: we make a separate plot for this to also compare creation and initialisation.
    ax.bar(x - (width * 2), i5_data_st3[0], width, yerr=i5_data_st3[1], color='lightcyan', edgecolor='k', bottom=i5_bottom2)
    ax.bar(x - width, gold_data_st3[0], width, yerr=gold_data_st3[1], color='palegreen', edgecolor='k', bottom=gold_bottom2)
    ax.bar(x, plat_data_st3[0], width, yerr=plat_data_st3[1], color='lightcoral', edgecolor='k', bottom=plat_bottom2)
    ax.bar(x + width, gen9_data_st3[0], width, yerr=gen9_data_st3[1], color='khaki', edgecolor='k', bottom=gen9_bottom2)
    ax.bar(x + (width * 2), iris_data_st3[0], width, yerr=iris_data_st3[1], color='lightpink', edgecolor='k', bottom=iris_bottom2)

    plt.yscale('linear')
    ax.set_title('Program duration averaged over 20 runs expressed in nanoseconds')
    fig.tight_layout()
    plt.savefig(path + "plots/" + data_size + "total.png")

    # plt.yscale('log')
    # ax.set_title('Program duration averaged over 20 runs expressed in nanoseconds in log scale')
    # plt.tight_layout()
    # plt.savefig(path + "plots/" + data_size + "total_log.png")

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

# Raw data, Plot performance data using 1 cluster per version, and 1 colored-bar per machine
i5_buf   = read_data(path_buffer + "i5/")
gold_buf = read_data(path_buffer + "gold/")
plat_buf = read_data(path_buffer + "platinum/")
gen9_buf = read_data(path_buffer + "gen9/")
iris_buf = read_data(path_buffer + "iris/")

i5_ref   = read_data(path_ref + "i5/", type="ref")
gold_ref = read_data(path_ref + "gold/", type="ref")
plat_ref = read_data(path_ref + "platinum/", type="ref")
gen9_ref = read_data(path_ref + "gen9/", type="ref")
iris_ref = read_data(path_ref + "iris/", type="ref")

i5_impl   = read_data(path_implicit + "i5/")
gold_impl = read_data(path_implicit + "gold/")
plat_impl = read_data(path_implicit + "platinum/")
gen9_impl = read_data(path_implicit + "gen9/")
iris_impl = read_data(path_implicit + "iris/")


plot_object_create(i5_buf[0], gold_buf[0], plat_buf[0], gen9_buf[0], iris_buf[0], \
                   i5_ref[0], gold_ref[0], plat_ref[0], gen9_ref[0], iris_ref[0], \
                   i5_impl[0], gold_impl[0], plat_impl[0], gen9_impl[0], iris_impl[0])

plot_object_init(i5_buf[1], gold_buf[1], plat_buf[1], gen9_buf[1], iris_buf[1], \
                 i5_buf[2], gold_buf[2], plat_buf[2], gen9_buf[2], iris_buf[2], \
                 i5_ref[1], gold_ref[1], plat_ref[1], gen9_ref[1], iris_ref[1], \
                 i5_impl[1], gold_impl[1], plat_impl[1], gen9_impl[1], iris_impl[1])

plot_kernel_duration(i5_buf[3], gold_buf[3], plat_buf[3], gen9_buf[3], iris_buf[3], \
                     i5_ref[2], gold_ref[2], plat_ref[2], gen9_ref[2], iris_ref[2], \
                     i5_impl[2], gold_impl[2], plat_impl[2], gen9_impl[2], iris_impl[2])

plot_total(i5_buf, gold_buf, plat_buf, gen9_buf, iris_buf, \
           i5_ref, gold_ref, plat_ref, gen9_ref, iris_ref, \
           i5_impl, gold_impl, plat_impl, gen9_impl, iris_impl)
