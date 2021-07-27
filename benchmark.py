"""Testing suite for the personal machine"""

from subprocess import Popen, PIPE


repeat = 10

# Personal computer testing
ref_dir = "/home/julius/Downloads/idg-fpga-master/thesis/reference/output/i5/"
buffer_dir = "/home/julius/Downloads/idg-fpga-master/thesis/buffer/output/i5/"
implicit_dir = "/home/julius/Downloads/idg-fpga-master/thesis/implicit/output/i5/"

# Files names are generated as: kernel_iteration | underscore | repeat number
for r in range(1, repeat + 1):
    out_ref = open(ref_dir + str(r), "w")
    out_buf = open(buffer_dir + str(r), "w")
    out_impl = open(implicit_dir + str(r), "w")

    p = Popen(["./reference/run-gridder-cpu 100"], shell=True, stdout=out_ref, universal_newlines=True)
    p.wait()

    p = Popen(["./buffer/run-gridder 100"], shell=True, stdout=out_buf, universal_newlines=True)
    p.wait()

    p = Popen(["./implicit/run-gridder 100"], shell=True, stdout=out_impl, universal_newlines=True)
    p.wait()
