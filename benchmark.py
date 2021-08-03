"""Testing suite for the personal machine"""

from subprocess import Popen, PIPE


repeat = 25

# Personal computer testing, NOTE: change either small or big for whatever params we use
ref_dir = "/home/julius/Downloads/idg-fpga-master/thesis/reference/output/big_param/i5/"
buffer_dir = "/home/julius/Downloads/idg-fpga-master/thesis/buffer/output/big_param/i5/"
implicit_dir = "/home/julius/Downloads/idg-fpga-master/thesis/implicit/output/big_param/i5/"

# Files names are generated as: kernel_iteration | underscore | repeat number
for r in range(1, repeat + 1):
    out_ref = open(ref_dir + str(r), "w")
    out_buf = open(buffer_dir + str(r), "w")
    out_impl = open(implicit_dir + str(r), "w")

    p = Popen(["./reference/run-gridder-cpu"], shell=True, stdout=out_ref, universal_newlines=True)
    p.wait()

    p = Popen(["./buffer/run-gridder"], shell=True, stdout=out_buf, universal_newlines=True)
    p.wait()

    p = Popen(["./implicit/run-gridder"], shell=True, stdout=out_impl, universal_newlines=True)
    p.wait()
