"""Easily run the instance with different parameters"""

from subprocess import Popen, PIPE

repeat = 1

# # Personal computer testing
# ref_dir = "/home/julius/Downloads/idg-fpga-master/testing/"
#
# # Files names are generated as: kernel_iteration | underscore | repeat number
# for r in range(1, repeat + 1):
#     out = open(buffer_dir + str(r), "w")
#     p = Popen(["./vref"], shell=True, stdout=out, universal_newlines=True)
#     p.wait()

p = Popen(["qsub -l nodes=1:gen9:ppn=2 -d . build.sh"], shell=True)
p.wait()

p = Popen(["qsub -l nodes=1:gen9:ppn=2 -d . run.sh"], shell=True)
p.wait()
