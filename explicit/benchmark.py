"""Easily run the instance with different parameters"""

from subprocess import Popen, PIPE


# kernel_iterations = [1, 10]
# repeat = 5
#
# # Personal computer testing
# directory_name = "/home/julius/Downloads/idg-fpga-master/juliusUSM/explicit/output/personal"
#
# # Files names are generated as: kernel_iteration | underscore | repeat number
# for k in kernel_iterations:
#     for r in range(1, repeat + 1):
#         file_name = directory_name + "/" + str(k) + "_" + str(r)
#         output_file = open(file_name, "w")
#
#         # Normal testing
#         cmd = ["./run-gridder.x " + str(k)]
#
#         p = Popen(cmd, shell=True, stdout=output_file, universal_newlines=True)
#         p.wait()


# Devcloud testing, need to specify the kernel iterations in the run.sh
for r in range(1, repeat + 1):
    cmd1 = ["qsub -l nodes=1:clx:ppn=2 -d . run.sh"]
    p = Popen(cmd2, shell=True)
    p.wait()

    cmd2 = ["qsub -l nodes=1:clx:ppn=2 -d . run10.sh"]
    p = Popen(cmd2, shell=True)
    p.wait()
