"""Testing suite for the devcloud"""

from subprocess import Popen, PIPE

repeat = 1

# NOTE: switch between: skl, clx, gen9 and iris_xe_max:quad_gpu
# NOTE: use rungpu.sh for gen9 and iris_xe_max:quad_gpu
for r in range(1, repeat + 1):
    p = Popen(["qsub -l nodes=1:skl:ppn=2 -d . run.sh"], shell=True)
    p.wait()
