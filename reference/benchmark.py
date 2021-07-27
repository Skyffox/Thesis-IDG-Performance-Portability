"""Testing suite for the reference model on the devcloud"""

from subprocess import Popen, PIPE

repeat = 1

p = Popen(["qsub -l nodes=1:skl:ppn=2 -d . build.sh"], shell=True)
p.wait()

for r in range(1, repeat + 1):
    p = Popen(["qsub -l nodes=1:skl:ppn=2 -d . run.sh"], shell=True)
    p.wait()


p = Popen(["qsub -l nodes=1:clx:ppn=2 -d . build.sh"], shell=True)
p.wait()

for r in range(1, repeat + 1):
    p = Popen(["qsub -l nodes=1:clx:ppn=2 -d . run.sh"], shell=True)
    p.wait().


p = Popen(["qsub -l nodes=1:gen9:ppn=2 -d . build.sh"], shell=True)
p.wait()

for r in range(1, repeat + 1):
    p = Popen(["qsub -l nodes=1:gen9:ppn=2 -d . rungpu.sh"], shell=True)
    p.wait()


p = Popen(["qsub -l nodes=1:iris_xe_max:ppn=2 -d . build.sh"], shell=True)
p.wait()

for r in range(1, repeat + 1):
    p = Popen(["qsub -l nodes=1:iris_xe_max:ppn=2 -d . rungpu.sh"], shell=True)
    p.wait()
