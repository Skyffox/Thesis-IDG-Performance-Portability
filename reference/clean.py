from subprocess import Popen, PIPE

p = Popen(["rm build.sh.*"], shell=True)
p.wait()

p = Popen(["rm run.sh.*"], shell=True)
p.wait()

p = Popen(["rm run_gpu.sh.*"], shell=True)
p.wait()
