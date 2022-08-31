"""Testing suite"""

from subprocess import Popen, PIPE

# home_dir = "/home/julius/Downloads/idg-fpga-master/thesis/"
home_dir = "/home/u136839/thesis/"
device = "i5"

# NOTE: switch between: skl, ram384gb (platinum), gen9 and iris_xe_max
# NOTE: make sure to have the gpu_selector for gpu tasks
# NOTE: USE export OverrideDefaultFP64Settings=1 and export IGC_EnableDPEmulation=1 for the iris execution
def benchmark(size_params, dir, machine):
    repeat = 20

    ref_dir = dir + "reference/output/" + size_params + "/" + machine
    buffer_dir = dir + "buffer/output/" + size_params + "/" + machine
    implicit_dir = dir + "implicit/output/" + size_params + "/" + machine

    for r in range(1, repeat + 1):
        out_ref = open(ref_dir + str(r), "w")
        out_buf = open(buffer_dir + str(r), "w")
        out_impl = open(implicit_dir + str(r), "w")

        # NOTE: change to -gpu if executing on the GPU
        p = Popen(["./reference/run-gridder-cpu"], shell=True, stdout=out_ref, universal_newlines=True)
        p.wait()

        p = Popen(["./buffer/run-gridder"], shell=True, stdout=out_buf, universal_newlines=True)
        p.wait()

        p = Popen(["./implicit/run-gridder"], shell=True, stdout=out_impl, universal_newlines=True)
        p.wait()

# NOTE: Change between "big_param" and "small_param"
benchmark("small_param", home_dir, device + "/")
