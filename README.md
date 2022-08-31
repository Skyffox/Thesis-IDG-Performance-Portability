# Introduction
This repository contains IDG implementations for CPUs and GPUs. The reference implementation uses OpenMP pragmas for parallelism the versions in buffer and implicit uses the oneAPI programming model for parallelism. The compare directory contains sample output from all implemented versions and tests the outputs for correctness against the reference CPU implementation.

# Environment
Make sure the dpcpp and icpx commands are usable, this is used to compile the code. Before we can invoke the compiler, we need to set certain environment variables that define the location of compiler-related components. These variables can be found in: /opt/intel/oneapi/setvars.sh.

This variable script is present in the OneAPI Base Toolkit which must be downloaded before use. Additionally, the files in this directory have only been tested on a 64-bit Linux operating system. A 32-bit operating system and the macOS operating system are not supported by the Intel DPC++ compiler.

# Build and run the programs
```
make
./run-gridder (buffer / implicit)
./run-gridder-cpu (reference)
./run-gridder-gpu (reference)
```
