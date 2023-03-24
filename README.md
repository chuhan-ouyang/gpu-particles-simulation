# CS5220 PUBLIC REPOSITORY (HW STARTER CODES)

## Overview

This assignment is an introduction to parallel programming using a shared memory model. In this assignment, we will be parallelizing a toy particle simulation (similar simulations are used in mechanics, biology, and astronomy). In our simulation, particles interact by repelling one another. 

The particles repel one another, but only when closer than a cutoff distance highlighted around one particle in grey.

## Asymptotic Complexity

# Serial Solution Time Complexity

If we were to naively compute the forces on the particles by iterating through every pair of particles, then we would expect the asymptotic complexity of our simulation to be O(n^2).

However, in our simulation, we have chosen a density of particles sufficiently low so that with n particles, we expect only O(n) interactions. An efficient implementation can reach this time complexity. The first part of your assignment will be to implement this linear time solution in a serial code, given a naive O(n^2) implementation.

# Parallel Speedup

Suppose we have a code that runs in time T = O(n) on a single processor. Then we'd hope to run close to time T/p when using p processors. After implementing an efficient serial O(n) solution, you will attempt to reach this speedup using OpenMP.

Due Date: Friday, February 24th 2023 at 11:59 PM

## Instructions

# Teams

Note that you will work individually for this assignment. 

# Getting Set Up

The starter code is available in the course Github repo and should work out of the box. To get started, we recommend you log in to Perlmutter and download the first part of the assignment. This will look something like the following:

```
student@local:~> ssh perlmutter-p1.nersc.gov
student@login04:~> git clone git@github.com:CS5220-SP23/CLASS.git
student@login04:~> cd hw2
student@login04:~/hw2> ls
CMakeLists.txt common.h job-openmp job-serial main.cpp openmp.cpp serial.cpp
```

There are five files in the base repository. Their purposes are as follows:

```
CMakeLists.txt
```

The build system that manages compiling your code.

```
main.cpp
```

A driver program that runs your code:

```
common.h
```

A header file with shared declarations:

```
job-openmp
```

A sample job script to run the OpenMP executable:

```
job-serial
```

A sample job script to run the serial executable:

```serial.cpp - - - You may modify this file.```

A simple O(n^2) particle simulation algorithm. It is your job to write an O(n) serial algorithm within the simulate_one_step function.

```openmp.cpp - - - You may modify this file.```

A skeleton file where you will implement your openmp simulation algorithm. It is your job to write an algorithm within the simulate_one_step function.

Please do not modify any of the files besides serial.cpp and openmp.cpp.

## Building our Code

First, we need to make sure that the CMake module is loaded.

```student@login04:~/hw2> module load cmake```

You should put the above command in your ~/.bash_profile file to avoid typing them every time you log in.

Next, let's build the code. CMake prefers out of tree builds, so we start by creating a build directory.

```
student@login04:~/hw2> mkdir build
student@login04:~/hw2> cd build
student@login04:~/hw2/build>
```

Next, we have to configure our build. We can either build our code in Debug mode or Release mode. In debug mode, optimizations are disabled and debug symbols are embedded in the binary for easier debugging with GDB. In release mode, optimizations are enabled, and debug symbols are omitted. For example:

```
student@login04:~/hw2/build> cmake -DCMAKE_BUILD_TYPE=Release ..
-- The C compiler identification is GNU 11.2.0
...
-- Configuring done
-- Generating done
-- Build files have been written to: /global/homes/s/student/hw2/build
```

Once our build is configured, we may actually execute the build:

```
student@login04:~/hw2/build> make
Scanning dependencies of target serial
[ 16%] Building CXX object CMakeFiles/serial.dir/main.cpp.o
[ 33%] Building CXX object CMakeFiles/serial.dir/serial.cpp.o
[ 50%] Linking CXX executable serial
[ 50%] Built target serial
```

Scanning dependencies of target openmp

```
[ 66%] Building CXX object CMakeFiles/openmp.dir/main.cpp.o
[ 83%] Building CXX object CMakeFiles/openmp.dir/openmp.cpp.o
[100%] Linking CXX executable openmp
[100%] Built target openmp
student@login04:~/hw2/build> ls

CMakeCache.txt CMakeFiles cmake_install.cmake Makefile openmp serial job-openmp job-serial
```

We now have two binaries (openmp and serial) and two job scripts (job-openmp and job-serial).

For info on running jobs and editing the code, refer to the HW1 page.
Running the Program

Both executables have the same command line interface. Without losing generality, we discuss how to operate the serial program here. Here's how to allocate an interactive node and run your program (warning: do not run on the login nodes. The benchmark will yield an incorrect result, and you will slow system performance for all users).

```
student@login04:~/hw2> salloc -N 1 -q interactive -t 01:00:00 --constraint cpu --account=mxxxx
salloc: Granted job allocation 53324632
salloc: Waiting for resource configuration
salloc: Nodes nid02346 are ready for job
student@nid02346:~/hw2> cd build
student@nid02346:~/hw2/build> ./serial

Simulation Time = 1.43277 seconds for 1000 particles.
```

You can also run the program using the batch scripts that you provide. By default, the program runs with 1000 particles. The number of particles can be changed with the "-n" command line parameter:

```
student@nid02346:~/hw2/build> ./serial -n 10000

Simulation Time = 195.029 seconds for 10000 particles.
```

If we rerun the program, the initial positions and velocities of the particles will be randomized because the particle seed is unspecified. By default, the particle seed will be unspecified; this can be changed with the "-s" command line parameter:

```
student@nid02346:~/hw2/build> ./serial -s 150

Simulation Time = 1.45459 seconds for 1000 particles.
```

This will set the particle seed to 150 which initializes the particles in a reproducible way. We will test the correctness of your code by randomly selecting several particle seeds and ensuring the particle positions are correct when printed with the "-o" command line parameter. You can print the particle positions to a file specified with the "-o" parameter:

```
student@nid02346:~/hw2/build> ./serial -o serial.parts.out

Simulation Time = 1.78357 seconds for 1000 particles.
```

This will create a serial.parts.out file with the particle positions after each step listed. You can use the hw2-rendering tool to convert this into a .gif file of your particles. See the below section on Rendering Output for more information.

You can use the "-h" command line parameter to print the help menu summarizing the parameter options:

```
student@nid02346:~/hw2/build> ./serial -h

Options:
-h: see this help
-n <int>: set number of particles
-o <filename>: set the output file name
-s <int>: set particle initialization seed
```

## Important notes for Performance:

There will be two types of scaling that are tested for your parallel codes:

In strong scaling we keep the problem size constant but increase the number of processors

In weak scaling we increase the problem size proportionally to the number of processors so the work/processor stays the same (Note that for the purposes of this assignment we will assume a linear scaling between work and processors)

While the scripts we are providing have small numbers of particles 1000 to allow for the O(n2) algorithm to finish execution, the final codes should be tested with values much larger (50000-1000000) to better see their performance.

## Grading

We will grade your assignment by reviewing your assignment write-up, measuring the scaling of both the openmp and serial implementations, and benchmarking your code's raw performance. To benchmark your code, we will compile it with the exact process detailed above, with the GNU compiler. We will run your submissions on Perlmutter's CPU processors.

There are usually some groups every year who come up with faster methods to compute the particle repulsion force function (i.e. rearranging the arithmetic, changing the formula, or using some fancy instructions). This is great, but small differences in the floating point position values begin to add up until the simulation output diverges from our ground truth (even though your method of computation might be more accurate than ours). Since (a) the point of the assignment is to explore OpenMP parallelism, and (b) we can't anticipate every possible way to compute this force function, here is the rule: if it doesn't pass the correctness check we provide you reliably, then it's not allowed.

## Submission Details (Similar to HW1)

1.  Make sure you have our most updated source code on your Permultter machine. We have updated the CMake file for this submission. 

2. Make sure you have only modified the file serial.cpp and openmp.cpp, and it compiles and runs as desired. 

3. Get your groupd ID, same as HW1. On Cavas, under the "People" section, there is a hw1 tab. Click on the tab and you'll see canvas has assigned a group id to each of you individually. Use the search bar to enter your name and find your group id. Treat it as a two digit number. (If you are group 4, your group id is "04").

4. Ensure that your write-up pdf is located in your source directory, next to serial.cpp It should be named CS5220Group04_hw2.pdf.

5. From your build directory, run:
```
            student@perlmutter:~/hw2/build> cmake -DGROUP_NO=04 ..
            student@perlmutter:~/hw2/build> make package
```

This second command will fail if the PDF is not present.

6. Confirm that it worked using the following command. You should see output like:

```
            student@perlmutter:~/hw2/build> tar tfz CS5220Group04_hw2.tar.gz
                       CS5220Group04_hw2/CS5220Group04_hw2.pdf
                       CS5220Group04_hw2/serial.cpp
                       CS5220Group04_hw2/openmp.cpp
```

7. Download and submit your .tar.gz through canvas. 

## Writeup Details

Your write-up should contain:

* Your name, cornell id (NetID), and perlmutter username,
* A plot in log-log scale that shows that your serial and parallel codes run in O(n) time and a description of the data structures that you used to achieve it.
* A description of the synchronization you used in the shared memory implementation.
* A description of the design choices that you tried and how did they affect the performance.
* Speedup plots that show how closely your OpenMP code approaches the idealized p-times speedup and a discussion on whether it is possible to do better.
* Where does the time go? Consider breaking down the runtime into computation time, synchronization time and/or communication time. How do they scale with p?

## Notes:

Your grade will mostly depend on three factors:

* Scaling sustained by your codes on the Perlmutter supercomputer (varying n).
* Performance sustained by your codes on the Perlmutter supercomputer.
* Explanations of your methodologies and the performance features you observed (including what didn't work).
* You must use the GNU C Compiler for this assignment. If your code does not compile and run with GCC, it will not be graded.
* If your code produces incorrect results, it will not be graded.

## Rendering Output

The output files that are produced from running the program with the "-o" command line parameter can be fed into the hw2-rendering tool made available to convert them into .gif files. These animations will be a useful tool in debugging. To get started clone the hw2-rendering repo:

```
student@login04:~> git clone git@github.com:CS5220-SP23/HW2_rendering.git
```

This tool uses python. This can be loaded on Perlmutter with the following command:

```
student@login04:~> module load python
```

We can then convert the output files to gifs with the following command: make sure to allocate an interactive node first!
```
student@login04:~/hw2/build> ~/HW2_rendering/render.py serial.parts.out particles.gif 0.01
```
Here serial.parts.out is an output file from the "-o" command line parameter. You should find a particles.gif file in your directory. The number 0.01 is the cutoff distance (will be drawn around each particle).

## Output Correctness

The output files that are produced from running the program with the "-o" command line parameter can be fed into the hw2-correctness tool made available to perform a correctness check. This is the same correctness check we will be performing when grading the homework, however, we will randomly select the particle seeds. To get started clone the hw2-correctness repo:

```
student@login04:~> git clone git@github.com:CS5220-SP23/HW2_correctness.git
```
This tool uses python. This can be loaded on Perlmutter with the following command:
```
student@login04:~> module load python
```
We can then test the output files for correctness with the following command: make sure to allocate an interactive node first!
```
student@nid02346:~/hw2/build> ~/HW2_correctness/correctness-check.py serial.parts.out correct.parts.out
```
If the program prints an error, then your output is incorrect. Here serial.parts.out is an output file from the "-o" command line parameter from your code. This can be substituted for any output you wish to test the correctness for. The correct.parts.out can be generated from the provided O(n^2) serial implementation. Remember to specify a particle seed with "-s" to ensure the same problem is solved between the two output files. The hw2-correctness repo provides the "verf.out" file which is the correct output with particle seed set to 1 "-s 1".

## Resources

* Programming in shared and distributed memory models are introduced in Lectures.

* Shared memory implementations may require using locks that are available as omp_lock_t in OpenMP (requires omp.h)

* You may consider using atomic operations such as __sync_lock_test_and_set with the GNU compiler.

* Other useful resources: pthreads tutorial, OpenMP tutorial, OpenMP specifications and MPI specifications.

* Hints on getting O(n) serial and Shared memory and MPI implementations (pdf)

* It can be very useful to use a performance measuring tool in this homework. Parallel profiling is a complicated business but there are a couple of tools that can help.

* TAU (Tuning and Analysis Utilities) is a source code instrumentation system to gather profiling information. You need to "module load tau" to access these capabilities. This system can profile MPI, OpenMP and PThread code, and mixtures, but it has a learning curve.

* HPCToolkit Is a sampling profiler for parallel programs. You need to "module load hpctoolkit". You can install the hpcviewer on your own computer for offline analysis, or use the one on NERSC by using the NX client to get X windows displayed back to your own machine.

* If you are using TAU or HPCToolkit you should run in your $SCRATCH directory which has faster disk access to the compute nodes (profilers can generate big profile files).
