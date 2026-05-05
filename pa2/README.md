# CSE291/DSC291 S26: PA 2
Welcome to PA 2. We recommend starting early. See the course site or Gradescope for the deadline and late-day policy.

Academic integrity is key. You may discuss ideas with classmates, but do not copy solutions.

### **Important**
You may work with groups of up to 3 and submit together on Gradescope.

## Part 1: Matmul Kernel Optimization (40 pts)
The file you must complete in this part is `student_kernel.py`. You will find the required interface in that file, together with a local test script and a remote submission script.

In this part of the assignment, you will implement a Triton-based matrix multiplication + ReLU + add kernel. The kernel computes the matrix function **D = ReLU(A × B + C)** where **A** is of shape *(M, K)*, **B** is of shape *(K, N)*, and **C & D** are of shape *(M, N)*. We will break the kernel down into four main steps:

1. **Tile Assignment**
2. **Shared Memory Tiling + Cooperative Fetching**
3. **Register Tiling (Accumulator)**
4. **Operator Fusion**
5. **Write Cache/Epilogue (Store Results)**

Each section below explains the purpose and the implementation details of each step.

### Setup
#### **VERY IMPORTANT:**
You do **not** need a local GPU to complete this part. Official evaluation for Part 1 uses the course remote grader running on an **A10 GPU**.

If you have a local CUDA GPU, you may use it for debugging with `student_local_test.py`, but the official speed numbers come from the remote grader.

If you don't have a local CUDA GPU, you can add print statements and submit with `--json` (e.g. `python student_submit.py student_kernel.py --json`) to see your printed output in the raw grader payload.

**Note:** Use fp16 inputs and fp16 output for this assignment. For the matmul accumulator, FP32 accumulation should be used for precision.

### 1.1 Tile Assignment
**Tile Assignment** is the process by which the overall matrix **C** is divided into smaller submatrices (tiles). Each kernel instance (a GPU “program” or thread block) is responsible for computing one tile of **C**. This division allows the computation to be parallelized across many threads.

Each kernel launch instance (denoted by a unique program id `pid`) should be mapped to a specific tile in **C**. The tile is defined by two indices: one for the row (denoted `pid_m`) and one for the column (`pid_n`).

### 1.2 Shared Memory Tiling + Cooperative Fetching
**Shared Memory Tiling** is used to load sub-tiles (smaller blocks) of matrices A and B into faster, on-chip memory. Threads within a kernel instance load these sub-tiles, reducing the number of global memory accesses.

Some things to keep in mind:
- You may need to use `tl.arange` to help compute offsets for the rows and columns.
- Use masks to make sure that out-of-bound memory accesses do not occur.

### 1.3 Register Tiling (Accumulator)
**Register Tiling** is the use of registers to hold intermediate results (partial sums) during the matrix multiplication. For this section, you will need to use an accumulator (a BLOCK_SIZE_M by BLOCK_SIZE_N matrix initialized to zeros) to accumulate results of dot products computed over tiles.

After accumulation you can optionally choose to fuse an activation function (like leaky ReLU) - this is used in practice to make architectures that use lots of matmuls and activation functions together (like transformers) much much faster!

### 1.4 Add and ReLU Fusion
In this step, we fuse the element-wise addition of matrix C and the final ReLU activation directly into the matmul kernel to optimize performance by reducing memory traffic and kernel launch overhead. After computing the matrix multiplication and storing the results in the accumulator, you must load the corresponding tile from matrix C and add it element-wise to the activated accumulator. Then apply the ReLU function using an element-wise maximum operation to set all negative values to zero.

This fusion of operations avoids writing intermediate results back to global memory and then reloading them for further processing, minimizing latency and making more efficient use of the GPU’s memory hierarchy.

### 1.5 Write Cache/Epilogue
In this step, we write the tile of C back to global memory. This final step ensures that the computed results are stored in the correct locations in the output matrix C. Be sure to also use a mask to prevent invalid indices from being written to. (Use `tl.store` to store your tiles.)

### 1.6 Grid Search
To achieve full credit on Part 1, you will likely need to search over launch configurations. In this release, those are provided through `KERNEL_CONFIGS`, where each config specifies:

- `BLOCK_M`
- `BLOCK_N`
- `BLOCK_K`
- `num_warps`
- `num_stages`

The remote grader will benchmark the configs you submit and pick the best one automatically.

### Part 1 Files
You will primarily use the following files:

- `student_kernel.py`: your actual submission file
- `student_local_test.py`: optional local CUDA test script
- `student_submit.py`: remote submission script

Your `student_kernel.py` file must define exactly two top-level objects:

- `KERNEL_CONFIGS`
- `@triton.jit def matmul_add_relu_kernel_fp16(...)`

### Part 1 Testing
If you have a local CUDA GPU, you can test locally with:

```bash
python student_local_test.py student_kernel.py
```

If you have a local CUDA GPU, `student_local_test.py` is useful for debugging correctness and basic kernel behavior, but the official speed numbers come from the remote grader on an **A10 GPU**.

If you do not have a local GPU, use the remote grader:

```bash
export GRADER_BASE_URL="https://<your-course-endpoint>"
export GRADER_TOKEN="<course-token>"
python student_submit.py student_kernel.py
```

You should see output similar to:

```text
Status: ok
Passed correctness: yes
Max abs diff: 0.062500
Selected config: BLOCK_M=16, BLOCK_N=16, BLOCK_K=16, num_warps=2, num_stages=2
Submitted configs: 1
Your kernel: 1.9396 ms
PyTorch ref: 0.3232 ms
Speedup vs ref: 0.1666x
Device: NVIDIA A10
```

`student_submit.py` prints a short human-readable summary by default. 

For remote debugging, you can ask `student_submit.py` to save the final output to a file:

```bash
python student_submit.py --json --output submit_debug.json student_kernel.py
```

This is useful when you want to inspect the raw `stdout`, `stderr`, `traceback`, or `config_failures` fields returned by the grader.

If you need to print values from inside a Triton kernel while debugging, do **not** use Python `print(...)` inside `@triton.jit`. Instead use Triton debugging helpers such as:

```python
tl.device_print("a=", a)
```

You should remove these debug prints before final benchmarking, since they will significantly slow down the kernel.

We will post the remote grader endpoint and token on **Piazza**.

### Part 1 Grading
This entire section combined will be 40 points total, with possible extra credit for very fast kernels. We will grade you on correctness and overall performance with respect to the PyTorch reference used by the remote grader.

The benchmark uses:

- `M = N = K = 2048`
- `dtype = fp16`
- fused matmul + add + relu

The grader reports whether your kernel passes correctness, which config was selected, your kernel latency in ms, the PyTorch reference latency in ms, and your speedup versus the reference.

The Part 1 score thresholds are:

- incorrect output: `0 / 40`
- correct output but `< 1.0x` speedup: `20 / 40`
- `>= 1.0x` speedup: `35 / 40`
- `>= 1.1x` speedup: `40 / 40`
- `>= 1.25x` speedup: `45 / 40`
- `>= 1.4x` speedup: `50 / 40`

## Part 2: Tensor Parallel Communication (60 pts)

In this part of the programming assignment, you will work on developing communication protocols for Data Parallel and Tensor Model Parallel training from the ground up, utilizing the Message Passing Interface ([MPI](https://mpi4py.readthedocs.io/en/stable/)) and NumPy.

Since the main focus will not be on the actual forward computation or backward propagation, your task will be to implement only selected communication steps used by the forward and backward passes.

### Setup Guide

You'll need need a multi-core machine with at least 8 cores for this assignment.

#### Installing MPI
You'll need to have MPI installed for this assignment. 

##### Linux
```bash
sudo apt install libopenmpi-dev
```
##### MacOS
```bash
brew install openmpi
```
##### Windows
Go to the following and install mpiexec. Note, you'll have to use `mpiexec` instead of `mpirun` in all the commands in this assignment.
https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi

#### Conda
The easiest way is to install an environment manager such as [Miniconda](https://docs.anaconda.com/free/miniconda/) and MPI for your OS (supported on MacOS, Linux, and Windows).

Once you have Conda installed and initialzied, create a new env with the following command:
```bash
conda create -n cse291pa2 python=3.10 -y
```
Then activate it:
```bash
conda activate cse291pa2
```
Now, from the assignment directory, install the requirements:
```bash
pip install -r requirements.txt
```

**You can also create and activate Conda environments through VSCode.**

Make sure your platform contains MPI support with at least 8 cores (or use `--oversubscribe` as noted below).

### 2.0. Warm-up

This assignment aims to guide you step by step through a 2D parallel training pipeline,incorporating both tensor model and data parallel training. For tensor model parallel training, we will delve into the naive approach. 

To become acquainted with our communication protocols, we will begin by experimenting with the MPI package that has been installed.

#### MPI Test
To verify that *mpi4py* has been setup correctly for distributed workloads, run:
```bash
mpirun -n 8 python mpi-test.py
```

Or if you are using Windows, use `mpiexec` instead of `mpirun` for all of follow commands:
```bash
mpiexec -n 8 python mpi-test.py
```

Depending on your machine, you can control the number of processes lanched with the `-n` argument.

> **Note:** If your machine has fewer physical cores than the requested process count (common on laptops), Open MPI will refuse to launch. Pass `--oversubscribe` to allow it, e.g. `mpirun --oversubscribe -n 8 python mpi-test.py`.

Additionally, we have included some simple examples of MPI functions in mpi-test.py, such as Allreduce(), Allgather(), Reduce_scatter(), Split() and Alltoall(). These are the only **collective** operations permitted for this assignment. Point-to-point primitives (`Send`, `Recv`, `Sendrecv`) are also allowed.

For the custom primitive implementations in Section 2.1, do not call built-in MPI collective operations inside `myAllreduce` or `myAlltoall`. You should implement those two methods using point-to-point communication. This restriction applies only to the custom primitives; in the later model-parallel helper functions, you may use the collective operations introduced in this assignment as appropriate.

- ##### All-Reduce

<p align="center">
<img src="figs/allreduce.png" alt="image" width="450" height="auto">
</p>

You can see an all-reduce example by running:
```bash
mpirun -n 8 python mpi-test.py --test_case allreduce
```

- ##### All-Gather

<p align="center">
<img src="figs/allgather.png" alt="image" width="450" height="auto">
</p>

You can see an all-gather example by running:
```bash
mpirun -n 8 python mpi-test.py --test_case allgather
```

- ##### Reduce-Scatter

<p align="center">
<img src="figs/reduce_scatter.png" alt="image" width="450" height="auto">
</p>

You can see a reduce-scatter example by running:
```bash
mpirun -n 8 python mpi-test.py --test_case reduce_scatter
```

- ##### Split

<p align="center">
<img src="figs/split.jpeg" alt="image" width="350" height="auto">
</p>


The Split function is particularly useful when applying MPI functions on a group basis. You can observe an example of group-wise reduction with the split function by running: 

```bash
mpirun -n 8 python mpi-test.py --test_case split
```

When playing with different test cases, try to get yourself familiar with the underline mpi functions 
and think about whether the output meets your expectation. 


#### Node Indexing Specifications

With a given data and model parallel size, we will assign nodes in a model parallel major for this assignment.
For instance, for `mp_size=2, dp_size=4` on 8 nodes we will group the nodes as shown below:

<p align="center">
<img src="figs/group.png" alt="image" width="350" height="auto">
</p>

### 2.1 Implementation of Collective Communication Primitives (20 pts + 10 pts bonus)

In previous part, we directly use primitives from the MPI library. In this task, you need to implement your own version of two primitives: all-reduce and all-to-all.

You only need to fill in the function `myAllreduce` and `myAlltoall` in the `comm.py` file!
Your `myAllreduce` implementation should respect the passed `op` argument. The required operators for this assignment are `MPI.MIN`, `MPI.SUM`, and `MPI.MAX`.

After implementing both function, you can use 

```bash
mpirun -n 8 python mpi-test.py --test_case myallreduce
```

and 

```bash
mpirun -n 8 python mpi-test.py --test_case myalltoall
```

to check the correctness of your implementation. We can also see the time consumption of both implementation. 
Discuss the time difference between your implementation and the MPI one, and give some possible reason if there is a gap. 
If your implementation is fast (within 150% time consumption of MPI version), you can get a bonus (+2.5 pts for each function)!
If your implementation is as fast as the MPI implementation (within 105% time consumption of MPI version), you can get a bonus (+5 pts for each function)!

Put your discussion (2-3 sentences) in the file discussion2-1.txt.

### 2.2 Data Split for Data Parallel Training (5 pts)

For this part, your task is to implement the `split_data` function in `data/data_parallel_preprocess.py`.

The function takes in the training data and returns the data split according to the given `mp_size, dp_size` 
and `rank`. You should split data uniformly across data parallel groups while the model parallel groups can share the 
same data split within the same data parallel group. The data length is guaranteed to be divided equally by the
`dp_size` in all our test cases.

Hints: 
For `mp_size=2, dp_size=4`, you should split the data this way:
 
<p align="center">
<img src="figs/part1-hint.png" alt="image" width="350" height="auto">
</p>

To test your implementation, please run
```bash
python3 -m pytest -l -v tests/test_data_split.py
```

### 2.3 Layer Initialization (15 pts)

In this part, your task is to get necessary information for model and data parallel training, which is then
used to initialize the corresponding layers in your model.

For this assignment we will work with a transformer layer.

You are only required to implement the communications within four fully connective layers within a transformer layer for forward and backward.
We have already taken care of the other stuffs i.e. the forward/backward computations and the training pipeline as these
are not relevant to the goal of this assignment.

For data parallel, we simply just split the batch of data equally across different data parallel groups.

For naive tensor model parallel training, we shard the weight matrices of the four fully connected layers across nodes as follows:
- `fc_q`, `fc_k`, `fc_v` are split along the **output** dimension.
- `fc_o` is split along the **input** dimension.

Given the above information, you need to implement the `get_info` function in `model/func_impl.py`.
The function gets essential information for later parts, including model/data parallel indexing,
model/data parallel communication groups, in/out dimensions for four FC layers. Please refers to the function for more information and hints.

To test your implementation, please run
```bash
mpirun -n 8 python3 -m pytest -l -v --with-mpi tests/test_get_info.py
```

### 2.4 Naive Model Parallel Forward Communication (10 pts)

Your task in this part is to implement the forward communications in W_o layer for the naive model parallel.
You need to implement the `naive_collect_forward_input` and `naive_collect_forward_output` functions in
`model/func_impl.py`. `naive_collect_forward_input` is an all-gather exercise: each MP rank starts with a slice of the input tensor, and the function reconstructs the full tensor. `naive_collect_forward_output` should match the `fc_o` input-dimension sharding from Section 2.3: each MP rank has a same-shaped partial output contribution, and the function should combine those partial outputs with an all-reduce sum. Please refer to the code for more details.

To test your implementations, please run
```bash
mpirun -n 4 python3 -m pytest -l -v --with-mpi tests/test_transformer_forward.py
```


### 2.5 Naive Model Parallel Backward Communication (10 pts)

Your task in this part is to implement the backward communications in W_o layer for the naive model parallel.
You need to implement the `naive_collect_backward_output` and `naive_collect_backward_x` functions in
`model/func_impl.py`. `naive_collect_backward_output` slices the full output gradient along the
partitioned output dimension so that each MP node receives its local shard, while
`naive_collect_backward_x` reduce-scatters the locally computed `grad_x` across MP nodes along the
input feature dimension. Please refer to the code for more details.

To test your implementations, please run
```bash
mpirun -n 4 python3 -m pytest -l -v --with-mpi tests/test_transformer_backward.py
```

## How to Submit Your Homework (Important!)

In your assignment root directory run
```bash
make handin.tar
```
This will first run `generate_pa2_report.py`, which writes:

- `pa2_report.json`: machine-readable combined output for Part 1 and Part 2, including score summaries

If you want to generate the reports yourself first, run:

```bash
export GRADER_BASE_URL="https://<your-course-endpoint>"
export GRADER_TOKEN="<course-token>"
python3 generate_pa2_report.py
```
This command writes `pa2_report.json` and also prints the human-readable score summary to stdout.

This creates a tarball containing:

- `student_kernel.py`
- `model/func_impl.py`
- `data/data_parallel_preprocess.py`
- `mpi_wrapper/comm.py`
- `discussion2-1.txt`
- `pa2_report.json`

Then you will see a `handin.tar` file under your root directory. Please go to Gradescope and submit that tarball.

#### References 

- some images credited to https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html
