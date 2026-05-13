"""MPI wrapper for PA3 Part 1.

This wrapper exposes both the buffered NumPy-style collectives that you used
in PA2 (`Allreduce`, `Allgather`, `Reduce_scatter`, `Alltoall`) and the
pickle-based Python-object collectives (`bcast`, `allgather`, `alltoall`, ...).

In Part 1 you will mostly use the pickle-based variants because token routing
in a Mixture-of-Experts produces variable-sized payloads per rank. The buffered
variants are still available if you want to use them in optimized code paths.

If you want to drop in your own implementations of all-reduce / all-to-all from
PA2 (Section 2.1), copy `myAllreduce` and `myAlltoall` from your PA2
`mpi_wrapper/comm.py` into the marked locations below. Doing so is optional but
recommended for the EP implementation; see the bonus rubric in the README.
"""

from mpi4py import MPI
import numpy as np


class Communicator(object):
    def __init__(self, comm: MPI.Comm):
        self.comm = comm
        self.total_bytes_transferred = 0

    # ---------- basic info ----------
    def Get_size(self):
        return self.comm.Get_size()

    def Get_rank(self):
        return self.comm.Get_rank()

    def Barrier(self):
        return self.comm.Barrier()

    # ---------- pickle-based (Python object) collectives ----------
    def bcast(self, data, root=0):
        return self.comm.bcast(data, root=root)

    def allgather(self, data):
        return self.comm.allgather(data)

    def alltoall(self, send_data):
        return self.comm.alltoall(send_data)

    def allreduce(self, data, op=MPI.SUM):
        return self.comm.allreduce(data, op=op)

    # ---------- buffered (NumPy) collectives ----------
    def Allreduce(self, src_array, dest_array, op=MPI.SUM):
        assert src_array.size == dest_array.size
        src_bytes = src_array.itemsize * src_array.size
        self.total_bytes_transferred += src_bytes * 2 * (self.comm.Get_size() - 1)
        self.comm.Allreduce(src_array, dest_array, op)

    def Allgather(self, src_array, dest_array):
        src_bytes = src_array.itemsize * src_array.size
        dest_bytes = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_bytes * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_bytes * (self.comm.Get_size() - 1)
        self.comm.Allgather(src_array, dest_array)

    def Reduce_scatter(self, src_array, dest_array, op=MPI.SUM):
        src_bytes = src_array.itemsize * src_array.size
        dest_bytes = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_bytes * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_bytes * (self.comm.Get_size() - 1)
        self.comm.Reduce_scatter_block(src_array, dest_array, op)

    def Alltoall(self, src_array, dest_array):
        nprocs = self.comm.Get_size()
        assert src_array.size % nprocs == 0
        assert dest_array.size % nprocs == 0
        send_seg_bytes = src_array.itemsize * (src_array.size // nprocs)
        recv_seg_bytes = dest_array.itemsize * (dest_array.size // nprocs)
        self.total_bytes_transferred += send_seg_bytes * (nprocs - 1)
        self.total_bytes_transferred += recv_seg_bytes * (nprocs - 1)
        self.comm.Alltoall(src_array, dest_array)

    def Split(self, key, color):
        return __class__(self.comm.Split(key=key, color=color))

    # ---------- optional: paste your PA2 implementations here ----------
    def myAllreduce(self, src_array, dest_array, op=MPI.SUM):
        """OPTIONAL: paste your PA2 implementation here to use it in this PA."""
        raise NotImplementedError(
            "myAllreduce is optional in PA3. Paste your PA2 implementation to enable it."
        )

    def myAlltoall(self, src_array, dest_array):
        """OPTIONAL: paste your PA2 implementation here to use it in this PA."""
        raise NotImplementedError(
            "myAlltoall is optional in PA3. Paste your PA2 implementation to enable it."
        )


# Default global communicator (mirrors the pa2 convention).
mpi = Communicator(MPI.COMM_WORLD)
