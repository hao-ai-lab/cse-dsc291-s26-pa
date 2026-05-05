from typing import Dict
from mpi4py import MPI
import numpy as np
import pytest

from model.func_impl import (
    naive_collect_forward_input,
    naive_collect_forward_output,
)

def check_naive_mp_forward_x(
    input_dict: Dict,
    expect_output_dict: Dict,
) -> None:
    x = input_dict["input_x"]

    output = naive_collect_forward_input(
        x=x,
        mp_size=input_dict["mp_size"],
        mp_comm=input_dict["mp_comm"],
    )

    # Verify that the data type is preserved.
    assert x.dtype == output.dtype

    np.testing.assert_allclose(
        actual=output, desired=expect_output_dict["output_array"]
    )


def check_naive_mp_forward_output(
    input_dict: Dict,
    expect_output_dict: Dict,
) -> None:
    x = input_dict["input_x"]

    output = naive_collect_forward_output(
        out=x,
        mp_size=input_dict["mp_size"],
        mp_comm=input_dict["mp_comm"],
    )

    # Verify that the data type is preserved.
    assert x.dtype == output.dtype

    np.testing.assert_allclose(
        actual=output, desired=expect_output_dict["output_array"]
    )


@pytest.mark.mpi
def test_fc_o_naive_mp_forward_x_3d():
    """
    Test for fc_o forward input collection with a 3D tensor.
    Create a global tensor of shape (4, 8, 8). With mp_size=4, each process gets
    a slice along the last axis of shape (4, 8, 2). After gathering, the full tensor
    should be reassembled.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Global tensor: shape (4, 8, 8)
    global_tensor = np.arange(4 * 8 * 8, dtype=np.float64).reshape((4, 8, 8))
    mp_size = 4
    part_size = global_tensor.shape[2] // mp_size  # 8 // 4 = 2

    # Each process gets its slice along the last axis.
    input_x = global_tensor[:, :, rank * part_size : (rank + 1) * part_size]

    input_dict = {
        "input_x": input_x,
        "mp_comm": comm,
        "mp_size": mp_size,
    }

    expect_output_dict = {
        "output_array": global_tensor,
    }

    check_naive_mp_forward_x(input_dict, expect_output_dict)


@pytest.mark.mpi
def test_fc_o_naive_mp_forward_output_3d():
    """
    Test for fc_o forward output collection with a 3D tensor.
    With fc_o sharded along the input dimension, each process computes a
    same-shaped partial output contribution. The communication should sum
    those contributions across the model-parallel group.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    mp_size = 4

    # Each process gets one full-width partial output contribution.
    global_tensor = np.arange(mp_size * 4 * 8 * 8, dtype=np.float64).reshape(
        (mp_size, 4, 8, 8)
    )
    input_x = global_tensor[rank]

    input_dict = {
        "input_x": input_x,
        "mp_comm": comm,
        "mp_size": mp_size,
    }

    expect_output_dict = {
        "output_array": np.sum(global_tensor, axis=0),
    }

    check_naive_mp_forward_output(input_dict, expect_output_dict)
