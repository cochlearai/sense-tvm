"""Cochl-specific TIR transform wrappers."""

from tvm.tir.transform import _ffi_api


def MakeUnpackedAPI():
    """Lower PrimFuncs to an unpacked C-style API."""
    return _ffi_api.MakeUnpackedAPI()  # type: ignore
