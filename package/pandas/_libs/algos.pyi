# Note: this covers algos.pyx and algos_common_helper but NOT algos_take_helper
from typing import Any

import numpy as np

class Infinity:
    """
    Provide a positive Infinity comparison method for ranking.
    """

    def __eq__(self, other) -> bool: ...
    def __ne__(self, other) -> bool: ...
    def __lt__(self, other) -> bool: ...
    def __le__(self, other) -> bool: ...
    def __gt__(self, other) -> bool: ...
    def __ge__(self, other) -> bool: ...

class NegInfinity:
    """
    Provide a negative Infinity comparison method for ranking.
    """

    def __eq__(self, other) -> bool: ...
    def __ne__(self, other) -> bool: ...
    def __lt__(self, other) -> bool: ...
    def __le__(self, other) -> bool: ...
    def __gt__(self, other) -> bool: ...
    def __ge__(self, other) -> bool: ...

def unique_deltas(
    arr: np.ndarray,  # const int64_t[:]
) -> np.ndarray: ...  # np.ndarray[np.int64, ndim=1]
def is_lexsorted(list_of_arrays: list[np.ndarray]) -> bool: ...
def groupsort_indexer(
    index: np.ndarray,  # const int64_t[:]
    ngroups: int,
) -> tuple[
    np.ndarray,  # ndarray[int64_t, ndim=1]
    np.ndarray,  # ndarray[int64_t, ndim=1]
]: ...
def kth_smallest(
    a: np.ndarray,  # numeric[:]
    k: int,
) -> Any: ...  # numeric

# ----------------------------------------------------------------------
# Pairwise correlation/covariance

def nancorr(
    mat: np.ndarray,  # const float64_t[:, :]
    cov: bool = False,
    minp=None,
) -> np.ndarray: ...  # ndarray[float64_t, ndim=2]
def nancorr_spearman(
    mat: np.ndarray,  # ndarray[float64_t, ndim=2]
    minp: int = 1,
) -> np.ndarray: ...  # ndarray[float64_t, ndim=2]
def nancorr_kendall(
    mat: np.ndarray,  # ndarray[float64_t, ndim=2]
    minp: int = 1,
) -> np.ndarray: ...  # ndarray[float64_t, ndim=2]

# ----------------------------------------------------------------------

# ctypedef fused algos_t:
#    float64_t
#    float32_t
#    object
#    int64_t
#    int32_t
#    int16_t
#    int8_t
#    uint64_t
#    uint32_t
#    uint16_t
#    uint8_t

def validate_limit(nobs: int | None, limit=None) -> int: ...
def pad(
    old: np.ndarray,  # ndarray[algos_t]
    new: np.ndarray,  # ndarray[algos_t]
    limit=None,
) -> np.ndarray: ...  # np.ndarray[np.intp, ndim=1]
def pad_inplace(
    values: np.ndarray,  # algos_t[:]
    mask: np.ndarray,  # uint8_t[:]
    limit=None,
) -> None: ...
def pad_2d_inplace(
    values: np.ndarray,  # algos_t[:, :]
    mask: np.ndarray,  # const uint8_t[:, :]
    limit=None,
) -> None: ...
def backfill(
    old: np.ndarray,  # ndarray[algos_t]
    new: np.ndarray,  # ndarray[algos_t]
    limit=None,
) -> np.ndarray: ...  # np.ndarray[np.intp, ndim=1]
def backfill_inplace(
    values: np.ndarray,  # algos_t[:]
    mask: np.ndarray,  # uint8_t[:]
    limit=None,
) -> None: ...
def backfill_2d_inplace(
    values: np.ndarray,  # algos_t[:, :]
    mask: np.ndarray,  # const uint8_t[:, :]
    limit=None,
) -> None: ...
def is_monotonic(
    arr: np.ndarray,  # ndarray[algos_t, ndim=1]
    timelike: bool,
) -> tuple[bool, bool, bool]: ...

# ----------------------------------------------------------------------
# rank_1d, rank_2d
# ----------------------------------------------------------------------

# ctypedef fused rank_t:
#    object
#    float64_t
#    uint64_t
#    int64_t

def rank_1d(
    values: np.ndarray,  # ndarray[rank_t, ndim=1]
    labels: np.ndarray,  # const int64_t[:]
    is_datetimelike: bool = ...,
    ties_method=...,
    ascending: bool = ...,
    pct: bool = ...,
    na_option=...,
) -> np.ndarray: ...  # np.ndarray[float64_t, ndim=1]
def rank_2d(
    in_arr: np.ndarray,  # ndarray[rank_t, ndim=2]
    axis: int = ...,
    is_datetimelike: bool = ...,
    ties_method=...,
    ascending: bool = ...,
    na_option=...,
    pct: bool = ...,
) -> np.ndarray: ...  # np.ndarray[float64_t, ndim=1]
def diff_2d(
    arr: np.ndarray,  # ndarray[diff_t, ndim=2]
    out: np.ndarray,  # ndarray[out_t, ndim=2]
    periods: int,
    axis: int,
    datetimelike: bool = ...,
) -> None: ...
def ensure_platform_int(arr: object) -> np.ndarray: ...
def ensure_object(arr: object) -> np.ndarray: ...
def ensure_float64(arr: object, copy=True) -> np.ndarray: ...
def ensure_float32(arr: object, copy=True) -> np.ndarray: ...
def ensure_int8(arr: object, copy=True) -> np.ndarray: ...
def ensure_int16(arr: object, copy=True) -> np.ndarray: ...
def ensure_int32(arr: object, copy=True) -> np.ndarray: ...
def ensure_int64(arr: object, copy=True) -> np.ndarray: ...
def ensure_uint8(arr: object, copy=True) -> np.ndarray: ...
def ensure_uint16(arr: object, copy=True) -> np.ndarray: ...
def ensure_uint32(arr: object, copy=True) -> np.ndarray: ...
def ensure_uint64(arr: object, copy=True) -> np.ndarray: ...
def take_1d_int8_int8(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_1d_int8_int32(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_1d_int8_int64(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_1d_int8_float64(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_1d_int16_int16(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_1d_int16_int32(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_1d_int16_int64(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_1d_int16_float64(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_1d_int32_int32(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_1d_int32_int64(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_1d_int32_float64(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_1d_int64_int64(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_1d_int64_float64(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_1d_float32_float32(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_1d_float32_float64(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_1d_float64_float64(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_1d_object_object(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_1d_bool_bool(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_1d_bool_object(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_axis0_int8_int8(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_axis0_int8_int32(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_axis0_int8_int64(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_axis0_int8_float64(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_axis0_int16_int16(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_axis0_int16_int32(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_axis0_int16_int64(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_axis0_int16_float64(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_axis0_int32_int32(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_axis0_int32_int64(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_axis0_int32_float64(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_axis0_int64_int64(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_axis0_int64_float64(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_axis0_float32_float32(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_axis0_float32_float64(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_axis0_float64_float64(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_axis0_object_object(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_axis0_bool_bool(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_axis0_bool_object(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_axis1_int8_int8(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_axis1_int8_int32(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_axis1_int8_int64(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_axis1_int8_float64(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_axis1_int16_int16(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_axis1_int16_int32(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_axis1_int16_int64(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_axis1_int16_float64(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_axis1_int32_int32(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_axis1_int32_int64(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_axis1_int32_float64(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_axis1_int64_int64(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_axis1_int64_float64(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_axis1_float32_float32(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_axis1_float32_float64(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_axis1_float64_float64(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_axis1_object_object(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_axis1_bool_bool(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_axis1_bool_object(
    values: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_multi_int8_int8(
    values: np.ndarray, indexer, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_multi_int8_int32(
    values: np.ndarray, indexer, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_multi_int8_int64(
    values: np.ndarray, indexer, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_multi_int8_float64(
    values: np.ndarray, indexer, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_multi_int16_int16(
    values: np.ndarray, indexer, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_multi_int16_int32(
    values: np.ndarray, indexer, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_multi_int16_int64(
    values: np.ndarray, indexer, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_multi_int16_float64(
    values: np.ndarray, indexer, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_multi_int32_int32(
    values: np.ndarray, indexer, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_multi_int32_int64(
    values: np.ndarray, indexer, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_multi_int32_float64(
    values: np.ndarray, indexer, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_multi_int64_float64(
    values: np.ndarray, indexer, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_multi_float32_float32(
    values: np.ndarray, indexer, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_multi_float32_float64(
    values: np.ndarray, indexer, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_multi_float64_float64(
    values: np.ndarray, indexer, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_multi_object_object(
    values: np.ndarray, indexer, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_multi_bool_bool(
    values: np.ndarray, indexer, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_multi_bool_object(
    values: np.ndarray, indexer, out: np.ndarray, fill_value=...
) -> None: ...
def take_2d_multi_int64_int64(
    values: np.ndarray, indexer, out: np.ndarray, fill_value=...
) -> None: ...