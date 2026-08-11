import numpy as np


def apply_legacy_numpy_aliases() -> None:
    np.bool = np.bool_
    np.int = np.int_
    np.float = np.float64
    np.str = np.str_
    np.complex = np.complex128
    np.object = np.object_
    np.unicode = np.str_
