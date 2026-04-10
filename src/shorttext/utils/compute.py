
from typing import Annotated

import numpy as np
import numpy.typing as npt
import numba as nb


@nb.njit(nb.float32(nb.float32[:], nb.float32[:]))
def cosine_similarity(
        vec1: Annotated[npt.NDArray[np.float64], "1D array"],
        vec2: Annotated[npt.NDArray[np.float64], "1D array"]
) -> float:
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
