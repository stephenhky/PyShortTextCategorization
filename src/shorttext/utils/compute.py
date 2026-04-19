
from typing import Annotated

import numpy as np
import numpy.typing as npt
import numba as nb


@nb.njit(nb.float64(nb.float64[::1], nb.float64[::1]))
def cosine_similarity(
        vec1: Annotated[npt.NDArray[np.float64], "1D array"],
        vec2: Annotated[npt.NDArray[np.float64], "1D array"]
) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        vec1: First vector.
        vec2: Second vector.

    Returns:
        Cosine similarity score between 0 and 1.
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
