import numpy as np

def noise_permutations_from_deformation(deformation: np.ndarray) -> np.ndarray:
    r"""
    Given a deformation matrix, return the corresponding noise model permutation.

    Args:
        deformation (np.ndarray): A 2D numpy array representing the deformation matrix.
    """
    deformation_to_noise_model_permutation = np.array([
    #    I  X  Y  Z
        [0, 1, 2, 3],  # I
        [0, 2, 1, 3],  # X-Y
        [0, 1, 3, 2],  # Y-Z
        [0, 3, 2, 1],  # X-Z
        [0, 3, 1, 2],  # X-Z-Y
        [0, 2, 3, 1],  # Y-Z-X
    ])
    return deformation_to_noise_model_permutation[deformation]