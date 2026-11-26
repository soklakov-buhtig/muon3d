import numpy as np
from scipy.optimize import minimize

from constants import (
    M_E_MEV,
    R_E_COMPTON_FM,
    R_CORE_VAC_FM,
)
from preprocessor import build_3d_grid, init_hopf_vortex_3d
from solution import relax_vortex_3d
from postprocessor import analyze_relaxed_vortex_3d

DEFAULT_TOPO_SEQUENCE = [
    (1, 2),
    (2, 3),
    (3, 4),
    (2, 5),
    (3, 5),
    (5, 8),
    (37, 73),
]
