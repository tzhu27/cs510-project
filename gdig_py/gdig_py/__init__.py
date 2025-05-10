"""
G-DIG: Gradient-based Diverse and high-quality Instruction data selection
"""

__version__ = "0.1.0"

from .config import GDIGConfig
from .pipeline import GDIGPipeline
from . import nngeometry
from . import dataset
from . import utils

__all__ = [
    'GDIGConfig',
    'GDIGPipeline',
    'nngeometry',
    'dataset',
    'utils',
] 