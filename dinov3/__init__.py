"""
DINOv3 Shadow Detection Package

A complete implementation of DINOv3-based shadow detection for overhead imagery.
"""

from .dinov3_backbone import DINOv3Backbone
from .dinov3_decoder import DINOv3Decoder
from .dinov3_model import DINOv3ShadowDetector

__version__ = '1.0.0'
__author__ = 'Anonymous'

__all__ = [
    'DINOv3Backbone',
    'DINOv3Decoder',
    'DINOv3ShadowDetector',
]