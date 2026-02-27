"""
Models package for AI-generated image detection.
"""

from .backbones import SimpleCNN, get_resnet18, get_resnet50

__all__ = ['SimpleCNN', 'get_resnet18', 'get_resnet50']
