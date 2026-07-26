"""Backward-compatible wrapper for the misspelled segmentation module.

New code should import from ``Project.ssl_pipeline.segmentation``.
"""

from Project.ssl_pipeline.segmentation import segment_audio

__all__ = ["segment_audio"]
