"""__init__ file for inference module."""

from .predict import SpeakerDatabase, SpeakerVerificationInference

__all__ = [
    "SpeakerVerificationInference",
    "SpeakerDatabase",
]
