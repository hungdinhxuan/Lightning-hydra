import logging.config
from .background_noise import BackgroundNoiseAugmentor
from .pitch import PitchAugmentor
from .reverb import ReverbAugmentor
from .speed import SpeedAugmentor
from .volume import VolumeAugmentor
from .telephone import TelephoneEncodingAugmentor
from .gaussian import GaussianAugmentor
from .copy_paste import CopyPasteAugmentor
from .base import BaseAugmentor
from .time_masking import TimeMaskingAugmentor
from .freq_masking import FrequencyMaskingAugmentor
from .masking import MaskingAugmentor
from .time_swap import TimeSwapAugmentor
from .freq_swap import FrequencySwapAugmentor
from .swapping import SwappingAugmentor
from .linear_filter import LinearFilterAugmentor
from .bandpass import BandpassAugmentor
from .time_stretch import TimeStretchAugmentor
from .highpass_filter import HighPassFilterAugmentor
from .autotune import AutoTuneAugmentor
from .echo import EchoAugmentor
from .amplitude_modulation import AmplitudeModulationAugmentor
from .gaussian_v1 import GaussianAugmentor as GaussianAugmentorV1
from .background_music_deepen import BackgroundMusicAugmentor as BackgroundMusicAugmentorDeepen
from .background_noise_deepen import BackgroundNoiseAugmentor as BackgroundNoiseAugmentorDeepen
from .freq_operation_deepen import FrequencyOperationAugmentor as FrequencyOperationAugmentorDeepen
from .lowpass_filter import LowPassFilterAugmentor
from .resample import ResampleAugmentor
from .background_noise_audiomentations import BackgroundNoiseAugmentor as BackgroundNoiseAugmentorAudiomentations
from .echo_deepen import EchoAugmentor as EchoAugmentorDeepen
# from . import utils

from .__version__ import (

    __author__,
    __author_email__,
    __description__,
    __license__,
    __title__,
    __url__,
    __version__,
)

SUPPORTED_AUGMENTORS = ['background_noise', 'pitch', 'speed', 'volume', 'reverb', 'telephone', 'gaussian_noise',
                        'time_masking', 'freq_masking', 'masking', 'time_swap', 'freq_swap', 'swapping', 'linear_filter', 'bandpass',
                        'time_stretch', 'highpass_filter', 'autotune', 'echo', 'amplitude_modulation', 'copy_paste']

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "std": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M",
        }
    },
    "handlers": {
        "default": {
            "class": "logging.NullHandler",
        },
        "test": {
            "class": "logging.StreamHandler",
            "formatter": "std",
            "level": logging.INFO,
        },
    },
    "loggers": {
        "art": {"handlers": ["default"]},
        "tests": {"handlers": ["test"], "level": "INFO", "propagate": True},
    },
}
logging.config.dictConfig(LOGGING)
logger = logging.getLogger(__name__)
