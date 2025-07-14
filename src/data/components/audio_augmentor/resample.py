from .base import BaseAugmentor
from audiomentations import Resample
import logging
from .utils import recursive_list_files, librosa_to_pydub
logger = logging.getLogger(__name__)


class ResampleAugmentor(BaseAugmentor):
    def __init__(self, config: dict):
        """
        Time stretch augmentor class requires these config:
        min_sample_rate: float, min factor
        max_sample_rate: float, max factor

        """
        super().__init__(config)
        self.my_transform = Resample(
            min_sample_rate=config["min_sample_rate"],
            max_sample_rate=config["max_sample_rate"],
            p=1.0
        )

    def transform(self):
        """
        Time stretch the audio using librosa time stretch method
        """
        self.augmented_audio = self.my_transform(
            samples=self.audio_data, sample_rate=self.sr)
        self.augmented_audio = librosa_to_pydub(self.augmented_audio)
