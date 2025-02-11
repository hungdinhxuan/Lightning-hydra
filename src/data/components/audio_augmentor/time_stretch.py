from .base import BaseAugmentor
from audiomentations import TimeStretch
import logging
from .utils import librosa_to_pydub
logger = logging.getLogger(__name__)


class TimeStretchAugmentor(BaseAugmentor):
    def __init__(self, config: dict):
        """
        Time stretch augmentor class requires these config:
        min_factor: float, min factor
        max_factor: float, max factor

        """
        super().__init__(config)
        self.my_transform = TimeStretch(
            min_rate=config["min_factor"],
            max_rate=config["max_factor"],
            leave_length_unchanged=True,
            p=1.0
        )
        self.audio_data = None

    def load(self, input_path: str):
        """
        :param input_path: path to the input audio file
        """
        # load with librosa
        super().load(input_path)
        self.audio_data = self.data

    def transform(self):
        """
        Time stretch the audio using librosa time stretch method
        """
        self.augmented_audio = self.my_transform(
            samples=self.audio_data, sample_rate=self.sr)
