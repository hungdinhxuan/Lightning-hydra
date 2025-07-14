from .base import BaseAugmentor
from audiomentations import HighPassFilter
import logging
from .utils import recursive_list_files, librosa_to_pydub
logger = logging.getLogger(__name__)


class HighPassFilterAugmentor(BaseAugmentor):
    def __init__(self, config: dict):
        """
        Time stretch augmentor class requires these config:
        min_cutoff_freq: float, min factor
        max_cutoff_freq: float, max factor

        """
        super().__init__(config)
        self.my_transform = HighPassFilter(
            min_cutoff_freq=config["min_cutoff_freq"],
            max_cutoff_freq=config["max_cutoff_freq"],
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
        self.augmented_audio = librosa_to_pydub(self.augmented_audio)
