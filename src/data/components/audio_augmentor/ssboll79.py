from .base import BaseAugmentor
from .utils import librosa_to_pydub
import numpy as np
from scipy.signal.windows import hamming
from scipy.fft import fft, ifft

import logging
logger = logging.getLogger(__name__)


class SSBoll79Augmentor(BaseAugmentor):
    """
    Spectral Subtraction based on Boll 79. Amplitude spectral subtraction 
    Includes Magnitude Averaging and Residual noise Reduction
    
    This augmentation applies spectral subtraction denoising to the audio signal.
    It's based on the Boll 1979 paper for noise reduction in speech signals.
    
    Config:
    initial_silence: float, initial silence (noise only) length in seconds (default: 0.25)
    """
    
    def __init__(self, config: dict):
        """
        Initialize the SSBoll79 spectral subtraction augmentor.
        
        Config parameters:
        - initial_silence: float, initial silence length in seconds for noise estimation (default: 0.25)
        """
        super().__init__(config)
        self.initial_silence = config.get("initial_silence", 0.25)
        
    def transform(self):
        """
        Apply spectral subtraction denoising to the audio signal.
        """
        # Apply SSBoll79 spectral subtraction
        denoised_data = self._ssboll79(self.data, self.sr, self.initial_silence)
        
        # Convert to pydub for saving (as required by base class)
        self.augmented_audio = librosa_to_pydub(denoised_data, sr=self.sr)
    
    def _ssboll79(self, signal, fs, IS=0.25):
        """
        Spectral Subtraction based on Boll 79. Amplitude spectral subtraction 
        Includes Magnitude Averaging and Residual noise Reduction
        
        Parameters:
        -----------
        signal : array-like
            The noisy signal
        fs : int
            Sampling frequency
        IS : float, optional
            Initial silence (noise only) length in seconds (default: 0.25 sec)
        
        Returns:
        --------
        output : ndarray
            Denoised signal
        """
        W = int(0.025 * fs)  # Window length is 25 ms
        nfft = W
        SP = 0.4  # Shift percentage is 40% (10ms)
        wnd = hamming(W)
        
        NIS = int((IS * fs - W) / (SP * W) + 1)  # Number of initial silence segments
        Gamma = 1  # Magnitude Power (1 for magnitude, 2 for power spectrum)
        
        y = self._segment(signal, W, SP, wnd)
        Y = fft(y, nfft, axis=0)
        YPhase = np.angle(Y[:int(Y.shape[0]/2)+1, :])  # Noisy Speech Phase
        Y = np.abs(Y[:int(Y.shape[0]/2)+1, :]) ** Gamma  # Spectrogram
        
        numberOfFrames = Y.shape[1]
        FreqResol = Y.shape[0]
        
        N = np.mean(Y[:, :NIS], axis=1)  # Initial Noise Power Spectrum mean
        NRM = np.zeros_like(N)  # Noise Residual Maximum
        NoiseCounter = 0
        NoiseLength = 9  # Smoothing factor for noise updating
        Beta = 0.03
        
        # Magnitude Averaging
        YS = Y.copy()
        for i in range(1, numberOfFrames - 1):
            YS[:, i] = (Y[:, i-1] + Y[:, i] + Y[:, i+1]) / 3
        
        X = np.zeros_like(Y)
        
        for i in range(numberOfFrames):
            NoiseFlag, SpeechFlag, NoiseCounter, Dist = self._vad(
                Y[:, i] ** (1/Gamma), 
                N ** (1/Gamma), 
                NoiseCounter
            )
            
            if SpeechFlag == 0:
                N = (NoiseLength * N + Y[:, i]) / (NoiseLength + 1)  # Update noise
                NRM = np.maximum(NRM, YS[:, i] - N)  # Update Maximum Noise Residue
                X[:, i] = Beta * Y[:, i]
            else:
                D = YS[:, i] - N  # Spectral Subtraction
                
                if i > 0 and i < numberOfFrames - 1:  # Residual Noise Reduction
                    for j in range(len(D)):
                        if D[j] < NRM[j]:
                            D[j] = min([D[j], YS[j, i-1] - N[j], YS[j, i+1] - N[j]])
                
                X[:, i] = np.maximum(D, 0)
        
        output = self._overlap_add2(X ** (1/Gamma), YPhase, W, int(SP * W))
        output = output / np.max(np.abs(output))
        
        return output

    def _overlap_add2(self, XNEW, yphase, windowLen, ShiftLen):
        """
        Reconstruct signal from spectrogram using overlap-add method
        
        Parameters:
        -----------
        XNEW : ndarray
            Magnitude spectrogram
        yphase : ndarray
            Phase information
        windowLen : int
            Window length
        ShiftLen : int
            Shift length between frames
        
        Returns:
        --------
        ReconstructedSignal : ndarray
            Reconstructed time-domain signal
        """
        ShiftLen = int(ShiftLen)
        FreqRes, FrameNum = XNEW.shape
        
        Spec = XNEW * np.exp(1j * yphase)
        
        # Mirror spectrum for real signal
        if windowLen % 2:  # if windowLen is odd
            Spec = np.vstack([Spec, np.flipud(np.conj(Spec[1:, :]))])
        else:
            Spec = np.vstack([Spec, np.flipud(np.conj(Spec[1:-1, :]))])
        
        sig = np.zeros((FrameNum - 1) * ShiftLen + windowLen)
        
        for i in range(FrameNum):
            start = i * ShiftLen
            spec = Spec[:, i]
            sig[start:start + windowLen] += np.real(ifft(spec, windowLen))
        
        return sig

    def _vad(self, signal, noise, NoiseCounter, NoiseMargin=3, Hangover=8):
        """
        Spectral Distance Voice Activity Detector
        
        Parameters:
        -----------
        signal : ndarray
            Current frame's magnitude spectrum
        noise : ndarray
            Noise magnitude spectrum template
        NoiseCounter : int
            Number of immediate previous noise frames
        NoiseMargin : float, optional
            Spectral distance threshold (default: 3)
        Hangover : int, optional
            Number of noise segments after which SpeechFlag resets (default: 8)
        
        Returns:
        --------
        NoiseFlag : int
            1 if segment is noise, 0 otherwise
        SpeechFlag : int
            1 if segment is speech, 0 otherwise
        NoiseCounter : int
            Updated noise counter
        Dist : float
            Spectral distance
        """
        FreqResol = len(signal)
        
        # Avoid log of zero
        signal = np.maximum(signal, 1e-10)
        noise = np.maximum(noise, 1e-10)
        
        SpectralDist = 20 * (np.log10(signal) - np.log10(noise))
        SpectralDist[SpectralDist < 0] = 0
        Dist = np.mean(SpectralDist)
        
        if Dist < NoiseMargin:
            NoiseFlag = 1
            NoiseCounter = NoiseCounter + 1
        else:
            NoiseFlag = 0
            NoiseCounter = 0
        
        # Detect noise only periods
        if NoiseCounter > Hangover:
            SpeechFlag = 0
        else:
            SpeechFlag = 1
        
        return NoiseFlag, SpeechFlag, NoiseCounter, Dist

    def _segment(self, signal, W=256, SP=0.4, Window=None):
        """
        Segment signal into overlapping windowed segments
        
        Parameters:
        -----------
        signal : array-like
            Input signal
        W : int, optional
            Number of samples per window (default: 256)
        SP : float, optional
            Shift percentage (default: 0.4)
        Window : array-like, optional
            Window function (default: Hamming window)
        
        Returns:
        --------
        Seg : ndarray
            Matrix where columns are windowed segments
        """
        if Window is None:
            Window = hamming(W)
        
        Window = Window.reshape(-1, 1)  # Make it a column vector
        L = len(signal)
        SP = int(W * SP)
        N = int((L - W) / SP + 1)  # Number of segments
        
        # Create index matrix
        Index = np.tile(np.arange(W), (N, 1)) + np.tile(np.arange(N).reshape(-1, 1) * SP, (1, W))
        Index = Index.T
        
        hw = np.tile(Window, (1, N))
        Seg = signal[Index] * hw
        
        return Seg
