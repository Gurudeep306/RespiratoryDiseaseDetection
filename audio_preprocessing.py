"""
Audio preprocessing module for respiratory disease detection.
Handles silence removal and audio chunking.
"""

import os
import numpy as np
import librosa
from pydub import AudioSegment
from pydub.silence import split_on_silence
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class AudioPreprocessor:
    """Preprocesses audio files by removing silence and chunking."""
    
    def __init__(self, min_silence_len=500, silence_thresh=-40, num_chunks=50):
        """
        Initialize audio preprocessor.
        
        Args:
            min_silence_len (int): Minimum silence duration in ms
            silence_thresh (int): Silence threshold in dB
            num_chunks (int): Number of chunks to divide audio into
        """
        self.min_silence_len = min_silence_len
        self.silence_thresh = silence_thresh
        self.num_chunks = num_chunks
    
    def remove_silence(self, audio_path, output_path=None):
        """
        Remove silence from audio file.
        
        Args:
            audio_path (str): Path to input audio file
            output_path (str): Path to save output (optional)
            
        Returns:
            AudioSegment: Audio with silence removed
        """
        try:
            audio = AudioSegment.from_wav(audio_path)
        except:
            audio = AudioSegment.from_file(audio_path)
        
        chunks = split_on_silence(
            audio,
            min_silence_len=self.min_silence_len,
            silence_thresh=self.silence_thresh
        )
        
        non_silent_audio = AudioSegment.empty()
        for chunk in chunks:
            non_silent_audio += chunk
        
        if output_path:
            non_silent_audio.export(output_path, format="wav")
        
        return non_silent_audio
    
    def chunk_audio(self, audio_path, output_dir=None):
        """
        Chunk audio into equal segments.
        
        Args:
            audio_path (str): Path to audio file
            output_dir (str): Directory to save chunks (optional)
            
        Returns:
            list: List of audio chunks
        """
        try:
            audio = AudioSegment.from_wav(audio_path)
        except:
            audio = AudioSegment.from_file(audio_path)
        
        # Remove silence first
        chunks = split_on_silence(audio, min_silence_len=self.min_silence_len,
                                  silence_thresh=self.silence_thresh)
        non_silent_audio = AudioSegment.empty()
        for chunk in chunks:
            non_silent_audio += chunk
        
        # Divide into equal segments
        total_duration = len(non_silent_audio)
        segment_duration = total_duration / self.num_chunks
        timestamps = [int(i * segment_duration) for i in range(self.num_chunks + 1)]
        
        segments = []
        for i in range(len(timestamps) - 1):
            start_time = timestamps[i]
            end_time = timestamps[i + 1]
            segment = non_silent_audio[start_time:end_time]
            segments.append(segment)
            
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"segment_{i}.wav")
                segment.export(output_path, format="wav")
        
        return segments
    
    def batch_process(self, input_dir, output_base_dir, remove_silence_only=False):
        """
        Process all audio files in a directory.
        
        Args:
            input_dir (str): Directory containing audio files
            output_base_dir (str): Base directory for outputs
            remove_silence_only (bool): Only remove silence without chunking
        """
        audio_files = list(Path(input_dir).glob("*.wav")) + \
                     list(Path(input_dir).glob("*.mp3"))
        
        for audio_file in audio_files:
            filename = audio_file.stem
            if remove_silence_only:
                output_path = os.path.join(output_base_dir, f"{filename}_no_silence.wav")
                self.remove_silence(str(audio_file), output_path)
            else:
                output_dir = os.path.join(output_base_dir, filename)
                self.chunk_audio(str(audio_file), output_dir)
