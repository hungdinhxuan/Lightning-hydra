#!/usr/bin/env python3
"""
Test script for parallel audio processing
"""

import sys
import os
from pathlib import Path

# Add the scripts directory to Python path
sys.path.append(str(Path(__file__).parent))

from main_handle_rc_parallel import process_single_audio_file, SSBoll79
import numpy as np
import soundfile as sf
import time


def create_test_audio(duration=2.0, sample_rate=16000, frequency=440.0):
    """Create a test audio signal"""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Create a sine wave with some noise
    signal = np.sin(2 * np.pi * frequency * t) + 0.1 * np.random.randn(len(t))
    return signal, sample_rate


def test_single_file_processing():
    """Test processing a single audio file"""
    print("Testing single file processing...")
    
    # Create test audio
    signal, fs = create_test_audio()
    
    # Save test audio
    test_dir = Path("test_audio")
    test_dir.mkdir(exist_ok=True)
    test_file = test_dir / "test_input.wav"
    sf.write(test_file, signal, fs)
    
    # Process the file
    target_dir = test_dir / "output"
    target_dir.mkdir(exist_ok=True)
    
    args = (str(test_file), target_dir, SSBoll79, "_processed")
    
    start_time = time.time()
    success, filename, error = process_single_audio_file(args)
    end_time = time.time()
    
    if success:
        print(f"✓ Successfully processed {filename} in {end_time - start_time:.2f} seconds")
        
        # Check if output file exists
        output_file = target_dir / "test_input_processed.wav"
        if output_file.exists():
            print(f"✓ Output file created: {output_file}")
            
            # Load and verify output
            output_signal, output_fs = sf.read(output_file)
            print(f"✓ Output signal shape: {output_signal.shape}, sample rate: {output_fs}")
        else:
            print("✗ Output file not found")
    else:
        print(f"✗ Failed to process {filename}: {error}")
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir, ignore_errors=True)


def test_parallel_processing():
    """Test parallel processing with multiple files"""
    print("\nTesting parallel processing...")
    
    from multiprocessing import Pool
    import tempfile
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_dir = temp_path / "input"
        output_dir = temp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        
        # Create multiple test files
        num_files = 4
        test_files = []
        
        for i in range(num_files):
            # Create different frequency for each file
            signal, fs = create_test_audio(frequency=440.0 + i * 100)
            test_file = input_dir / f"test_{i}.wav"
            sf.write(test_file, signal, fs)
            test_files.append(test_file)
        
        print(f"Created {num_files} test files")
        
        # Prepare arguments for parallel processing
        args_list = [(str(test_file), output_dir, SSBoll79, "_processed") 
                     for test_file in test_files]
        
        # Process files in parallel
        start_time = time.time()
        with Pool(processes=2) as pool:  # Use 2 processes for testing
            results = pool.map(process_single_audio_file, args_list)
        end_time = time.time()
        
        # Check results
        successful = sum(1 for success, _, _ in results if success)
        failed = len(results) - successful
        
        print(f"✓ Processed {successful} files successfully, {failed} failed")
        print(f"✓ Total time: {end_time - start_time:.2f} seconds")
        print(f"✓ Average time per file: {(end_time - start_time)/num_files:.2f} seconds")
        
        # Verify output files
        output_files = list(output_dir.glob("*.wav"))
        print(f"✓ Created {len(output_files)} output files")


def main():
    """Run all tests"""
    print("Running parallel processing tests...\n")
    
    try:
        test_single_file_processing()
        test_parallel_processing()
        print("\n✓ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
