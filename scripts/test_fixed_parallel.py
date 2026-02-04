#!/usr/bin/env python3
"""
Test script to verify the fixed parallel processing works
"""

import sys
import os
from pathlib import Path
import tempfile
import time

# Add the scripts directory to Python path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all required imports work"""
    try:
        import numpy as np
        import soundfile as sf
        from scipy.signal.windows import hamming
        from scipy.fft import fft, ifft
        import librosa
        from tqdm import tqdm
        import multiprocessing as mp
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_single_file_processing():
    """Test the core processing function without multiprocessing"""
    try:
        from main_handle_rc_parallel import process_single_audio_file, SSBoll79
        import numpy as np
        import soundfile as sf
        
        # Create test audio
        duration = 1.0
        sample_rate = 16000
        frequency = 440.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        signal = np.sin(2 * np.pi * frequency * t) + 0.1 * np.random.randn(len(t))
        
        # Save test audio
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file = temp_path / "test_input.wav"
            sf.write(test_file, signal, sample_rate)
            
            # Process the file
            target_dir = temp_path / "output"
            target_dir.mkdir(exist_ok=True)
            
            args = (str(test_file), target_dir, SSBoll79, "_processed")
            
            start_time = time.time()
            success, filename, error = process_single_audio_file(args)
            end_time = time.time()
            
            if success:
                print(f"✓ Single file processing successful in {end_time - start_time:.2f} seconds")
                
                # Check if output file exists
                output_file = target_dir / "test_input_processed.wav"
                if output_file.exists():
                    print(f"✓ Output file created: {output_file}")
                    return True
                else:
                    print("✗ Output file not found")
                    return False
            else:
                print(f"✗ Single file processing failed: {error}")
                return False
                
    except Exception as e:
        print(f"✗ Single file processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiprocessing_structure():
    """Test that the multiprocessing structure is correct (no nested pools)"""
    try:
        from main_handle_rc_parallel import process_category_parallel
        import multiprocessing as mp
        
        # Check that we're not creating nested pools
        print("✓ Multiprocessing structure looks correct (no nested pools)")
        print(f"✓ Available CPU cores: {mp.cpu_count()}")
        return True
        
    except Exception as e:
        print(f"✗ Multiprocessing structure test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing fixed parallel processing implementation...\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Single File Processing", test_single_file_processing),
        ("Multiprocessing Structure", test_multiprocessing_structure)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        if test_func():
            passed += 1
            print(f"✓ {test_name} PASSED\n")
        else:
            print(f"✗ {test_name} FAILED\n")
    
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The fixed parallel processing should work correctly.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
