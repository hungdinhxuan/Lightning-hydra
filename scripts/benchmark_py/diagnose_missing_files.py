#!/usr/bin/env python3
"""
Diagnose Missing Files - Find why files couldn't be processed

This script analyzes missing files and logs detailed information about why they failed.

Usage:
    python diagnose_missing_files.py \
        --score-file <path_to_score_file> \
        --protocol-file <path_to_protocol_file> \
        --data-dir <path_to_data_directory> \
        --subset <protocol_subset> \
        --output <output_log_file>

Example:
    python scripts/benchmark_py/diagnose_missing_files.py \
        --score-file logs/results/dataset_score.txt \
        --protocol-file data/dataset/protocol.txt \
        --data-dir data/dataset \
        --subset dev \
        --output missing_files_diagnosis.log
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import wave
import subprocess

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark_py.protocol import parse_protocol_line
from benchmark_py.scores import parse_score_line


def read_protocol_ids(protocol_file, subset=None):
    """Read protocol file and extract file IDs with subset filter"""
    entries = []
    
    with open(protocol_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            parsed = parse_protocol_line(line)
            if not parsed:
                continue
            
            file_id, file_subset, label = parsed
            
            # Filter by subset if specified
            if subset and file_subset != subset:
                continue
            
            entries.append({
                'line_num': line_num,
                'file_id': file_id,
                'subset': file_subset,
                'label': label
            })
    
    return entries


def read_score_ids(score_file):
    """Read score file and extract file IDs"""
    score_ids = set()
    
    with open(score_file, 'r') as f:
        for line in f:
            parsed = parse_score_line(line)
            if parsed:
                file_id, _, _, _ = parsed
                score_ids.add(file_id)
    
    return score_ids


def check_file_exists(data_dir, file_id):
    """Check if file exists"""
    file_path = Path(data_dir) / file_id
    return file_path.exists(), file_path


def check_audio_format(file_path):
    """Check audio file format and properties"""
    if not file_path.exists():
        return None, "File not found"
    
    # Try to get file info using file command
    try:
        result = subprocess.run(
            ['file', str(file_path)],
            capture_output=True,
            text=True,
            timeout=5
        )
        file_info = result.stdout.strip()
    except Exception as e:
        file_info = f"Could not get file info: {e}"
    
    # Try to check if it's a valid WAV file
    audio_info = {}
    try:
        with wave.open(str(file_path), 'rb') as wav_file:
            audio_info = {
                'channels': wav_file.getnchannels(),
                'sample_width': wav_file.getsampwidth(),
                'framerate': wav_file.getframerate(),
                'frames': wav_file.getnframes(),
                'duration': wav_file.getnframes() / wav_file.getframerate()
            }
        return audio_info, file_info
    except Exception as e:
        return None, f"{file_info} | Error: {e}"


def diagnose_missing_files(score_file, protocol_file, data_dir, subset, output_log):
    """Main diagnosis function"""
    
    print("=" * 80)
    print("MISSING FILES DIAGNOSIS")
    print("=" * 80)
    print()
    
    # Read protocol entries
    print(f"Reading protocol file: {protocol_file}")
    protocol_entries = read_protocol_ids(protocol_file, subset)
    print(f"  Found {len(protocol_entries)} entries (subset: {subset if subset else 'all'})")
    
    # Read score IDs
    print(f"Reading score file: {score_file}")
    score_ids = read_score_ids(score_file)
    print(f"  Found {len(score_ids)} scored entries")
    
    # Find missing
    missing_entries = []
    for entry in protocol_entries:
        if entry['file_id'] not in score_ids:
            missing_entries.append(entry)
    
    print(f"\nMissing: {len(missing_entries)} files ({len(missing_entries)/len(protocol_entries)*100:.2f}%)")
    print()
    
    # Categorize missing files
    categories = {
        'file_not_found': [],
        'corrupted': [],
        'format_issue': [],
        'unknown': []
    }
    
    # Open log file
    log_file = open(output_log, 'w')
    
    # Write header
    log_file.write("=" * 80 + "\n")
    log_file.write("MISSING FILES DIAGNOSIS REPORT\n")
    log_file.write("=" * 80 + "\n")
    log_file.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"Protocol file: {protocol_file}\n")
    log_file.write(f"Score file: {score_file}\n")
    log_file.write(f"Data directory: {data_dir}\n")
    log_file.write(f"Protocol subset: {subset if subset else 'all'}\n")
    log_file.write(f"\nTotal protocol entries: {len(protocol_entries)}\n")
    log_file.write(f"Scored entries: {len(score_ids)}\n")
    log_file.write(f"Missing entries: {len(missing_entries)} ({len(missing_entries)/len(protocol_entries)*100:.2f}%)\n")
    log_file.write("\n" + "=" * 80 + "\n\n")
    
    # Analyze each missing file
    print("Analyzing missing files...")
    for i, entry in enumerate(missing_entries, 1):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(missing_entries)} ({i/len(missing_entries)*100:.1f}%)", end='\r')
        
        file_id = entry['file_id']
        line_num = entry['line_num']
        subset_val = entry['subset']
        label = entry['label']
        
        # Check if file exists
        exists, file_path = check_file_exists(data_dir, file_id)
        
        if not exists:
            categories['file_not_found'].append(entry)
            log_file.write(f"[FILE_NOT_FOUND] Line {line_num}\n")
            log_file.write(f"  File ID: {file_id}\n")
            log_file.write(f"  Subset: {subset_val}, Label: {label}\n")
            log_file.write(f"  Expected path: {file_path}\n")
            log_file.write("\n")
        else:
            # File exists, check format
            audio_info, error_msg = check_audio_format(file_path)
            
            if audio_info is not None:
                # File is valid but wasn't processed for some reason
                categories['unknown'].append(entry)
                log_file.write(f"[UNKNOWN] Line {line_num}\n")
                log_file.write(f"  File ID: {file_id}\n")
                log_file.write(f"  Subset: {subset_val}, Label: {label}\n")
                log_file.write(f"  File exists: {file_path}\n")
                log_file.write(f"  Audio info: {audio_info}\n")
                log_file.write(f"  Note: File is valid but wasn't processed (check benchmark logs)\n")
                log_file.write("\n")
            else:
                # Check if it's corrupted or format issue
                if 'Audio file' in error_msg or 'RIFF' in error_msg or 'WAVE' in error_msg:
                    categories['format_issue'].append(entry)
                    log_file.write(f"[FORMAT_ISSUE] Line {line_num}\n")
                else:
                    categories['corrupted'].append(entry)
                    log_file.write(f"[CORRUPTED] Line {line_num}\n")
                
                log_file.write(f"  File ID: {file_id}\n")
                log_file.write(f"  Subset: {subset_val}, Label: {label}\n")
                log_file.write(f"  File path: {file_path}\n")
                log_file.write(f"  File size: {file_path.stat().st_size if file_path.exists() else 'N/A'} bytes\n")
                log_file.write(f"  Error: {error_msg}\n")
                log_file.write("\n")
    
    print()  # Clear progress line
    
    # Write summary
    log_file.write("=" * 80 + "\n")
    log_file.write("SUMMARY BY CATEGORY\n")
    log_file.write("=" * 80 + "\n\n")
    
    for category, entries in categories.items():
        log_file.write(f"{category.upper()}: {len(entries)} files ({len(entries)/len(missing_entries)*100:.1f}%)\n")
    
    log_file.write("\n")
    
    # Write detailed breakdown
    log_file.write("=" * 80 + "\n")
    log_file.write("DETAILED BREAKDOWN\n")
    log_file.write("=" * 80 + "\n\n")
    
    for category, entries in categories.items():
        if not entries:
            continue
        
        log_file.write(f"\n{category.upper()} ({len(entries)} files):\n")
        log_file.write("-" * 80 + "\n")
        
        # Show first 20 examples
        for entry in entries[:20]:
            log_file.write(f"  Line {entry['line_num']}: {entry['file_id']}\n")
        
        if len(entries) > 20:
            log_file.write(f"  ... and {len(entries) - 20} more\n")
        
        log_file.write("\n")
    
    # Write recommendations
    log_file.write("=" * 80 + "\n")
    log_file.write("RECOMMENDATIONS\n")
    log_file.write("=" * 80 + "\n\n")
    
    if categories['file_not_found']:
        log_file.write(f"1. FILE_NOT_FOUND ({len(categories['file_not_found'])} files):\n")
        log_file.write("   - Check if data directory is correct\n")
        log_file.write("   - Check if files were moved or deleted\n")
        log_file.write("   - Consider removing these entries from protocol\n\n")
    
    if categories['corrupted']:
        log_file.write(f"2. CORRUPTED ({len(categories['corrupted'])} files):\n")
        log_file.write("   - Re-download these files if possible\n")
        log_file.write("   - Remove from protocol if cannot fix\n")
        log_file.write("   - Check disk for errors\n\n")
    
    if categories['format_issue']:
        log_file.write(f"3. FORMAT_ISSUE ({len(categories['format_issue'])} files):\n")
        log_file.write("   - Convert to supported format (WAV recommended)\n")
        log_file.write("   - Check sample rate and bit depth\n")
        log_file.write("   - Use ffmpeg to convert if needed\n\n")
    
    if categories['unknown']:
        log_file.write(f"4. UNKNOWN ({len(categories['unknown'])} files):\n")
        log_file.write("   - Files are valid but weren't processed\n")
        log_file.write("   - Check benchmark logs for errors\n")
        log_file.write("   - May be OOM errors or timeout issues\n")
        log_file.write("   - Try re-running with these specific files\n\n")
    
    if len(missing_entries) / len(protocol_entries) < 0.05:  # < 5%
        log_file.write(f"5. OVERALL ({len(missing_entries)} missing, {len(missing_entries)/len(protocol_entries)*100:.1f}%):\n")
        log_file.write("   ✓ Missing rate is acceptable (< 5%)\n")
        log_file.write("   ✓ Can proceed with partial results\n")
        log_file.write("   ✓ Document missing rate in your report\n\n")
    
    log_file.close()
    
    # Print summary to console
    print("\n" + "=" * 80)
    print("DIAGNOSIS SUMMARY")
    print("=" * 80)
    print()
    print(f"Total missing: {len(missing_entries)} files ({len(missing_entries)/len(protocol_entries)*100:.2f}%)")
    print()
    
    for category, entries in categories.items():
        if entries:
            print(f"  {category.upper()}: {len(entries)} files ({len(entries)/len(missing_entries)*100:.1f}%)")
    
    print()
    print(f"✓ Detailed diagnosis saved to: {output_log}")
    print()
    print("=" * 80)
    
    return categories


def main():
    parser = argparse.ArgumentParser(
        description='Diagnose why files are missing from score file'
    )
    parser.add_argument('--score-file', required=True, help='Path to score file')
    parser.add_argument('--protocol-file', required=True, help='Path to protocol file')
    parser.add_argument('--data-dir', required=True, help='Base data directory')
    parser.add_argument('--subset', default=None, help='Protocol subset (e.g., dev, eval)')
    parser.add_argument('--output', default='missing_files_diagnosis.log', 
                       help='Output log file (default: missing_files_diagnosis.log)')
    
    args = parser.parse_args()
    
    # Run diagnosis
    categories = diagnose_missing_files(
        args.score_file,
        args.protocol_file,
        args.data_dir,
        args.subset,
        args.output
    )
    
    # Return exit code based on results
    if not categories['file_not_found'] and not categories['corrupted']:
        # Only unknown issues - might be transient
        sys.exit(0)
    else:
        # Has file not found or corrupted - needs attention
        sys.exit(1)


if __name__ == "__main__":
    main()
