#!/usr/bin/env python3
"""
File transfer script for WildSpoof dataset.
Reads protocol.txt and copies files to destination folder with parallel processing.
"""

import os
import shutil
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional
import logging

try:
    from rich.progress import Progress, TaskID, BarColumn, TextColumn, TimeRemainingColumn
    from rich.console import Console
    from rich.logging import RichHandler
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: Rich library not available. Install with: pip install rich")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()] if RICH_AVAILABLE else [logging.StreamHandler()]
)
logger = logging.getLogger("transfer_file")


class FileTransfer:
    def __init__(self, source_root: str, destination_root: str, max_workers: int = 8):
        self.source_root = Path(source_root)
        self.destination_root = Path(destination_root)
        self.max_workers = max_workers
        self.console = Console() if RICH_AVAILABLE else None
        
        # Create destination directory if it doesn't exist
        self.destination_root.mkdir(parents=True, exist_ok=True)
        
    def parse_protocol(self, protocol_file: str) -> List[Tuple[str, str, str]]:
        """Parse protocol.txt file and return list of (filename, subset, label) tuples."""
        protocol_path = Path(protocol_file)
        
        if not protocol_path.exists():
            raise FileNotFoundError(f"Protocol file not found: {protocol_file}")
            
        logger.info(f"Reading protocol file: {protocol_file}")
        
        entries = []
        with open(protocol_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                parts = line.split()
                if len(parts) != 3:
                    logger.warning(f"Line {line_num}: Invalid format, expected 3 parts, got {len(parts)}")
                    continue
                    
                filename, subset, label = parts
                entries.append((filename, subset, label))
                
        logger.info(f"Parsed {len(entries)} entries from protocol file")
        return entries
    
    def copy_single_file(self, entry: Tuple[str, str, str], source_root: Path, dest_root: Path) -> bool:
        """Copy a single file from source to destination."""
        filename, subset, label = entry
        
        # Construct source and destination paths
        source_file = source_root / filename
        dest_file = dest_root / subset / label / Path(filename).name
        
        try:
            # Create destination directory if it doesn't exist
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy the file
            shutil.copy2(source_file, dest_file)
            return True
            
        except FileNotFoundError:
            logger.error(f"Source file not found: {source_file}")
            return False
        except Exception as e:
            logger.error(f"Error copying {source_file}: {e}")
            return False
    
    def transfer_files(self, protocol_file: str, show_progress: bool = True) -> dict:
        """Transfer files based on protocol with parallel processing."""
        # Parse protocol file
        entries = self.parse_protocol(protocol_file)
        
        if not entries:
            logger.warning("No valid entries found in protocol file")
            return {"total": 0, "success": 0, "failed": 0}
        
        # Statistics
        stats = {"total": len(entries), "success": 0, "failed": 0}
        
        # Prepare for progress tracking
        if show_progress and RICH_AVAILABLE:
            with Progress(
                TextColumn("[bold blue]Transferring files", justify="right"),
                BarColumn(bar_width=None),
                "[progress.percentage]{task.percentage:>3.1f}%",
                "•",
                TextColumn("[bold green]{task.completed}/{task.total}"),
                "•",
                TimeRemainingColumn(),
                console=self.console,
                expand=True
            ) as progress:
                task = progress.add_task("Copying files", total=len(entries))
                
                # Process files in parallel
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # Submit all tasks
                    future_to_entry = {
                        executor.submit(self.copy_single_file, entry, self.source_root, self.destination_root): entry
                        for entry in entries
                    }
                    
                    # Process completed tasks
                    for future in as_completed(future_to_entry):
                        success = future.result()
                        if success:
                            stats["success"] += 1
                        else:
                            stats["failed"] += 1
                            
                        progress.update(task, advance=1)
                        
                        # Update progress description with current stats
                        progress.update(
                            task, 
                            description=f"Copied {stats['success']} files, {stats['failed']} failed"
                        )
        else:
            # Fallback without progress bar
            logger.info(f"Starting parallel file transfer with {self.max_workers} workers...")
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_entry = {
                    executor.submit(self.copy_single_file, entry, self.source_root, self.destination_root): entry
                    for entry in entries
                }
                
                # Process completed tasks
                for i, future in enumerate(as_completed(future_to_entry), 1):
                    success = future.result()
                    if success:
                        stats["success"] += 1
                    else:
                        stats["failed"] += 1
                    
                    # Log progress every 1000 files
                    if i % 1000 == 0 or i == len(entries):
                        logger.info(f"Processed {i}/{len(entries)} files: {stats['success']} success, {stats['failed']} failed")
        
        return stats
    
    def print_summary(self, stats: dict):
        """Print transfer summary."""
        logger.info("\n" + "="*50)
        logger.info("TRANSFER SUMMARY")
        logger.info("="*50)
        logger.info(f"Total files: {stats['total']}")
        logger.info(f"Successfully copied: {stats['success']}")
        logger.info(f"Failed: {stats['failed']}")
        logger.info(f"Success rate: {stats['success']/stats['total']*100:.1f}%" if stats['total'] > 0 else "Success rate: N/A")
        logger.info(f"Destination: {self.destination_root}")
        logger.info("="*50)


def main():
    parser = argparse.ArgumentParser(description="Transfer WildSpoof files based on protocol")
    parser.add_argument("--protocol", "-p", 
                       default="/AISRC3/data/WildSpoof/protocol.txt",
                       help="Path to protocol.txt file")
    parser.add_argument("--source", "-s",
                       default="/AISRC3/data/WildSpoof",
                       help="Source root directory")
    parser.add_argument("--destination", "-d",
                       required=True,
                       help="Destination root directory")
    parser.add_argument("--workers", "-w",
                       type=int, default=8,
                       help="Number of parallel workers (default: 8)")
    parser.add_argument("--no-progress", action="store_true",
                       help="Disable progress bar")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not Path(args.source).exists():
        logger.error(f"Source directory does not exist: {args.source}")
        return 1
        
    if not Path(args.protocol).exists():
        logger.error(f"Protocol file does not exist: {args.protocol}")
        return 1
    
    # Create transfer instance
    transfer = FileTransfer(
        source_root=args.source,
        destination_root=args.destination,
        max_workers=args.workers
    )
    
    try:
        # Perform transfer
        stats = transfer.transfer_files(
            protocol_file=args.protocol,
            show_progress=not args.no_progress
        )
        
        # Print summary
        transfer.print_summary(stats)
        
        return 0 if stats["failed"] == 0 else 1
        
    except Exception as e:
        logger.error(f"Transfer failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
