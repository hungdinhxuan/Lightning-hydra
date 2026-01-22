#!/usr/bin/env python3
"""
Example usage of the file transfer script.
"""

from transfer_file import FileTransfer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_usage():
    """Example of how to use the FileTransfer class programmatically."""
    
    # Configuration
    source_root = "/AISRC3/data/WildSpoof"
    destination_root = "/path/to/destination"
    protocol_file = "/AISRC3/data/WildSpoof/protocol.txt"
    max_workers = 8
    
    # Create transfer instance
    transfer = FileTransfer(
        source_root=source_root,
        destination_root=destination_root,
        max_workers=max_workers
    )
    
    try:
        # Perform transfer
        stats = transfer.transfer_files(protocol_file, show_progress=True)
        
        # Print summary
        transfer.print_summary(stats)
        
    except Exception as e:
        logger.error(f"Transfer failed: {e}")

if __name__ == "__main__":
    example_usage()
