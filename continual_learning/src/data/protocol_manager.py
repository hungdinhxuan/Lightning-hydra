import os
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class ProtocolEntry:
    path: str
    subset: str
    label: str
    task_id: int

class ProtocolManager:
    def __init__(self, protocol_dir: str):
        """Initialize protocol manager.
        
        Args:
            protocol_dir: Directory containing task protocol files
        """
        self.protocol_dir = protocol_dir
        self.task_protocols = {}  # task_id -> protocol file path
        self._load_protocols()
        
    def _load_protocols(self):
        """Load all task protocol files from the directory."""
        for file in os.listdir(self.protocol_dir):
            if file.startswith("task") and file.endswith("_protocol.txt"):
                try:
                    task_id = int(file.split("_")[0][4:])  # Extract task number from filename
                    self.task_protocols[task_id] = os.path.join(self.protocol_dir, file)
                except ValueError:
                    print(f"Warning: Could not parse task ID from {file}")
                    
    def get_task_protocols(self) -> Dict[int, str]:
        """Get all task protocol files."""
        return self.task_protocols
    
    def get_task_data(self, task_id: int, subset: str = "train") -> List[ProtocolEntry]:
        """Get data for a specific task and subset.
        
        Args:
            task_id: Task identifier
            subset: Data subset (train, dev, eval)
            
        Returns:
            List of ProtocolEntry objects
        """
        if task_id not in self.task_protocols:
            raise ValueError(f"No protocol found for task {task_id}")
            
        entries = []
        with open(self.task_protocols[task_id], 'r') as f:
            for line in f:
                path, data_subset, label = line.strip().split()
                if data_subset == subset:
                    entries.append(ProtocolEntry(
                        path=path,
                        subset=data_subset,
                        label=label,
                        task_id=task_id
                    ))
        return entries
    
    def get_all_tasks_data(self, subset: str = "train") -> Dict[int, List[ProtocolEntry]]:
        """Get data for all tasks and a specific subset.
        
        Args:
            subset: Data subset (train, dev, eval)
            
        Returns:
            Dictionary mapping task_id to list of ProtocolEntry objects
        """
        return {
            task_id: self.get_task_data(task_id, subset)
            for task_id in sorted(self.task_protocols.keys())
        }
    
    def get_task_labels(self, task_id: int) -> Dict[str, int]:
        """Get label mapping for a specific task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Dictionary mapping label strings to integers
        """
        entries = self.get_task_data(task_id)
        unique_labels = sorted(set(entry.label for entry in entries))
        return {label: idx for idx, label in enumerate(unique_labels)}
    
    def get_all_task_labels(self) -> Dict[int, Dict[str, int]]:
        """Get label mappings for all tasks.
        
        Returns:
            Dictionary mapping task_id to label mapping
        """
        return {
            task_id: self.get_task_labels(task_id)
            for task_id in sorted(self.task_protocols.keys())
        } 