import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional, Set

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable, *args, **kwargs):
        return iterable


def infer_label_from_rel_path(rel_path: str) -> str:
    lower = rel_path.lower()
    has_bonafide = "bonafide" in lower or "benign" in lower or "real" in lower
    has_spoof = "spoof" in lower or "fake" in lower or "synthetic" in lower

    if has_bonafide and not has_spoof:
        return "bonafide"
    if has_spoof and not has_bonafide:
        return "spoof"
    if has_bonafide and has_spoof:
        raise ValueError(
            f"Ambiguous label inference for rel_path (matches both types): {rel_path}"
        )
    raise ValueError(
        "Cannot infer label from rel_path; expected one of "
        '"bonafide"/"benign" or "spoof"/"fake": '
        f"{rel_path}"
    )


def find_audio_files_in_dir(args: tuple[str, Set[str]]) -> List[str]:
    """
    Worker function for parallel file discovery.
    Args: (dir_path_str, ext_set)
    Returns: List of file paths as strings (for pickling).
    """
    dir_path_str, ext_set = args
    dir_path = Path(dir_path_str)
    files: List[str] = []
    
    try:
        for path in dir_path.rglob("*"):
            if path.is_file():
                if path.suffix.lstrip(".").lower() in ext_set:
                    files.append(str(path))
    except (PermissionError, OSError):
        # Skip directories we can't access
        pass
    
    return files


def find_audio_files(
    root_dir: Path,
    exts: Iterable[str],
    num_workers: int = 1,
) -> List[Path]:
    """
    Recursively find all audio files under root_dir with the given extensions.
    Returned paths are sorted for reproducibility.
    
    Optimized: parallel processing of subdirectories with progress bar.
    Deduplicates files based on resolved absolute path to handle symlinks/hardlinks.
    """
    # Resolve root_dir to absolute path for consistent path handling
    root_dir = root_dir.resolve()
    
    # Normalize extensions: lowercase, no leading dot
    ext_set: Set[str] = {ext.lower().lstrip(".") for ext in exts}
    
    if num_workers == 1:
        # Sequential processing with progress bar
        # Use set to track unique files (handles symlinks/hardlinks)
        seen_files: Set[str] = set()
        files: List[Path] = []
        duplicates = 0
        
        # Use rglob iterator with progress bar (memory efficient)
        # tqdm will show rate and elapsed time even without total count
        for path in tqdm(
            root_dir.rglob("*"),
            desc="Scanning files",
            unit="items",
        ):
            if path.is_file():
                if path.suffix.lstrip(".").lower() in ext_set:
                    # Use resolved absolute path as unique identifier
                    # This handles symlinks pointing to the same file
                    file_key = str(path.resolve())
                    if file_key in seen_files:
                        duplicates += 1
                    else:
                        seen_files.add(file_key)
                        files.append(path)
        
        if duplicates > 0:
            print(
                f"[INFO] Removed {duplicates} duplicate file entries (symlinks/hardlinks)",
                file=sys.stderr,
            )
        
        return sorted(files)
    
    # Parallel processing: process subdirectories in parallel
    # Collect all immediate subdirectories and the root itself
    subdirs: List[Path] = [root_dir]
    
    try:
        # Add immediate subdirectories for better parallelization
        for item in root_dir.iterdir():
            if item.is_dir():
                subdirs.append(item)
    except (PermissionError, OSError):
        pass
    
    # If we only have the root dir, fall back to sequential
    if len(subdirs) == 1:
        return find_audio_files(root_dir, exts, num_workers=1)
    
    all_files: List[Path] = []
    
    # Prepare arguments for workers
    worker_args = [(str(subdir.resolve()), ext_set) for subdir in subdirs]
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_dir = {
            executor.submit(find_audio_files_in_dir, args): args[0]
            for args in worker_args
        }
        
        # Collect results with progress bar
        for future in tqdm(
            as_completed(future_to_dir),
            total=len(future_to_dir),
            desc="Scanning directories",
            unit="dirs",
        ):
            try:
                file_strs = future.result()
                all_files.extend(Path(f) for f in file_strs)
            except Exception as e:  # noqa: BLE001
                dir_name = future_to_dir[future]
                print(
                    f"[WARN] Error scanning {dir_name}: {e}",
                    file=sys.stderr,
                )
    
    # Deduplicate files (important for parallel processing with overlapping subdirs)
    seen_files: Set[str] = set()
    unique_files: List[Path] = []
    duplicates = 0
    
    for path in all_files:
        file_key = str(Path(path).resolve())
        if file_key in seen_files:
            duplicates += 1
        else:
            seen_files.add(file_key)
            unique_files.append(path)
    
    if duplicates > 0:
        print(
            f"[INFO] Removed {duplicates} duplicate file entries (symlinks/hardlinks/overlapping subdirs)",
            file=sys.stderr,
        )
    
    # Sort for reproducibility
    return sorted(unique_files)


def write_protocol(
    root_dir: Path,
    output_path: Path,
    subset: str = "eval",
    label: str = "bonafide",
    exts: Iterable[str] = ("wav", "flac", "mp3", "ogg", "m4a"),
    num_workers: int = 1,
    prefix: Optional[str] = None,
) -> None:
    """
    Create a protocol file where each line has the form:

        <relative_path> <subset> <label>

    - relative_path: path to the audio file relative to root_dir (with optional prefix)
    - subset: e.g. train / dev / eval
    - label: e.g. bonafide / spoof (or "auto" to infer from rel_path)
    - prefix: optional prefix to prepend to relative_path (e.g., "M-AILABS" -> "M-AILABS/...")
    """
    # Resolve root_dir to absolute path to ensure consistent path handling
    root_dir_abs = root_dir.resolve()
    
    audio_files = find_audio_files(root_dir_abs, exts, num_workers=num_workers)

    if not audio_files:
        print(f"[WARN] No audio files found under {root_dir}", file=sys.stderr)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    auto_label = label == "auto"

    # Track unique relative paths to avoid duplicates in protocol
    seen_rel_paths: Set[str] = set()
    duplicate_count = 0

    # Write with progress bar
    with output_path.open("w", encoding="utf-8") as f:
        for audio_path in tqdm(
            audio_files,
            desc="Writing protocol",
            unit="files",
            total=len(audio_files),
        ):
            # Convert to Path if needed (from parallel workers)
            if isinstance(audio_path, str):
                audio_path = Path(audio_path)
            
            # Ensure audio_path is absolute for relative_to to work correctly
            # IMPORTANT: Don't use resolve() here as it resolves symlinks,
            # which can cause issues when symlinks point outside root_dir_abs
            if not audio_path.is_absolute():
                # Convert relative path to absolute without resolving symlinks
                # Use parent / name to preserve symlink structure
                audio_path = root_dir_abs / audio_path
            
            # Compute relative path, handling symlinks that may point outside root_dir
            # Treat symlinks like hardlinks: always include them using their path in the tree
            audio_str = str(audio_path)
            root_str = str(root_dir_abs)
            
            # Try pathlib's relative_to first (preferred, fast)
            try:
                rel_path = audio_path.relative_to(root_dir_abs).as_posix()
            except ValueError:
                # Fallback: relative_to failed (e.g., symlink target resolves outside root_dir)
                # Since rglob found this file, it must be in the directory tree under root
                # Extract the relative path manually from the string representation
                # This treats symlinks like hardlinks - use their path in the tree structure
                
                if audio_str.startswith(root_str):
                    # Direct substring extraction: portion after root
                    rel_path = audio_str[len(root_str):].lstrip("/").replace(os.sep, "/")
                elif root_str in audio_str:
                    # Root appears as substring - extract portion after root
                    idx = audio_str.find(root_str)
                    rel_path = audio_str[idx + len(root_str):].lstrip("/").replace(os.sep, "/")
                else:
                    # Path doesn't contain root - might be resolved symlink pointing outside
                    # Use path components to find common prefix and extract relative path
                    audio_parts = Path(audio_str).parts
                    root_parts = Path(root_str).parts
                    
                    # Find common prefix (directory components that match)
                    common_len = 0
                    for i in range(min(len(root_parts), len(audio_parts))):
                        if root_parts[i] == audio_parts[i]:
                            common_len = i + 1
                        else:
                            break
                    
                    if common_len >= len(root_parts) and common_len < len(audio_parts):
                        # Root is a prefix of audio_path - extract relative portion
                        rel_path = "/".join(audio_parts[common_len:])
                    elif common_len > 0:
                        # Partial match - extract from common point
                        # This handles cases where paths share some common ancestor
                        rel_path = "/".join(audio_parts[common_len:])
                    else:
                        # No common prefix - use basename as last resort
                        # This shouldn't happen if rglob found the file, but handle it anyway
                        rel_path = os.path.basename(audio_str)
                        print(
                            f"[WARN] Using basename for file (no common path with root): {audio_path}",
                            file=sys.stderr,
                        )
            
            # Apply prefix if provided
            if prefix:
                # Ensure prefix doesn't end with / and rel_path doesn't start with /
                prefix_clean = prefix.rstrip("/")
                rel_path_clean = rel_path.lstrip("/")
                rel_path_with_prefix = f"{prefix_clean}/{rel_path_clean}" if rel_path_clean else prefix_clean
            else:
                rel_path_with_prefix = rel_path
            
            # Check for duplicate relative paths before writing
            if rel_path_with_prefix in seen_rel_paths:
                duplicate_count += 1
                continue
            
            seen_rel_paths.add(rel_path_with_prefix)
            
            resolved_label = (
                infer_label_from_rel_path(rel_path_with_prefix) if auto_label else label
            )
            
            # Quote path if it contains spaces for proper parsing
            if " " in rel_path_with_prefix:
                quoted_path = f'"{rel_path_with_prefix}"'
                line = f"{quoted_path} {subset} {resolved_label}\n"
            else:
                line = f"{rel_path_with_prefix} {subset} {resolved_label}\n"
            
            f.write(line)

    unique_entries = len(seen_rel_paths)
    print(
        f"[INFO] Wrote {unique_entries} unique entries to protocol file: {output_path}",
        file=sys.stderr,
    )
    
    if duplicate_count > 0:
        print(
            f"[INFO] Removed {duplicate_count} duplicate entries from protocol output",
            file=sys.stderr,
        )


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a protocol.txt file with lines of the form:\n"
            "  <relative_path> <subset> <label>\n\n"
            "Example:\n"
            "  python scripts/make_protocol.py "
            "--root-dir pool/spoofceleb/flac/train "
            "--output pool/spoofceleb/protocol.txt "
            "--subset train --label bonafide"
        )
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        required=True,
        help="Root directory to recursively search for audio files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("protocol.txt"),
        help="Path to output protocol file (default: ./protocol.txt).",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="eval",
        choices=["train", "dev", "eval"],
        help="Subset name to use in protocol file (default: eval).",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="bonafide",
        choices=["bonafide", "spoof", "auto"],
        help=(
            "Label to use in protocol file (default: bonafide). "
            'Use "auto" to infer label from rel_path.'
        ),
    )
    parser.add_argument(
        "--exts",
        type=str,
        default="wav,flac,mp3,ogg,m4a",
        help=(
            "Comma-separated list of audio file extensions to include "
            "(default: wav,flac,mp3,ogg,m4a)."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help=(
            "Number of parallel workers for file discovery "
            f"(default: {os.cpu_count() or 4} = parallel, or 1 = sequential). "
            "Set to 1 to disable parallel processing."
        ),
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help=(
            "Optional prefix to prepend to relative paths in the protocol file. "
            "For example, if prefix is 'M-AILABS', relative paths will be "
            "like 'M-AILABS/subdir/file.wav' instead of 'subdir/file.wav'."
        ),
    )

    args = parser.parse_args(list(argv))
    
    # Set default num_workers based on CPU count
    if args.num_workers is None:
        args.num_workers = os.cpu_count() or 4

    return args


def main(argv: Optional[Iterable[str]] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    args = parse_args(argv)

    root_dir: Path = args.root_dir
    output_path: Path = args.output
    subset: str = args.subset
    label: str = args.label

    exts = [ext.strip().lstrip(".") for ext in args.exts.split(",") if ext.strip()]

    if not root_dir.exists() or not root_dir.is_dir():
        print(f"[ERROR] root-dir does not exist or is not a directory: {root_dir}", file=sys.stderr)
        sys.exit(1)

    write_protocol(
        root_dir=root_dir,
        output_path=output_path,
        subset=subset,
        label=label,
        exts=exts,
        num_workers=args.num_workers,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
