# Create Softlinks Script

This script creates symbolic links (softlinks) for audio files based on the SpoofCeleb protocol.txt file, maintaining the same directory structure while filtering for specific datasets and file paths.

## Features

- ✅ Parse protocol.txt and filter entries
- ✅ Filter by dataset split (train, dev, etc.)
- ✅ Filter by path pattern (e.g., files containing "a00")
- ✅ Maintain directory structure
- ✅ Dry run mode to preview before creating links
- ✅ Progress tracking for large files
- ✅ Statistics reporting

## Usage

### Command Line Interface

#### Basic Usage (Dry Run)
```bash
cd /home/hungdx/code/Lightning-hydra

python scripts/create_softlinks.py \
    --protocol data/spoofceleb/protocol.txt \
    --source-base data/spoofceleb \
    --target-base data/spoofceleb_filtered \
    --dry-run
```

#### Create Links (Train and Dev sets with 'a00')
```bash
python scripts/create_softlinks.py \
    --protocol data/spoofceleb/protocol.txt \
    --source-base data/spoofceleb \
    --target-base data/spoofceleb_filtered
```

#### Only Train Set
```bash
python scripts/create_softlinks.py \
    --protocol data/spoofceleb/protocol.txt \
    --source-base data/spoofceleb \
    --target-base data/spoofceleb_train_only \
    --sets train
```

#### Custom Path Filter
```bash
python scripts/create_softlinks.py \
    --protocol data/spoofceleb/protocol.txt \
    --source-base data/spoofceleb \
    --target-base data/spoofceleb_custom \
    --path-filter "a01"
```

### Programmatic Usage

```python
from create_softlinks import create_softlinks_from_protocol

# Example: Create links for train and dev sets with 'a00' in path
stats = create_softlinks_from_protocol(
    protocol_file="/path/to/protocol.txt",
    source_base_dir="/path/to/source",
    target_base_dir="/path/to/target",
    sets_to_include=("train", "dev"),
    path_filter="a00",
    dry_run=False  # Set to True to preview without creating
)

print(f"Created {stats['links_created']} links")
```

## Parameters

- `--protocol`: Path to the protocol.txt file (required)
- `--source-base`: Base directory where actual audio files are located (required)
- `--target-base`: Base directory where softlinks will be created (required)
- `--sets`: Dataset splits to include (default: train dev)
- `--path-filter`: String that must be present in file path (default: a00)
- `--dry-run`: Preview what would be done without creating links

## Protocol File Format

The protocol.txt file should have the following format:
```
flac/train/a00/id11178/LAHad7VLKXw-00007-001.flac train bonafide
flac/train/a00/id10580/VR8MUnB_MTI-00007-003.flac train bonafide
flac/development/a00/id10318/YYsxcZ5saac-00002-006.flac dev bonafide
```

Each line contains: `filepath set_name label`

## Output Statistics

The function returns a dictionary with statistics:
- `total_lines`: Total lines processed from protocol file
- `filtered_lines`: Lines matching the filter criteria
- `links_created`: Number of symbolic links created
- `links_skipped`: Links skipped (already exist)
- `errors`: Number of errors encountered

## Directory Structure

The script maintains the exact directory structure from the protocol file. For example:

**Source:**
```
data/spoofceleb/
└── flac/
    ├── train/
    │   └── a00/
    │       └── id11178/
    │           └── LAHad7VLKXw-00007-001.flac
    └── development/
        └── a00/
            └── id10318/
                └── YYsxcZ5saac-00002-006.flac
```

**Target (softlinks):**
```
data/spoofceleb_filtered/
└── flac/
    ├── train/
    │   └── a00/
    │       └── id11178/
    │           └── LAHad7VLKXw-00007-001.flac -> ../../../../../../spoofceleb/flac/train/a00/id11178/LAHad7VLKXw-00007-001.flac
    └── development/
        └── a00/
            └── id10318/
                └── YYsxcZ5saac-00002-006.flac -> ../../../../../../spoofceleb/flac/development/a00/id10318/YYsxcZ5saac-00002-006.flac
```

## Notes

- The script creates parent directories automatically
- Existing links are skipped (not overwritten)
- Use dry run mode first to verify the operation
- For large protocol files (millions of lines), progress is printed every 100,000 lines

