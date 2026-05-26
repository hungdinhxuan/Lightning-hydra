"""Adapter around existing benchmark_py workflow."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional

from src.benchmark_service.schemas import BenchmarkJob

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from benchmark_py import benchmark as legacy_benchmark
from benchmark_py.execution import BenchmarkConfig
from benchmark_py.utils import Color, cleanup_temp_files, display_progress, print_color


class LegacyBenchmarkAdapter:
    """Run legacy benchmark orchestration with injectable per-dataset executor."""

    def __init__(self, execute_benchmark_fn: Callable[[BenchmarkConfig], bool]) -> None:
        self.execute_benchmark_fn = execute_benchmark_fn

    def run(self, job: BenchmarkJob) -> bool:
        job.validate()
        benchmark_folder = Path(job.dataset_path)
        results_root = Path(job.result_dir)
        eval_config_path = legacy_benchmark.resolve_eval_config_path(
            benchmark_folder,
            Path(job.eval_config) if job.eval_config else None,
        )
        results_folder, summary_file, normalized_yaml = legacy_benchmark.initialize_results(
            results_root,
            job.run_name,
            job.config_path,
            job.model_path,
            job.adapter_path,
            job.is_ln,
            job.trim_length,
            eval_config_path,
        )

        print_color(Color.CYAN, f"Checking subdirectories in '{benchmark_folder}'...")
        subdirs = legacy_benchmark.get_subdirectories(benchmark_folder)
        if not subdirs:
            raise RuntimeError(f"No subdirectories found in '{benchmark_folder}'")

        all_success = True
        total_subfolders = len(subdirs)
        for idx, subfolder in enumerate(subdirs, start=1):
            print_color(Color.YELLOW, "┌─────────────────────────────────────────────────────────────────┐")
            print_color(Color.YELLOW, f"│ Processing dataset: {subfolder.name}")
            print_color(Color.YELLOW, "└─────────────────────────────────────────────────────────────────┘")
            display_progress(idx, total_subfolders, legacy_benchmark.CONSTANTS.progress_bar_width)
            success = legacy_benchmark.process_dataset(
                subfolder=subfolder,
                gpu_number=job.gpu_id,
                yaml_config=job.config_path,
                base_model_path=job.model_path,
                results_folder=results_folder,
                normalized_yaml=normalized_yaml,
                comment=job.run_name,
                adapter_paths=job.adapter_path,
                is_ln=job.is_ln,
                is_random_start=job.random_start,
                trim_length=job.trim_length,
                batch_size=job.batch_size,
                summary_file=summary_file,
                extra_overrides=job.extra_overrides,
                eval_config_path=eval_config_path,
                benchmark_root=benchmark_folder,
                missing_protocol_label=job.missing_protocol_label,
                execute_benchmark_fn=self.execute_benchmark_fn,
            )
            all_success = all_success and success

        legacy_benchmark.calculate_pooled_eer(
            results_folder,
            normalized_yaml,
            job.run_name,
            summary_file,
            subdirs,
            eval_config_path,
        )
        legacy_benchmark.create_merged_protocol(
            results_folder,
            normalized_yaml,
            job.run_name,
            job.config_path,
            job.model_path,
            summary_file,
            subdirs,
        )
        cleanup_temp_files(results_folder)
        return all_success
