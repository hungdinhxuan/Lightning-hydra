#!/bin/bash
# Quick script to run benchmark with partial results acceptance
#
# Usage:
#   ./RUN_WITH_PARTIAL.sh [MIN_COMPLETION_RATE]
#
# Example:
#   ./RUN_WITH_PARTIAL.sh 78   # Accept >= 78%
#   ./RUN_WITH_PARTIAL.sh 95   # Accept >= 95% (default)
#   ./RUN_WITH_PARTIAL.sh 100  # Only accept 100%

MIN_RATE=${1:-78}  # Default to 78% if not specified

echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║         Running Benchmark with Partial Results Feature           ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""
echo "⚙️  Configuration:"
echo "  MIN_COMPLETION_RATE: ${MIN_RATE}%"
echo "  PROTOCOL_SUBSET: ${PROTOCOL_SUBSET:-eval}"
echo ""
echo "📝 This will:"
echo "  • Resume incomplete datasets"
echo "  • Accept results with >= ${MIN_RATE}% completion"
echo "  • Mark partial results in summary"
echo ""
echo "Press Ctrl+C to cancel, or Enter to continue..."
read

export MIN_COMPLETION_RATE=$MIN_RATE

# Run the benchmark with your previous command
# Modify this line with your actual command:
PROTOCOL_SUBSET=dev ./scripts/benchmark_py/benchmark.py \
    -g 0 \
    -c cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_lora_infer \
    -b $(pwd)/data/CNSL_Q1_2026_benchmarks \
    -m /nvme2/hungdx/logs/train/runs/2026-01-15_14-26-30/checkpoints/averaged_top5.ckpt \
    -r logs/results/CNSL_Q1_2026_benchmarks_dev \
    -n "XLSR_ConformerTCM_MDT_RawboostLA_DF" \
    -l true

echo ""
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║                      Benchmark Complete!                         ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
