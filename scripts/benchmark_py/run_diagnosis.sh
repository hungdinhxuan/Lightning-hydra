#!/bin/bash
# Automatic diagnosis for missing files in 2026_JAN_14_CNSL_DATA

echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║           🔍 DIAGNOSING MISSING FILES (824 files)                ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
SCORE_FILE="logs/results/CNSL_Q1_2026_benchmarks_dev/XLSR_ConformerTCM_MDT_RawboostLA_DF/2026_JAN_14_CNSL_DATA_cnsl_lora_elevenlabs_xlsr_conformertcm_mdt_lora_infer_XLSR_ConformerTCM_MDT_RawboostLA_DF.txt"
PROTOCOL_FILE="data/CNSL_Q1_2026_benchmarks/2026_JAN_14_CNSL_DATA/protocol.txt"
DATA_DIR="data/CNSL_Q1_2026_benchmarks/2026_JAN_14_CNSL_DATA"
SUBSET="dev"
OUTPUT_LOG="logs/results/CNSL_Q1_2026_benchmarks_dev/XLSR_ConformerTCM_MDT_RawboostLA_DF/missing_files_diagnosis_$(date +%Y%m%d_%H%M%S).log"

echo "Configuration:"
echo "  Score file: $SCORE_FILE"
echo "  Protocol file: $PROTOCOL_FILE"
echo "  Data directory: $DATA_DIR"
echo "  Subset: $SUBSET"
echo "  Output log: $OUTPUT_LOG"
echo ""
echo "Starting diagnosis..."
echo ""

# Run diagnosis
python scripts/benchmark_py/diagnose_missing_files.py \
    --score-file "$SCORE_FILE" \
    --protocol-file "$PROTOCOL_FILE" \
    --data-dir "$DATA_DIR" \
    --subset "$SUBSET" \
    --output "$OUTPUT_LOG"

EXIT_CODE=$?

echo ""
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║                     DIAGNOSIS COMPLETE                           ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""
echo "📄 Full report saved to:"
echo "   $OUTPUT_LOG"
echo ""
echo "📊 To view:"
echo "   cat $OUTPUT_LOG"
echo "   less $OUTPUT_LOG"
echo "   tail -100 $OUTPUT_LOG  # Last 100 lines"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Status: Only transient issues (OOM, timeout, etc.)"
    echo "   → Can accept 98.8% completion rate"
else
    echo "⚠️  Status: Found file_not_found or corrupted files"
    echo "   → Review log file for details"
fi

exit $EXIT_CODE
