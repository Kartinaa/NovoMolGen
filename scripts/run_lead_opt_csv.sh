#!/bin/bash
# Convenience script to run lead optimization from CSV
# This script handles the LD_LIBRARY_PATH issue automatically

# Set library path for matplotlib
export LD_LIBRARY_PATH=$HOME/anaconda3/envs/NovoMol/lib:$LD_LIBRARY_PATH

# Default parameters
MODEL_PATH="outputs/11_21_25_SAFEGen_lr_KLnormclaped0.1_unfreezeall_noloar_xattn67891011_newmodel/checkpoint-33400/full_model"
VALIDATION_SET="finetune_data/molgenbench_dataset/validation"
CSV_FILE="finetune_data/molgenbench_dataset/h2l_scaffold/h2l_breakpoint_scaffolds_with_safe.csv"
NUM_SAMPLES=50
OUTPUT_DIR="outputs/lead_optimization_h2l_scaffold"
BATCH_SIZE=10
MAX_LENGTH=64

# Additional optional parameters
SEED=2025
SAMPLE_POSTERIOR=""
ADDITIONAL_ARGS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --validation_set)
            VALIDATION_SET="$2"
            shift 2
            ;;
        --csv_file)
            CSV_FILE="$2"
            shift 2
            ;;
        --num_samples_per_breakpoint)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --sample_posterior)
            SAMPLE_POSTERIOR="--sample_posterior"
            shift 1
            ;;
        *)
            # Pass through any other arguments
            ADDITIONAL_ARGS="$ADDITIONAL_ARGS $1 $2"
            shift 2
            ;;
    esac
done

echo "================================"
echo "Lead Optimization from CSV"
echo "================================"
echo "Model: $MODEL_PATH"
echo "Validation set: $VALIDATION_SET"
echo "CSV file: $CSV_FILE"
echo "Samples per breakpoint: $NUM_SAMPLES"
echo "Output directory: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Seed: $SEED"
echo "Sample posterior: ${SAMPLE_POSTERIOR:-No}"
echo "================================"
echo ""

# Run the script
conda run -n NovoMol python scripts/lead_optimization_from_csv.py \
    --model_path "$MODEL_PATH" \
    --validation_set "$VALIDATION_SET" \
    --csv_file "$CSV_FILE" \
    --num_samples_per_breakpoint "$NUM_SAMPLES" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --max_length "$MAX_LENGTH" \
    --seed "$SEED" \
    --log_level INFO \
    $SAMPLE_POSTERIOR \
    $ADDITIONAL_ARGS

