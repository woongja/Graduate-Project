#!/bin/bash

# Auto-tune generation script with dynamic parameters
# Usage: ./generate.sh [options]

# Default values
PROTOCOL_CSV="/home/woongjae/noise-tracing/muti-feature_fusion/Datasets/meta_file/meta_clean.csv"
BASE_OUTPUT_DIR="/home/woongjae/noise-tracing/muti-feature_fusion/Datasets"
NOISE_TYPE="auto_tune"
NUM_WORKERS=8  # Reduced default workers
SAMPLE_RATE=16000
USE_COMMON_SCALES=false
FRAME_LENGTH=2048
SEQUENTIAL=false  # New option for sequential processing
SEED=""

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -p, --protocol_csv PATH       Path to protocol CSV file (default: $PROTOCOL_CSV)"
    echo "  -o, --output_dir PATH         Base output directory (default: $BASE_OUTPUT_DIR)"
    echo "  -n, --noise_type NAME         Noise type identifier (default: $NOISE_TYPE)"
    echo "  -w, --workers NUM             Number of workers (default: $NUM_WORKERS)"
    echo "  -r, --sample_rate NUM         Sample rate (default: $SAMPLE_RATE)"
    echo "  -c, --common_scales           Use only common scales"
    echo "  -f, --frame_length NUM        Frame length (default: $FRAME_LENGTH)"
    echo "  -q, --sequential              Use sequential processing (most stable)"
    echo "  -s, --seed NUM                Random seed for reproducibility"
    echo "  -h, --help                    Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                                    # Use all defaults (threaded)"
    echo "  $0 -q                                                 # Sequential processing (safest)"
    echo "  $0 -p /path/to/protocol.csv -o /path/to/output       # Custom paths"
    echo "  $0 -c -w 2 -s 42                                     # Conservative threading settings"
    echo "  $0 -q -c -s 42                                       # Sequential with common scales"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--protocol_csv)
            PROTOCOL_CSV="$2"
            shift 2
            ;;
        -o|--output_dir)
            BASE_OUTPUT_DIR="$2"
            shift 2
            ;;
        -n|--noise_type)
            NOISE_TYPE="$2"
            shift 2
            ;;
        -w|--workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        -r|--sample_rate)
            SAMPLE_RATE="$2"
            shift 2
            ;;
        -c|--common_scales)
            USE_COMMON_SCALES=true
            shift
            ;;
        -f|--frame_length)
            FRAME_LENGTH="$2"
            shift 2
            ;;
        -q|--sequential)
            SEQUENTIAL=true
            shift
            ;;
        -s|--seed)
            SEED="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Build command
CMD="python 1_generate_auto_tune.py"
CMD="$CMD --protocol_csv \"$PROTOCOL_CSV\""
CMD="$CMD --base_output_dir \"$BASE_OUTPUT_DIR\""
CMD="$CMD --noise_type \"$NOISE_TYPE\""
CMD="$CMD --num_workers $NUM_WORKERS"
CMD="$CMD --sample_rate $SAMPLE_RATE"
CMD="$CMD --frame_length $FRAME_LENGTH"

if [ "$USE_COMMON_SCALES" = true ]; then
    CMD="$CMD --use_common_scales"
fi

if [ "$SEQUENTIAL" = true ]; then
    CMD="$CMD --sequential"
fi

if [ ! -z "$SEED" ]; then
    CMD="$CMD --seed $SEED"
fi

# Print configuration
echo "🚀 Starting Auto-Tune Generation"
echo "================================="
echo "Protocol CSV: $PROTOCOL_CSV"
echo "Output directory: $BASE_OUTPUT_DIR"
echo "Noise type: $NOISE_TYPE"
echo "Workers: $NUM_WORKERS"
echo "Sample rate: $SAMPLE_RATE"
echo "Use common scales: $USE_COMMON_SCALES"
echo "Frame length: $FRAME_LENGTH"
echo "Sequential processing: $SEQUENTIAL"
if [ ! -z "$SEED" ]; then
    echo "Random seed: $SEED"
fi
echo "================================="
echo ""

# Execute command
echo "Running: $CMD"
eval $CMD

echo ""
echo "✅ Auto-tune generation completed!"