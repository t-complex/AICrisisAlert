#!/bin/bash

# Crisis Classification Training Script
# This script runs the complete training pipeline for crisis classification

set -e  # Exit on any error

echo "🚀 Starting Crisis Classification Training Pipeline"
echo "=================================================="

# Configuration
DATA_DIR="data/processed"
OUTPUT_DIR="outputs/models/$(date +%Y%m%d_%H%M%S)"
EPOCHS=3
BATCH_SIZE=16
LEARNING_RATE=2e-5
LORA_RANK=8
LORA_ALPHA=32
WARMUP_STEPS=100
SEED=42

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Data directory $DATA_DIR not found!"
    echo "Please ensure you have the processed datasets:"
    echo "  - $DATA_DIR/train.csv"
    echo "  - $DATA_DIR/validation.csv" 
    echo "  - $DATA_DIR/test.csv"
    exit 1
fi

# Check for required CSV files
for file in "train.csv" "validation.csv" "test.csv"; do
    if [ ! -f "$DATA_DIR/$file" ]; then
        echo "❌ Error: Required file $DATA_DIR/$file not found!"
        exit 1
    fi
done

echo "✅ Data validation passed"
echo "📊 Dataset files found:"
echo "  - Train: $(wc -l < $DATA_DIR/train.csv) samples"
echo "  - Validation: $(wc -l < $DATA_DIR/validation.csv) samples"
echo "  - Test: $(wc -l < $DATA_DIR/test.csv) samples"

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo "📁 Output directory: $OUTPUT_DIR"

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "🔍 Checking GPU availability..."
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    echo "✅ GPU detected - will use mixed precision training"
    MIXED_PRECISION="--use_mixed_precision"
else
    echo "⚠️  No GPU detected - using CPU training"
    MIXED_PRECISION=""
fi

# Training configuration
echo ""
echo "📋 Training Configuration:"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  LoRA Rank: $LORA_RANK"
echo "  LoRA Alpha: $LORA_ALPHA"
echo "  Warmup Steps: $WARMUP_STEPS"
echo "  Random Seed: $SEED"

# Check for optional monitoring
WANDB_ARG=""
if command -v wandb &> /dev/null; then
    echo "📈 Weights & Biases available - enabling monitoring"
    WANDB_ARG="--use_wandb"
fi

# Run training
echo ""
echo "🏃 Starting training..."
echo "=================================================="

python3 src/training/train.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --warmup_steps "$WARMUP_STEPS" \
    --seed "$SEED" \
    --experiment_name "crisis_classification_$(date +%Y%m%d_%H%M%S)" \
    $WANDB_ARG \
    $MIXED_PRECISION

# Check training success
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Training completed successfully!"
    echo "📁 Model artifacts saved to: $OUTPUT_DIR"
    echo ""
    echo "📋 Generated files:"
    echo "  - best_model.pt (best checkpoint)"
    echo "  - merged_model.pt (final merged model)"
    echo "  - lora_adapters/ (LoRA adapter weights)"
    echo "  - tokenizer/ (tokenizer files)"
    echo "  - training_config.json (configuration)"
    echo "  - training_history.json (training metrics)"
    echo "  - test_results.json (final evaluation)"
    echo "  - logs/ (training visualizations)"
    echo ""
    echo "🚀 Ready for inference and deployment!"
else
    echo ""
    echo "❌ Training failed!"
    echo "Check the logs above for error details."
    exit 1
fi