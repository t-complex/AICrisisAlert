#!/bin/bash

# Enhanced Crisis Classification Training Script
# Runs BERTweet with Focal Loss on Leak-Free Datasets
# Expected Performance: 87-92% Accuracy, 0.82-0.87 Macro F1

echo "🚀 Starting Enhanced Crisis Classification Training"
echo "=" * 60
echo "Features:"
echo "  ✅ BERTweet-base (social media optimized)"
echo "  ✅ Focal Loss (γ=2.0)"
echo "  ✅ Leak-free balanced datasets"
echo "  ✅ Advanced training techniques"
echo "  ✅ 5 epochs with early stopping"
echo "  ✅ Learning rate: 1e-5 with warmup"
echo "  ✅ Effective batch size: 64 (16×4)"
echo "=" * 60

# Change to project directory
cd "$(dirname "$0")/.."

# Create necessary directories
mkdir -p outputs/models/bertweet_enhanced
mkdir -p logs

# Set CUDA device if available
if command -v nvidia-smi &> /dev/null; then
    export CUDA_VISIBLE_DEVICES=0
    echo "🔥 GPU detected - using CUDA"
else
    echo "💻 Using CPU training"
fi

# Run enhanced training with configuration file
python src/training/enhanced_train.py \
    --config configs/enhanced_training_config.json \
    --experiment_name "bertweet_enhanced_$(date +%Y%m%d_%H%M%S)" \
    2>&1 | tee logs/enhanced_training_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "✅ Enhanced training completed!"
echo "📊 Check outputs/models/bertweet_enhanced/ for results"
echo "📈 TensorBoard logs: outputs/models/bertweet_enhanced/tensorboard/"
echo "📋 Training log: logs/enhanced_training_*.log"