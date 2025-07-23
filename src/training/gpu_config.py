# src/training/gpu_config.py
import torch
import platform
import subprocess
import logging
from typing import Dict, Optional, Tuple
import os

logger = logging.getLogger(__name__)

class GPUConfig:
    """GPU configuration for Windows NVIDIA setup."""
    
    def __init__(self):
        self.platform = platform.system()
        self.device = None
        self.gpu_info = {}
        
    def setup_cuda_environment(self) -> bool:
        """Setup CUDA environment for Windows."""
        if self.platform != "Windows":
            logger.info(f"Running on {self.platform}, not Windows")
            return True
            
        try:
            # Check CUDA availability
            if not torch.cuda.is_available():
                logger.error("CUDA is not available. Please install CUDA toolkit.")
                return False
                
            # Get CUDA version
            cuda_version = torch.version.cuda
            logger.info(f"CUDA version: {cuda_version}")
            
            # Check cuDNN
            cudnn_version = torch.backends.cudnn.version()
            logger.info(f"cuDNN version: {cudnn_version}")
            
            # Set Windows-specific environment variables
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # For better error messages
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
            
            # Enable TensorFloat-32 for RTX 30 series
            if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends, 'cudnn'):
                torch.backends.cudnn.allow_tf32 = True
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup CUDA environment: {e}")
            return False
    
    def get_gpu_info(self) -> Dict:
        """Get detailed GPU information for RTX 3060."""
        try:
            if not torch.cuda.is_available():
                return {"error": "CUDA not available"}
                
            device_id = 0
            self.device = torch.device(f'cuda:{device_id}')
            
            # Get GPU properties
            props = torch.cuda.get_device_properties(device_id)
            
            # Calculate memory in GB
            total_memory_gb = props.total_memory / (1024**3)
            allocated_memory_gb = torch.cuda.memory_allocated(device_id) / (1024**3)
            reserved_memory_gb = torch.cuda.memory_reserved(device_id) / (1024**3)
            
            self.gpu_info = {
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_gb": round(total_memory_gb, 2),
                "allocated_memory_gb": round(allocated_memory_gb, 2),
                "reserved_memory_gb": round(reserved_memory_gb, 2),
                "available_memory_gb": round(total_memory_gb - allocated_memory_gb, 2),
                "multi_processor_count": props.multi_processor_count,
                "is_integrated": props.is_integrated,
                "is_multi_gpu_board": props.is_multi_gpu_board,
            }
            
            # Use nvidia-smi for additional info
            if self.platform == "Windows":
                self._get_nvidia_smi_info()
                
            return self.gpu_info
            
        except Exception as e:
            logger.error(f"Failed to get GPU info: {e}")
            return {"error": str(e)}
    
    def _get_nvidia_smi_info(self):
        """Get additional GPU info using nvidia-smi on Windows."""
        try:
            # Run nvidia-smi command securely without shell=True
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,utilization.gpu,utilization.memory", 
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=30  # Add timeout for security
            )
            
            if result.returncode == 0:
                values = result.stdout.strip().split(", ")
                if len(values) >= 4:
                    self.gpu_info.update({
                        "temperature_c": float(values[0]),
                        "power_draw_w": float(values[1]),
                        "gpu_utilization_%": float(values[2]),
                        "memory_utilization_%": float(values[3])
                    })
        except Exception as e:
            logger.warning(f"Could not get nvidia-smi info: {e}")
    
    def optimize_for_rtx3060(self, batch_size: int = 8) -> Dict:
        """Optimize settings for RTX 3060 with 12GB VRAM."""
        recommendations = {
            "batch_size": batch_size,
            "gradient_accumulation_steps": 1,
            "mixed_precision": True,
            "gradient_checkpointing": False,  # Not needed with 12GB for most models
            "num_workers": 4,  # Windows optimal
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 2,
            "max_memory_usage_gb": 10,  # Leave 2GB buffer
        }
        
        # Adjust for available memory
        available_memory = self.gpu_info.get("available_memory_gb", 12)
        if available_memory < 10:
            recommendations["gradient_checkpointing"] = True
            recommendations["batch_size"] = max(2, batch_size // 2)
            recommendations["max_memory_usage_gb"] = available_memory - 2
            
        return recommendations
    
    def get_training_config(self) -> Dict:
        """Get optimized training configuration for Windows."""
        if not self.setup_cuda_environment():
            raise RuntimeError("Failed to setup CUDA environment")
            
        gpu_info = self.get_gpu_info()
        if "error" in gpu_info:
            raise RuntimeError(f"GPU error: {gpu_info['error']}")
            
        config = self.optimize_for_rtx3060()
        
        return {
            "device": "cuda",
            "gpu_info": gpu_info,
            "training_config": config,
            "windows_specific": {
                "num_threads": min(8, os.cpu_count() or 4),  # Limit threads on Windows
                "use_mkl": True,  # Intel MKL for Windows
                "cudnn_benchmark": True,  # Enable cuDNN auto-tuner
            }
        }

# Windows-specific training script modifications
def setup_windows_training():
    """Setup training environment for Windows."""
    # Security: Removed Windows Defender exclusion functionality
    # This was a security risk as it modified system security settings
    logger.info("Windows training setup - Defender exclusion disabled for security")
    
    # Set Windows-specific PyTorch settings
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    
    # Security: Removed power plan modification functionality  
    # This was a security risk as it could modify system power settings
    logger.info("Windows training setup - Power plan modification disabled for security")