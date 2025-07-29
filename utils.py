"""
Utility functions for the AI Interview System
"""

import os
import json
import time
import hashlib
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

def convert_to_serializable(obj: Any) -> Any:
    """Convert object to be JSON serializable, handling NumPy types"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(obj, 'isoformat'):  
        return obj.isoformat()
    else:
        return obj

def generate_cache_key(text: str, max_length: int = 1000) -> str:
    """Generate a cache key from text using hash for memory efficiency"""
    truncated_text = text[:max_length]
    return hashlib.md5(truncated_text.encode()).hexdigest()

def safe_file_operation(operation: callable, *args, **kwargs) -> Any:
    """Safely perform file operations with error handling"""
    try:
        return operation(*args, **kwargs)
    except Exception as e:
        logger.error(f"File operation failed: {e}")
        return None

def save_json_file(data: Dict[str, Any], filepath: str) -> bool:
    """Save data to JSON file with error handling"""
    try:
        serializable_data = convert_to_serializable(data)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Failed to save JSON file {filepath}: {e}")
        return False

def load_json_file(filepath: str) -> Optional[Dict[str, Any]]:
    """Load data from JSON file with error handling"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON file {filepath}: {e}")
    return None

def generate_filename(prefix: str, extension: str, position: str = "") -> str:
    """Generate a timestamped filename"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if position:
        return f"{prefix}_{position}_{timestamp}.{extension}"
    return f"{prefix}_{timestamp}.{extension}"

def ensure_directory(directory: str) -> bool:
    """Ensure directory exists, create if it doesn't"""
    try:
        os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {directory}: {e}")
        return False

def cleanup_old_files(directory: str, max_age_days: int = 30) -> int:
    """Clean up old files in directory"""
    if not os.path.exists(directory):
        return 0
    
    cleaned_count = 0
    current_time = time.time()
    max_age_seconds = max_age_days * 24 * 3600
    
    try:
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                file_age = current_time - os.path.getmtime(filepath)
                if file_age > max_age_seconds:
                    os.remove(filepath)
                    cleaned_count += 1
    except Exception as e:
        logger.error(f"Error cleaning up files in {directory}: {e}")
    
    return cleaned_count

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable format"""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

def calculate_confidence_score(scores: List[float], weights: Optional[List[float]] = None) -> float:
    """Calculate weighted confidence score"""
    if not scores:
        return 0.0
    
    if weights and len(weights) == len(scores):
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        weight_sum = sum(weights)
        return weighted_sum / weight_sum if weight_sum > 0 else 0.0
    
    return sum(scores) / len(scores)

def validate_text_input(text: str, min_length: int = 10, max_length: int = 10000) -> tuple[bool, str]:
    """Validate text input with length constraints"""
    if not text or not text.strip():
        return False, "Text cannot be empty"
    
    text_length = len(text.strip())
    if text_length < min_length:
        return False, f"Text must be at least {min_length} characters long"
    
    if text_length > max_length:
        return False, f"Text must be no more than {max_length} characters long"
    
    return True, "Valid"

def get_system_info() -> Dict[str, Any]:
    """Get basic system information for debugging"""
    import platform
    import psutil
    
    try:
        return {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "disk_free_gb": round(psutil.disk_usage('.').free / (1024**3), 2)
        }
    except Exception:
        return {"error": "Could not retrieve system info"}

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_cached_system_metrics() -> Dict[str, Any]:
    """Get cached system metrics for performance monitoring"""
    try:
        import psutil
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('.').percent,
            "timestamp": datetime.now().isoformat()
        }
    except Exception:
        return {"error": "Could not retrieve metrics"}

def log_performance_metric(operation: str, duration: float, success: bool = True) -> None:
    """Log performance metrics for monitoring"""
    logger.info(f"Performance: {operation} took {duration:.2f}s, success: {success}")

class PerformanceTimer:
    """Context manager for timing operations"""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        success = exc_type is None
        log_performance_metric(self.operation_name, duration, success)

def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers with default for zero division"""
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ValueError):
        return default

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length with suffix"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix
