"""
Configuration file for the AI Interview System
"""

import os
from typing import Dict, List

# Application Configuration
APP_CONFIG = {
    "title": "AI Interview System",
    "page_icon": "🎤",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# AI Detection Configuration
AI_DETECTION_CONFIG = {
    "confidence_thresholds": {
        "ai_score": 0.7,
        "plagiarism": 0.5
    },
    "cache_size": 100,
    "min_text_length": 50,
    "eden_ai_min_length": 100
}

# Security Monitor Configuration
SECURITY_CONFIG = {
    "warning_cooldown": 5,
    "face_detection_confidence": 0.5,
    "face_tracking_confidence": 0.5,
    "max_face_offset": 0.1
}

# Interview Configuration
INTERVIEW_CONFIG = {
    "stages": {
        'introduction': {
            'name': 'Introduction',
            'description': 'Getting to know you and your background'
        },
        'technical': {
            'name': 'Technical Assessment',
            'description': 'Evaluating technical skills and knowledge'
        },
        'behavioral': {
            'name': 'Behavioral Questions',
            'description': 'Understanding your work style and experiences'
        },
        'role_specific': {
            'name': 'Role-Specific Discussion',
            'description': 'Questions specific to the position'
        },
        'closing': {
            'name': 'Closing Discussion',
            'description': 'Final thoughts and next steps'
        }
    },
    "questions_per_stage": 3,
    "max_response_time": 300,  # 5 minutes
    "skip_phrases": [
        "i don't know",
        "i dont know",
        "don't know",
        "dont know",
        "ask something else",
        "different question",
        "skip this",
        "another question",
        "can we skip",
        "next question"
    ]
}

# OpenAI Configuration
OPENAI_CONFIG = {
    "model": "gpt-4",
    "max_tokens": 1000,
    "temperature": 0.7,
    "timeout": 30
}

# Eden AI Configuration
EDEN_AI_CONFIG = {
    "url": "https://api.edenai.run/v2/text/ai_detection",
    "default_providers": "sapling,writer,originalityai",
    "timeout": 15,
    "fallback_providers": ""
}

# Prohibited Objects for YOLO Detection
PROHIBITED_OBJECTS = {
    'cell phone': 'Mobile phone detected',
    'laptop': 'Laptop detected',
    'book': 'Book detected',
    'tablet': 'Tablet detected',
    'remote': 'Remote control detected',
    'keyboard': 'External keyboard detected',
    'tv': 'Television detected'
}

# File Paths
PATHS = {
    "reports_dir": "interview_reports",
    "session_data_dir": "session_data",
    "yolo_model": "yolov8n.pt",
    "haarcascade": "haarcascade_frontalface_default.xml"
}

# Performance Settings
PERFORMANCE_CONFIG = {
    "video_fps": 10,  # Reduced FPS for better performance
    "cache_ttl": 3600,  # 1 hour cache TTL
    "max_video_width": 640,
    "max_video_height": 480,
    "audio_timeout": 15,
    "max_concurrent_threads": 3
}

# UI Configuration
UI_CONFIG = {
    "sidebar_width": 300,
    "video_width": 640,
    "video_height": 480,
    "chart_width": 600,
    "chart_height": 400
}

def get_env_var(key: str, default: str = "") -> str:
    """Get environment variable with default value"""
    return os.getenv(key, default)

def validate_config() -> bool:
    """Validate configuration settings"""
    try:
        # Check required directories exist or can be created
        for path in PATHS.values():
            if not os.path.exists(path) and not path.endswith('.pt') and not path.endswith('.xml'):
                os.makedirs(path, exist_ok=True)
        return True
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False
