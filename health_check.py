"""
Health check module for the AI Interview System
Provides system status and diagnostics
"""

import streamlit as st
import time
import psutil
import os
from datetime import datetime
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def get_system_health() -> Dict[str, Any]:
    """Get comprehensive system health information"""
    try:
        # Basic system info
        health_data = {
            "timestamp": datetime.now().isoformat(),
            "status": "healthy",
            "uptime": time.time(),
            "system": {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('.').percent,
                "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None
            },
            "application": {
                "streamlit_version": st.__version__,
                "python_version": f"{psutil.version_info}",
                "process_count": len(psutil.pids())
            }
        }
        
        # Check critical thresholds
        if health_data["system"]["cpu_percent"] > 90:
            health_data["status"] = "warning"
            health_data["warnings"] = health_data.get("warnings", [])
            health_data["warnings"].append("High CPU usage")
        
        if health_data["system"]["memory_percent"] > 90:
            health_data["status"] = "warning"
            health_data["warnings"] = health_data.get("warnings", [])
            health_data["warnings"].append("High memory usage")
        
        if health_data["system"]["disk_percent"] > 95:
            health_data["status"] = "critical"
            health_data["errors"] = health_data.get("errors", [])
            health_data["errors"].append("Disk space critical")
        
        return health_data
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error": str(e)
        }

def check_dependencies() -> Dict[str, bool]:
    """Check if all required dependencies are available"""
    dependencies = {
        "streamlit": False,
        "opencv": False,
        "mediapipe": False,
        "openai": False,
        "ultralytics": False,
        "nltk": False,
        "pygame": False,
        "sklearn": False
    }
    
    try:
        import streamlit
        dependencies["streamlit"] = True
    except ImportError:
        pass
    
    try:
        import cv2
        dependencies["opencv"] = True
    except ImportError:
        pass
    
    try:
        import mediapipe
        dependencies["mediapipe"] = True
    except ImportError:
        pass
    
    try:
        import openai
        dependencies["openai"] = True
    except ImportError:
        pass
    
    try:
        import ultralytics
        dependencies["ultralytics"] = True
    except ImportError:
        pass
    
    try:
        import nltk
        dependencies["nltk"] = True
    except ImportError:
        pass
    
    try:
        import pygame
        dependencies["pygame"] = True
    except ImportError:
        pass
    
    try:
        import sklearn
        dependencies["sklearn"] = True
    except ImportError:
        pass
    
    return dependencies

def check_file_permissions() -> Dict[str, bool]:
    """Check file system permissions"""
    permissions = {
        "read_config": False,
        "write_reports": False,
        "write_sessions": False,
        "read_models": False
    }
    
    try:
        # Check config read
        if os.path.exists("config.py"):
            permissions["read_config"] = os.access("config.py", os.R_OK)
        
        # Check reports write
        os.makedirs("interview_reports", exist_ok=True)
        permissions["write_reports"] = os.access("interview_reports", os.W_OK)
        
        # Check sessions write
        os.makedirs("session_data", exist_ok=True)
        permissions["write_sessions"] = os.access("session_data", os.W_OK)
        
        # Check models read
        if os.path.exists("yolov8n.pt"):
            permissions["read_models"] = os.access("yolov8n.pt", os.R_OK)
        else:
            permissions["read_models"] = True  # Will be downloaded if needed
            
    except Exception as e:
        logger.error(f"Permission check failed: {e}")
    
    return permissions

def display_health_dashboard():
    """Display health dashboard in Streamlit"""
    st.title("🏥 System Health Dashboard")
    
    # Get health data
    health_data = get_system_health()
    dependencies = check_dependencies()
    permissions = check_file_permissions()
    
    # Status overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status = health_data.get("status", "unknown")
        if status == "healthy":
            st.success(f"Status: {status.upper()}")
        elif status == "warning":
            st.warning(f"Status: {status.upper()}")
        else:
            st.error(f"Status: {status.upper()}")
    
    with col2:
        st.metric("CPU Usage", f"{health_data.get('system', {}).get('cpu_percent', 0):.1f}%")
    
    with col3:
        st.metric("Memory Usage", f"{health_data.get('system', {}).get('memory_percent', 0):.1f}%")
    
    # System metrics
    st.subheader("📊 System Metrics")
    
    if "system" in health_data:
        metrics_col1, metrics_col2 = st.columns(2)
        
        with metrics_col1:
            st.metric("Disk Usage", f"{health_data['system'].get('disk_percent', 0):.1f}%")
            st.metric("Process Count", health_data.get('application', {}).get('process_count', 0))
        
        with metrics_col2:
            if health_data['system'].get('load_average'):
                load_avg = health_data['system']['load_average']
                st.metric("Load Average", f"{load_avg[0]:.2f}, {load_avg[1]:.2f}, {load_avg[2]:.2f}")
    
    # Dependencies status
    st.subheader("📦 Dependencies")
    
    dep_cols = st.columns(4)
    dep_items = list(dependencies.items())
    
    for i, (dep, status) in enumerate(dep_items):
        with dep_cols[i % 4]:
            if status:
                st.success(f"✅ {dep}")
            else:
                st.error(f"❌ {dep}")
    
    # Permissions status
    st.subheader("🔐 File Permissions")
    
    perm_cols = st.columns(2)
    perm_items = list(permissions.items())
    
    for i, (perm, status) in enumerate(perm_items):
        with perm_cols[i % 2]:
            if status:
                st.success(f"✅ {perm.replace('_', ' ').title()}")
            else:
                st.error(f"❌ {perm.replace('_', ' ').title()}")
    
    # Warnings and errors
    if health_data.get("warnings"):
        st.subheader("⚠️ Warnings")
        for warning in health_data["warnings"]:
            st.warning(warning)
    
    if health_data.get("errors"):
        st.subheader("🚨 Errors")
        for error in health_data["errors"]:
            st.error(error)
    
    # Raw data
    with st.expander("🔍 Raw Health Data"):
        st.json(health_data)
    
    # Auto-refresh
    if st.button("🔄 Refresh"):
        st.rerun()
    
    # Auto-refresh every 30 seconds
    time.sleep(30)
    st.rerun()

if __name__ == "__main__":
    display_health_dashboard()
