"""
Tools for Surveillance Agent
Camera analysis, motion detection, and alerting
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from tools.base_tool import BaseTool
from typing import Dict, Any, List
import random
from datetime import datetime
import numpy as np


class CameraAnalyzerTool(BaseTool):
    """Analyzes camera feeds for objects and anomalies"""
    
    def __init__(self):
        super().__init__(
            name="camera_analyzer",
            description="Analyzes camera feed and detects objects, people, and anomalies"
        )
        self.confidence_threshold = 0.7
    
    def execute(self, camera_id: str = "cam_001", **kwargs) -> Dict[str, Any]:
        """
        Simulate camera analysis
        
        Args:
            camera_id: Camera identifier
        
        Returns:
            Analysis results with detected objects
        """
        # Simulate camera feed analysis
        # In production, this would use YOLO, OpenCV, etc.
        
        detected_objects = []
        
        # Simulate object detection
        possible_objects = [
            {"type": "person", "confidence": 0.95, "location": "entrance"},
            {"type": "vehicle", "confidence": 0.88, "location": "parking"},
            {"type": "package", "confidence": 0.76, "location": "doorstep"},
        ]
        
        # Randomly detect some objects
        for obj in possible_objects:
            if random.random() < 0.3:  # 30% chance to detect
                detected_objects.append(obj)
        
        # Check for anomalies
        anomalies = []
        
        # Simulate anomaly detection (person at odd hours, etc.)
        hour = datetime.now().hour
        if detected_objects and (hour < 6 or hour > 22):
            anomalies.append({
                "type": "unusual_activity",
                "description": f"Person detected at {hour}:00 (off-hours)",
                "severity": "medium"
            })
        
        return {
            "success": True,
            "data": {
                "camera_id": camera_id,
                "timestamp": datetime.now().isoformat(),
                "detected_objects": detected_objects,
                "anomalies": anomalies,
                "frame_quality": random.uniform(0.8, 1.0),
                "summary": f"Detected {len(detected_objects)} objects, {len(anomalies)} anomalies"
            }
        }


class MotionDetectorTool(BaseTool):
    """Detects motion in camera zones"""
    
    def __init__(self):
        super().__init__(
            name="motion_detector",
            description="Detects motion in specified camera zones and tracks movement patterns"
        )
        self.sensitivity = 0.5
    
    def execute(self, zone: str = "entrance", **kwargs) -> Dict[str, Any]:
        """
        Detect motion in a zone
        
        Args:
            zone: Zone to monitor (entrance, parking, hallway, etc.)
        
        Returns:
            Motion detection results
        """
        # Simulate motion detection
        motion_detected = random.random() < 0.2  # 20% chance
        
        if motion_detected:
            intensity = random.uniform(0.3, 1.0)
            duration = random.uniform(1.0, 10.0)
            
            # Classify motion type based on intensity and duration
            if intensity > 0.8 and duration > 5:
                motion_type = "significant"
            elif intensity > 0.5:
                motion_type = "moderate"
            else:
                motion_type = "minor"
            
            return {
                "success": True,
                "data": {
                    "motion_detected": True,
                    "zone": zone,
                    "intensity": round(intensity, 2),
                    "duration": round(duration, 2),
                    "motion_type": motion_type,
                    "timestamp": datetime.now().isoformat(),
                    "requires_investigation": motion_type == "significant"
                }
            }
        else:
            return {
                "success": True,
                "data": {
                    "motion_detected": False,
                    "zone": zone,
                    "timestamp": datetime.now().isoformat()
                }
            }


class AlertSenderTool(BaseTool):
    """Sends alerts for security events"""
    
    def __init__(self):
        super().__init__(
            name="alert_sender",
            description="Sends alerts via various channels (SMS, email, push notification)"
        )
        self.alert_history = []
    
    def execute(
        self, 
        message: str, 
        severity: str = "medium",
        channels: List[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send an alert
        
        Args:
            message: Alert message
            severity: low, medium, high, critical
            channels: List of channels (sms, email, push)
        
        Returns:
            Alert sending status
        """
        if channels is None:
            channels = ["push"]
        
        # Validate severity
        valid_severities = ["low", "medium", "high", "critical"]
        if severity not in valid_severities:
            severity = "medium"
        
        alert = {
            "id": f"alert_{len(self.alert_history) + 1}",
            "message": message,
            "severity": severity,
            "channels": channels,
            "timestamp": datetime.now().isoformat(),
            "status": "sent"
        }
        
        self.alert_history.append(alert)
        
        return {
            "success": True,
            "data": {
                "alert_id": alert["id"],
                "message": message,
                "severity": severity,
                "channels_notified": len(channels),
                "timestamp": alert["timestamp"]
            }
        }
    
    def get_alert_history(self, limit: int = 10) -> List[Dict]:
        """Get recent alerts"""
        return self.alert_history[-limit:]


def demo():
    """Demo surveillance tools"""
    print("="*60)
    print("Surveillance Tools Demo")
    print("="*60 + "\n")
    
    # Test Camera Analyzer
    print("1. Testing Camera Analyzer...")
    camera_tool = CameraAnalyzerTool()
    result = camera_tool(camera_id="cam_entrance_01")
    
    print(f"   Success: {result['success']}")
    print(f"   Objects detected: {len(result['data']['detected_objects'])}")
    print(f"   Anomalies: {len(result['data']['anomalies'])}")
    if result['data']['detected_objects']:
        print(f"   First object: {result['data']['detected_objects'][0]}")
    print()
    
    # Test Motion Detector
    print("2. Testing Motion Detector...")
    motion_tool = MotionDetectorTool()
    
    for i in range(5):
        result = motion_tool(zone="entrance")
        if result['data']['motion_detected']:
            print(f"   ⚠️  Motion detected! Type: {result['data']['motion_type']}, "
                  f"Intensity: {result['data']['intensity']}")
        else:
            print(f"   ✅ No motion detected")
    print()
    
    # Test Alert Sender
    print("3. Testing Alert Sender...")
    alert_tool = AlertSenderTool()
    
    result = alert_tool(
        message="Unauthorized access detected at entrance",
        severity="high",
        channels=["sms", "email", "push"]
    )
    
    print(f"   Alert sent: {result['success']}")
    print(f"   Alert ID: {result['data']['alert_id']}")
    print(f"   Channels: {result['data']['channels_notified']}")
    print()
    
    # Show tool stats
    print("4. Tool Statistics:")
    for tool in [camera_tool, motion_tool, alert_tool]:
        stats = tool.get_stats()
        print(f"   {stats['name']}: {stats['executions']} executions, "
              f"avg {stats['avg_time']}s")
    
    print("\n" + "="*60)
    print("✅ Surveillance Tools Demo Complete!")
    print("="*60)


if __name__ == "__main__":
    demo()