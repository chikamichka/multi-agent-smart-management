"""
Tools for Access Control Agent
RFID reading, facial recognition, access logging, and door control
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from tools.base_tool import BaseTool
from typing import Dict, Any, List, Optional
import random
from datetime import datetime, timedelta


class RFIDReaderTool(BaseTool):
    """Reads RFID badges and validates access"""
    
    def __init__(self):
        super().__init__(
            name="rfid_reader",
            description="Reads RFID badges and validates user credentials"
        )
        
        # Simulated badge database
        self.valid_badges = {
            "BADGE001": {"name": "John Doe", "role": "employee", "access_level": 2},
            "BADGE002": {"name": "Jane Smith", "role": "manager", "access_level": 3},
            "BADGE003": {"name": "Bob Wilson", "role": "security", "access_level": 4},
            "BADGE004": {"name": "Alice Brown", "role": "visitor", "access_level": 1},
        }
    
    def execute(self, badge_id: str = None, location: str = "entrance", **kwargs) -> Dict[str, Any]:
        """
        Read and validate RFID badge
        
        Args:
            badge_id: Badge identifier (if None, simulates random scan)
            location: Access point location
        
        Returns:
            Validation results
        """
        # Simulate badge scan
        if badge_id is None:
            # Random badge or invalid scan
            if random.random() < 0.8:  # 80% valid scans
                badge_id = random.choice(list(self.valid_badges.keys()))
            else:
                badge_id = f"INVALID{random.randint(100, 999)}"
        
        # Validate badge
        if badge_id in self.valid_badges:
            user_info = self.valid_badges[badge_id]
            
            return {
                "success": True,
                "data": {
                    "badge_id": badge_id,
                    "valid": True,
                    "user_name": user_info["name"],
                    "role": user_info["role"],
                    "access_level": user_info["access_level"],
                    "location": location,
                    "timestamp": datetime.now().isoformat(),
                    "message": f"Access granted for {user_info['name']}"
                }
            }
        else:
            return {
                "success": True,
                "data": {
                    "badge_id": badge_id,
                    "valid": False,
                    "location": location,
                    "timestamp": datetime.now().isoformat(),
                    "message": "Invalid badge - access denied"
                }
            }


class FacialRecognitionTool(BaseTool):
    """Performs facial recognition for access control"""
    
    def __init__(self):
        super().__init__(
            name="facial_recognition",
            description="Performs facial recognition and verifies identity"
        )
        
        # Simulated face database
        self.known_faces = {
            "face_001": {"name": "John Doe", "confidence_threshold": 0.85},
            "face_002": {"name": "Jane Smith", "confidence_threshold": 0.85},
            "face_003": {"name": "Bob Wilson", "confidence_threshold": 0.85},
        }
    
    def execute(self, image_path: str = None, location: str = "entrance", **kwargs) -> Dict[str, Any]:
        """
        Perform facial recognition
        
        Args:
            image_path: Path to image (simulated)
            location: Camera location
        
        Returns:
            Recognition results
        """
        # Simulate facial recognition
        if random.random() < 0.7:  # 70% recognition rate
            face_id = random.choice(list(self.known_faces.keys()))
            face_info = self.known_faces[face_id]
            confidence = random.uniform(0.8, 0.99)
            
            recognized = confidence >= face_info["confidence_threshold"]
            
            return {
                "success": True,
                "data": {
                    "recognized": recognized,
                    "face_id": face_id if recognized else "unknown",
                    "name": face_info["name"] if recognized else "Unknown",
                    "confidence": round(confidence, 2),
                    "location": location,
                    "timestamp": datetime.now().isoformat(),
                    "message": f"Recognized: {face_info['name']}" if recognized else "Face not recognized"
                }
            }
        else:
            return {
                "success": True,
                "data": {
                    "recognized": False,
                    "face_id": "unknown",
                    "name": "Unknown",
                    "confidence": 0.0,
                    "location": location,
                    "timestamp": datetime.now().isoformat(),
                    "message": "No face detected"
                }
            }


class AccessLoggerTool(BaseTool):
    """Logs access events and maintains audit trail"""
    
    def __init__(self):
        super().__init__(
            name="access_logger",
            description="Logs all access events and maintains audit trail"
        )
        self.access_log = []
    
    def execute(
        self,
        user_id: str,
        action: str,
        location: str,
        result: str,
        details: Dict[str, Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Log an access event
        
        Args:
            user_id: User identifier
            action: Action performed (entry, exit, denied)
            location: Location of access attempt
            result: Result (granted, denied)
            details: Additional details
        
        Returns:
            Logging status
        """
        log_entry = {
            "log_id": f"log_{len(self.access_log) + 1:06d}",
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "action": action,
            "location": location,
            "result": result,
            "details": details or {}
        }
        
        self.access_log.append(log_entry)
        
        return {
            "success": True,
            "data": {
                "log_id": log_entry["log_id"],
                "logged": True,
                "total_entries": len(self.access_log)
            }
        }
    
    def get_logs(
        self,
        user_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """Get access logs with optional filtering"""
        logs = self.access_log
        
        if user_id:
            logs = [log for log in logs if log["user_id"] == user_id]
        
        return logs[-limit:]
    
    def get_suspicious_activity(self) -> List[Dict]:
        """Identify suspicious access patterns"""
        suspicious = []
        
        # Check for multiple failed attempts
        failed_attempts = {}
        for log in self.access_log[-50:]:  # Check last 50 entries
            if log["result"] == "denied":
                user = log["user_id"]
                failed_attempts[user] = failed_attempts.get(user, 0) + 1
        
        for user, count in failed_attempts.items():
            if count >= 3:
                suspicious.append({
                    "type": "multiple_failed_attempts",
                    "user_id": user,
                    "count": count,
                    "severity": "high"
                })
        
        return suspicious


class DoorControllerTool(BaseTool):
    """Controls electronic door locks"""
    
    def __init__(self):
        super().__init__(
            name="door_controller",
            description="Controls electronic door locks and monitors door status"
        )
        
        self.door_states = {
            "entrance": "locked",
            "exit": "locked",
            "parking": "locked",
            "warehouse": "locked"
        }
    
    def execute(
        self,
        door_id: str,
        action: str,
        duration: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Control door lock
        
        Args:
            door_id: Door identifier
            action: Action to perform (unlock, lock, status)
            duration: Auto-lock duration in seconds (for unlock)
        
        Returns:
            Door control result
        """
        if action == "status":
            state = self.door_states.get(door_id, "unknown")
            return {
                "success": True,
                "data": {
                    "door_id": door_id,
                    "state": state,
                    "timestamp": datetime.now().isoformat()
                }
            }
        
        elif action == "unlock":
            self.door_states[door_id] = "unlocked"
            
            auto_lock_time = None
            if duration:
                auto_lock_time = (datetime.now() + timedelta(seconds=duration)).isoformat()
            
            return {
                "success": True,
                "data": {
                    "door_id": door_id,
                    "action": "unlocked",
                    "duration": duration,
                    "auto_lock_at": auto_lock_time,
                    "timestamp": datetime.now().isoformat()
                }
            }
        
        elif action == "lock":
            self.door_states[door_id] = "locked"
            
            return {
                "success": True,
                "data": {
                    "door_id": door_id,
                    "action": "locked",
                    "timestamp": datetime.now().isoformat()
                }
            }
        
        else:
            return {
                "success": False,
                "error": f"Invalid action: {action}"
            }


def demo():
    """Demo access control tools"""
    print("="*60)
    print("Access Control Tools Demo")
    print("="*60 + "\n")
    
    # Test RFID Reader
    print("1. Testing RFID Reader...")
    rfid_tool = RFIDReaderTool()
    
    for i in range(3):
        result = rfid_tool(location="main_entrance")
        if result['data']['valid']:
            print(f"   ✅ {result['data']['message']} (Level {result['data']['access_level']})")
        else:
            print(f"   ❌ {result['data']['message']}")
    print()
    
    # Test Facial Recognition
    print("2. Testing Facial Recognition...")
    face_tool = FacialRecognitionTool()
    
    for i in range(3):
        result = face_tool(location="entrance_camera")
        if result['data']['recognized']:
            print(f"   ✅ {result['data']['message']} (Confidence: {result['data']['confidence']})")
        else:
            print(f"   ❌ {result['data']['message']}")
    print()
    
    # Test Access Logger
    print("3. Testing Access Logger...")
    logger_tool = AccessLoggerTool()
    
    # Log some events
    logger_tool(user_id="BADGE001", action="entry", location="entrance", result="granted")
    logger_tool(user_id="INVALID999", action="entry", location="entrance", result="denied")
    logger_tool(user_id="INVALID999", action="entry", location="entrance", result="denied")
    logger_tool(user_id="INVALID999", action="entry", location="entrance", result="denied")
    
    logs = logger_tool.get_logs(limit=5)
    print(f"   Total logs: {len(logs)}")
    print(f"   Last log: {logs[-1]['action']} by {logs[-1]['user_id']} - {logs[-1]['result']}")
    
    suspicious = logger_tool.get_suspicious_activity()
    if suspicious:
        print(f"   ⚠️  Suspicious activity detected: {suspicious[0]}")
    print()
    
    # Test Door Controller
    print("4. Testing Door Controller...")
    door_tool = DoorControllerTool()
    
    # Check status
    result = door_tool(door_id="entrance", action="status")
    print(f"   Door status: {result['data']['state']}")
    
    # Unlock door
    result = door_tool(door_id="entrance", action="unlock", duration=10)
    print(f"   Action: {result['data']['action']}, Auto-lock in {result['data']['duration']}s")
    
    # Lock door
    result = door_tool(door_id="entrance", action="lock")
    print(f"   Action: {result['data']['action']}")
    print()
    
    # Show tool stats
    print("5. Tool Statistics:")
    for tool in [rfid_tool, face_tool, logger_tool, door_tool]:
        stats = tool.get_stats()
        print(f"   {stats['name']}: {stats['executions']} executions, "
              f"avg {stats['avg_time']}s")
    
    print("\n" + "="*60)
    print("✅ Access Control Tools Demo Complete!")
    print("="*60)


if __name__ == "__main__":
    demo()