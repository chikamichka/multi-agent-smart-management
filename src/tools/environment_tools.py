"""
Tools for Environmental Monitoring Agent
Sensor reading, HVAC control, irrigation, and data analysis
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from tools.base_tool import BaseTool
from typing import Dict, Any, List, Optional
import random
from datetime import datetime


class SensorReaderTool(BaseTool):
    """Reads environmental sensors"""
    
    def __init__(self):
        super().__init__(
            name="sensor_reader",
            description="Reads temperature, humidity, CO2, and other environmental sensors"
        )
        
        # Baseline values for simulation
        self.baselines = {
            "temperature": 22.0,
            "humidity": 65.0,
            "co2": 450.0,
            "light": 500.0,
            "soil_moisture": 45.0
        }
    
    def execute(
        self,
        zone: str = "main_area",
        sensor_types: List[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Read environmental sensors
        
        Args:
            zone: Zone to read (main_area, greenhouse, warehouse, etc.)
            sensor_types: List of sensor types to read
        
        Returns:
            Sensor readings
        """
        if sensor_types is None:
            sensor_types = ["temperature", "humidity", "co2"]
        
        readings = {}
        
        for sensor_type in sensor_types:
            if sensor_type not in self.baselines:
                continue
            
            baseline = self.baselines[sensor_type]
            
            # Add random variation
            variation = random.uniform(-0.1, 0.1) * baseline
            value = baseline + variation
            
            # Occasionally create anomalies
            if random.random() < 0.1:  # 10% chance
                value += random.uniform(-0.2, 0.3) * baseline
            
            # Determine status
            status = "normal"
            if sensor_type == "temperature":
                if value < 18 or value > 26:
                    status = "warning"
                if value < 15 or value > 30:
                    status = "critical"
            elif sensor_type == "humidity":
                if value < 40 or value > 80:
                    status = "warning"
                if value < 30 or value > 90:
                    status = "critical"
            elif sensor_type == "co2":
                if value > 800:
                    status = "warning"
                if value > 1200:
                    status = "critical"
            
            readings[sensor_type] = {
                "value": round(value, 2),
                "unit": self._get_unit(sensor_type),
                "status": status,
                "timestamp": datetime.now().isoformat()
            }
        
        return {
            "success": True,
            "data": {
                "zone": zone,
                "readings": readings,
                "summary": f"Read {len(readings)} sensors in {zone}"
            }
        }
    
    def _get_unit(self, sensor_type: str) -> str:
        """Get unit for sensor type"""
        units = {
            "temperature": "°C",
            "humidity": "%",
            "co2": "ppm",
            "light": "lux",
            "soil_moisture": "%"
        }
        return units.get(sensor_type, "")


class HVACControllerTool(BaseTool):
    """Controls HVAC system"""
    
    def __init__(self):
        super().__init__(
            name="hvac_controller",
            description="Controls heating, ventilation, and air conditioning systems"
        )
        
        self.hvac_state = {
            "main_area": {"mode": "auto", "target_temp": 22.0, "fan_speed": "medium"},
            "greenhouse": {"mode": "cooling", "target_temp": 24.0, "fan_speed": "high"},
            "warehouse": {"mode": "off", "target_temp": 20.0, "fan_speed": "low"}
        }
    
    def execute(
        self,
        zone: str,
        action: str = "status",
        target_temp: Optional[float] = None,
        mode: Optional[str] = None,
        fan_speed: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Control HVAC system
        
        Args:
            zone: Zone to control
            action: Action (status, set_temperature, set_mode, set_fan)
            target_temp: Target temperature
            mode: HVAC mode (auto, heating, cooling, off)
            fan_speed: Fan speed (low, medium, high)
        
        Returns:
            HVAC control result
        """
        if zone not in self.hvac_state:
            self.hvac_state[zone] = {"mode": "auto", "target_temp": 22.0, "fan_speed": "medium"}
        
        state = self.hvac_state[zone]
        
        if action == "status":
            return {
                "success": True,
                "data": {
                    "zone": zone,
                    "current_state": state,
                    "timestamp": datetime.now().isoformat()
                }
            }
        
        elif action == "set_temperature":
            if target_temp is not None:
                old_temp = state["target_temp"]
                state["target_temp"] = target_temp
                
                return {
                    "success": True,
                    "data": {
                        "zone": zone,
                        "action": "temperature_updated",
                        "old_temperature": old_temp,
                        "new_temperature": target_temp,
                        "timestamp": datetime.now().isoformat()
                    }
                }
        
        elif action == "set_mode":
            if mode in ["auto", "heating", "cooling", "off"]:
                old_mode = state["mode"]
                state["mode"] = mode
                
                return {
                    "success": True,
                    "data": {
                        "zone": zone,
                        "action": "mode_updated",
                        "old_mode": old_mode,
                        "new_mode": mode,
                        "timestamp": datetime.now().isoformat()
                    }
                }
        
        elif action == "set_fan":
            if fan_speed in ["low", "medium", "high"]:
                old_speed = state["fan_speed"]
                state["fan_speed"] = fan_speed
                
                return {
                    "success": True,
                    "data": {
                        "zone": zone,
                        "action": "fan_updated",
                        "old_speed": old_speed,
                        "new_speed": fan_speed,
                        "timestamp": datetime.now().isoformat()
                    }
                }
        
        return {
            "success": False,
            "error": f"Invalid action or parameters"
        }


class IrrigationControllerTool(BaseTool):
    """Controls irrigation systems"""
    
    def __init__(self):
        super().__init__(
            name="irrigation_controller",
            description="Controls automated irrigation and watering systems"
        )
        
        self.irrigation_zones = {
            "greenhouse_zone1": {"status": "off", "schedule": "morning", "duration": 15},
            "greenhouse_zone2": {"status": "off", "schedule": "evening", "duration": 10},
            "outdoor_garden": {"status": "off", "schedule": "morning", "duration": 20}
        }
    
    def execute(
        self,
        zone: str,
        action: str = "status",
        duration: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Control irrigation system
        
        Args:
            zone: Irrigation zone
            action: Action (status, start, stop, schedule)
            duration: Watering duration in minutes
        
        Returns:
            Irrigation control result
        """
        if zone not in self.irrigation_zones:
            self.irrigation_zones[zone] = {"status": "off", "schedule": "manual", "duration": 10}
        
        zone_state = self.irrigation_zones[zone]
        
        if action == "status":
            return {
                "success": True,
                "data": {
                    "zone": zone,
                    "status": zone_state["status"],
                    "schedule": zone_state["schedule"],
                    "duration": zone_state["duration"],
                    "timestamp": datetime.now().isoformat()
                }
            }
        
        elif action == "start":
            zone_state["status"] = "active"
            if duration:
                zone_state["duration"] = duration
            
            return {
                "success": True,
                "data": {
                    "zone": zone,
                    "action": "irrigation_started",
                    "duration": zone_state["duration"],
                    "estimated_completion": f"in {zone_state['duration']} minutes",
                    "timestamp": datetime.now().isoformat()
                }
            }
        
        elif action == "stop":
            zone_state["status"] = "off"
            
            return {
                "success": True,
                "data": {
                    "zone": zone,
                    "action": "irrigation_stopped",
                    "timestamp": datetime.now().isoformat()
                }
            }
        
        return {
            "success": False,
            "error": f"Invalid action: {action}"
        }


class DataAnalyzerTool(BaseTool):
    """Analyzes environmental data trends"""
    
    def __init__(self):
        super().__init__(
            name="data_analyzer",
            description="Analyzes environmental data patterns and trends"
        )
        self.historical_data = []
    
    def execute(
        self,
        data: Dict[str, Any],
        analysis_type: str = "trend",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze environmental data
        
        Args:
            data: Data to analyze
            analysis_type: Type of analysis (trend, anomaly, comparison)
        
        Returns:
            Analysis results
        """
        # Store data
        self.historical_data.append({
            "timestamp": datetime.now().isoformat(),
            "data": data
        })
        
        # Keep only recent data
        if len(self.historical_data) > 100:
            self.historical_data = self.historical_data[-100:]
        
        if analysis_type == "trend":
            # Analyze trends
            trends = {}
            
            if "readings" in data:
                for sensor, reading in data["readings"].items():
                    if reading["status"] != "normal":
                        trends[sensor] = {
                            "status": reading["status"],
                            "value": reading["value"],
                            "recommendation": self._get_recommendation(sensor, reading)
                        }
            
            return {
                "success": True,
                "data": {
                    "analysis_type": "trend",
                    "trends": trends,
                    "data_points": len(self.historical_data),
                    "concerns": len(trends),
                    "timestamp": datetime.now().isoformat()
                }
            }
        
        elif analysis_type == "summary":
            # Provide summary statistics
            return {
                "success": True,
                "data": {
                    "analysis_type": "summary",
                    "total_datapoints": len(self.historical_data),
                    "latest_reading": data,
                    "timestamp": datetime.now().isoformat()
                }
            }
        
        return {
            "success": False,
            "error": f"Unknown analysis type: {analysis_type}"
        }
    
    def _get_recommendation(self, sensor: str, reading: Dict) -> str:
        """Get recommendation based on sensor reading"""
        if sensor == "temperature":
            if reading["value"] > 26:
                return "Consider increasing cooling or ventilation"
            elif reading["value"] < 18:
                return "Consider increasing heating"
        elif sensor == "humidity":
            if reading["value"] > 80:
                return "Increase ventilation to reduce humidity"
            elif reading["value"] < 40:
                return "Consider using humidifier"
        elif sensor == "co2":
            if reading["value"] > 800:
                return "Increase fresh air circulation"
        
        return "Monitor closely"


def demo():
    """Demo environmental tools"""
    print("="*60)
    print("Environmental Monitoring Tools Demo")
    print("="*60 + "\n")
    
    # Test Sensor Reader
    print("1. Testing Sensor Reader...")
    sensor_tool = SensorReaderTool()
    
    result = sensor_tool(
        zone="greenhouse",
        sensor_types=["temperature", "humidity", "co2", "soil_moisture"]
    )
    
    print(f"   Zone: {result['data']['zone']}")
    for sensor, reading in result['data']['readings'].items():
        status_emoji = "✅" if reading['status'] == 'normal' else "⚠️"
        print(f"   {status_emoji} {sensor}: {reading['value']}{reading['unit']} ({reading['status']})")
    print()
    
    # Test HVAC Controller
    print("2. Testing HVAC Controller...")
    hvac_tool = HVACControllerTool()
    
    # Check status
    result = hvac_tool(zone="greenhouse", action="status")
    print(f"   Current: {result['data']['current_state']}")
    
    # Set temperature
    result = hvac_tool(zone="greenhouse", action="set_temperature", target_temp=26.0)
    print(f"   {result['data']['action']}: {result['data']['old_temperature']}°C → {result['data']['new_temperature']}°C")
    
    # Set mode
    result = hvac_tool(zone="greenhouse", action="set_mode", mode="cooling")
    print(f"   Mode changed: {result['data']['old_mode']} → {result['data']['new_mode']}")
    print()
    
    # Test Irrigation Controller
    print("3. Testing Irrigation Controller...")
    irrigation_tool = IrrigationControllerTool()
    
    # Check status
    result = irrigation_tool(zone="greenhouse_zone1", action="status")
    print(f"   Status: {result['data']['status']}, Schedule: {result['data']['schedule']}")
    
    # Start irrigation
    result = irrigation_tool(zone="greenhouse_zone1", action="start", duration=15)
    print(f"   {result['data']['action']}, Duration: {result['data']['duration']} min")
    
    # Stop irrigation
    result = irrigation_tool(zone="greenhouse_zone1", action="stop")
    print(f"   {result['data']['action']}")
    print()
    
    # Test Data Analyzer
    print("4. Testing Data Analyzer...")
    analyzer_tool = DataAnalyzerTool()
    
    # Analyze sensor data
    sensor_data = sensor_tool(zone="greenhouse", sensor_types=["temperature", "humidity"])
    result = analyzer_tool(data=sensor_data['data'], analysis_type="trend")
    
    print(f"   Analysis type: {result['data']['analysis_type']}")
    print(f"   Data points: {result['data']['data_points']}")
    print(f"   Concerns: {result['data']['concerns']}")
    
    if result['data']['trends']:
        print(f"   Trends detected:")
        for sensor, trend in result['data']['trends'].items():
            print(f"      - {sensor}: {trend['recommendation']}")
    print()
    
    # Show tool stats
    print("5. Tool Statistics:")
    for tool in [sensor_tool, hvac_tool, irrigation_tool, analyzer_tool]:
        stats = tool.get_stats()
        print(f"   {stats['name']}: {stats['executions']} executions, "
              f"avg {stats['avg_time']}s")
    
    print("\n" + "="*60)
    print("✅ Environmental Tools Demo Complete!")
    print("="*60)


if __name__ == "__main__":
    demo()