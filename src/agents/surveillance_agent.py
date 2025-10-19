"""
Surveillance Agent
Monitors camera feeds, detects anomalies, and manages security alerts
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent
from agents.llm_manager import AgentLLMManager
from state.state_manager import StateManager
from tools.surveillance_tools import CameraAnalyzerTool, MotionDetectorTool, AlertSenderTool
from typing import Dict, Any
from loguru import logger
import json


class SurveillanceAgent(BaseAgent):
    """Agent for surveillance and security monitoring"""
    
    def __init__(
        self,
        llm_manager: AgentLLMManager,
        state_manager: StateManager
    ):
        # Initialize tools
        tools = [
            CameraAnalyzerTool(),
            MotionDetectorTool(),
            AlertSenderTool()
        ]
        
        super().__init__(
            agent_id="surveillance_agent",
            name="Surveillance Agent",
            description="Monitors security cameras, detects suspicious activity, and manages alerts",
            tools=tools,
            llm_manager=llm_manager,
            state_manager=state_manager,
            priority="high"
        )
    
    def analyze_situation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze surveillance data and detect anomalies
        
        Args:
            context: Contains camera_id, zone, or monitoring request
        
        Returns:
            Analysis results
        """
        logger.info(f"{self.name}: Analyzing situation")
        
        # Extract context
        camera_id = context.get('camera_id', 'cam_001')
        zone = context.get('zone', 'entrance')
        
        # Check camera feed
        camera_result = self.execute_tool('camera_analyzer', camera_id=camera_id)
        
        # Check for motion
        motion_result = self.execute_tool('motion_detector', zone=zone)
        
        # Combine results
        combined_data = {
            'camera': camera_result.get('data', {}),
            'motion': motion_result.get('data', {})
        }
        
        # Use LLM to analyze the situation
        prompt = self._generate_analysis_prompt(combined_data)
        llm_analysis = self.llm.generate(prompt, max_tokens=200, temperature=0.3)
        
        # Parse LLM response
        analysis_result = self._parse_analysis(llm_analysis, combined_data)
        
        return {
            'success': True,
            'data': combined_data,
            'assessment': analysis_result,
            'requires_action': analysis_result.get('threat_level', 'low') in ['medium', 'high', 'critical']
        }
    
    def decide_action(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide what action to take based on analysis
        
        Args:
            analysis: Analysis results
        
        Returns:
            Action decision
        """
        logger.info(f"{self.name}: Deciding action")
        
        assessment = analysis.get('assessment', {})
        threat_level = assessment.get('threat_level', 'low')
        
        # Determine if action is needed
        if not analysis.get('requires_action', False):
            return {
                'success': True,
                'action_required': False,
                'reasoning': 'No security concerns detected'
            }
        
        # Use LLM to decide action
        prompt = self._generate_decision_prompt(analysis)
        llm_decision = self.llm.generate(prompt, max_tokens=150, temperature=0.2)
        
        # Parse decision
        decision = self._parse_decision(llm_decision, threat_level)
        
        return decision
    
    def _generate_analysis_prompt(self, data: Dict[str, Any]) -> str:
        """Generate prompt for situation analysis"""
        camera_data = data.get('camera', {})
        motion_data = data.get('motion', {})
        
        prompt = f"""You are {self.name}. Analyze this surveillance data:

Camera Analysis:
- Objects detected: {len(camera_data.get('detected_objects', []))}
- Anomalies: {len(camera_data.get('anomalies', []))}
{f"- Details: {camera_data.get('anomalies', [])}" if camera_data.get('anomalies') else ""}

Motion Detection:
- Motion detected: {motion_data.get('motion_detected', False)}
{f"- Type: {motion_data.get('motion_type', 'none')}" if motion_data.get('motion_detected') else ""}
{f"- Intensity: {motion_data.get('intensity', 0)}" if motion_data.get('motion_detected') else ""}

Assess the threat level (low/medium/high/critical) and provide brief reasoning.

Response format:
THREAT_LEVEL: [level]
REASONING: [brief explanation]

Response:"""
        
        return prompt
    
    def _generate_decision_prompt(self, analysis: Dict[str, Any]) -> str:
        """Generate prompt for action decision"""
        assessment = analysis.get('assessment', {})
        
        prompt = f"""You are {self.name}. Based on this assessment, decide what action to take:

Threat Level: {assessment.get('threat_level', 'unknown')}
Reasoning: {assessment.get('reasoning', 'No details')}

Available actions:
1. alert_sender - Send security alert
2. No action needed

Should you send an alert? If yes, specify severity (low/medium/high/critical).

Response format:
ACTION: [alert_sender or none]
SEVERITY: [if alert, specify severity]
MESSAGE: [alert message if needed]

Response:"""
        
        return prompt
    
    def _parse_analysis(self, llm_response: str, data: Dict) -> Dict[str, Any]:
        """Parse LLM analysis response"""
        lines = llm_response.strip().split('\n')
        
        result = {
            'threat_level': 'low',
            'reasoning': '',
            'raw_response': llm_response
        }
        
        for line in lines:
            if line.startswith('THREAT_LEVEL:'):
                level = line.split(':', 1)[1].strip().lower()
                if level in ['low', 'medium', 'high', 'critical']:
                    result['threat_level'] = level
            elif line.startswith('REASONING:'):
                result['reasoning'] = line.split(':', 1)[1].strip()
        
        # If reasoning is empty, extract from anomalies
        if not result['reasoning'] and data.get('camera', {}).get('anomalies'):
            anomalies = data['camera']['anomalies']
            result['reasoning'] = f"Detected {len(anomalies)} anomalies"
        
        return result
    
    def _parse_decision(self, llm_response: str, threat_level: str) -> Dict[str, Any]:
        """Parse LLM decision response"""
        lines = llm_response.strip().split('\n')
        
        decision = {
            'success': True,
            'action_required': False,
            'tool_name': None,
            'tool_params': {},
            'reasoning': llm_response
        }
        
        action = None
        severity = threat_level
        message = "Security alert"
        
        for line in lines:
            if line.startswith('ACTION:'):
                action = line.split(':', 1)[1].strip().lower()
            elif line.startswith('SEVERITY:'):
                severity = line.split(':', 1)[1].strip().lower()
            elif line.startswith('MESSAGE:'):
                message = line.split(':', 1)[1].strip()
        
        if action and 'alert' in action:
            decision['action_required'] = True
            decision['tool_name'] = 'alert_sender'
            decision['tool_params'] = {
                'message': message if message != "Security alert" else f"Security concern detected - {threat_level} threat level",
                'severity': severity if severity in ['low', 'medium', 'high', 'critical'] else threat_level,
                'channels': ['push', 'sms'] if severity in ['high', 'critical'] else ['push']
            }
        
        return decision


def demo():
    """Demo the surveillance agent"""
    print("="*60)
    print("Surveillance Agent Demo")
    print("="*60 + "\n")
    
    # Initialize dependencies
    from state.state_manager import StateManager
    
    state_manager = StateManager()
    llm_manager = AgentLLMManager(model_name='qwen-1.8b')
    
    # Create agent
    agent = SurveillanceAgent(llm_manager, state_manager)
    
    print(f"Agent initialized: {agent.name}")
    print(f"Available tools: {', '.join(agent.get_available_tools())}")
    print()
    
    # Test task 1: Normal monitoring
    print("1. Testing Normal Monitoring...")
    task1 = {
        'task_id': 'mon_001',
        'camera_id': 'cam_entrance',
        'zone': 'entrance'
    }
    
    result1 = agent.process_task(task1)
    print(f"   Success: {result1['success']}")
    print(f"   Threat Level: {result1['analysis']['assessment']['threat_level']}")
    print(f"   Action Required: {result1['analysis']['requires_action']}")
    print()
    
    # Test task 2: With anomaly (simulate by running multiple times)
    print("2. Testing Anomaly Detection (may take a few attempts)...")
    for i in range(3):
        task2 = {
            'task_id': f'mon_00{i+2}',
            'camera_id': 'cam_entrance',
            'zone': 'entrance'
        }
        
        result2 = agent.process_task(task2)
        if result2['analysis']['assessment']['threat_level'] != 'low':
            print(f"   ⚠️  Anomaly detected! Threat: {result2['analysis']['assessment']['threat_level']}")
            if result2['decision'].get('action_required'):
                print(f"   📢 Alert sent: {result2['execution']['data'].get('message', 'N/A')}")
            break
        else:
            print(f"   ✅ Attempt {i+1}: Normal")
    print()
    
    # Show agent stats
    print("3. Agent Statistics:")
    stats = agent.get_stats()
    for key, value in stats.items():
        if key != 'tools_used':
            print(f"   {key}: {value}")
    
    print("\n   Tools usage:")
    for tool, count in stats['tools_used'].items():
        print(f"      {tool}: {count} times")
    
    print("\n" + "="*60)
    print("✅ Surveillance Agent Demo Complete!")
    print("="*60)
    
    # Cleanup
    state_manager.close()


if __name__ == "__main__":
    demo()