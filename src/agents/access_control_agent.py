"""
Access Control Agent
Manages entry/exit authorization and tracks access patterns
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent
from agents.llm_manager import AgentLLMManager
from state.state_manager import StateManager
from tools.access_tools import (
    RFIDReaderTool, FacialRecognitionTool, 
    AccessLoggerTool, DoorControllerTool
)
from typing import Dict, Any
from loguru import logger


class AccessControlAgent(BaseAgent):
    """Agent for access control and authorization"""
    
    def __init__(
        self,
        llm_manager: AgentLLMManager,
        state_manager: StateManager
    ):
        # Initialize tools
        tools = [
            RFIDReaderTool(),
            FacialRecognitionTool(),
            AccessLoggerTool(),
            DoorControllerTool()
        ]
        
        super().__init__(
            agent_id="access_control_agent",
            name="Access Control Agent",
            description="Manages access authorization, tracks entry/exit, and controls door locks",
            tools=tools,
            llm_manager=llm_manager,
            state_manager=state_manager,
            priority="critical"
        )
    
    def analyze_situation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze access request and validate credentials"""
        logger.info(f"{self.name}: Analyzing access request")
        
        request_type = context.get('request_type', 'rfid')
        location = context.get('location', 'entrance')
        
        # Read credentials based on type
        if request_type == 'rfid':
            auth_result = self.execute_tool('rfid_reader', 
                                           badge_id=context.get('badge_id'),
                                           location=location)
        elif request_type == 'facial':
            auth_result = self.execute_tool('facial_recognition',
                                           image_path=context.get('image_path'),
                                           location=location)
        else:
            return {'success': False, 'error': 'Unknown request type'}
        
        # Check for suspicious patterns
        suspicious = self.tools['access_logger'].get_suspicious_activity()
        
        # Use LLM to analyze
        prompt = self._generate_analysis_prompt(auth_result, suspicious, context)
        llm_analysis = self.llm.generate(prompt, max_tokens=150, temperature=0.2)
        
        assessment = self._parse_analysis(llm_analysis, auth_result)
        
        return {
            'success': True,
            'auth_result': auth_result,
            'suspicious_activity': suspicious,
            'assessment': assessment,
            'requires_action': True  # Always log and decide
        }
    
    def decide_action(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Decide whether to grant access"""
        logger.info(f"{self.name}: Making access decision")
        
        auth_data = analysis['auth_result'].get('data', {})
        assessment = analysis['assessment']
        
        # Use LLM for decision
        prompt = self._generate_decision_prompt(analysis)
        llm_decision = self.llm.generate(prompt, max_tokens=100, temperature=0.1)
        
        decision = self._parse_decision(llm_decision, auth_data, assessment)
        
        return decision
    
    def _generate_analysis_prompt(self, auth_result: Dict, suspicious: list, context: Dict) -> str:
        """Generate analysis prompt"""
        auth_data = auth_result.get('data', {})
        
        prompt = f"""You are {self.name}. Analyze this access request:

Credentials:
- Valid: {auth_data.get('valid', False)}
- User: {auth_data.get('user_name', 'Unknown')}
- Location: {auth_data.get('location', 'Unknown')}
- Access Level: {auth_data.get('access_level', 0)}

Suspicious Activity:
{f"- {len(suspicious)} suspicious patterns detected" if suspicious else "- None detected"}

Assess the access request (approve/deny) and provide reasoning.

Response format:
DECISION: [APPROVE or DENY]
CONFIDENCE: [low/medium/high]
REASONING: [brief explanation]

Response:"""
        
        return prompt
    
    def _generate_decision_prompt(self, analysis: Dict) -> str:
        """Generate decision prompt"""
        assessment = analysis['assessment']
        auth_data = analysis['auth_result'].get('data', {})
        
        prompt = f"""You are {self.name}. Make final access decision:

Assessment: {assessment.get('decision', 'unknown')}
Confidence: {assessment.get('confidence', 'low')}
Reasoning: {assessment.get('reasoning', 'No details')}

Decide actions needed:
1. Log access attempt? (always yes)
2. Grant access (unlock door)? (yes/no)
3. Duration if granted (in seconds, e.g., 10)

Response format:
GRANT_ACCESS: [YES or NO]
DOOR_UNLOCK_DURATION: [seconds if granted, else 0]
LOG_RESULT: [granted or denied]

Response:"""
        
        return prompt
    
    def _parse_analysis(self, llm_response: str, auth_result: Dict) -> Dict[str, Any]:
        """Parse LLM analysis"""
        lines = llm_response.strip().split('\n')
        
        result = {
            'decision': 'DENY',
            'confidence': 'low',
            'reasoning': '',
            'raw_response': llm_response
        }
        
        for line in lines:
            if line.startswith('DECISION:'):
                result['decision'] = line.split(':', 1)[1].strip().upper()
            elif line.startswith('CONFIDENCE:'):
                result['confidence'] = line.split(':', 1)[1].strip().lower()
            elif line.startswith('REASONING:'):
                result['reasoning'] = line.split(':', 1)[1].strip()
        
        return result
    
    def _parse_decision(self, llm_response: str, auth_data: Dict, assessment: Dict) -> Dict[str, Any]:
        """Parse LLM decision"""
        lines = llm_response.strip().split('\n')
        
        grant_access = False
        duration = 0
        log_result = 'denied'
        
        for line in lines:
            if line.startswith('GRANT_ACCESS:'):
                grant_access = 'YES' in line.upper()
            elif line.startswith('DOOR_UNLOCK_DURATION:'):
                try:
                    duration = int(''.join(filter(str.isdigit, line)))
                except:
                    duration = 10 if grant_access else 0
            elif line.startswith('LOG_RESULT:'):
                log_result = line.split(':', 1)[1].strip().lower()
        
        # Build decision
        decision = {
            'success': True,
            'action_required': True,
            'reasoning': llm_response
        }
        
        # First, always log
        decision['tool_name'] = 'access_logger'
        decision['tool_params'] = {
            'user_id': auth_data.get('badge_id', auth_data.get('face_id', 'unknown')),
            'action': 'entry',
            'location': auth_data.get('location', 'unknown'),
            'result': 'granted' if grant_access else 'denied',
            'details': {'assessment': assessment}
        }
        
        # Store door action for later
        decision['grant_access'] = grant_access
        decision['door_action'] = {
            'door_id': auth_data.get('location', 'entrance'),
            'action': 'unlock' if grant_access else 'lock',
            'duration': duration if grant_access else None
        }
        
        return decision


def demo():
    """Demo access control agent"""
    print("="*60)
    print("Access Control Agent Demo")
    print("="*60 + "\n")
    
    from state.state_manager import StateManager
    
    state_manager = StateManager()
    llm_manager = AgentLLMManager(model_name='qwen-1.8b')
    
    agent = AccessControlAgent(llm_manager, state_manager)
    
    print(f"Agent initialized: {agent.name}")
    print(f"Available tools: {', '.join(agent.get_available_tools())}\n")
    
    # Test 1: Valid badge
    print("1. Testing Valid Badge Access...")
    task1 = {
        'task_id': 'access_001',
        'request_type': 'rfid',
        'badge_id': 'BADGE001',
        'location': 'main_entrance'
    }
    
    result1 = agent.process_task(task1)
    print(f"   Success: {result1['success']}")
    print(f"   Decision: {result1['analysis']['assessment']['decision']}")
    print(f"   Access Granted: {result1.get('decision', {}).get('grant_access', False)}")
    print()
    
    # Test 2: Invalid badge
    print("2. Testing Invalid Badge Access...")
    task2 = {
        'task_id': 'access_002',
        'request_type': 'rfid',
        'badge_id': 'INVALID999',
        'location': 'main_entrance'
    }
    
    result2 = agent.process_task(task2)
    print(f"   Success: {result2['success']}")
    print(f"   Decision: {result2['analysis']['assessment']['decision']}")
    print()
    
    # Show stats
    print("3. Agent Statistics:")
    stats = agent.get_stats()
    for key, value in stats.items():
        if key != 'tools_used':
            print(f"   {key}: {value}")
    
    print("\n" + "="*60)
    print("✅ Access Control Agent Demo Complete!")
    print("="*60)
    
    state_manager.close()


if __name__ == "__main__":
    demo()