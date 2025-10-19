"""
Environmental Monitoring Agent
Monitors and optimizes environmental conditions
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent
from agents.llm_manager import AgentLLMManager
from state.state_manager import StateManager
from tools.environment_tools import (
    SensorReaderTool, HVACControllerTool,
    IrrigationControllerTool, DataAnalyzerTool
)
from typing import Dict, Any
from loguru import logger


class EnvironmentAgent(BaseAgent):
    """Agent for environmental monitoring and control"""
    
    def __init__(
        self,
        llm_manager: AgentLLMManager,
        state_manager: StateManager
    ):
        # Initialize tools
        tools = [
            SensorReaderTool(),
            HVACControllerTool(),
            IrrigationControllerTool(),
            DataAnalyzerTool()
        ]
        
        super().__init__(
            agent_id="environment_agent",
            name="Environmental Agent",
            description="Monitors environmental conditions and optimizes HVAC and irrigation systems",
            tools=tools,
            llm_manager=llm_manager,
            state_manager=state_manager,
            priority="medium"
        )
    
    def analyze_situation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze environmental conditions"""
        logger.info(f"{self.name}: Analyzing environmental conditions")
        
        zone = context.get('zone', 'main_area')
        sensor_types = context.get('sensor_types', ['temperature', 'humidity', 'co2'])
        
        # Read sensors
        sensor_result = self.execute_tool('sensor_reader',
                                         zone=zone,
                                         sensor_types=sensor_types)
        
        # Analyze data
        analysis_result = self.execute_tool('data_analyzer',
                                           data=sensor_result.get('data', {}),
                                           analysis_type='trend')
        
        # Use LLM to assess situation
        prompt = self._generate_analysis_prompt(sensor_result, analysis_result)
        llm_analysis = self.llm.generate(prompt, max_tokens=200, temperature=0.3)
        
        assessment = self._parse_analysis(llm_analysis, sensor_result, analysis_result)
        
        return {
            'success': True,
            'sensor_data': sensor_result,
            'analysis': analysis_result,
            'assessment': assessment,
            'requires_action': assessment.get('action_needed', False)
        }
    
    def decide_action(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Decide what environmental action to take"""
        logger.info(f"{self.name}: Deciding environmental action")
        
        if not analysis.get('requires_action', False):
            return {
                'success': True,
                'action_required': False,
                'reasoning': 'All environmental conditions normal'
            }
        
        assessment = analysis['assessment']
        
        # Use LLM to decide action
        prompt = self._generate_decision_prompt(analysis)
        llm_decision = self.llm.generate(prompt, max_tokens=150, temperature=0.2)
        
        decision = self._parse_decision(llm_decision, assessment)
        
        return decision
    
    def _generate_analysis_prompt(self, sensor_data: Dict, analysis: Dict) -> str:
        """Generate analysis prompt"""
        readings = sensor_data.get('data', {}).get('readings', {})
        
        readings_str = "\n".join([
            f"- {sensor}: {data['value']}{data['unit']} ({data['status']})"
            for sensor, data in readings.items()
        ])
        
        trends = analysis.get('data', {}).get('trends', {})
        
        prompt = f"""You are {self.name}. Analyze these environmental conditions:

Sensor Readings:
{readings_str}

Detected Trends:
{f"- {len(trends)} concerning trends" if trends else "- All normal"}

Assess if environmental action is needed.

Response format:
ACTION_NEEDED: [YES or NO]
PRIORITY: [low/medium/high]
REASONING: [brief explanation]

Response:"""
        
        return prompt
    
    def _generate_decision_prompt(self, analysis: Dict) -> str:
        """Generate decision prompt"""
        assessment = analysis['assessment']
        sensor_data = analysis['sensor_data'].get('data', {})
        
        prompt = f"""You are {self.name}. Decide environmental control action:

Assessment: {assessment.get('reasoning', 'No details')}
Priority: {assessment.get('priority', 'low')}

Available actions:
1. hvac_controller - Adjust temperature/ventilation
2. irrigation_controller - Control watering
3. No action needed

Choose ONE action and specify parameters.

Response format:
ACTION: [hvac_controller, irrigation_controller, or none]
PARAMETERS: [if action needed, specify: e.g., "set_temperature 22" or "start_irrigation 15"]

Response:"""
        
        return prompt
    
    def _parse_analysis(self, llm_response: str, sensor_data: Dict, analysis: Dict) -> Dict[str, Any]:
        """Parse LLM analysis"""
        lines = llm_response.strip().split('\n')
        
        result = {
            'action_needed': False,
            'priority': 'low',
            'reasoning': '',
            'raw_response': llm_response
        }
        
        for line in lines:
            if line.startswith('ACTION_NEEDED:'):
                result['action_needed'] = 'YES' in line.upper()
            elif line.startswith('PRIORITY:'):
                priority = line.split(':', 1)[1].strip().lower()
                if priority in ['low', 'medium', 'high']:
                    result['priority'] = priority
            elif line.startswith('REASONING:'):
                result['reasoning'] = line.split(':', 1)[1].strip()
        
        return result
    
    def _parse_decision(self, llm_response: str, assessment: Dict) -> Dict[str, Any]:
        """Parse LLM decision"""
        lines = llm_response.strip().split('\n')
        
        action = None
        parameters = ""
        
        for line in lines:
            if line.startswith('ACTION:'):
                action_text = line.split(':', 1)[1].strip().lower()
                if 'hvac' in action_text:
                    action = 'hvac_controller'
                elif 'irrigation' in action_text:
                    action = 'irrigation_controller'
            elif line.startswith('PARAMETERS:'):
                parameters = line.split(':', 1)[1].strip()
        
        decision = {
            'success': True,
            'action_required': action is not None,
            'reasoning': llm_response
        }
        
        if action == 'hvac_controller':
            # Parse HVAC parameters
            if 'temperature' in parameters.lower():
                try:
                    temp = float(''.join(c for c in parameters if c.isdigit() or c == '.'))
                    decision['tool_name'] = 'hvac_controller'
                    decision['tool_params'] = {
                        'zone': 'main_area',
                        'action': 'set_temperature',
                        'target_temp': temp
                    }
                except:
                    decision['tool_params'] = {
                        'zone': 'main_area',
                        'action': 'set_mode',
                        'mode': 'auto'
                    }
            else:
                decision['tool_params'] = {
                    'zone': 'main_area',
                    'action': 'set_mode',
                    'mode': 'auto'
                }
        
        elif action == 'irrigation_controller':
            # Parse irrigation parameters
            try:
                duration = int(''.join(c for c in parameters if c.isdigit()))
                if duration == 0:
                    duration = 15
            except:
                duration = 15
            
            decision['tool_name'] = 'irrigation_controller'
            decision['tool_params'] = {
                'zone': 'greenhouse_zone1',
                'action': 'start',
                'duration': duration
            }
        
        return decision


def demo():
    """Demo environmental agent"""
    print("="*60)
    print("Environmental Agent Demo")
    print("="*60 + "\n")
    
    from state.state_manager import StateManager
    
    state_manager = StateManager()
    llm_manager = AgentLLMManager(model_name='qwen-1.8b')
    
    agent = EnvironmentAgent(llm_manager, state_manager)
    
    print(f"Agent initialized: {agent.name}")
    print(f"Available tools: {', '.join(agent.get_available_tools())}\n")
    
    # Test 1: Normal conditions
    print("1. Testing Normal Conditions...")
    task1 = {
        'task_id': 'env_001',
        'zone': 'greenhouse',
        'sensor_types': ['temperature', 'humidity', 'soil_moisture']
    }
    
    result1 = agent.process_task(task1)
    print(f"   Success: {result1['success']}")
    print(f"   Action Needed: {result1['analysis']['assessment']['action_needed']}")
    print()
    
    # Test 2: Multiple readings (might trigger action)
    print("2. Testing Multiple Readings...")
    for i in range(3):
        task = {
            'task_id': f'env_00{i+2}',
            'zone': 'greenhouse',
            'sensor_types': ['temperature', 'humidity']
        }
        
        result = agent.process_task(task)
        if result['analysis']['assessment']['action_needed']:
            print(f"   ⚠️  Action needed! Priority: {result['analysis']['assessment']['priority']}")
            if result['decision'].get('action_required'):
                print(f"   🔧 Taking action: {result['decision'].get('tool_name', 'N/A')}")
            break
        else:
            print(f"   ✅ Attempt {i+1}: Normal")
    print()
    
    # Show stats
    print("3. Agent Statistics:")
    stats = agent.get_stats()
    for key, value in stats.items():
        if key != 'tools_used':
            print(f"   {key}: {value}")
    
    print("\n" + "="*60)
    print("✅ Environmental Agent Demo Complete!")
    print("="*60)
    
    state_manager.close()


if __name__ == "__main__":
    demo()