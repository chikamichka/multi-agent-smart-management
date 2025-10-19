"""
Multi-Agent Orchestrator using LangGraph
Coordinates Surveillance, Access Control, and Environmental agents
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import Dict, Any, List, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from loguru import logger
import operator
from datetime import datetime
import time

from state.state_manager import StateManager
from agents.llm_manager import AgentLLMManager
from agents.surveillance_agent import SurveillanceAgent
from agents.access_control_agent import AccessControlAgent
from agents.environment_agent import EnvironmentAgent


# Define the shared state between agents
class MultiAgentState(TypedDict):
    """Shared state across all agents"""
    scenario: str
    context: Dict[str, Any]
    surveillance_result: Dict[str, Any]
    access_result: Dict[str, Any]
    environment_result: Dict[str, Any]
    coordinator_decision: Dict[str, Any]
    messages: Annotated[List[str], operator.add]
    final_report: Dict[str, Any]


class MultiAgentOrchestrator:
    """Orchestrates multiple specialized agents"""
    
    def __init__(
        self,
        llm_manager: AgentLLMManager,
        state_manager: StateManager
    ):
        """
        Initialize orchestrator
        
        Args:
            llm_manager: Shared LLM manager
            state_manager: Shared state manager
        """
        self.llm = llm_manager
        self.state_manager = state_manager
        
        # Initialize agents
        logger.info("Initializing agents...")
        self.surveillance_agent = SurveillanceAgent(llm_manager, state_manager)
        self.access_agent = AccessControlAgent(llm_manager, state_manager)
        self.environment_agent = EnvironmentAgent(llm_manager, state_manager)
        
        # Build orchestration graph
        self.graph = self._build_graph()
        
        logger.success("✅ Multi-Agent Orchestrator initialized")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create the graph
        workflow = StateGraph(MultiAgentState)
        
        # Add agent nodes
        workflow.add_node("route", self._route_scenario)
        workflow.add_node("surveillance", self._surveillance_node)
        workflow.add_node("access_control", self._access_control_node)
        workflow.add_node("environment", self._environment_node)
        workflow.add_node("coordinator", self._coordinator_node)
        workflow.add_node("report", self._generate_report)
        
        # Define edges based on scenario routing
        workflow.set_entry_point("route")
        
        # Route can go to any agent or coordinator
        workflow.add_conditional_edges(
            "route",
            self._should_run_agents,
            {
                "surveillance": "surveillance",
                "access": "access_control",
                "environment": "environment",
                "all": "surveillance",  # Run all agents in sequence
                "coordinator": "coordinator"
            }
        )
        
        # Surveillance can go to coordinator or end
        workflow.add_edge("surveillance", "coordinator")
        
        # Access control can go to coordinator or end
        workflow.add_edge("access_control", "coordinator")
        
        # Environment can go to coordinator or end
        workflow.add_edge("environment", "coordinator")
        
        # Coordinator generates final report
        workflow.add_edge("coordinator", "report")
        
        # Report is the end
        workflow.add_edge("report", END)
        
        return workflow.compile()
    
    def _route_scenario(self, state: MultiAgentState) -> MultiAgentState:
        """Route scenario to appropriate agents"""
        scenario = state["scenario"]
        logger.info(f"Routing scenario: {scenario}")
        
        state["messages"].append(f"[Router] Analyzing scenario: {scenario}")
        return state
    
    def _should_run_agents(self, state: MultiAgentState) -> str:
        """Decide which agents to run based on scenario"""
        scenario = state["scenario"].lower()
        context = state.get("context", {})
        
        # Determine routing based on keywords
        if "security" in scenario or "intrusion" in scenario or "camera" in scenario:
            return "surveillance"
        elif "access" in scenario or "entry" in scenario or "badge" in scenario:
            return "access"
        elif "temperature" in scenario or "environment" in scenario or "hvac" in scenario:
            return "environment"
        elif "comprehensive" in scenario or "all" in scenario:
            return "all"
        else:
            # Default: run coordinator to decide
            return "coordinator"
    
    def _surveillance_node(self, state: MultiAgentState) -> MultiAgentState:
        """Run surveillance agent"""
        logger.info("Running Surveillance Agent...")
        
        state["messages"].append("[Surveillance Agent] Processing...")
        
        task = {
            'task_id': f"surv_{int(time.time())}",
            'camera_id': state["context"].get('camera_id', 'cam_001'),
            'zone': state["context"].get('zone', 'entrance')
        }
        
        result = self.surveillance_agent.process_task(task)
        state["surveillance_result"] = result
        
        state["messages"].append(
            f"[Surveillance Agent] {'✅ Complete' if result['success'] else '❌ Failed'}"
        )
        
        return state
    
    def _access_control_node(self, state: MultiAgentState) -> MultiAgentState:
        """Run access control agent"""
        logger.info("Running Access Control Agent...")
        
        state["messages"].append("[Access Control Agent] Processing...")
        
        task = {
            'task_id': f"access_{int(time.time())}",
            'request_type': state["context"].get('request_type', 'rfid'),
            'badge_id': state["context"].get('badge_id'),
            'location': state["context"].get('location', 'entrance')
        }
        
        result = self.access_agent.process_task(task)
        state["access_result"] = result
        
        state["messages"].append(
            f"[Access Control Agent] {'✅ Complete' if result['success'] else '❌ Failed'}"
        )
        
        return state
    
    def _environment_node(self, state: MultiAgentState) -> MultiAgentState:
        """Run environment agent"""
        logger.info("Running Environmental Agent...")
        
        state["messages"].append("[Environmental Agent] Processing...")
        
        task = {
            'task_id': f"env_{int(time.time())}",
            'zone': state["context"].get('zone', 'greenhouse'),
            'sensor_types': state["context"].get('sensor_types', ['temperature', 'humidity'])
        }
        
        result = self.environment_agent.process_task(task)
        state["environment_result"] = result
        
        state["messages"].append(
            f"[Environmental Agent] {'✅ Complete' if result['success'] else '❌ Failed'}"
        )
        
        return state
    
    def _coordinator_node(self, state: MultiAgentState) -> MultiAgentState:
        """Coordinate agent results and make final decisions"""
        logger.info("Coordinator analyzing results...")
        
        state["messages"].append("[Coordinator] Analyzing agent results...")
        
        # Collect all agent results
        results = {
            'surveillance': state.get('surveillance_result'),
            'access': state.get('access_result'),
            'environment': state.get('environment_result')
        }
        
        # Filter out None values
        active_results = {k: v for k, v in results.items() if v is not None}
        
        # Use LLM to coordinate
        prompt = self._generate_coordinator_prompt(active_results, state["scenario"])
        coordination = self.llm.generate(prompt, max_tokens=300, temperature=0.3)
        
        state["coordinator_decision"] = {
            'timestamp': datetime.now().isoformat(),
            'coordination': coordination,
            'agents_involved': list(active_results.keys())
        }
        
        state["messages"].append("[Coordinator] ✅ Coordination complete")
        
        return state
    
    def _generate_report(self, state: MultiAgentState) -> MultiAgentState:
        """Generate final comprehensive report"""
        logger.info("Generating final report...")
        
        # Compile results
        report = {
            'scenario': state["scenario"],
            'timestamp': datetime.now().isoformat(),
            'agents_used': [],
            'summary': {},
            'coordination': state.get('coordinator_decision', {}),
            'recommendations': []
        }
        
        # Add surveillance results
        if state.get('surveillance_result'):
            report['agents_used'].append('Surveillance')
            surv = state['surveillance_result']
            if surv.get('success'):
                threat_level = surv.get('analysis', {}).get('assessment', {}).get('threat_level', 'unknown')
                report['summary']['surveillance'] = f"Threat Level: {threat_level}"
        
        # Add access control results
        if state.get('access_result'):
            report['agents_used'].append('Access Control')
            access = state['access_result']
            if access.get('success'):
                decision = access.get('analysis', {}).get('assessment', {}).get('decision', 'unknown')
                report['summary']['access_control'] = f"Decision: {decision}"
        
        # Add environment results
        if state.get('environment_result'):
            report['agents_used'].append('Environmental')
            env = state['environment_result']
            if env.get('success'):
                action_needed = env.get('analysis', {}).get('assessment', {}).get('action_needed', False)
                report['summary']['environment'] = f"Action Needed: {action_needed}"
        
        # Extract recommendations from coordinator
        coordination = state.get('coordinator_decision', {}).get('coordination', '')
        if 'recommend' in coordination.lower():
            report['recommendations'].append(coordination)
        
        state["final_report"] = report
        state["messages"].append("[System] 📊 Final report generated")
        
        return state
    
    def _generate_coordinator_prompt(self, results: Dict, scenario: str) -> str:
        """Generate prompt for coordinator"""
        
        results_summary = []
        for agent, result in results.items():
            if result and result.get('success'):
                results_summary.append(f"\n{agent.upper()} Agent:")
                if 'analysis' in result:
                    analysis = result['analysis']
                    if 'assessment' in analysis:
                        for key, value in analysis['assessment'].items():
                            results_summary.append(f"  - {key}: {value}")
        
        results_text = "\n".join(results_summary)
        
        prompt = f"""You are the Multi-Agent System Coordinator. Multiple specialized agents have analyzed a situation.

Scenario: {scenario}

Agent Results:
{results_text}

Your task:
1. Synthesize the findings from all agents
2. Identify any conflicts or dependencies between agent recommendations
3. Provide a coordinated action plan
4. Assess overall system status

Provide a brief coordinated response focusing on key actions needed.

Response:"""
        
        return prompt
    
    def run_scenario(self, scenario: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run a complete multi-agent scenario
        
        Args:
            scenario: Scenario description
            context: Additional context
        
        Returns:
            Final report
        """
        logger.info(f"🚀 Running scenario: {scenario}")
        
        # Initialize state
        initial_state = MultiAgentState(
            scenario=scenario,
            context=context or {},
            surveillance_result=None,
            access_result=None,
            environment_result=None,
            coordinator_decision=None,
            messages=[],
            final_report={}
        )
        
        # Run the graph
        start_time = time.time()
        
        try:
            final_state = self.graph.invoke(initial_state)
            
            elapsed = time.time() - start_time
            
            logger.success(f"✅ Scenario completed in {elapsed:.2f}s")
            
            return {
                'success': True,
                'report': final_state.get('final_report', {}),
                'messages': final_state.get('messages', []),
                'execution_time': round(elapsed, 2)
            }
            
        except Exception as e:
            logger.error(f"❌ Scenario failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': round(time.time() - start_time, 2)
            }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get statistics for all agents"""
        return {
            'surveillance': self.surveillance_agent.get_stats(),
            'access_control': self.access_agent.get_stats(),
            'environment': self.environment_agent.get_stats(),
            'llm_usage': self.llm.get_token_usage(),
            'cost_savings': self.llm.estimate_cost('gpt4')
        }


def demo():
    """Demo the multi-agent orchestrator"""
    print("="*70)
    print("     🤖 MULTI-AGENT ORCHESTRATOR DEMO")
    print("="*70 + "\n")
    
    # Initialize
    state_manager = StateManager()
    llm_manager = AgentLLMManager(model_name='qwen-1.8b')
    
    orchestrator = MultiAgentOrchestrator(llm_manager, state_manager)
    
    print("System initialized with 3 specialized agents\n")
    
    # Scenario 1: Security incident
    print("="*70)
    print("SCENARIO 1: Security Monitoring")
    print("="*70)
    
    result1 = orchestrator.run_scenario(
        scenario="Security monitoring at entrance",
        context={'camera_id': 'cam_entrance', 'zone': 'entrance'}
    )
    
    print(f"\n✅ Status: {'Success' if result1['success'] else 'Failed'}")
    print(f"⏱️  Time: {result1['execution_time']}s")
    print(f"📋 Agents Used: {', '.join(result1['report'].get('agents_used', []))}")
    print(f"\n📊 Summary:")
    for key, value in result1['report'].get('summary', {}).items():
        print(f"   - {key}: {value}")
    
    # Scenario 2: Access request
    print("\n" + "="*70)
    print("SCENARIO 2: Access Control Request")
    print("="*70)
    
    result2 = orchestrator.run_scenario(
        scenario="Access request at main entrance",
        context={'request_type': 'rfid', 'badge_id': 'BADGE002', 'location': 'main_entrance'}
    )
    
    print(f"\n✅ Status: {'Success' if result2['success'] else 'Failed'}")
    print(f"⏱️  Time: {result2['execution_time']}s")
    print(f"📋 Agents Used: {', '.join(result2['report'].get('agents_used', []))}")
    
    # Scenario 3: Environmental monitoring
    print("\n" + "="*70)
    print("SCENARIO 3: Environmental Control")
    print("="*70)
    
    result3 = orchestrator.run_scenario(
        scenario="Temperature monitoring in greenhouse",
        context={'zone': 'greenhouse', 'sensor_types': ['temperature', 'humidity']}
    )
    
    print(f"\n✅ Status: {'Success' if result3['success'] else 'Failed'}")
    print(f"⏱️  Time: {result3['execution_time']}s")
    print(f"📋 Agents Used: {', '.join(result3['report'].get('agents_used', []))}")
    
    # Show system statistics
    print("\n" + "="*70)
    print("SYSTEM STATISTICS")
    print("="*70)
    
    stats = orchestrator.get_system_stats()
    
    print("\nAgent Performance:")
    for agent_name, agent_stats in stats.items():
        if agent_name not in ['llm_usage', 'cost_savings']:
            print(f"\n{agent_stats['name']}:")
            print(f"  Tasks: {agent_stats['tasks_completed']} completed, {agent_stats['tasks_failed']} failed")
            print(f"  Success Rate: {agent_stats['success_rate']:.1%}")
            print(f"  Avg Time: {agent_stats['avg_execution_time']}s")
    
    print("\n💰 Cost Comparison:")
    cost = stats['cost_savings']
    print(f"  Local (Edge): ${cost['local_cost']:.4f}")
    print(f"  Cloud (GPT-4): ${cost['cloud_cost']:.4f}")
    print(f"  💵 Savings: ${cost['savings']:.4f}")
    print(f"  Tokens: {cost['input_tokens']} in, {cost['output_tokens']} out")
    
    print("\n" + "="*70)
    print("✅ Multi-Agent Orchestrator Demo Complete!")
    print("="*70)
    
    # Cleanup
    state_manager.close()


if __name__ == "__main__":
    demo()