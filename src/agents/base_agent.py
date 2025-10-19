"""
Base Agent Class for Multi-Agent System
Provides common functionality for all specialized agents
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
from loguru import logger
import time

from state.state_manager import StateManager, AgentState
from agents.llm_manager import AgentLLMManager


class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str,
        tools: List[Any],
        llm_manager: AgentLLMManager,
        state_manager: StateManager,
        priority: str = "medium"
    ):
        """
        Initialize agent
        
        Args:
            agent_id: Unique agent identifier
            name: Agent name
            description: Agent description
            tools: List of tools agent can use
            llm_manager: Shared LLM manager
            state_manager: Shared state manager
            priority: Agent priority (low, medium, high, critical)
        """
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.tools = {tool.name: tool for tool in tools}
        self.llm = llm_manager
        self.state = AgentState(state_manager, agent_id)
        self.priority = priority
        
        # Agent statistics
        self.stats = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_execution_time': 0.0,
            'tools_used': {},
            'decisions_made': 0
        }
        
        # Set initial status
        self.state.set_status("initialized")
        
        logger.info(f"✅ {self.name} initialized with {len(self.tools)} tools")
    
    @abstractmethod
    def analyze_situation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze current situation and determine if action is needed
        
        Args:
            context: Current context/data
        
        Returns:
            Analysis results
        """
        pass
    
    @abstractmethod
    def decide_action(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide what action to take based on analysis
        
        Args:
            analysis: Analysis results
        
        Returns:
            Action decision
        """
        pass
    
    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a tool
        
        Args:
            tool_name: Name of tool to execute
            **kwargs: Tool parameters
        
        Returns:
            Tool execution result
        """
        if tool_name not in self.tools:
            logger.error(f"Tool not found: {tool_name}")
            return {
                'success': False,
                'error': f"Tool '{tool_name}' not available to {self.name}"
            }
        
        tool = self.tools[tool_name]
        
        try:
            logger.info(f"{self.name} executing: {tool_name}")
            result = tool(**kwargs)
            
            # Track tool usage
            self.stats['tools_used'][tool_name] = self.stats['tools_used'].get(tool_name, 0) + 1
            
            # Log action
            self.state.log_action(
                action=f"executed_tool_{tool_name}",
                details={'result': result, 'params': kwargs}
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main task processing pipeline
        
        Args:
            task: Task to process
        
        Returns:
            Task result
        """
        start_time = time.time()
        task_id = task.get('task_id', f"task_{int(time.time())}")
        
        logger.info(f"🤖 {self.name} processing task: {task_id}")
        
        try:
            self.state.set_status("processing")
            
            # 1. Analyze situation
            analysis = self.analyze_situation(task)
            
            if not analysis.get('success', False):
                raise Exception(f"Analysis failed: {analysis.get('error', 'Unknown error')}")
            
            # 2. Decide action
            decision = self.decide_action(analysis)
            self.stats['decisions_made'] += 1
            
            if not decision.get('success', False):
                raise Exception(f"Decision failed: {decision.get('error', 'Unknown error')}")
            
            # 3. Execute action
            if decision.get('action_required', False):
                tool_name = decision.get('tool_name')
                tool_params = decision.get('tool_params', {})
                
                execution_result = self.execute_tool(tool_name, **tool_params)
                
                if not execution_result.get('success', False):
                    raise Exception(f"Tool execution failed: {execution_result.get('error', 'Unknown')}")
            else:
                execution_result = {'success': True, 'data': 'No action required'}
            
            # Success
            elapsed = time.time() - start_time
            self.stats['tasks_completed'] += 1
            self.stats['total_execution_time'] += elapsed
            self.state.set_status("idle")
            
            result = {
                'success': True,
                'agent': self.name,
                'task_id': task_id,
                'analysis': analysis,
                'decision': decision,
                'execution': execution_result,
                'execution_time': round(elapsed, 3),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.success(f"✅ {self.name} completed task in {elapsed:.2f}s")
            
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            self.stats['tasks_failed'] += 1
            self.state.set_status("error")
            
            logger.error(f"❌ {self.name} task failed: {e}")
            
            return {
                'success': False,
                'agent': self.name,
                'task_id': task_id,
                'error': str(e),
                'execution_time': round(elapsed, 3),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names"""
        return list(self.tools.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        avg_time = (self.stats['total_execution_time'] / self.stats['tasks_completed'] 
                   if self.stats['tasks_completed'] > 0 else 0)
        
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'status': self.state.get_status(),
            'priority': self.priority,
            'tasks_completed': self.stats['tasks_completed'],
            'tasks_failed': self.stats['tasks_failed'],
            'success_rate': (self.stats['tasks_completed'] / 
                           (self.stats['tasks_completed'] + self.stats['tasks_failed'])
                           if (self.stats['tasks_completed'] + self.stats['tasks_failed']) > 0 
                           else 0),
            'avg_execution_time': round(avg_time, 3),
            'decisions_made': self.stats['decisions_made'],
            'tools_used': self.stats['tools_used']
        }
    
    def _generate_prompt(self, template: str, **kwargs) -> str:
        """
        Generate prompt from template
        
        Args:
            template: Prompt template
            **kwargs: Template variables
        
        Returns:
            Formatted prompt
        """
        return template.format(
            agent_name=self.name,
            agent_description=self.description,
            available_tools=', '.join(self.get_available_tools()),
            **kwargs
        )
    
    def __repr__(self):
        return f"<{self.__class__.__name__}(id={self.agent_id}, name={self.name}, status={self.state.get_status()})>"