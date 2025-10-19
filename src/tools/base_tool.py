"""
Base Tool Class for Agent Tools
Provides common interface and utilities for all tools
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from datetime import datetime
from loguru import logger
import time


class BaseTool(ABC):
    """Abstract base class for agent tools"""
    
    def __init__(self, name: str, description: str):
        """
        Initialize tool
        
        Args:
            name: Tool name
            description: Tool description
        """
        self.name = name
        self.description = description
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.last_execution = None
    
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool
        
        Returns:
            Dictionary with 'success', 'data', and optional 'error'
        """
        pass
    
    def __call__(self, **kwargs) -> Dict[str, Any]:
        """Make tool callable"""
        start_time = time.time()
        
        try:
            logger.info(f"🔧 Executing tool: {self.name}")
            result = self.execute(**kwargs)
            
            # Track metrics
            execution_time = time.time() - start_time
            self.execution_count += 1
            self.total_execution_time += execution_time
            self.last_execution = datetime.now()
            
            # Add metadata
            if isinstance(result, dict):
                result['_meta'] = {
                    'tool': self.name,
                    'execution_time': round(execution_time, 3),
                    'timestamp': self.last_execution.isoformat()
                }
            
            logger.success(f"✅ {self.name} completed in {execution_time:.3f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ {self.name} failed: {e}")
            
            return {
                'success': False,
                'error': str(e),
                '_meta': {
                    'tool': self.name,
                    'execution_time': round(execution_time, 3),
                    'timestamp': datetime.now().isoformat()
                }
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tool execution statistics"""
        avg_time = (self.total_execution_time / self.execution_count 
                    if self.execution_count > 0 else 0)
        
        return {
            'name': self.name,
            'executions': self.execution_count,
            'total_time': round(self.total_execution_time, 2),
            'avg_time': round(avg_time, 3),
            'last_execution': self.last_execution.isoformat() if self.last_execution else None
        }
    
    def to_langchain_tool(self):
        """Convert to LangChain tool format"""
        from langchain.tools import Tool
        
        return Tool(
            name=self.name,
            description=self.description,
            func=self.__call__
        )