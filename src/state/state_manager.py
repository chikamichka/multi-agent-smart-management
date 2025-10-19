"""
State Manager using Redis for Multi-Agent Coordination
Handles shared state, message passing, and agent coordination
"""

import redis
import json
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from loguru import logger
import threading


class StateManager:
    """Manages shared state across agents using Redis"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None
    ):
        """
        Initialize Redis state manager
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password (optional)
        """
        try:
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True,
                socket_keepalive=True,
                socket_connect_timeout=5
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info(f"✅ Connected to Redis at {host}:{port}")
            
        except redis.ConnectionError as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            logger.warning("💡 Make sure Redis is running: brew services start redis")
            raise
        
        self.lock = threading.Lock()
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set a value in state
        
        Args:
            key: State key
            value: Value to store (will be JSON serialized)
            ttl: Time to live in seconds (optional)
        
        Returns:
            True if successful
        """
        try:
            # Serialize complex objects to JSON
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            
            if ttl:
                return self.redis_client.setex(key, ttl, value)
            else:
                return self.redis_client.set(key, value)
                
        except Exception as e:
            logger.error(f"Error setting key {key}: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from state
        
        Args:
            key: State key
            default: Default value if key not found
        
        Returns:
            Value from state or default
        """
        try:
            value = self.redis_client.get(key)
            
            if value is None:
                return default
            
            # Try to parse JSON
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
                
        except Exception as e:
            logger.error(f"Error getting key {key}: {e}")
            return default
    
    def delete(self, key: str) -> bool:
        """Delete a key from state"""
        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            logger.error(f"Error deleting key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists"""
        try:
            return bool(self.redis_client.exists(key))
        except Exception as e:
            logger.error(f"Error checking key {key}: {e}")
            return False
    
    def increment(self, key: str, amount: int = 1) -> int:
        """Increment a counter"""
        try:
            return self.redis_client.incrby(key, amount)
        except Exception as e:
            logger.error(f"Error incrementing key {key}: {e}")
            return 0
    
    def push_to_list(self, key: str, value: Any) -> int:
        """Push value to a list"""
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            return self.redis_client.rpush(key, value)
        except Exception as e:
            logger.error(f"Error pushing to list {key}: {e}")
            return 0
    
    def get_list(self, key: str, start: int = 0, end: int = -1) -> List[Any]:
        """Get list values"""
        try:
            values = self.redis_client.lrange(key, start, end)
            
            # Try to parse JSON for each value
            parsed_values = []
            for value in values:
                try:
                    parsed_values.append(json.loads(value))
                except (json.JSONDecodeError, TypeError):
                    parsed_values.append(value)
            
            return parsed_values
            
        except Exception as e:
            logger.error(f"Error getting list {key}: {e}")
            return []
    
    def set_hash(self, key: str, mapping: Dict[str, Any]) -> bool:
        """Set multiple fields in a hash"""
        try:
            # Serialize complex values
            serialized = {}
            for k, v in mapping.items():
                if isinstance(v, (dict, list)):
                    serialized[k] = json.dumps(v)
                else:
                    serialized[k] = str(v)
            
            return bool(self.redis_client.hset(key, mapping=serialized))
            
        except Exception as e:
            logger.error(f"Error setting hash {key}: {e}")
            return False
    
    def get_hash(self, key: str) -> Dict[str, Any]:
        """Get all fields from a hash"""
        try:
            data = self.redis_client.hgetall(key)
            
            # Try to parse JSON values
            parsed = {}
            for k, v in data.items():
                try:
                    parsed[k] = json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    parsed[k] = v
            
            return parsed
            
        except Exception as e:
            logger.error(f"Error getting hash {key}: {e}")
            return {}
    
    def publish(self, channel: str, message: Any) -> int:
        """Publish message to a channel"""
        try:
            if isinstance(message, (dict, list)):
                message = json.dumps(message)
            return self.redis_client.publish(channel, message)
        except Exception as e:
            logger.error(f"Error publishing to channel {channel}: {e}")
            return 0
    
    def subscribe(self, channel: str):
        """Subscribe to a channel"""
        try:
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe(channel)
            return pubsub
        except Exception as e:
            logger.error(f"Error subscribing to channel {channel}: {e}")
            return None
    
    def clear_all(self) -> bool:
        """Clear all data (use with caution!)"""
        try:
            self.redis_client.flushdb()
            logger.warning("⚠️  Cleared all Redis data")
            return True
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Redis statistics"""
        try:
            info = self.redis_client.info()
            return {
                'connected': True,
                'used_memory': info.get('used_memory_human', 'N/A'),
                'total_keys': self.redis_client.dbsize(),
                'uptime_seconds': info.get('uptime_in_seconds', 0),
                'version': info.get('redis_version', 'unknown')
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {'connected': False, 'error': str(e)}
    
    def close(self):
        """Close Redis connection"""
        try:
            self.redis_client.close()
            logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing connection: {e}")


# Agent-specific state helpers
class AgentState:
    """Helper class for agent-specific state management"""
    
    def __init__(self, state_manager: StateManager, agent_id: str):
        self.state = state_manager
        self.agent_id = agent_id
        self.prefix = f"agent:{agent_id}"
    
    def set_status(self, status: str):
        """Set agent status"""
        self.state.set(f"{self.prefix}:status", status)
    
    def get_status(self) -> str:
        """Get agent status"""
        return self.state.get(f"{self.prefix}:status", "idle")
    
    def log_action(self, action: str, details: Dict[str, Any]):
        """Log an action"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'agent_id': self.agent_id,
            'action': action,
            'details': details
        }
        self.state.push_to_list(f"{self.prefix}:actions", log_entry)
    
    def get_action_history(self, limit: int = 10) -> List[Dict]:
        """Get recent actions"""
        return self.state.get_list(f"{self.prefix}:actions", start=-limit)
    
    def increment_task_counter(self) -> int:
        """Increment task counter"""
        return self.state.increment(f"{self.prefix}:task_count")
    
    def get_task_count(self) -> int:
        """Get total task count"""
        count = self.state.get(f"{self.prefix}:task_count", "0")
        return int(count) if isinstance(count, str) else count


def demo():
    """Demo the state manager"""
    print("="*60)
    print("State Manager Demo")
    print("="*60 + "\n")
    
    # Initialize
    try:
        state = StateManager()
    except Exception as e:
        print(f"\n❌ Could not connect to Redis")
        print(f"Error: {e}")
        print("\n💡 Make sure Redis is installed and running:")
        print("   brew install redis")
        print("   brew services start redis")
        return
    
    # Test basic operations
    print("1. Testing basic set/get...")
    state.set("test:key", "Hello Redis!")
    value = state.get("test:key")
    print(f"   Stored: 'Hello Redis!'")
    print(f"   Retrieved: '{value}'")
    print(f"   ✅ Basic operations work\n")
    
    # Test JSON serialization
    print("2. Testing JSON serialization...")
    data = {"name": "Agent1", "tasks": [1, 2, 3]}
    state.set("test:json", data)
    retrieved = state.get("test:json")
    print(f"   Stored: {data}")
    print(f"   Retrieved: {retrieved}")
    print(f"   ✅ JSON serialization works\n")
    
    # Test list operations
    print("3. Testing list operations...")
    state.push_to_list("test:list", "Item 1")
    state.push_to_list("test:list", {"item": 2})
    items = state.get_list("test:list")
    print(f"   List items: {items}")
    print(f"   ✅ List operations work\n")
    
    # Test agent state
    print("4. Testing agent state...")
    agent_state = AgentState(state, "surveillance")
    agent_state.set_status("active")
    agent_state.log_action("motion_detected", {"location": "entrance"})
    agent_state.increment_task_counter()
    
    status = agent_state.get_status()
    count = agent_state.get_task_count()
    history = agent_state.get_action_history()
    
    print(f"   Agent status: {status}")
    print(f"   Task count: {count}")
    print(f"   Recent actions: {len(history)}")
    print(f"   ✅ Agent state works\n")
    
    # Show stats
    print("5. Redis Statistics:")
    stats = state.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n" + "="*60)
    print("✅ State Manager Demo Complete!")
    print("="*60)
    
    # Cleanup
    state.close()


if __name__ == "__main__":
    demo()