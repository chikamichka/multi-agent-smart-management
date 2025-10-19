"""
LLM Manager for Multi-Agent System
Optimized for M1 Mac with local model support
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Optional, Dict, List
from loguru import logger
import time


class AgentLLMManager:
    """Manages LLM for agent reasoning and decision making"""
    
    MODELS = {
        'qwen-1.8b': 'Qwen/Qwen2.5-1.5B-Instruct',
        'phi-2': 'microsoft/phi-2',
        'phi-3-mini': 'microsoft/Phi-3-mini-4k-instruct',
    }
    
    def __init__(
        self,
        model_name: str = 'qwen-1.8b',
        device: str = 'auto',
        cache_dir: str = './data/models'
    ):
        """
        Initialize LLM Manager
        
        Args:
            model_name: Model to use
            device: Device (auto, mps, cpu, cuda)
            cache_dir: Model cache directory
        """
        self.model_name = self.MODELS.get(model_name, model_name)
        self.device = self._setup_device(device)
        
        logger.info(f"Loading LLM: {self.model_name} on {self.device}")
        
        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device in ['mps', 'cuda'] else torch.float32,
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        
        if self.device in ['mps', 'cuda']:
            self.model.to(self.device)
        
        # Create pipeline
        self.pipe = pipeline(
            'text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device if self.device != 'mps' else -1
        )
        
        logger.success(f"✅ LLM loaded: {self.model.num_parameters() / 1e9:.2f}B parameters")
        
        # Track token usage for cost comparison
        self.token_usage = {
            'input_tokens': 0,
            'output_tokens': 0,
            'total_requests': 0
        }
    
    def _setup_device(self, device: str) -> str:
        """Setup device for M1 compatibility"""
        if device == 'auto':
            if torch.backends.mps.is_available():
                return 'mps'
            elif torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        return device
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.3,
        stop_sequences: List[str] = None
    ) -> str:
        """
        Generate text from prompt
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop_sequences: Stop generation at these sequences
        
        Returns:
            Generated text
        """
        start_time = time.time()
        
        try:
            # Count input tokens (approximate)
            input_tokens = len(self.tokenizer.encode(prompt))
            
            # Generate with MPS-optimized parameters
            generation_kwargs = {
                'max_new_tokens': max_tokens,
                'pad_token_id': self.tokenizer.eos_token_id,
                'return_full_text': False,
                'do_sample': True,
                'temperature': max(0.1, min(temperature, 1.0)),
                'top_p': 0.9,
                'top_k': 50,
                'repetition_penalty': 1.1
            }
            
            outputs = self.pipe(prompt, **generation_kwargs)
            generated_text = outputs[0]['generated_text'].strip()
            
            # Count output tokens (approximate)
            output_tokens = len(self.tokenizer.encode(generated_text))
            
            # Track usage
            self.token_usage['input_tokens'] += input_tokens
            self.token_usage['output_tokens'] += output_tokens
            self.token_usage['total_requests'] += 1
            
            elapsed = time.time() - start_time
            logger.debug(f"Generated {output_tokens} tokens in {elapsed:.2f}s")
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            # Fallback to greedy decoding
            try:
                logger.warning("Attempting greedy fallback...")
                inputs = self.tokenizer(prompt, return_tensors="pt")
                if self.device in ['mps', 'cuda']:
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generated_text = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                
                return generated_text.strip()
                
            except Exception as e2:
                logger.error(f"Fallback failed: {e2}")
                return "Error: Unable to generate response"
    
    def get_token_usage(self) -> Dict[str, int]:
        """Get token usage statistics"""
        return self.token_usage.copy()
    
    def estimate_cost(self, cloud_provider: str = 'gpt4') -> Dict[str, float]:
        """
        Estimate cost comparison with cloud APIs
        
        Args:
            cloud_provider: Cloud provider for comparison (gpt4, claude)
        
        Returns:
            Cost comparison data
        """
        # Cloud API pricing (per 1K tokens)
        pricing = {
            'gpt4': {'input': 0.03, 'output': 0.06},
            'claude': {'input': 0.008, 'output': 0.024},
            'gpt3.5': {'input': 0.0015, 'output': 0.002}
        }
        
        if cloud_provider not in pricing:
            cloud_provider = 'gpt4'
        
        prices = pricing[cloud_provider]
        
        input_cost = (self.token_usage['input_tokens'] / 1000) * prices['input']
        output_cost = (self.token_usage['output_tokens'] / 1000) * prices['output']
        total_cloud_cost = input_cost + output_cost
        
        return {
            'provider': cloud_provider,
            'local_cost': 0.0,  # Free!
            'cloud_cost': round(total_cloud_cost, 4),
            'savings': round(total_cloud_cost, 4),
            'input_tokens': self.token_usage['input_tokens'],
            'output_tokens': self.token_usage['output_tokens'],
            'total_requests': self.token_usage['total_requests']
        }


def demo():
    """Demo the LLM manager"""
    print("="*60)
    print("Agent LLM Manager Demo")
    print("="*60 + "\n")
    
    # Initialize
    llm = AgentLLMManager(model_name='qwen-1.8b')
    
    # Test simple reasoning
    print("1. Testing Agent Reasoning...")
    prompt = """You are a surveillance agent. A motion sensor detected movement at the entrance at 2 AM.

Analyze the situation and decide what action to take.

Response (be concise):"""
    
    response = llm.generate(prompt, max_tokens=150, temperature=0.3)
    print(f"   Prompt: Motion detected at entrance at 2 AM")
    print(f"   Response: {response[:200]}...")
    print()
    
    # Test decision making
    print("2. Testing Decision Making...")
    prompt = """You are an access control agent. 

Situation:
- RFID badge: INVALID123
- Location: Main entrance
- Time: 14:30
- Failed attempts: 3

Should you grant access? Respond with YES or NO and brief reason.

Response:"""
    
    response = llm.generate(prompt, max_tokens=100, temperature=0.2)
    print(f"   Decision: {response}")
    print()
    
    # Test tool selection
    print("3. Testing Tool Selection...")
    prompt = """You are an environmental agent. Current readings:
- Temperature: 28°C (normal: 20-24°C)
- Humidity: 75%

Available tools:
1. hvac_controller - Control temperature
2. sensor_reader - Read more sensors
3. alert_sender - Send alerts

Which tool should you use? Respond with just the tool name.

Tool:"""
    
    response = llm.generate(prompt, max_tokens=50, temperature=0.2)
    print(f"   Selected tool: {response}")
    print()
    
    # Show token usage
    print("4. Token Usage Statistics:")
    usage = llm.get_token_usage()
    for key, value in usage.items():
        print(f"   {key}: {value}")
    print()
    
    # Show cost comparison
    print("5. Cost Comparison (vs Cloud APIs):")
    for provider in ['gpt4', 'claude', 'gpt3.5']:
        cost_data = llm.estimate_cost(provider)
        print(f"   {cost_data['provider'].upper()}:")
        print(f"      Local: ${cost_data['local_cost']:.4f}")
        print(f"      Cloud: ${cost_data['cloud_cost']:.4f}")
        print(f"      Savings: ${cost_data['savings']:.4f}")
    
    print("\n" + "="*60)
    print("✅ Agent LLM Manager Demo Complete!")
    print("="*60)


if __name__ == "__main__":
    demo()