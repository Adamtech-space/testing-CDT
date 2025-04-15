import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Callable, Any

class SubtopicRegistry:
    """Registry for managing subtopic activation functions."""
    
    def __init__(self):
        self.subtopics: List[Dict[str, Any]] = []
    
    def register(self, code_range: str, activate_func: Callable, name: str):
        """Register a subtopic with its activation function."""
        self.subtopics.append({
            "code_range": code_range,
            "activate_func": activate_func,
            "name": name
        })
    
    async def activate_all(self, scenario: str, code_ranges: str) -> Dict[str, Any]:
        """Activate all relevant subtopics in parallel."""
        specific_codes = []
        activated_subtopics = []
        
        async def run_subtopic(subtopic: Dict[str, Any]) -> Dict[str, Any]:
            if subtopic["code_range"] in code_ranges:
                print(f"Activating subtopic: {subtopic['name']}")
                # Run the activation function in a thread pool to avoid blocking
                loop = asyncio.get_running_loop()
                with ThreadPoolExecutor() as pool:
                    code = await loop.run_in_executor(pool, lambda: subtopic["activate_func"](scenario))
                return {
                    "code": code,
                    "name": subtopic["name"]
                }
            return None
        
        # Run all relevant subtopics concurrently
        tasks = [run_subtopic(subtopic) for subtopic in self.subtopics]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if result and result["code"]:
                specific_codes.append(result["code"])
                activated_subtopics.append(result["name"])
        
        return {
            "specific_codes": specific_codes,
            "activated_subtopics": activated_subtopics
        }