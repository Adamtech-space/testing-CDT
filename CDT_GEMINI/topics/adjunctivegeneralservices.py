import os
import sys
import asyncio
from langchain.prompts import PromptTemplate
from prompt import PROMPT
from llm_services import LLMService, get_service, set_model, set_temperature, generate_response
from llm_services import DEFAULT_MODEL, DEFAULT_TEMP
from subtopic_registry import SubtopicRegistry

# Add parent directory to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import subtopics with fallback mechanism
try:
    from subtopics.AdjunctiveGeneralServices.anesthesia import anesthesia_service
    from subtopics.AdjunctiveGeneralServices.drugs import drugs_service
    from subtopics.AdjunctiveGeneralServices.miscellaneous_services import misc_service
    from subtopics.AdjunctiveGeneralServices.non_clinical_procedures import non_clinical_service
    from subtopics.AdjunctiveGeneralServices.professional_consultation import consultation_service
    from subtopics.AdjunctiveGeneralServices.professional_visits import visits_service
    from subtopics.AdjunctiveGeneralServices.unclassified_treatment import unclassified_service
except ImportError:
    print("Warning: Could not import subtopics for AdjunctiveGeneralServices. Using fallback functions.")
    # Define fallback functions if needed
    def activate_unclassified_treatment(scenario): return None
    def activate_anesthesia(scenario): return None
    def activate_professional_consultation(scenario): return None
    def activate_professional_visits(scenario): return None
    def activate_drugs(scenario): return None
    def activate_miscellaneous_services(scenario): return None
    def activate_non_clinical_procedures(scenario): return None

class AdjunctiveGeneralServices:
    """Class to analyze and activate adjunctive general services based on dental scenarios."""
    
    def __init__(self, llm_service: LLMService = None):
        """Initialize with an optional LLMService instance."""
        self.llm_service = llm_service or get_service()
        self.prompt_template = self._create_prompt_template()
        self.registry = SubtopicRegistry()
        self._register_subtopics()
    
    def _register_subtopics(self):
        """Register all subtopics for parallel activation."""
        self.registry.register("D9110-D9130", unclassified_service.activate_unclassified_treatment, 
                            "Unclassified Treatment (D9110-D9130)")
        self.registry.register("D9210-D9248", anesthesia_service.activate_anesthesia, 
                            "Anesthesia (D9210-D9248)")
        self.registry.register("D9310-D9311", consultation_service.activate_professional_consultation, 
                            "Professional Consultation (D9310-D9311)")
        self.registry.register("D9410-D9450", visits_service.activate_professional_visits, 
                            "Professional Visits (D9410-D9450)")
        self.registry.register("D9610-D9630", drugs_service.activate_drugs, 
                            "Drugs (D9610-D9630)")
        self.registry.register("D9910-D9973", misc_service.activate_miscellaneous_services, 
                            "Miscellaneous Services (D9910-D9973)")
        self.registry.register("D9961-D9999", non_clinical_service.activate_non_clinical_procedures, 
                            "Non-clinical Procedures (D9961-D9999)")
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create the prompt template for analyzing adjunctive general services."""
        return PromptTemplate(
            template=f"""
You are a highly experienced dental coding expert with over 15 years of expertise in ADA dental codes. 
Your task is to analyze the given scenario and determine the most applicable adjunctive general services code range(s) based on the following detailed classifications:
[... Prompt content remains unchanged ...]
### **Scenario:**
{{scenario}}
{PROMPT}
RESPOND WITH ALL APPLICABLE CODE RANGES from the options above, even if they are only slightly relevant.
List them in order of relevance, with the most relevant first.
Example: "D9210-D9248, D9110-D9130, D9610-D9630"
""",
            input_variables=["scenario"]
        )
    
    def analyze_adjunctive_general_services(self, scenario: str) -> str:
        """Analyze the scenario to determine applicable code ranges."""
        try:
            print(f"Analyzing adjunctive general services scenario: {scenario[:100]}...")
            result = self.llm_service.invoke_chain(self.prompt_template, {"scenario": scenario})
            code_range = result.strip()
            print(f"Adjunctive analyze_adjunctive_general_services result: {code_range}")
            return code_range
        except Exception as e:
            print(f"Error in analyze_adjunctive_general_services: {str(e)}")
            return ""
    
    async def activate_adjunctive_general_services(self, scenario: str) -> dict:
        """Activate relevant subtopics in parallel and return detailed results."""
        try:
            # Get the code range from the analysis
            adjunctive_result = self.analyze_adjunctive_general_services(scenario)
            if not adjunctive_result:
                print("No adjunctive result returned")
                return {}
            
            print(f"Adjunctive Result in activate_adjunctive_general_services: {adjunctive_result}")
            
            # Activate subtopics in parallel using the registry
            result = await self.registry.activate_all(scenario, adjunctive_result)
            
            # Choose the primary subtopic only if there are activated subtopics
            primary_subtopic = result["activated_subtopics"][0] if result["activated_subtopics"] else None
            
            # Return a dictionary with the required fields
            return {
                "code_range": adjunctive_result,
                "subtopic": primary_subtopic,
                "activated_subtopics": result["activated_subtopics"],
                "codes": result["specific_codes"]
            }
        except Exception as e:
            print(f"Error in adjunctive general services analysis: {str(e)}")
            return {}
    
    async def run_analysis(self, scenario: str) -> None:
        """Run the analysis and print results."""
        print(f"Using model: {self.llm_service.model} with temperature: {self.llm_service.temperature}")
        result = await self.activate_adjunctive_general_services(scenario)
        print(f"\n=== ADJUNCTIVE GENERAL SERVICES ANALYSIS RESULT ===")
        print(f"CODE RANGE: {result.get('code_range', 'None')}")
        print(f"PRIMARY SUBTOPIC: {result.get('subtopic', 'None')}")
        print(f"ACTIVATED SUBTOPICS: {', '.join(result.get('activated_subtopics', []))}")
        print(f"SPECIFIC CODES: {', '.join(result.get('codes', []))}")

# Example usage
if __name__ == "__main__":
    async def main():
        adj_service = AdjunctiveGeneralServices()
        scenario = input("Enter an adjunctive general services dental scenario: ")
        await adj_service.run_analysis(scenario)
    
    asyncio.run(main())