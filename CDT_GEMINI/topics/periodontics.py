import os
import sys
from langchain.prompts import PromptTemplate
from llm_services import LLMService, get_service, set_model, set_temperature
from llm_services import DEFAULT_MODEL, DEFAULT_TEMP

# Add the root directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

# Import modules
from topics.prompt import PROMPT
from subtopics.Periodontics import (
    activate_surgical_services,
    activate_non_surgical_services,
    activate_other_periodontal_services,
)

class PeriodonticServices:
    """Class to analyze and activate periodontic services based on dental scenarios."""
    
    def __init__(self, llm_service: LLMService = None):
        """Initialize with an optional LLMService instance."""
        self.llm_service = llm_service or get_service()
        self.prompt_template = self._create_prompt_template()
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create the prompt template for analyzing periodontic services."""
        return PromptTemplate(
            template=f"""
You are a highly experienced dental coding expert with over 15 years of expertise in ADA dental codes. 
Your task is to analyze the given scenario and determine the most applicable periodontal code range(s) based on the following classifications:

## **Surgical Services (D4210-D4286)**
**Use when:** Performing invasive periodontal procedures involving incisions and flap elevation.
**Check:** Documentation details the specific surgical approach, tissues involved, and treatment goals.
**Note:** These procedures address moderate to severe periodontal disease through surgical intervention.
**Activation trigger:** Scenario mentions OR implies any gum surgery, periodontal surgery, flap procedure, gingivectomy, graft placement, or surgical treatment of periodontal disease. INCLUDE this range if there's any indication of invasive treatment of gum or periodontal tissues.

## **Non-Surgical Periodontal Services (D4322-D4381)**
**Use when:** Providing non-invasive treatment for periodontal disease.
**Check:** Documentation specifies the extent of treatment and instruments/methods used.
**Note:** These procedures treat periodontal disease without surgical intervention.
**Activation trigger:** Scenario mentions OR implies any scaling and root planing, deep cleaning, periodontal debridement, non-surgical periodontal therapy, or treatment of gum disease without surgery. INCLUDE this range if there's any hint of non-surgical treatment of periodontal conditions.

## **Other Periodontal Services (D4910-D4999)**
**Use when:** Providing specialized periodontal care beyond routine treatment.
**Check:** Documentation details the specific service and its therapeutic purpose.
**Note:** These include maintenance treatments following active therapy and specialized interventions.
**Activation trigger:** Scenario mentions OR implies any periodontal maintenance, antimicrobial delivery, gingival irrigation, local drug delivery, or follow-up periodontal care. INCLUDE this range if there's any suggestion of ongoing periodontal management or specialized treatments.

### **Scenario:**
{{scenario}}
{PROMPT}

RESPOND WITH ALL APPLICABLE CODE RANGES from the options above, even if they are only slightly relevant.
List them in order of relevance, with the most relevant first.
Example: "D4322-D4381, D4910-D4999, D4210-D4286"
""",
            input_variables=["scenario"]
        )
    
    def analyze_periodontic(self, scenario: str) -> str:
        """Analyze the scenario to determine applicable code ranges."""
        try:
            print(f"Analyzing periodontic scenario: {scenario[:100]}...")
            result = self.llm_service.invoke_chain(self.prompt_template, {"scenario": scenario})
            code_range = result.strip()
            print(f"Periodontic analyze_periodontic result: {code_range}")
            return code_range
        except Exception as e:
            print(f"Error in analyze_periodontic: {str(e)}")
            return ""
    
    def activate_periodontic(self, scenario: str) -> dict:
        """Activate relevant subtopics and return detailed results."""
        try:
            # Get the code range from the analysis
            periodontic_result = self.analyze_periodontic(scenario)
            if not periodontic_result:
                print("No periodontic result returned")
                return {}
            
            print(f"Periodontic Result in activate_periodontic: {periodontic_result}")
            
            # Process specific periodontic subtopics based on the result
            specific_codes = []
            activated_subtopics = []
            
            # Check for each subtopic and activate if applicable
            subtopic_map = [
                ("D4210-D4286", activate_surgical_services, "Surgical Services (D4210-D4286)"),
                ("D4322-D4381", activate_non_surgical_services, "Non-Surgical Services (D4322-D4381)"),
                ("D4910-D4999", activate_other_periodontal_services, "Other Periodontal Services (D4910-D4999)")
            ]
            
            for code_range, activate_func, subtopic_name in subtopic_map:
                if code_range in periodontic_result:
                    print(f"Activating subtopic: {subtopic_name}")
                    code = activate_func(scenario)
                    if code:
                        specific_codes.append(code)
                        activated_subtopics.append(subtopic_name)
            
            # Choose the primary subtopic (either the first activated or a default)
            primary_subtopic = activated_subtopics[0] if activated_subtopics else "Non-Surgical Services (D4322-D4381)"
            
            # Return a dictionary with the required fields
            return {
                "code_range": periodontic_result,
                "subtopic": primary_subtopic,
                "activated_subtopics": activated_subtopics,
                "codes": specific_codes
            }
        except Exception as e:
            print(f"Error in periodontic analysis: {str(e)}")
            return {}
    
    def run_analysis(self, scenario: str) -> None:
        """Run the analysis and print results."""
        print(f"Using model: {self.llm_service.model} with temperature: {self.llm_service.temperature}")
        result = self.activate_periodontic(scenario)
        print(f"\n=== PERIODONTIC ANALYSIS RESULT ===")
        print(f"CODE RANGE: {result.get('code_range', 'None')}")
        print(f"PRIMARY SUBTOPIC: {result.get('subtopic', 'None')}")
        print(f"ACTIVATED SUBTOPICS: {', '.join(result.get('activated_subtopics', []))}")
        print(f"SPECIFIC CODES: {', '.join(result.get('codes', []))}")

# Example usage
if __name__ == "__main__":
    periodontic_service = PeriodonticServices()
    scenario = input("Enter a periodontic scenario: ")
    periodontic_service.run_analysis(scenario)