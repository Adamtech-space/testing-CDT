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
from subtopics.Prosthodontics_Fixed import (
    activate_fixed_partial_denture_pontics,
    activate_fixed_partial_denture_retainers_inlays_onlays,
    activate_fixed_partial_denture_retainers_crowns,
    activate_other_fixed_partial_denture_services
)

class FixedProsthodonticsServices:
    """Class to analyze and activate fixed prosthodontics services based on dental scenarios."""
    
    def __init__(self, llm_service: LLMService = None):
        """Initialize with an optional LLMService instance."""
        self.llm_service = llm_service or get_service()
        self.prompt_template = self._create_prompt_template()
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create the prompt template for analyzing fixed prosthodontics services."""
        return PromptTemplate(
            template=f"""
You are a highly experienced dental coding expert with over 15 years of expertise in ADA dental codes. 
Your task is to analyze the given scenario and determine the most applicable fixed prosthodontics code range(s) based on the following classifications:

## **Fixed Partial Denture Pontics (D6205-D6253)**
**Use when:** Providing artificial replacement teeth in a fixed bridge.
**Check:** Documentation specifies pontic material and design for the edentulous area.
**Note:** These are the artificial teeth in a bridge that replace missing natural teeth.
**Activation trigger:** Scenario mentions OR implies any bridge pontic, artificial tooth in a bridge, tooth replacement in fixed prosthesis, or pontic design/material. INCLUDE this range if there's any indication of replacement teeth in a fixed bridge.

## **Fixed Partial Denture Retainers — Inlays/Onlays (D6545-D6634)**
**Use when:** Using inlays or onlays as the retaining elements for a fixed bridge.
**Check:** Documentation details the inlay/onlay material and design as a bridge retainer.
**Note:** These are more conservative than full crowns but still provide retention for the bridge.
**Activation trigger:** Scenario mentions OR implies any inlay retainer, onlay abutment, partial coverage retainer for bridge, or conservative bridge attachment. INCLUDE this range if there's any hint of inlays or onlays being used to support a fixed bridge.

## **Fixed Partial Denture Retainers — Crowns (D6710-D6793)**
**Use when:** Using full coverage crowns as the retaining elements for a fixed bridge.
**Check:** Documentation specifies crown material and design as bridge abutments.
**Note:** These provide maximum retention but require more tooth reduction.
**Activation trigger:** Scenario mentions OR implies any crown retainer, abutment crown, full coverage bridge support, or crown preparation for bridge. INCLUDE this range if there's any suggestion of full crowns being used to support a fixed bridge.

## **Other Fixed Partial Denture Services (D6920-D6999)**
**Use when:** Providing additional services related to fixed bridges.
**Check:** Documentation details the specific service and its purpose for the bridge.
**Note:** These include repairs, recementations, and specialized bridge components.
**Activation trigger:** Scenario mentions OR implies any bridge repair, recementation, stress breaker, precision attachment, or maintenance of existing bridge. INCLUDE this range if there's any indication of services for fixed bridges beyond the initial fabrication.

### **Scenario:**
{{scenario}}
{PROMPT}

RESPOND WITH ALL APPLICABLE CODE RANGES from the options above, even if they are only slightly relevant.
List them in order of relevance, with the most relevant first.
""",
            input_variables=["scenario"]
        )
    
    def analyze_prosthodontics_fixed(self, scenario: str) -> str:
        """Analyze the scenario to determine applicable code ranges."""
        try:
            print(f"Analyzing fixed prosthodontics scenario: {scenario[:100]}...")
            result = self.llm_service.invoke_chain(self.prompt_template, {"scenario": scenario})
            code_range = result.strip()
            print(f"Prosthodontics Fixed analyze_prosthodontics_fixed result: {code_range}")
            return code_range
        except Exception as e:
            print(f"Error in analyze_prosthodontics_fixed: {str(e)}")
            return ""
    
    def activate_prosthodontics_fixed(self, scenario: str) -> dict:
        """Activate relevant subtopics and return detailed results."""
        try:
            # Get the code range from the analysis
            prosthodontics_result = self.analyze_prosthodontics_fixed(scenario)
            if not prosthodontics_result:
                print("No prosthodontics result returned")
                return {}
            
            print(f"Prosthodontics Fixed Result in activate_prosthodontics_fixed: {prosthodontics_result}")
            
            # Process specific prosthodontics subtopics based on the result
            specific_codes = []
            activated_subtopics = []
            
            # Check for each subtopic and activate if applicable
            subtopic_map = [
                ("D6205-D6253", activate_fixed_partial_denture_pontics, "Fixed Partial Denture Pontics (D6205-D6253)"),
                ("D6545-D6634", activate_fixed_partial_denture_retainers_inlays_onlays, "Fixed Partial Denture Retainers — Inlays/Onlays (D6545-D6634)"),
                ("D6710-D6793", activate_fixed_partial_denture_retainers_crowns, "Fixed Partial Denture Retainers — Crowns (D6710-D6793)"),
                ("D6920-D6999", activate_other_fixed_partial_denture_services, "Other Fixed Partial Denture Services (D6920-D6999)")
            ]
            
            for code_range, activate_func, subtopic_name in subtopic_map:
                if code_range in prosthodontics_result:
                    print(f"Activating subtopic: {subtopic_name}")
                    code = activate_func(scenario)
                    if code:
                        specific_codes.append(code)
                        activated_subtopics.append(subtopic_name)
            
            # Choose the primary subtopic (either the first activated or a default)
            primary_subtopic = activated_subtopics[0] if activated_subtopics else "Fixed Partial Denture Pontics (D6205-D6253)"
            
            # Return a dictionary with the required fields
            return {
                "code_range": prosthodontics_result,
                "subtopic": primary_subtopic,
                "activated_subtopics": activated_subtopics,
                "codes": specific_codes
            }
        except Exception as e:
            print(f"Error in prosthodontics fixed analysis: {str(e)}")
            return {}
    
    def run_analysis(self, scenario: str) -> None:
        """Run the analysis and print results."""
        print(f"Using model: {self.llm_service.model} with temperature: {self.llm_service.temperature}")
        result = self.activate_prosthodontics_fixed(scenario)
        print(f"\n=== FIXED PROSTHODONTICS ANALYSIS RESULT ===")
        print(f"CODE RANGE: {result.get('code_range', 'None')}")
        print(f"PRIMARY SUBTOPIC: {result.get('subtopic', 'None')}")
        print(f"ACTIVATED SUBTOPICS: {', '.join(result.get('activated_subtopics', []))}")
        print(f"SPECIFIC CODES: {', '.join(result.get('codes', []))}")

# Example usage
if __name__ == "__main__":
    prosthodontics_service = FixedProsthodonticsServices()
    scenario = input("Enter a fixed prosthodontics dental scenario: ")
    prosthodontics_service.run_analysis(scenario)