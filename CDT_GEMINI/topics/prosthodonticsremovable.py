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
from subtopics.Prosthodontics_Removable import (
    activate_complete_dentures,
    activate_partial_denture,
    activate_adjustments_to_dentures,
    activate_repairs_to_complete_dentures,
    activate_repairs_to_partial_dentures,
    activate_denture_rebase_procedures,
    activate_denture_reline_procedures,
    activate_interim_prosthesis,
    activate_other_removable_prosthetic_services
)

class RemovableProsthodonticsServices:
    """Class to analyze and activate removable prosthodontics services based on dental scenarios."""
    
    def __init__(self, llm_service: LLMService = None):
        """Initialize with an optional LLMService instance."""
        self.llm_service = llm_service or get_service()
        self.prompt_template = self._create_prompt_template()
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create the prompt template for analyzing removable prosthodontics services."""
        return PromptTemplate(
            template=f"""
You are a highly experienced dental coding expert with over 15 years of expertise in ADA dental codes. 
Your task is to analyze the given scenario and determine the most applicable removable prosthodontics code range(s) based on the following classifications:

## IMPORTANT GUIDELINES:
- You should activate ALL code ranges that have any potential relevance to the scenario
- Even if a code range is only slightly related, include it in your response
- Only exclude a code range if it is DEFINITELY NOT relevant to the scenario
- When in doubt, INCLUDE the code range rather than exclude it
- Multiple code ranges can and should be activated if they have any potential applicability
- Your goal is to ensure no potentially relevant codes are missed

## **Complete Dentures (D5110-D5140)**
**Use when:** Providing full arch prostheses for edentulous patients.
**Check:** Documentation specifies maxillary/mandibular and immediate/conventional status.
**Note:** These address complete tooth loss in an arch with full tissue coverage prostheses.
**Activation trigger:** Scenario mentions OR implies any full denture, complete denture, edentulous treatment, immediate denture, or replacement of all teeth in an arch. INCLUDE this range if there's any indication of complete tooth replacement in either arch.

## **Partial Denture (D5211-D5286)**
**Use when:** Providing removable prostheses for partially edentulous patients.
**Check:** Documentation details the framework material and arch being restored.
**Note:** These replace some but not all teeth in an arch while utilizing remaining natural teeth for support.
**Activation trigger:** Scenario mentions OR implies any partial denture, RPD, removable partial, cast framework prosthesis, or flexible base partial. INCLUDE this range if there's any hint of replacing some but not all teeth with a removable appliance.

## **Adjustments to Dentures (D5410-D5422)**
**Use when:** Modifying existing dentures to improve fit or function.
**Check:** Documentation specifies the type of denture being adjusted.
**Note:** These address minor issues without remaking or significantly altering the prosthesis.
**Activation trigger:** Scenario mentions OR implies any denture adjustment, fit correction, comfort adjustment, occlusal adjustment of prosthesis, or minor modification of denture. INCLUDE this range if there's any suggestion of minor alterations to existing dentures.

## **Repairs to Complete Dentures (D5511-D5520)**
**Use when:** Fixing damaged complete dentures.
**Check:** Documentation identifies the specific damage and repair performed.
**Note:** These restore function to damaged complete dentures without replacement.
**Activation trigger:** Scenario mentions OR implies any denture repair, broken denture, cracked denture base, replacement of broken teeth in denture, or fixing complete denture. INCLUDE this range if there's any indication of repairing damage to a complete denture.

## **Repairs to Partial Dentures (D5611-D5671)**
**Use when:** Fixing damaged partial dentures.
**Check:** Documentation details the specific component repaired or replaced.
**Note:** These restore function to damaged partial dentures by addressing specific components.
**Activation trigger:** Scenario mentions OR implies any partial denture repair, broken clasp, damaged framework, resin base repair, or adding components to existing partial. INCLUDE this range if there's any hint of repairing or modifying components of a partial denture.

## **Denture Rebase Procedures (D5710-D5725)**
**Use when:** Completely replacing the base material of an existing denture.
**Check:** Documentation indicates complete replacement of the base while maintaining the original teeth.
**Note:** These procedures address significant changes in ridge morphology requiring new base adaptation.
**Activation trigger:** Scenario mentions OR implies any denture rebase, replacing entire denture base, new base for existing denture, or complete base replacement. INCLUDE this range if there's any suggestion of replacing the entire base material while keeping the original teeth.

## **Denture Reline Procedures (D5730-D5761)**
**Use when:** Adding new material to the tissue surface of a denture to improve fit.
**Check:** Documentation specifies whether chairside or laboratory reline and type of denture.
**Note:** These procedures add material rather than completely replacing the base.
**Activation trigger:** Scenario mentions OR implies any denture reline, adding material to denture base, improving fit with new lining, chairside or lab reline. INCLUDE this range if there's any indication of adding material to the tissue surface of a denture.

## **Interim Prosthesis (D5810-D5821)**
**Use when:** Providing temporary dentures during treatment phases.
**Check:** Documentation clarifies the interim nature and purpose of the prosthesis.
**Note:** These are not intended as definitive restorations but as transitional appliances.
**Activation trigger:** Scenario mentions OR implies any temporary denture, interim prosthesis, transitional denture, provisional appliance, or temporary tooth replacement. INCLUDE this range if there's any hint of temporary dentures during transition to final prostheses.

## **Other Removable Prosthetic Services (D5765-D5899)**
**Use when:** Providing specialized prosthetic services not covered in other categories.
**Check:** Documentation details the specific service and its therapeutic purpose.
**Note:** These include tissue conditioning, precision attachment, and other advanced procedures.
**Activation trigger:** Scenario mentions OR implies any tissue conditioning, precision attachment, specialized denture procedure, overdenture, or unusual prosthetic technique. INCLUDE this range if there's any suggestion of specialized removable prosthetic services beyond standard dentures and partials.

### **Scenario:**
{{scenario}}
{PROMPT}

RESPOND WITH ALL APPLICABLE CODE RANGES from the options above, even if they are only slightly relevant.
List them in order of relevance, with the most relevant first.
""",
            input_variables=["scenario"]
        )
    
    def analyze_prosthodontics_removable(self, scenario: str) -> str:
        """Analyze the scenario to determine applicable code ranges."""
        try:
            print(f"Analyzing removable prosthodontics scenario: {scenario[:100]}...")
            result = self.llm_service.invoke_chain(self.prompt_template, {"scenario": scenario})
            code_range = result.strip()
            print(f"Prosthodontics Removable analyze_prosthodontics_removable result: {code_range}")
            return code_range
        except Exception as e:
            print(f"Error in analyze_prosthodontics_removable: {str(e)}")
            return ""
    
    def activate_prosthodontics_removable(self, scenario: str) -> dict:
        """Activate relevant subtopics and return detailed results."""
        try:
            # Get the code range from the analysis
            prosthodontics_result = self.analyze_prosthodontics_removable(scenario)
            if not prosthodontics_result:
                print("No removable prosthodontics result returned")
                return {}
            
            print(f"Prosthodontics Removable Result in activate_prosthodontics_removable: {prosthodontics_result}")
            
            # Process specific prosthodontics subtopics based on the result
            specific_codes = []
            activated_subtopics = []
            
            # Check for each subtopic and activate if applicable
            subtopic_map = [
                ("D5110-D5140", activate_complete_dentures, "Complete Dentures (D5110-D5140)"),
                ("D5211-D5286", activate_partial_denture, "Partial Denture (D5211-D5286)"),
                ("D5410-D5422", activate_adjustments_to_dentures, "Adjustments to Dentures (D5410-D5422)"),
                ("D5511-D5520", activate_repairs_to_complete_dentures, "Repairs to Complete Dentures (D5511-D5520)"),
                ("D5611-D5671", activate_repairs_to_partial_dentures, "Repairs to Partial Dentures (D5611-D5671)"),
                ("D5710-D5725", activate_denture_rebase_procedures, "Denture Rebase Procedures (D5710-D5725)"),
                ("D5730-D5761", activate_denture_reline_procedures, "Denture Reline Procedures (D5730-D5761)"),
                ("D5810-D5821", activate_interim_prosthesis, "Interim Prosthesis (D5810-D5821)"),
                ("D5765-D5899", activate_other_removable_prosthetic_services, "Other Removable Prosthetic Services (D5765-D5899)")
            ]
            
            for code_range, activate_func, subtopic_name in subtopic_map:
                if code_range in prosthodontics_result:
                    print(f"Activating subtopic: {subtopic_name}")
                    code = activate_func(scenario)
                    if code:
                        specific_codes.append(code)
                        activated_subtopics.append(subtopic_name)
            
            # Choose the primary subtopic (either the first activated or a default)
            primary_subtopic = activated_subtopics[0] if activated_subtopics else "Complete Dentures (D5110-D5140)"
            
            # Return a dictionary with the required fields
            return {
                "code_range": prosthodontics_result,
                "subtopic": primary_subtopic,
                "activated_subtopics": activated_subtopics,
                "codes": specific_codes
            }
        except Exception as e:
            print(f"Error in removable prosthodontics analysis: {str(e)}")
            return {}
    
    def run_analysis(self, scenario: str) -> None:
        """Run the analysis and print results."""
        print(f"Using model: {self.llm_service.model} with temperature: {self.llm_service.temperature}")
        result = self.activate_prosthodontics_removable(scenario)
        print(f"\n=== REMOVABLE PROSTHODONTICS ANALYSIS RESULT ===")
        print(f"CODE RANGE: {result.get('code_range', 'None')}")
        print(f"PRIMARY SUBTOPIC: {result.get('subtopic', 'None')}")
        print(f"ACTIVATED SUBTOPICS: {', '.join(result.get('activated_subtopics', []))}")
        print(f"SPECIFIC CODES: {', '.join(result.get('codes', []))}")

# Example usage
if __name__ == "__main__":
    prosthodontics_service = RemovableProsthodonticsServices()
    scenario = input("Enter a removable prosthodontics dental scenario: ")
    prosthodontics_service.run_analysis(scenario)