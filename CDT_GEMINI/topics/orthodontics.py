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
from subtopics.Orthodontics import (
    activate_limited_orthodontic_treatment,
    activate_comprehensive_orthodontic_treatment,
    activate_minor_treatment_harmful_habits,
    activate_other_orthodontic_services
)

class OrthodonticServices:
    """Class to analyze and activate orthodontic services based on dental scenarios."""
    
    def __init__(self, llm_service: LLMService = None):
        """Initialize with an optional LLMService instance."""
        self.llm_service = llm_service or get_service()
        self.prompt_template = self._create_prompt_template()
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create the prompt template for analyzing orthodontic services."""
        return PromptTemplate(
            template=f"""
You are a highly experienced dental coding expert with over 15 years of expertise in ADA dental codes. 
Your task is to analyze the given scenario and determine the most applicable orthodontic code range(s) based on the following classifications:

## **Limited Orthodontic Treatment (D8010-D8040)**
**Use when:** Providing partial correction or addressing a specific orthodontic problem.
**Check:** Documentation specifies which dentition stage (primary, transitional, adolescent, adult) is being treated.
**Note:** These procedures focus on limited treatment goals rather than comprehensive correction.
**Activation trigger:** Scenario mentions OR implies any partial orthodontic treatment, minor tooth movement, single arch treatment, or interceptive orthodontics. INCLUDE this range if there's any indication of focused orthodontic care rather than full correction.

## **Comprehensive Orthodontic Treatment (D8070-D8090)**
**Use when:** Providing complete orthodontic correction for the entire dentition.
**Check:** Documentation identifies the dentition stage (transitional, adolescent, adult) being treated.
**Note:** These involve full banding/bracketing of the dentition with regular adjustments.
**Activation trigger:** Scenario mentions OR implies any full orthodontic treatment, complete braces, comprehensive correction, full arch treatment, or extensive alignment. INCLUDE this range if there's any hint of complete orthodontic care addressing overall occlusion.

## **Minor Treatment to Control Harmful Habits (D8210-D8220)**
**Use when:** Correcting deleterious oral habits through appliance therapy.
**Check:** Documentation specifies the habit being addressed and type of appliance used.
**Note:** These procedures target specific habits rather than overall malocclusion.
**Activation trigger:** Scenario mentions OR implies any thumb-sucking, tongue thrusting, habit appliance, habit breaking, or interceptive treatment for parafunctional habits. INCLUDE this range if there's any suggestion of treating harmful oral habits through specialized appliances.

## **Other Orthodontic Services (D8660-D8999)**
**Use when:** Providing supplementary orthodontic services or treatments not specified elsewhere.
**Check:** Documentation details the specific service provided and its purpose.
**Note:** These include consultations, retention, repairs, and additional orthodontic services.
**Activation trigger:** Scenario mentions OR implies any pre-orthodontic visit, retainer placement, bracket repair, adjustment visit, or specialized orthodontic service. INCLUDE this range if there's any indication of orthodontic care beyond the initial appliance placement.

### **Scenario:**
{{scenario}}
{PROMPT}

RESPOND WITH ALL APPLICABLE CODE RANGES from the options above, even if they are only slightly relevant.
List them in order of relevance, with the most relevant first.
""",
            input_variables=["scenario"]
        )
    
    def analyze_orthodontic(self, scenario: str) -> str:
        """Analyze the scenario to determine applicable code ranges."""
        try:
            print(f"Analyzing orthodontic scenario: {scenario[:100]}...")
            result = self.llm_service.invoke_chain(self.prompt_template, {"scenario": scenario})
            code_range = result.strip()
            print(f"Orthodontic analyze_orthodontic result: {code_range}")
            return code_range
        except Exception as e:
            print(f"Error in analyze_orthodontic: {str(e)}")
            return ""
    
    def activate_orthodontic(self, scenario: str) -> dict:
        """Activate relevant subtopics and return detailed results."""
        try:
            # Get the code range from the analysis
            orthodontic_result = self.analyze_orthodontic(scenario)
            if not orthodontic_result:
                print("No orthodontic result returned")
                return {}
            
            print(f"Orthodontic Result in activate_orthodontic: {orthodontic_result}")
            
            # Process specific orthodontic subtopics based on the result
            specific_codes = []
            activated_subtopics = []
            
            # Check for each subtopic and activate if applicable
            subtopic_map = [
                ("D8010-D8040", activate_limited_orthodontic_treatment, "Limited Orthodontic Treatment (D8010-D8040)"),
                ("D8070-D8090", activate_comprehensive_orthodontic_treatment, "Comprehensive Orthodontic Treatment (D8070-D8090)"),
                ("D8210-D8220", activate_minor_treatment_harmful_habits, "Minor Treatment to Control Harmful Habits (D8210-D8220)"),
                ("D8660-D8999", activate_other_orthodontic_services, "Other Orthodontic Services (D8660-D8999)")
            ]
            
            for code_range, activate_func, subtopic_name in subtopic_map:
                if code_range in orthodontic_result:
                    print(f"Activating subtopic: {subtopic_name}")
                    code = activate_func(scenario)
                    if code:
                        specific_codes.append(code)
                        activated_subtopics.append(subtopic_name)
            
            # Choose the primary subtopic (either the first activated or a default)
            primary_subtopic = activated_subtopics[0] if activated_subtopics else "Comprehensive Orthodontic Treatment (D8070-D8090)"
            
            # Return a dictionary with the required fields
            return {
                "code_range": orthodontic_result,
                "subtopic": primary_subtopic,
                "activated_subtopics": activated_subtopics,
                "codes": specific_codes
            }
        except Exception as e:
            print(f"Error in orthodontic analysis: {str(e)}")
            return {}
    
    def run_analysis(self, scenario: str) -> None:
        """Run the analysis and print results."""
        print(f"Using model: {self.llm_service.model} with temperature: {self.llm_service.temperature}")
        result = self.activate_orthodontic(scenario)
        print(f"\n=== ORTHODONTIC ANALYSIS RESULT ===")
        print(f"CODE RANGE: {result.get('code_range', 'None')}")
        print(f"PRIMARY SUBTOPIC: {result.get('subtopic', 'None')}")
        print(f"ACTIVATED SUBTOPICS: {', '.join(result.get('activated_subtopics', []))}")
        print(f"SPECIFIC CODES: {', '.join(result.get('codes', []))}")

# Example usage
if __name__ == "__main__":
    orthodontic_service = OrthodonticServices()
    scenario = input("Enter an orthodontic scenario: ")
    orthodontic_service.run_analysis(scenario)