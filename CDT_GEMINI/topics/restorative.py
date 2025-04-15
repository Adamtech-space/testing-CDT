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
from subtopics.Restorative import (
    AmalgamRestorationsServices,
    ResinBasedCompositeRestorationsServices,
    GoldFoilRestorationsServices,
    InlaysAndOnlaysServices,
    CrownsServices,
    OtherRestorativeServices
)

class RestorativeServices:
    """Class to analyze and activate restorative services based on dental scenarios."""
    
    def __init__(self, llm_service: LLMService = None):
        """Initialize with an optional LLMService instance."""
        self.llm_service = llm_service or get_service()
        self.prompt_template = self._create_prompt_template()
        
        # Initialize the subtopic service classes
        self.amalgam_restorations = AmalgamRestorationsServices(llm_service)
        self.resin_based_composite_restorations = ResinBasedCompositeRestorationsServices(llm_service)
        self.gold_foil_restorations = GoldFoilRestorationsServices(llm_service)
        self.inlays_and_onlays = InlaysAndOnlaysServices(llm_service)
        self.crowns = CrownsServices(llm_service)
        self.other_restorative_services = OtherRestorativeServices(llm_service)
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create the prompt template for analyzing restorative services."""
        return PromptTemplate(
            template=f"""
You are a highly experienced dental coding expert with over 15 years of expertise in ADA dental codes. 
Your task is to analyze the given scenario and determine the most applicable restorative code range(s) based on the following classifications:

## **Amalgam Restorations (D2140-D2161)**
**Use when:** Placing silver-colored metal alloy fillings.
**Check:** Documentation specifies the number of surfaces involved and tooth location.
**Note:** These durable restorations are typically used for posterior teeth.
**Activation trigger:** Scenario mentions OR implies any amalgam filling, silver filling, metallic restoration, posterior filling, or direct restoration with amalgam. INCLUDE this range if there's any indication of silver-colored fillings being placed.

## **Resin-Based Composite Restorations (D2330-D2394)**
**Use when:** Placing tooth-colored fillings made of composite resin.
**Check:** Documentation details the number of surfaces involved and tooth location.
**Note:** These esthetic restorations can be used in anterior or posterior teeth.
**Activation trigger:** Scenario mentions OR implies any composite filling, tooth-colored restoration, resin restoration, bonded filling, or esthetic direct restoration. INCLUDE this range if there's any hint of tooth-colored fillings being placed.

## **Gold Foil Restorations (D2410-D2430)**
**Use when:** Placing pure gold restorations through direct techniques.
**Check:** Documentation identifies the number of surfaces and specific gold foil technique.
**Note:** These are specialized, less common restorations using pure gold.
**Activation trigger:** Scenario mentions OR implies any gold foil, direct gold restoration, or traditional gold filling technique. INCLUDE this range if there's any suggestion of pure gold being used in a direct restoration technique.

## **Inlays and Onlays (D2510-D2664)**
**Use when:** Providing laboratory-fabricated partial coverage restorations.
**Check:** Documentation specifies the material used and extent of cuspal coverage.
**Note:** These indirect restorations offer precision fit for moderate to large lesions.
**Activation trigger:** Scenario mentions OR implies any inlay, onlay, indirect restoration, laboratory-fabricated partial coverage, or precision-fitted restoration. INCLUDE this range if there's any indication of indirect partial coverage restorations.

## **Crowns (D2710-D2799)**
**Use when:** Providing full-coverage restorations for damaged teeth.
**Check:** Documentation details the crown material and reason for full coverage.
**Note:** These restore severely damaged teeth requiring complete coronal protection.
**Activation trigger:** Scenario mentions OR implies any crown, full coverage restoration, cap, coronal coverage, or complete tooth restoration. INCLUDE this range if there's any hint of full coverage restorations being provided.

## **Other Restorative Services (D2910-D2999)**
**Use when:** Providing additional services related to restorations.
**Check:** Documentation specifies the exact service and its purpose.
**Note:** These include repairs, recementations, cores, posts, and specialized services.
**Activation trigger:** Scenario mentions OR implies any core buildup, post and core, resin infiltration, temporary restoration, crown repair, veneer, reattachment of fragment, or specialized restorative service. INCLUDE this range if there's any suggestion of restorative services beyond standard fillings or crowns.

### **Scenario:**
{{scenario}}
{PROMPT}

RESPOND WITH ALL APPLICABLE CODE RANGES from the options above, even if they are only slightly relevant.
List them in order of relevance, with the most relevant first.
""",
            input_variables=["scenario"]
        )
    
    def analyze_restorative(self, scenario: str) -> str:
        """Analyze the scenario to determine applicable code ranges."""
        try:
            print(f"Analyzing restorative scenario: {scenario[:100]}...")
            result = self.llm_service.invoke_chain(self.prompt_template, {"scenario": scenario})
            code_range = result.strip()
            print(f"Restorative analyze_restorative result: {code_range}")
            return code_range
        except Exception as e:
            print(f"Error in analyze_restorative: {str(e)}")
            return ""
    
    def activate_restorative(self, scenario: str) -> dict:
        """Activate relevant subtopics and return detailed results."""
        try:
            # Get the code range from the analysis
            restorative_result = self.analyze_restorative(scenario)
            if not restorative_result:
                print("No restorative result returned")
                return {}
            
            print(f"Restorative Result in activate_restorative: {restorative_result}")
            
            # Process specific restorative subtopics based on the result
            specific_codes = []
            activated_subtopics = []
            
            # Check for each subtopic and activate if applicable
            subtopic_map = [
                ("D2140-D2161", self.amalgam_restorations.activate_amalgam_restorations, "Amalgam Restorations (D2140-D2161)"),
                ("D2330-D2394", self.resin_based_composite_restorations.activate_resin_based_composite_restorations, "Resin-Based Composite Restorations (D2330-D2394)"),
                ("D2410-D2430", self.gold_foil_restorations.activate_gold_foil_restorations, "Gold Foil Restorations (D2410-D2430)"),
                ("D2510-D2664", self.inlays_and_onlays.activate_inlays_and_onlays, "Inlays and Onlays (D2510-D2664)"),
                ("D2710-D2799", self.crowns.activate_crowns, "Crowns (D2710-D2799)"),
                ("D2910-D2999", self.other_restorative_services.activate_other_restorative_services, "Other Restorative Services (D2910-D2999)")
            ]
            
            for code_range, activate_method, subtopic_name in subtopic_map:
                if code_range in restorative_result:
                    print(f"Activating subtopic: {subtopic_name}")
                    code = activate_method(scenario)
                    if code:
                        specific_codes.append(code)
                        activated_subtopics.append(subtopic_name)
            
            # Choose the primary subtopic (either the first activated or a default)
            primary_subtopic = activated_subtopics[0] if activated_subtopics else "Resin-Based Composite Restorations (D2330-D2394)"
            
            # Return a dictionary with the required fields
            return {
                "code_range": restorative_result,
                "subtopic": primary_subtopic,
                "activated_subtopics": activated_subtopics,
                "codes": specific_codes
            }
        except Exception as e:
            print(f"Error in restorative analysis: {str(e)}")
            return {}
    
    def run_analysis(self, scenario: str) -> None:
        """Run the analysis and print results."""
        print(f"Using model: {self.llm_service.model} with temperature: {self.llm_service.temperature}")
        result = self.activate_restorative(scenario)
        print(f"\n=== RESTORATIVE ANALYSIS RESULT ===")
        print(f"CODE RANGE: {result.get('code_range', 'None')}")
        print(f"PRIMARY SUBTOPIC: {result.get('subtopic', 'None')}")
        print(f"ACTIVATED SUBTOPICS: {', '.join(result.get('activated_subtopics', []))}")
        print(f"SPECIFIC CODES: {', '.join(result.get('codes', []))}")

# Example usage
if __name__ == "__main__":
    restorative_service = RestorativeServices()
    scenario = input("Enter a restorative dental scenario: ")
    restorative_service.run_analysis(scenario)