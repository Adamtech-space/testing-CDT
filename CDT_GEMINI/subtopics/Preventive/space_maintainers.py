"""
Module for extracting space maintainers codes.
"""

import os
import sys
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from subtopics.prompt.prompt import PROMPT
from llm_services import get_llm_service

# Load environment variables
load_dotenv()

class SpaceMaintainersServices:
    """
    Class for extracting space maintainers codes.
    """
    
    def __init__(self, temperature=0.0):
        """
        Initialize the SpaceMaintainersServices class.
        
        Args:
            temperature (float, optional): Temperature setting for the LLM. Defaults to 0.0.
        """
        self.temperature = temperature
        self.llm_service = get_llm_service(temperature=temperature)
        self.prompt_template = self._create_prompt_template()
        
    def _create_prompt_template(self):
        """
        Create a LangChain-based prompt template for space maintainers code extraction.
        
        Returns:
            PromptTemplate: A configured prompt template for space maintainers code extraction.
        """
        return PromptTemplate(
            template=f"""
You are a highly experienced dental coding expert

 Before picking a code, ask:
- What was the primary reason the patient came in? Was it to guide the eruption of a permanent tooth due to premature primary tooth loss, or for another issue?
- Is the space maintainer a distal shoe design, fixed, and unilateral?
- Which quadrant is involved, and is the first permanent molar unerupted?
- Is this for the initial fabrication and delivery, or does it involve follow-up, adjustments, or replacement?
- Does the patient's dental history (e.g., early tooth loss, eruption patterns) justify the use of a distal shoe appliance?

---

### Preventive Dental Codes: Space Maintainers

#### Code: D1575 - Distal Shoe Space Maintainer — Fixed, Unilateral — Per Quadrant
- **When to use:**
  - Fabrication and delivery of a fixed, unilateral distal shoe space maintainer in one quadrant.
  - Designed to extend subgingivally and distally to guide the eruption of the first permanent molar after premature loss of a primary molar (typically the second primary molar).
- **What to check:**
  - Confirm premature loss of a primary molar and the first permanent molar is unerupted, requiring guidance.
  - Verify the appliance is fixed, unilateral, and uses a distal shoe design (e.g., a metal extension into the tissue).
  - Assess the quadrant involved and ensure proper space for the erupting molar.
  - Check radiographs to confirm the position of the unerupted molar and surrounding bone structure.
  - Ensure this is for initial placement only, not follow-up or replacement.
- **Notes:**
  - Per-quadrant code—specify quadrant (e.g., UR, UL, LR, LL) in documentation.
  - Distinct from other space maintainers (e.g., D1510) due to its subgingival extension and specific purpose.
  - Does not include ongoing adjustments, follow-up visits, or replacement appliances after eruption—those are separate services.
  - Typically used in pediatric patients with mixed dentition; requires careful monitoring due to tissue interaction.
  - Documentation should include tooth number lost, molar eruption status, and appliance design details.

---

### Key Takeaways:
- *Distal Shoe Specificity:* D1575 is unique for its subgingival and distal extension to guide an unerupted first permanent molar—don't confuse with other space maintainers.
- *Initial Placement Only:* Covers fabrication and delivery; subsequent adjustments or replacements aren't included and may require narrative or different coding.
- *Quadrant-Based:* Always identify the specific quadrant treated for accurate billing and tracking.
- *Patient Monitoring:* Due to its invasive design, regular follow-ups are critical (though not billable under D1575) to ensure proper eruption and tissue health.
- *Documentation Precision:* Link the code to premature tooth loss, unerupted molar position, and appliance specifics to justify its use.

Scenario:
"{{question}}"

{PROMPT}
""",
            input_variables=["question"]
        )
        
    def extract_space_maintainers_code(self, scenario):
        """
        Extract space maintainers code(s) for a given scenario.
        
        Args:
            scenario (str): The dental scenario to analyze.
            
        Returns:
            str: The extracted space maintainers code(s).
        """
        try:
            # First check if this is about bilateral maxillary space maintainer removal
            scenario_lower = scenario.lower()
            if ("maxillary" in scenario_lower or "upper" in scenario_lower) and \
               ("bilateral" in scenario_lower or "both sides" in scenario_lower) and \
               ("remov" in scenario_lower or "take off" in scenario_lower or "take out" in scenario_lower):
                print("Space maintainers module: This is a maxillary bilateral removal scenario - deferring to space_maintenance.py")
                return ""  # Return empty so that the space_maintenance module's D1557 is used

            # Only proceed with chain if not a maxillary bilateral removal scenario
            result = self.llm_service.invoke(
                self.prompt_template.format(question=scenario)
            )
            print(f"Space maintainers code result: {result}")
            return result.strip()
        except Exception as e:
            print(f"Error in extract_space_maintainers_code: {str(e)}")
            return ""
            
    def activate_space_maintainers(self, scenario):
        """
        Activate space maintainers analysis and return results.
        
        Args:
            scenario (str): The dental scenario to analyze.
            
        Returns:
            str: The extracted space maintainers code(s).
        """
        try:
            return self.extract_space_maintainers_code(scenario)
        except Exception as e:
            print(f"Error in activate_space_maintainers: {str(e)}")
            return ""
            
    def run_analysis(self, scenario):
        """
        Run the space maintainers analysis for a given scenario.
        
        Args:
            scenario (str): The dental scenario to analyze.
            
        Returns:
            str: The extracted space maintainers code(s).
        """
        return self.activate_space_maintainers(scenario)

# For backwards compatibility
def extract_space_maintainers_code(scenario, temperature=0.0):
    """
    Extract space maintainers code(s) for a given scenario.
    """
    service = SpaceMaintainersServices(temperature=temperature)
    return service.extract_space_maintainers_code(scenario)

def activate_space_maintainers(scenario):
    """
    Activate space maintainers analysis and return results.
    """
    service = SpaceMaintainersServices()
    return service.activate_space_maintainers(scenario) 