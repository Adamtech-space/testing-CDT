"""
Module for extracting space maintenance codes.
"""

import os
import sys
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from subtopics.prompt.prompt import PROMPT
from llm_services import get_llm_service

# Load environment variables
load_dotenv()

class SpaceMaintenanceServices:
    """
    Class for extracting space maintenance codes.
    """
    
    def __init__(self, temperature=0.0):
        """
        Initialize the SpaceMaintenanceServices class.
        
        Args:
            temperature (float, optional): Temperature setting for the LLM. Defaults to 0.0.
        """
        self.temperature = temperature
        self.llm_service = get_llm_service(temperature=temperature)
        self.prompt_template = self._create_prompt_template()
        
    def _create_prompt_template(self):
        """
        Create a LangChain-based prompt template for space maintenance code extraction.
        
        Returns:
            PromptTemplate: A configured prompt template for space maintenance code extraction.
        """
        return PromptTemplate(
            template=f"""
You are a highly experienced dental coding expert

### Before picking a code, ask:
- What was the primary reason the patient came in? Was it to maintain space due to premature tooth loss, or for another issue?
- Is the space maintainer fixed or removable?
- Is it unilateral (one side) or bilateral (both sides), and which arch is involved (maxillary or mandibular)?
- Is this a new placement, a repair (re-cement/re-bond), or a removal procedure?
- Does the patient's dental history or current condition (e.g., crowding, eruption patterns) support the need for space maintenance?

---

### Preventive Dental Codes: Space Maintenance (Passive Appliances)

#### Code: D1510 - Space Maintainer — Fixed, Unilateral — Per Quadrant
- **When to use:**
  - Fixed unilateral space maintainer placed in one quadrant to preserve space after premature tooth loss.
  - Excludes distal shoe space maintainers (used for unerupted teeth).
- **What to check:**
  - Confirm premature loss of a primary tooth in the quadrant and need to maintain space.
  - Verify the appliance is fixed (e.g., band and loop) and unilateral (one side only).
  - Ensure it's not a distal shoe design, which requires a different approach.
- **Notes:**
  - Per-quadrant code—specify quadrant (e.g., UR, UL, LR, LL) in documentation.
  - Used primarily in mixed dentition to prevent drifting of adjacent teeth.
  - Requires radiographic evidence of tooth loss and space assessment.

#### Code: D1516 - Space Maintainer — Fixed — Bilateral, Maxillary
- **When to use:**
  - Fixed bilateral space maintainer placed on the maxillary arch to preserve space on both sides.
  - Typically used after bilateral primary tooth loss in the upper jaw.
- **What to check:**
  - Confirm bilateral tooth loss in the maxillary arch and need for space preservation.
  - Verify the appliance is fixed (e.g., transpalatal arch) and spans both sides.
  - Assess occlusion to ensure the appliance won't interfere with bite.
- **Notes:**
  - Specific to maxillary arch—document appliance type and tooth numbers involved.
  - Common in pediatric cases with multiple early tooth losses.
  - Check for stability and patient comfort post-placement.

#### Code: D1517 - Space Maintainer — Fixed — Bilateral, Mandibular
- **When to use:**
  - Fixed bilateral space maintainer placed on the mandibular arch to preserve space on both sides.
  - Used after bilateral primary tooth loss in the lower jaw.
- **What to check:**
  - Confirm bilateral tooth loss in the mandibular arch and space maintenance need.
  - Verify the appliance is fixed (e.g., lingual holding arch) and bilateral.
  - Evaluate mandibular growth patterns to ensure proper fit over time.
- **Notes:**
  - Specific to mandibular arch—note appliance design and affected teeth.
  - Often used to prevent molar tipping or crowding in mixed dentition.
  - Requires periodic monitoring for fit as jaw develops.

#### Code: D1520 - Space Maintainer — Removable, Unilateral — Per Quadrant
- **When to use:**
  - Removable unilateral space maintainer placed in one quadrant to maintain space.
  - Suitable when patient compliance allows for a removable appliance.
- **What to check:**
  - Confirm premature tooth loss in the quadrant and suitability for a removable device.
  - Verify the appliance is unilateral and removable (e.g., partial denture-like design).
  - Assess patient's ability to maintain and wear the appliance consistently.
- **Notes:**
  - Per-quadrant code—specify quadrant and appliance details.
  - Less common than fixed options due to compliance concerns in children.
  - Patient/parent education on care and wear time is critical.

#### Code: D1526 - Space Maintainer — Removable — Bilateral, Maxillary
- **When to use:**
  - Removable bilateral space maintainer placed on the maxillary arch for both sides.
  - Used after bilateral tooth loss in the upper jaw with a removable design.
- **What to check:**
  - Confirm bilateral maxillary tooth loss and need for space maintenance.
  - Verify the appliance is removable and spans both sides of the upper arch.
  - Check fit and patient tolerance for a removable device.
- **Notes:**
  - Specific to maxillary arch—document tooth numbers and appliance type.
  - May include clasps or acrylic components; note in records.
  - Requires patient cooperation for effectiveness.

#### Code: D1527 - Space Maintainer — Removable — Bilateral, Mandibular
- **When to use:**
  - Removable bilateral space maintainer placed on the mandibular arch for both sides.
  - Applied after bilateral tooth loss in the lower jaw with a removable appliance.
- **What to check:**
  - Confirm bilateral mandibular tooth loss and space preservation need.
  - Verify the appliance is removable and bilateral in the lower arch.
  - Assess oral hygiene and patient ability to manage a removable device.
- **Notes:**
  - Specific to mandibular arch—record appliance details and teeth involved.
  - Less frequent than fixed mandibular options (e.g., D1517) due to stability issues.
  - Document instructions for removal, cleaning, and wear.

#### Code: D1551 - Re-cement or Re-bond Bilateral Space Maintainer — Maxillary
- **When to use:**
  - Re-cementing or re-bonding a previously placed bilateral maxillary space maintainer.
  - Used when the fixed appliance has loosened or detached.
- **What to check:**
  - Confirm the original appliance is bilateral and maxillary (e.g., D1516).
  - Assess why it failed (e.g., cement washout, trauma) and current fit.
  - Verify no damage to the appliance requiring replacement.
- **Notes:**
  - Specific to maxillary bilateral fixed maintainers—document prior placement.
  - Not for new placements or removable appliances.
  - Check occlusion post-repair to ensure stability.

#### Code: D1552 - Re-cement or Re-bond Bilateral Space Maintainer — Mandibular
- **When to use:**
  - Re-cementing or re-bonding a previously placed bilateral mandibular space maintainer.
  - Applied when the fixed appliance has become loose or dislodged.
- **What to check:**
  - Confirm the original appliance is bilateral and mandibular (e.g., D1517).
  - Evaluate cause of detachment and integrity of the appliance.
  - Ensure proper alignment and function after re-cementing.
- **Notes:**
  - Specific to mandibular bilateral fixed maintainers—note prior code used.
  - Requires documentation of repair process and materials (e.g., cement type).
  - Not for unilateral or removable devices.

#### Code: D1553 - Re-cement or Re-bond Unilateral Space Maintainer — Per Quadrant
- **When to use:**
  - Re-cementing or re-bonding a previously placed unilateral space maintainer in one quadrant.
  - Used for fixed unilateral appliances that have loosened.
- **What to check:**
  - Confirm the original appliance is unilateral and fixed (e.g., D1510).
  - Identify quadrant and reason for failure (e.g., adhesive breakdown).
  - Verify appliance condition before reattachment.
- **Notes:**
  - Per-quadrant code—specify quadrant and prior placement details.
  - Distinct from bilateral repair codes (D1551/D1552).
  - Ensure space is still maintained post-repair.

#### Code: D1556 - Removal of Fixed Unilateral Space Maintainer — Per Quadrant
- **When to use:**
  - Removal of a fixed unilateral space maintainer from one quadrant.
  - Typically when the permanent tooth erupts or space maintenance is no longer needed.
- **What to check:**
  - Confirm the appliance is unilateral and fixed (e.g., D1510) in the specified quadrant.
  - Verify eruption status of succedaneous tooth or treatment plan change.
  - Assess for any complications (e.g., embedded bands) during removal.
- **Notes:**
  - Per-quadrant code—document quadrant and removal reason.
  - Not for bilateral or removable appliances.
  - Record post-removal oral health status.

#### Code: D1557 - Removal of Fixed Bilateral Space Maintainer — Maxillary
- **When to use:**
  - Removal of a fixed bilateral maxillary space maintainer.
  - Used when space maintenance is complete or no longer required.
- **What to check:**
  - Confirm the appliance is bilateral and maxillary (e.g., D1516).
  - Check for permanent tooth eruption or orthodontic plan updates.
  - Evaluate appliance condition and ease of removal.
- **Notes:**
  - Specific to maxillary bilateral fixed maintainers—note prior placement.
  - Requires documentation of removal process and arch condition.
  - Distinct from unilateral or mandibular codes.

#### Code: D1558 - Removal of Fixed Bilateral Space Maintainer — Mandibular
- **When to use:**
  - Removal of a fixed bilateral mandibular space maintainer.
  - Applied when space preservation is no longer necessary.
- **What to check:**
  - Confirm the appliance is bilateral and mandibular (e.g., D1517).
  - Assess eruption of permanent teeth or changes in treatment needs.
  - Ensure no damage to surrounding teeth during removal.
- **Notes:**
  - Specific to mandibular bilateral fixed maintainers—document details.
  - Not for unilateral or removable appliances.
  - Record any follow-up care or observations post-removal.

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
- *Fixed vs. Removable:* Match the appliance design to the code—fixed codes (D1510-D1517) vs. removable codes (D1520-D1527).
- *Unilateral vs. Bilateral:* Carefully distinguish between single-quadrant devices (D1510/D1520) and those spanning both sides of an arch (D1516-D1517/D1526-D1527).
- *Arch Specificity:* For bilateral appliances, always specify maxillary (D1516/D1526) or mandibular (D1517/D1527).
- *Service Type:* Initial placement is distinct from repair (D1551-D1553) or removal (D1556-D1558)—don't bundle with placement codes.
- *Documentation Precision:* Note quadrant, appliance design, and purpose (e.g., tooth loss prevention vs. distal guidance for D1575).

Scenario:
"{{question}}"

{PROMPT}
""",
            input_variables=["question"]
        )
        
    def extract_space_maintenance_code(self, scenario):
        """
        Extract space maintenance code(s) for a given scenario.
        
        Args:
            scenario (str): The dental scenario to analyze.
            
        Returns:
            str: The extracted space maintenance code(s).
        """
        try:
            result = self.llm_service.invoke(
                self.prompt_template.format(question=scenario)
            )
            print(f"Space maintenance code result: {result}")
            return result.strip()
        except Exception as e:
            print(f"Error in extract_space_maintenance_code: {str(e)}")
            return ""
            
    def activate_space_maintenance(self, scenario):
        """
        Activate space maintenance analysis and return results.
        
        Args:
            scenario (str): The dental scenario to analyze.
            
        Returns:
            str: The extracted space maintenance code(s).
        """
        try:
            return self.extract_space_maintenance_code(scenario)
        except Exception as e:
            print(f"Error in activate_space_maintenance: {str(e)}")
            return ""
            
    def run_analysis(self, scenario):
        """
        Run the space maintenance analysis for a given scenario.
        
        Args:
            scenario (str): The dental scenario to analyze.
            
        Returns:
            str: The extracted space maintenance code(s).
        """
        return self.activate_space_maintenance(scenario)

# For backwards compatibility
def extract_space_maintenance_code(scenario, temperature=0.0):
    """
    Extract space maintenance code(s) for a given scenario.
    """
    service = SpaceMaintenanceServices(temperature=temperature)
    return service.extract_space_maintenance_code(scenario)

def activate_space_maintenance(scenario):
    """
    Activate space maintenance analysis and return results.
    """
    service = SpaceMaintenanceServices()
    return service.activate_space_maintenance(scenario) 