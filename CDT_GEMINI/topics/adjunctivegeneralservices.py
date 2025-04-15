import os
import sys
from langchain.prompts import PromptTemplate
from prompt import PROMPT
from llm_services import LLMService, get_service, set_model, set_temperature, generate_response
from llm_services import DEFAULT_MODEL, DEFAULT_TEMP

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
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create the prompt template for analyzing adjunctive general services."""
        return PromptTemplate(
            template=f"""
You are a highly experienced dental coding expert with over 15 years of expertise in ADA dental codes. 
Your task is to analyze the given scenario and determine the most applicable adjunctive general services code range(s) based on the following detailed classifications:

## **Unclassified Treatment (D9110-D9130)**

### **Before picking a code, ask:**
- Is the patient experiencing dental pain or discomfort requiring immediate relief?
- Is the treatment being provided as an emergency or palliative measure?
- Is this a minor, temporary procedure to stabilize a condition until definitive treatment?
- Is the patient's condition acute or chronic?

#### **Code Range: D9110-D9130** - *Palliative and Emergency Treatments*
**Use when:** Patient presents with acute pain, swelling, infection, or trauma requiring immediate attention but not definitive treatment.
**Check:** Documentation clearly describes the nature of the emergency, specific palliative measures provided, and plan for definitive care.
**Note:** These codes are appropriate for temporary relief of symptoms or conditions, not for comprehensive treatment.
**Activation trigger:** Scenario mentions OR implies any emergency attention, pain relief, temporary measures, acute conditions, palliative care, or situations where immediate intervention is needed before definitive treatment. INCLUDE this range if there's any indication of urgency or temporary relief measures.

## **Anesthesia (D9210-D9248)**

### **Before picking a code, ask:**
- What level of anesthesia is being administered (local, regional, moderate sedation, deep sedation, general)?
- Is the anesthesia being provided in conjunction with another procedure or as a standalone service?
- What is the duration of anesthesia administration?
- What is the method of delivery (injection, inhalation, IV)?

#### **Code Range: D9210-D9248** - *Anesthesia and Sedation Services*
**Use when:** Providing pain control, anxiety management, or sedation for dental procedures.
**Check:** Documentation specifies type of anesthesia, method of administration, duration, and patient monitoring details.
**Note:** Different codes apply based on whether anesthesia is provided with or without an operative procedure.
**Activation trigger:** Scenario mentions OR implies any form of pain control, sedation, anesthesia, anxiety management, comfort measures during procedures, or patient fear/anxiety. INCLUDE this range if there's any hint that pain control or sedation might be needed or beneficial.

## **Professional Consultation (D9310-D9311)**

### **Before picking a code, ask:**
- Is a specialist or consultant providing a diagnostic service at the request of another dentist?
- Is this consultation separate from the treating dentist's services?
- Is the consultation being provided for a specific condition requiring specialized expertise?
- Will a written report be provided to the referring dentist?

#### **Code Range: D9310-D9311** - *Diagnostic Consultation Services*
**Use when:** A dentist other than the treating dentist provides evaluation services at the request of another dentist.
**Check:** Documentation includes the reason for consultation, findings, diagnosis, and recommendations provided to the referring provider.
**Note:** These codes are not used for interdepartmental consultations within the same practice.
**Activation trigger:** Scenario mentions OR implies any referral, second opinion, specialist involvement, evaluation by someone other than the primary dentist, or complex cases that might benefit from additional expertise. INCLUDE this range if there's any indication that more than one provider is or should be involved in diagnosis or treatment planning.

## **Professional Visits (D9410-D9450)**

### **Before picking a code, ask:**
- Is the dental treatment being provided outside the standard dental office setting?
- Does the patient require care in a hospital, long-term care facility, or at home?
- Is the visit occurring during non-regular business hours?
- Is there additional case presentation or treatment planning needed?

#### **Code Range: D9410-D9450** - *Professional Services Outside Office Setting*
**Use when:** Providing dental services in alternative settings or during non-standard hours.
**Check:** Documentation specifies the location of service, reason for alternative setting, travel time, and additional resources required.
**Note:** These codes account for the additional time, resources, and expertise needed for treatment in non-traditional settings.
**Activation trigger:** Scenario mentions OR implies any non-office setting, special circumstances regarding timing or location, patient mobility issues, institutional care, or situations requiring dentist travel. INCLUDE this range if there's any hint that care might be delivered outside standard office hours or locations.

## **Drugs (D9610-D9630)**

### **Before picking a code, ask:**
- Is a therapeutic drug being administered or dispensed as part of treatment?
- What is the route of administration (injection, topical, oral)?
- Is the medication being provided for home use or administered in-office?
- What is the purpose of the medication (antibiotics, analgesics, anti-inflammatories)?

#### **Code Range: D9610-D9630** - *Medication Administration and Dispensing*
**Use when:** Providing therapeutic drugs as part of dental treatment either in-office or for home use.
**Check:** Documentation includes drug name, dosage, route of administration, and medical necessity.
**Note:** These codes are used when medication is a significant component of treatment, not when medications are incidental to a procedure.
**Activation trigger:** Scenario mentions OR implies any medication use, infection concerns, pain management needs, inflammation, or conditions that typically require pharmacological intervention. INCLUDE this range if there's any indication that medications might be needed as part of the treatment approach.

## **Miscellaneous Services (D9910-D9973)**

### **Before picking a code, ask:**
- Does the service involve specialized application of materials for non-carious sensitivity?
- Is a custom appliance being provided for TMJ or bruxism?
- Does the patient require specialized behavior management?
- Is esthetic enhancement (bleaching, microabrasion) being performed?
- Is repair of a prosthetic or other appliance needed?

#### **Code Range: D9910-D9973** - *Specialized Adjunctive Services*
**Use when:** Providing specialized treatments that don't fit into other categories, such as desensitization, occlusal guards, behavior management, or esthetic procedures.
**Check:** Documentation clearly specifies the nature of the service, materials used, and justification for the procedure.
**Note:** These services often require additional expertise, materials, or time beyond standard dental procedures.
**Activation trigger:** Scenario mentions OR implies any sensitivity issues, TMJ problems, grinding/clenching, patient management challenges, cosmetic concerns, appliance issues, or any specialized service that doesn't clearly fit other categories. INCLUDE this range if there's any hint of these special circumstances or unique patient needs.

## **Non-clinical Procedures (D9961-D9999)**

### **Before picking a code, ask:**
- Is the service administrative rather than clinical in nature?
- Does it involve record duplication, reporting, or specialized documentation?
- Is translation or interpretation required for patient communication?
- Is teledentistry being utilized for patient care?
- Is this an unlisted procedure requiring a narrative description?

#### **Code Range: D9961-D9999** - *Administrative and Supporting Services*
**Use when:** Providing non-clinical services that support dental care but aren't treatment procedures themselves.
**Check:** Documentation explains why the service was necessary and how it facilitates patient care.
**Note:** These codes capture important administrative and support services that enable effective dental treatment.
**Activation trigger:** Scenario mentions OR implies any administrative service, language barriers, remote care delivery, special documentation needs, or unique circumstances not covered by standard clinical codes. INCLUDE this range if there's any indication of non-treatment services that support or facilitate care.

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
    
    def activate_adjunctive_general_services(self, scenario: str) -> dict:
        """Activate relevant subtopics and return detailed results."""
        try:
            # Get the code range from the analysis
            adjunctive_result = self.analyze_adjunctive_general_services(scenario)
            if not adjunctive_result:
                print("No adjunctive result returned")
                return {}
            
            print(f"Adjunctive Result in activate_adjunctive_general_services: {adjunctive_result}")
            
            # Process specific adjunctive subtopics based on the result
            specific_codes = []
            activated_subtopics = []
            
            # Check for each subtopic and activate if applicable
            subtopic_map = [
                ("D9110-D9130", unclassified_service.activate_unclassified_treatment, "Unclassified Treatment (D9110-D9130)"),
                ("D9210-D9248", anesthesia_service.activate_anesthesia, "Anesthesia (D9210-D9248)"),
                ("D9310-D9311", consultation_service.activate_professional_consultation, "Professional Consultation (D9310-D9311)"),
                ("D9410-D9450", visits_service.activate_professional_visits, "Professional Visits (D9410-D9450)"),
                ("D9610-D9630", drugs_service.activate_drugs, "Drugs (D9610-D9630)"),
                ("D9910-D9973", misc_service.activate_miscellaneous_services, "Miscellaneous Services (D9910-D9973)"),
                ("D9961-D9999", non_clinical_service.activate_non_clinical_procedures, "Non-clinical Procedures (D9961-D9999)")
            ]
            
            for code_range, activate_func, subtopic_name in subtopic_map:
                if code_range in adjunctive_result:
                    print(f"Activating subtopic: {subtopic_name}")
                    code = activate_func(scenario)
                    if code:
                        specific_codes.append(code)
                        activated_subtopics.append(subtopic_name)
            
            # Choose the primary subtopic (either the first activated or a default)
            primary_subtopic = activated_subtopics[0] if activated_subtopics else "Unclassified Treatment (D9110-D9130)"
            
            # Return a dictionary with the required fields
            return {
                "code_range": adjunctive_result,
                "subtopic": primary_subtopic,
                "activated_subtopics": activated_subtopics,
                "codes": specific_codes
            }
        except Exception as e:
            print(f"Error in adjunctive general services analysis: {str(e)}")
            return {}
    
    def run_analysis(self, scenario: str) -> None:
        """Run the analysis and print results."""
        print(f"Using model: {self.llm_service.model} with temperature: {self.llm_service.temperature}")
        result = self.activate_adjunctive_general_services(scenario)
        print(f"\n=== ADJUNCTIVE GENERAL SERVICES ANALYSIS RESULT ===")
        print(f"CODE RANGE: {result.get('code_range', 'None')}")
        print(f"PRIMARY SUBTOPIC: {result.get('subtopic', 'None')}")
        print(f"ACTIVATED SUBTOPICS: {', '.join(result.get('activated_subtopics', []))}")
        print(f"SPECIFIC CODES: {', '.join(result.get('codes', []))}")

# Example usage
if __name__ == "__main__":
    adj_service = AdjunctiveGeneralServices()
    scenario = input("Enter an adjunctive general services dental scenario: ")
    adj_service.run_analysis(scenario)