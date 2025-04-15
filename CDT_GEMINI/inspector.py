import os
import logging
from dotenv import load_dotenv
from llm_services import generate_response, get_service, set_model, set_temperature
from typing import Dict, Any, Optional
from llm_services import DEFAULT_MODEL, DEFAULT_TEMP

# Load environment variables
load_dotenv()

# Set a specific model for this file (optional)
# set_model_for_file("gemini-2.0-flash-thinking-exp-01-21")

class DentalInspector:
    """Class to handle dental code inspection with configurable prompts and settings"""
    
    PROMPT_TEMPLATE = """
You are the final code selector ("Inspector") with extensive expertise in dental coding. Your task is to perform a thorough analysis of the provided scenario along with the candidate CDT code outputs—including all explanations and doubts—from previous subtopics. Your final output must include only the CDT code(s) that are justified by the scenario, with minimal assumptions.

Scenario:
{scenario}

Topic Analysis Results:
{topic_analysis}

Additional Information from Questions (if any):
{questioner_data}

Instructions:

1) Carefully read the complete clinical scenario provided.

2) Review all candidate CDT codes suggested by previous subtopics along with their explanations and any doubts raised.

3) Use Provided Codes: Only consider and select from the candidate CDT codes that were actually analyzed in the topic analysis results. Do not reject codes that weren't part of the original analysis.

4) Reasonable Assumptions: You may make basic clinical assumptions that are standard in dental practice, but avoid making significant assumptions about unstated procedures.

5) Justification: Select codes that are reasonably supported by the scenario. If a code has minor doubts but is likely correct based on the context, include it.

6) Mutually Exclusive Codes: When presented with mutually exclusive codes, choose the one that is best justified for the specific visit. Do not bill for the same procedure twice.

7) Revenue & Defensibility: Your selection should maximize revenue while ensuring billing is defensible, but don't reject codes unnecessarily.

8) Consider Additional QA: Incorporate any additional information or clarifications provided through the question-answer process.

9) Output the same code multiple times if it is applicable (e.g., 8 scans would include the code 8 times).

10) Coding Rules:
    - Consider standard bundling rules but don't be overly strict
    - Assume medical necessity for standard procedures
    - Consider emergency/post-op status if implied by context
    - Only reject codes that were explicitly analyzed in the topic analysis

IMPORTANT: You must format your response exactly as follows:

EXPLANATION: [provide a detailed explanation for why each code was selected or rejected. Include specific reasoning for each code mentioned in the topic analysis.]

CODES: [comma-separated list of CDT codes that are accepted, with no square brackets around individual codes]

REJECTED CODES: [comma-separated list of CDT codes that were considered but rejected, with no square brackets around individual codes. Only list codes that were actually analyzed in the topic analysis and are explicitly contradicted by the scenario.]

For example:

EXPLANATION: D0120 (periodic oral evaluation) is appropriate as this was a regular dental visit. D0274 (bitewings-four radiographs) is included because the scenario mentions taking four bitewing x-rays. D1110 (prophylaxis-adult) is included as the scenario describes cleaning of teeth for an adult patient. D0140 was rejected because this was not an emergency visit.
CODES: D0120, D0274, D1110
REJECTED CODES: D0140,D0220,D0230
"""

    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMP):
        """Initialize the inspector with model and temperature settings"""
        self.service = get_service()
        self.configure(model, temperature)
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Configure logging for the inspector module"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        return logging.getLogger(__name__)

    def configure(self, model: Optional[str] = None, temperature: Optional[float] = None) -> None:
        """Configure model and temperature settings"""
        if model:
            set_model(model)
        if temperature is not None:
            set_temperature(temperature)

    def format_prompt(self, scenario: str, topic_analysis: Any, questioner_data: Any = None) -> str:
        """Format the prompt template with the given inputs"""
        topic_analysis_str = self._format_topic_analysis(topic_analysis)
        questioner_data_str = self._format_questioner_data(questioner_data)
        
        return self.PROMPT_TEMPLATE.format(
            scenario=scenario,
            topic_analysis=topic_analysis_str,
            questioner_data=questioner_data_str
        )

    def _format_topic_analysis(self, topic_analysis: Any) -> str:
        """Format topic analysis data"""
        if topic_analysis is None:
            return "No CDT data analysis data available in DB"
        
        if isinstance(topic_analysis, dict):
            formatted_topics = []
            for code_range, topic_data in topic_analysis.items():
                topic_name = topic_data.get("name", "Unknown")
                topic_result = topic_data.get("result", "No result")
                formatted_topics.append(f"{topic_name} ({code_range}): {topic_result}")
            return "\n".join(formatted_topics)
        
        return str(topic_analysis)

    def _format_questioner_data(self, questioner_data: Any) -> str:
        """Format questioner data"""
        if questioner_data is None:
            return "No additional information provided."
        
        if isinstance(questioner_data, dict):
            qa_pairs = []
            if questioner_data.get("has_questions", False) and questioner_data.get("has_answers", False):
                answers = questioner_data.get("answers", {})
                
                for q_type in ["cdt_questions", "icd_questions"]:
                    questions = questioner_data.get(q_type, {}).get("questions", [])
                    prefix = q_type.split("_")[0].upper()
                    for question in questions:
                        answer = answers.get(question, "No answer provided")
                        qa_pairs.append(f"{prefix} Q: {question}\nA: {answer}")
                
                return "\n".join(qa_pairs) if qa_pairs else "Questions were asked but no answers were provided."
            elif questioner_data.get("has_questions", False):
                return "Questions were identified but not yet answered."
            return "No additional questions were needed."
        
        return str(questioner_data)

    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into structured format"""
        codes_line = ""
        explanation_line = ""
        rejected_codes_line = ""
        
        lines = response.strip().split('\n')
        in_explanation = False
        
        for line in lines:
            line = line.strip()
            
            if line.upper().startswith("CODES:"):
                codes_line = line.split(":", 1)[1].strip()
            elif line.upper().startswith("REJECTED CODES:"):
                rejected_codes_line = line.split(":", 1)[1].strip()
            elif line.upper().startswith("EXPLANATION:"):
                in_explanation = True
                explanation_line = line.split(":", 1)[1].strip()
            elif in_explanation and line:
                explanation_line += " " + line

        cleaned_codes = self._clean_codes(codes_line)
        rejected_codes = self._clean_codes(rejected_codes_line)
        
        return {
            "codes": cleaned_codes,
            "rejected_codes": rejected_codes,
            "explanation": explanation_line
        }

    def _clean_codes(self, codes_line: str) -> list:
        """Clean and format codes from response"""
        cleaned = []
        if codes_line:
            codes_line = codes_line.strip('[]')
            for code in codes_line.split(','):
                clean_code = code.strip().strip('[]')
                if clean_code and clean_code.lower() != 'none' and '*' not in clean_code:
                    cleaned.append(clean_code)
        return cleaned

    def process(self, scenario: str, topic_analysis: Any = None, questioner_data: Any = None) -> Dict[str, Any]:
        """Process a dental scenario and return inspection results"""
        try:
            formatted_prompt = self.format_prompt(scenario, topic_analysis, questioner_data)
            response = generate_response(formatted_prompt)
            result = self.parse_response(response)
            
            self.logger.info(f"Dental analysis completed for scenario")
            self.logger.info(f"Extracted codes: {result['codes']}")
            self.logger.info(f"Extracted rejected codes: {result['rejected_codes']}")
            self.logger.info(f"Extracted explanation: {result['explanation']}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in process: {str(e)}")
            return {
                "error": str(e),
                "codes": [],
                "rejected_codes": [],
                "explanation": f"Error occurred: {str(e)}",
                "type": "error",
                "data_source": "error"
            }

    @property
    def current_settings(self) -> Dict[str, Any]:
        """Get current model settings"""
        return {
            "model": self.service.model,
            "temperature": self.service.temperature
        }

class InspectorCLI:
    """Command Line Interface for the DentalInspector"""
    
    def __init__(self):
        self.inspector = DentalInspector()

    def print_settings(self):
        """Print current model settings"""
        settings = self.inspector.current_settings
        print(f"Using model: {settings['model']} with temperature: {settings['temperature']}")

    def print_results(self, result: Dict[str, Any]):
        """Print inspection results in a formatted way"""
        print("\n=== INSPECTION RESULTS ===")
        print("\nAccepted Codes:")
        for code in result["codes"]:
            print(f"- {code}")
        
        print("\nRejected Codes:")
        for code in result["rejected_codes"]:
            print(f"- {code}")
        
        print("\nExplanation:")
        print(result["explanation"])

    def run(self):
        """Run the CLI interface"""
        self.print_settings()
        scenario = input("Enter dental scenario: ")
        topic_analysis = input("Enter topic analysis (or press Enter to skip): ") or None
        questioner_data = input("Enter questioner data (or press Enter to skip): ") or None
        
        result = self.inspector.process(scenario, topic_analysis, questioner_data)
        self.print_results(result)

def main():
    """Main entry point for the script"""
    cli = InspectorCLI()
    cli.run()

if __name__ == "__main__":
    main()





