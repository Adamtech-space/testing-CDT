from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import asyncio
from typing import Dict, Any, List

# Import the data cleaner and cdt classifier
from data_cleaner import DentalScenarioProcessor
from cdt_classifier import CDTClassifier
from sub_topic_registry import SubtopicRegistry

# Import topic functions
from topics.diagnostics import diagnostic_service
from topics.preventive import preventive_service
from topics.restorative import restorative_service
from topics.endodontics import endodontic_service
from topics.periodontics import periodontic_service
from topics.prosthodonticsremovable import prosthodontics_service
from topics.maxillofacialprosthetics import maxillofacial_service
from topics.implantservices import implant_service
from topics.prosthodonticsfixed import prosthodontics_service
from topics.oralandmaxillofacialsurgery import oral_surgery_service
from topics.orthodontics import orthodontic_service
from topics.adjunctivegeneralservices import adjunctive_general_services_service

# Initialize FastAPI app
app = FastAPI(title="Dental Code Extractor API")

# Initialize data cleaner and cdt classifier
cleaner = DentalScenarioProcessor()
cdt_classifier = CDTClassifier()

# Create TopicRegistry for parallel activation
topic_registry = SubtopicRegistry()

# Map CDT code ranges to topic functions
CDT_TOPIC_MAPPING = {
    "D0100-D0999": {"func": diagnostic_service.activate_diagnostic, "name": "Diagnostic"},
    "D1000-D1999": {"func": preventive_service.activate_preventive, "name": "Preventive"},
    "D2000-D2999": {"func": restorative_service.activate_restorative, "name": "Restorative"},
    "D3000-D3999": {"func": endodontic_service.activate_endodontic, "name": "Endodontics"},
    "D4000-D4999": {"func": periodontic_service.activate_periodontic, "name": "Periodontics"},
    "D5000-D5899": {"func": prosthodontics_service.activate_prosthodontics_fixed, "name": "Prosthodontics Removable"},
    "D5900-D5999": {"func": maxillofacial_service.activate_maxillofacial_prosthetics, "name": "Maxillofacial Prosthetics"},
    "D6000-D6199": {"func": implant_service.activate_implant_services, "name": "Implant Services"},
    "D6200-D6999": {"func": prosthodontics_service.activate_prosthodontics_fixed, "name": "Prosthodontics Fixed"},
    "D7000-D7999": {"func": oral_surgery_service.activate_oral_maxillofacial_surgery, "name": "Oral and Maxillofacial Surgery"},
    "D8000-D8999": {"func": orthodontic_service.activate_orthodontic, "name": "Orthodontics"},
    "D9000-D9999": {"func": adjunctive_general_services_service.activate_adjunctive_general_services, "name": "Adjunctive General Services"}
}

# Register all topics with the registry
for code_range, topic_info in CDT_TOPIC_MAPPING.items():
    topic_registry.register(code_range, topic_info["func"], topic_info["name"])

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://dentalcoder.vercel.app", 
        "https://automed.adamtechnologies.in", 
        "http://automed.adamtechnologies.in",
        os.getenv("FRONTEND_URL", "")  # Get from environment variable
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper function to map specific CDT codes to their broader category
def map_to_cdt_category(specific_code: str) -> str:
    """Maps a specific CDT code or range to its broader category."""
    # Extract the base code without the range
    if "-" in specific_code:
        base_code = specific_code.split("-")[0]
    else:
        base_code = specific_code
    
    # Get the first 2 characters
    category_prefix = base_code[:2]
    
    # Map to the corresponding category
    if category_prefix == "D0":
        return "D0100-D0999"
    elif category_prefix == "D1":
        return "D1000-D1999"
    elif category_prefix == "D2":
        return "D2000-D2999"
    elif category_prefix == "D3":
        return "D3000-D3999"
    elif category_prefix == "D4":
        return "D4000-D4999"
    elif category_prefix == "D5":
        # Check if it's in the range D5000-D5899 or D5900-D5999
        if base_code[2:4] < "90":
            return "D5000-D5899"
        else:
            return "D5900-D5999"
    elif category_prefix == "D6":
        # Check if it's in the range D6000-D6199 or D6200-D6999
        if base_code[2:4] < "20":
            return "D6000-D6199"
        else:
            return "D6200-D6999"
    elif category_prefix == "D7":
        return "D7000-D7999"
    elif category_prefix == "D8":
        return "D8000-D8999"
    elif category_prefix == "D9":
        return "D9000-D9999"
    else:
        return None

# Request model
class ScenarioRequest(BaseModel):
    scenario: str

# API endpoint to process user input
@app.post("/api/analyze")
async def analyze_web(request: ScenarioRequest):
    """Process the dental scenario through the data cleaner, CDT classifier, and topic activators."""
    try:
        # Step 1: Process the input through data_cleaner
        print(f"🔍 INPUT SCENARIO: {request.scenario}")
        processed_result = cleaner.process(request.scenario)
        processed_scenario = processed_result["standardized_scenario"]
        print(f"✅ PROCESSED SCENARIO: {processed_scenario}")
        
        # Step 2: Process the cleaned scenario with CDT classifier
        print(f"⏳ RUNNING CDT CLASSIFICATION...")
        cdt_result = cdt_classifier.process(processed_scenario)
        print(f"🏆 CDT CLASSIFICATION COMPLETE with {len(cdt_result.get('formatted_results', []))} code ranges")
        
        # Step 3: Activate topics in parallel based on code ranges
        print(f"⚡ ACTIVATING TOPICS IN PARALLEL...")
        range_codes = cdt_result["range_codes_string"].split(",")
        
        # Process code ranges to get the standardized categories
        category_ranges = set()
        for range_code in range_codes:
            category = map_to_cdt_category(range_code.strip())
            if category:
                category_ranges.add(category)
        
        # Convert to comma-separated string for the registry
        category_ranges_str = ",".join(category_ranges)
        
        # Run all relevant topics in parallel
        topic_results = await topic_registry.activate_all(processed_scenario, category_ranges_str)
        
        print(f"🎯 TOPICS ACTIVATED: {topic_results.get('activated_subtopics', [])}")
        print(f"📋 SPECIFIC CODES IDENTIFIED: {len(topic_results.get('specific_codes', []))}")
        
        # Format the topics data for response
        topics_data = {}
        for code_range in category_ranges:
            if code_range in CDT_TOPIC_MAPPING:
                topic_name = CDT_TOPIC_MAPPING[code_range]["name"]
                topics_data[code_range] = {
                    "name": topic_name,
                    "activated": topic_name in topic_results.get('activated_subtopics', [])
                }
        
        return {
            "status": "success",
            "data": {
                "processed_scenario": processed_scenario,
                "cdt_classification": cdt_result,
                "topics_results": topic_results,
                "topics_data": topics_data
            }
        }
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/")
def test():
    return {"message": "Dental Code Extractor API is running"}

# Run the application
if __name__ == "__main__":
    import uvicorn
        # Get port and host from environment variables with fallbacks
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
        
    print(f"🚀 STARTING SERVER on {host}:{port}")
    uvicorn.run("app:app", host=host, port=port, reload=True)