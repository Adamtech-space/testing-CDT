from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

# Import the data cleaner and cdt classifier
from data_cleaner import DentalScenarioProcessor
from cdt_classifier import CDTClassifier

# Initialize FastAPI app
app = FastAPI(title="Dental Code Extractor API")

# Initialize data cleaner and cdt classifier
cleaner = DentalScenarioProcessor()
cdt_classifier = CDTClassifier()

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

# Request model
class ScenarioRequest(BaseModel):
    scenario: str

# API endpoint to process user input
@app.post("/api/analyze")
async def analyze_web(request: ScenarioRequest):
    """Process the dental scenario through the data cleaner and CDT classifier."""
    try:
        # Step 1: Process the input through data_cleaner
        print(f"Input scenario: {request.scenario}")
        processed_result = cleaner.process(request.scenario)
        processed_scenario = processed_result["standardized_scenario"]
        print(f"Processed scenario: {processed_scenario}")
        
        # Step 2: Process the cleaned scenario with CDT classifier
        print(f"Running CDT classification...")
        cdt_result = cdt_classifier.process(processed_scenario)
        print(f"CDT classification complete with {len(cdt_result.get('formatted_results', []))} code ranges")
        
        return {
            "status": "success",
            "data": {
                "processed_scenario": processed_scenario,
                "cdt_classification": cdt_result
            }
        }
    except Exception as e:
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
    
    print(f"Starting server on {host}:{port}")
    uvicorn.run("app:app", host=host, port=port, reload=True)