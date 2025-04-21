from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import os
from src.data_loader import generate_random_data
from src.som.som import SelfOrganisingMap
# from src.visualization import save_som_image

app = FastAPI(title="Self-Organizing Map API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class SOMRequest(BaseModel):
    width: int = 10
    height: int = 10
    input_dim: int = 3
    num_samples: int = 10
    num_iterations: int = 100

@app.post("/run-som")
def run_som(params: SOMRequest):
    # Generating input data
    data = generate_random_data(params.num_samples, params.input_dim)

    # Training SOM
    som = SelfOrganisingMap(params.width, params.height, params.input_dim)
    som.train(data, params.num_iterations)

    return {"message": "SOM generated successfully.", "weights": som.weights.tolist()}

# @app.get("/get-som-image")
# def get_som_image(file_name: str = Query(..., description="Filename to retrieve from the output directory")):
#     full_path = os.path.join(OUTPUT_DIR, file_name)
#     if not os.path.exists(full_path):
#         return {"error": "File not found."}
#     return FileResponse(full_path, media_type="image/png", filename=file_name)

## COmmand to run in terminal uvicorn src.main:app --reload --port 3000