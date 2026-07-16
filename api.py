# api_real.py

from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.api_client import fetch_customer_by_id
from src.rag_engine import NailRAGEngine
from src.utils import load_config


app = FastAPI(title="Nailify RAG API")

config = load_config()
engine = NailRAGEngine(config)


class CustomerProfile(BaseModel):
    skinTone: Optional[str] = "Neutral"
    skinShade: Optional[str] = "Medium"
    handShape: Optional[str] = "Average"
    occupation: Optional[str] = None
    nailCondition: Optional[str] = "Healthy"
    preferredComplexity: Optional[str] = None
    preferredColors: List[str] = Field(default_factory=list)
    preferredStyles: List[str] = Field(default_factory=list)
    preferredOccasions: List[str] = Field(default_factory=list)
    preferredNailShapeId: Optional[int] = None

class RecommendationResponse(BaseModel):
    status: str
    data: dict


def _available_products_summary():
    shapes_cache = engine._get_shapes_cache()
    surfaces_cache = engine._get_surfaces_cache()
    components_cache = engine._get_components_cache()
    
    return {
        "shapes": [
            {"id": shape.get("nailShapeId"), "name": shape.get("name")}
            for shape in shapes_cache
        ],
        "surfaces": [
            {
                "id": surface.get("nailSurfaceId"),
                "name": surface.get("name"),
                "price": surface.get("price"),
            }
            for surface in surfaces_cache
        ],
        "components": [
            {
                "id": component.get("componentId"),
                "name": component.get("name"),
                "type": component.get("componentType"),
            }
            for component in components_cache
        ],
    }


@app.post("/api/recommend", response_model=RecommendationResponse)
async def get_recommendation(profile: CustomerProfile):
    try:
        profile_data = profile.model_dump() if hasattr(profile, "model_dump") else profile.dict()
        result = engine.recommend(profile_data)
        return {"status": "success", "data": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/recommend/customer/{customer_id}", response_model=RecommendationResponse)
async def get_customer_recommendation(customer_id: str):
    try:
        customer = fetch_customer_by_id(customer_id)
        result = engine.recommend(customer)
        return {
            "status": "success",
            "data": result
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/products")
async def get_products():
    shapes_cache = engine._get_shapes_cache()
    surfaces_cache = engine._get_surfaces_cache()
    components_cache = engine._get_components_cache()
    
    return {
        "shapes": shapes_cache,
        "surfaces": surfaces_cache,
        "components": components_cache,
    }


@app.post("/api/products/refresh")
async def refresh_products():
    engine._load_api_data()
    shapes_cache = engine._get_shapes_cache()
    surfaces_cache = engine._get_surfaces_cache()
    components_cache = engine._get_components_cache()
    
    return {
        "status": "success",
        "components_count": len(components_cache),
        "shapes_count": len(shapes_cache),
        "surfaces_count": len(surfaces_cache),
    }


@app.get("/api/health")
async def health_check():
    components_cache = engine._get_components_cache()
    return {
        "status": "healthy",
        "model": config["ollama"]["model"],
        "components_count": len(components_cache),
        "shapes_count": len(engine.shapes_cache),
        "surfaces_count": len(engine.surfaces_cache),
        "api_cache_ttl_seconds": 600,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
