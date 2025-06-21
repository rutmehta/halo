from pydantic import BaseModel, Field
from typing import List

class SimilarFace(BaseModel):
    image_url: str = Field(..., description="URL or path to the similar face image.")
    similarity_score: float = Field(..., description="A distance metric indicating similarity. Lower is more similar for L2 distance.")

class SearchResponse(BaseModel):
    top_matches: List[SimilarFace] = Field(..., description="A list of the top 5 most similar faces, ranked by similarity.")
    query_face_found: bool = Field(..., description="Indicates if a face was successfully detected in the input image.") 