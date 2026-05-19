from pydantic import BaseModel
from typing import List

class InputData(BaseModel):
    data: List[List[float]]