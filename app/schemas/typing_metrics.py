# app/models/typing.py
from pydantic import BaseModel
from typing import Any

class TypingData(BaseModel):
    D1U1_mean: float
    D1U2_mean: float
    D1D2_mean: float
    U1D2_mean: float
    U1U2_mean: float
    D1U1_std: float
    D1U2_std: float
    D1D2_std: float
    U1D2_std: float
    U1U2_std: float
    editDistance: int
    nbKeystroke: int
    gender: bool
    ageRange: int
    degree: int
    pcTimeAverage: int
    status: int
    typeWith: int
    country_US: int
    country_UK: int
    typistType_touch: int
    typistType_hunt_and_peck: int
    text_input: str

class AnalysisResult(BaseModel):
    analysis: Any