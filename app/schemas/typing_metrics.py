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
    delFreq: int   # Changed from float to int to match JSON data
    leftFreq: int  # Changed from float to int to match JSON data
    gender: bool   # Boolean based on the JSON data (True/False)
    ageRange: int  # Integer based on the JSON data
    degree: int    # Integer based on the JSON data
    pcTimeAverage: int  # Changed from float to int to match JSON data
    status: int    # Integer based on the JSON data
    typeWith: int  # Integer based on the JSON data

    # Country fields as integers based on the JSON data
    country_Angola: bool
    country_Arabie_Saoudite: bool
    country_Bahamas: bool
    country_Barbade: bool
    country_Belgique: bool
    country_Belize: bool
    country_Canada: bool
    country_France: bool
    country_Germany: bool
    country_Italy: bool
    country_Portugal: bool
    country_Spain: bool
    country_Suisse: bool
    country_Tunisia: bool
    country_United_States: bool
    country_United_Arab_Emirates: bool

    # TypistType fields as integers based on the JSON data
    typistType_One_Finger_Typist: bool
    typistType_Two_Finger_Typist: bool
    typistType_Touch_Typist: bool
    text_input:str

class AnalysisResult(BaseModel):
    analysis: Any
