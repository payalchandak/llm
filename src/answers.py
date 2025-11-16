from pydantic import BaseModel, conint
from enum import Enum
from typing import Type, TypeVar, Generic

class PatientChoice(str, Enum):
    patient_1 = "Patient 1"
    patient_2 = "Patient 2"

class YesNo(str, Enum):
    yes = "Yes"
    no = "No"

class Intervention(str, Enum):
    yes = "Intervene"
    no = "Monitor"

AnswerType = TypeVar("AnswerType", bound=Enum)

class Answer(BaseModel, Generic[AnswerType]):
    reasoning: str
    decision: AnswerType
    reflection: str
    confidence: conint(ge=0, le=5)