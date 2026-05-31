from pydantic import BaseModel
from typing import Literal, Optional


class VisionProfile(BaseModel):
    age: Optional[int] = None
    gender: Optional[Literal["male", "female", "unknown"]] = "unknown"
    height_cm: Optional[float] = None
    shoulder_width_cm: Optional[float] = None
    body_type: Optional[Literal["thin", "medium", "fat", "unknown"]] = "unknown"
    upper_color: Optional[str] = "unknown"
    presence: bool = False