from pydantic import BaseModel, Field
from typing import Optional
from langchain_core.messages import BaseMessage

class AgentState(BaseModel):
    messages: list[BaseMessage] = Field(default_factory=list)
    user_input: str = ""
    current_step: str = "start"
    profile_complete: bool = False
    job_posting: str = ""
    cv_sections: dict = Field(default_factory=dict)
    missing_info: list[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True  # būtina dėl BaseMessage tipo