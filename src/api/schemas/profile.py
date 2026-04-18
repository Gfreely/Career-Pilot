from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class OfferStatusModel(BaseModel):
    received: List[str] = Field(default_factory=list)
    pending: List[str] = Field(default_factory=list)
    rejected: List[str] = Field(default_factory=list)


class EducationBackgroundItem(BaseModel):
    school: Optional[str] = None
    level: Optional[str] = None
    degree: Optional[str] = None
    major: Optional[str] = None


class ProjectExperienceItem(BaseModel):
    name: Optional[str] = None
    tech_stack: List[str] = Field(default_factory=list)
    description: Optional[str] = None


class InternshipExperienceItem(BaseModel):
    company: Optional[str] = None
    position: Optional[str] = None
    duration: Optional[str] = None


class UserProfilePayload(BaseModel):
    major: Optional[str] = None
    degree: Optional[str] = None
    graduation_year: Optional[str] = None
    target_cities: List[str] = Field(default_factory=list)
    tech_stack: List[str] = Field(default_factory=list)
    job_preferences: List[str] = Field(default_factory=list)
    offer_status: OfferStatusModel = Field(default_factory=OfferStatusModel)
    experience_level: Optional[str] = None
    concerns: List[str] = Field(default_factory=list)
    education_background: List[EducationBackgroundItem] = Field(default_factory=list)
    project_experience: List[ProjectExperienceItem] = Field(default_factory=list)
    internship_experience: List[InternshipExperienceItem] = Field(default_factory=list)


class ProfileBundleResponse(BaseModel):
    profile: Dict[str, Any]
    profile_text: str
    filter_metadata: Dict[str, Any]


class ProfileFileResponse(BaseModel):
    path: str
    format: str
    content: str


class ProfileFileUpdateRequest(BaseModel):
    content: str


class ProfileFileUpdateResponse(ProfileBundleResponse):
    file: ProfileFileResponse


class ProfileAnalysisRequest(BaseModel):
    model_name: str = "Qwen/Qwen3.5-397B-A17B"
    target_position: str
    target_city: str = ""
    target_direction: str = ""
    notes: str = ""
    resume_content: Optional[str] = None
    conversation_id: Optional[str] = None

class ResumeUploadResponse(BaseModel):
    markdown_content: str
    file_path: str

class ProfileAnalysisResponse(BaseModel):
    summary: str
    match_score: int
    strengths: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    action_plan: List[str] = Field(default_factory=list)
    suggested_roles: List[str] = Field(default_factory=list)
    interview_focus: List[str] = Field(default_factory=list)
