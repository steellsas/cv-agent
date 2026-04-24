import json
import os
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional


# ── Data Models ──

class PersonalInfo(BaseModel):
    name: str = ""
    location: str = ""
    email: str = ""
    phone: str = ""
    linkedin: str = ""
    github: str = ""
    headline: str = ""


class Experience(BaseModel):
    company: str = ""
    role: str = ""
    period: str = ""
    location: str = ""
    description: str = ""
    achievements: list[str] = Field(default_factory=list)
    skills_used: list[str] = Field(default_factory=list)
    # ← NAUJI LAUKAI:
    projects: list[str] = Field(default_factory=list)
    impact: str = ""
    key_phrases: list[str] = Field(default_factory=list)


class Education(BaseModel):
    institution: str = ""
    degree: str = ""
    period: str = ""      # ← jau turi default ""
    field: str = ""

class Project(BaseModel):
    name: str = ""
    description: str = ""
    tech_stack: list[str] = Field(default_factory=list)
    outcome: str = ""
    url: str = ""
    source: str = ""  # github / cv / conversation


class Skills(BaseModel):
    technical: list[str] = Field(default_factory=list)
    familiar: list[str] = Field(default_factory=list)
    soft: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)


class Personality(BaseModel):
    traits: list[str] = Field(default_factory=list)
    work_style: str = ""
    motivation: str = ""
    authentic_phrases: list[str] = Field(default_factory=list)


class UserProfile(BaseModel):
    personal: PersonalInfo = Field(default_factory=PersonalInfo)
    summary: str = ""
    experience: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    skills: Skills = Field(default_factory=Skills)
    certifications: list[str] = Field(default_factory=list)
    personality: Personality = Field(default_factory=Personality)
    languages: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)  # cv_pdf / github / conversation


class ProfileStore:
    def __init__(self, profile_path: str = "data/profile.json"):
        self.profile_path = Path(profile_path)
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        self.profile = self._load()

    def _load(self) -> UserProfile:
        """Loads profile from JSON file"""
        if self.profile_path.exists():
            with open(self.profile_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"✅ Profile loaded from {self.profile_path}")
            return UserProfile(**data)
        return UserProfile()

    def save(self):
        """Saves current profile to JSON file"""
        with open(self.profile_path, "w", encoding="utf-8") as f:
            json.dump(self.profile.model_dump(), f, indent=2, ensure_ascii=False)
        print(f"💾 Profile saved to {self.profile_path}")

    def is_empty(self) -> bool:
        """Checks if profile has any data"""
        return (
            not self.profile.personal.name and
            not self.profile.experience and
            not self.profile.education
        )

    def has_source(self, source: str) -> bool:
        """Checks if a source has been imported"""
        return source in self.profile.sources

    def add_source(self, source: str):
        """Marks a source as imported"""
        if source not in self.profile.sources:
            self.profile.sources.append(source)

    # ── Experience ──

    def add_experience(self, exp: Experience):
        """Adds work experience — avoids duplicates by company+role"""
        existing = [
            e for e in self.profile.experience
            if e.company.lower() == exp.company.lower()
            and e.role.lower() == exp.role.lower()
        ]
        if not existing:
            self.profile.experience.append(exp)
        else:
            # Merge achievements if duplicate
            idx = self.profile.experience.index(existing[0])
            for ach in exp.achievements:
                if ach not in self.profile.experience[idx].achievements:
                    self.profile.experience[idx].achievements.append(ach)

    def update_experience(self, company: str, role: str, updates: dict):
        """Updates specific experience entry"""
        for exp in self.profile.experience:
            if exp.company.lower() == company.lower():
                for key, value in updates.items():
                    if hasattr(exp, key):
                        setattr(exp, key, value)
                break

    # ── Projects ──

    def add_project(self, project: Project):
        """Adds project — avoids duplicates by name"""
        existing = [p for p in self.profile.projects
                    if p.name.lower() == project.name.lower()]
        if not existing:
            self.profile.projects.append(project)
        else:
            # Update if GitHub version has more info
            if project.source == "github" and existing[0].source != "github":
                idx = self.profile.projects.index(existing[0])
                self.profile.projects[idx] = project

    # ── Skills ──

    def add_skills(self, technical: list = None, familiar: list = None,
                   soft: list = None, tools: list = None):
        """Adds skills — avoids duplicates"""
        def merge(existing: list, new: list) -> list:
            if not new:
                return existing
            combined = existing.copy()
            for s in new:
                if s.lower() not in [e.lower() for e in combined]:
                    combined.append(s)
            return combined

        if technical:
            self.profile.skills.technical = merge(
                self.profile.skills.technical, technical)
        if familiar:
            self.profile.skills.familiar = merge(
                self.profile.skills.familiar, familiar)
        if soft:
            self.profile.skills.soft = merge(
                self.profile.skills.soft, soft)
        if tools:
            self.profile.skills.tools = merge(
                self.profile.skills.tools, tools)

    # ── Education ──

    def add_education(self, edu: Education):
        """Adds education — avoids duplicates"""
        existing = [e for e in self.profile.education
                    if e.institution.lower() == edu.institution.lower()]
        if not existing:
            self.profile.education.append(edu)

    # ── Personality ──

    def update_personality(self, traits: list = None, work_style: str = None,
                           motivation: str = None, phrases: list = None):
        """Updates personality section"""
        if traits:
            for t in traits:
                if t not in self.profile.personality.traits:
                    self.profile.personality.traits.append(t)
        if work_style:
            self.profile.personality.work_style = work_style
        if motivation:
            self.profile.personality.motivation = motivation
        if phrases:
            for p in phrases:
                if p not in self.profile.personality.authentic_phrases:
                    self.profile.personality.authentic_phrases.append(p)

    # ── Profile Summary ──

    def get_completeness(self) -> dict:
        """Returns profile completeness overview"""
        p = self.profile
        return {
            "personal":      {"complete": bool(p.personal.name), "count": 1 if p.personal.name else 0},
            "experience":    {"complete": len(p.experience) > 0, "count": len(p.experience)},
            "education":     {"complete": len(p.education) > 0, "count": len(p.education)},
            "projects":      {"complete": len(p.projects) > 0, "count": len(p.projects)},
            "skills":        {"complete": len(p.skills.technical) > 0, "count": len(p.skills.technical)},
            "personality":   {"complete": bool(p.personality.work_style), "count": len(p.personality.traits)},
            "certifications": {"complete": len(p.certifications) > 0, "count": len(p.certifications)},
        }

    def display_summary(self):
        """Displays profile summary in terminal"""
        print("\n" + "="*50)
        print("👤 PROFILE SUMMARY")
        print("="*50)

        p = self.profile
        if p.personal.name:
            print(f"\n  Name:     {p.personal.name}")
        if p.personal.headline:
            print(f"  Headline: {p.personal.headline}")
        if p.summary:
            print(f"\n  Summary: {p.summary[:120]}...")

        completeness = self.get_completeness()
        print("\n📊 Completeness:")
        for section, data in completeness.items():
            icon = "✅" if data["complete"] else "❌"
            print(f"  {icon} {section:<15} ({data['count']} entries)")

        if p.sources:
            print(f"\n📥 Sources: {', '.join(p.sources)}")

    def get_for_llm(self, sections: list = None) -> dict:
        """Returns only requested sections for LLM prompt"""
        p = self.profile
        all_sections = {
            "personal":      p.personal.model_dump(),
            "summary":       p.summary,
            "experience":    [e.model_dump() for e in p.experience],
            "education":     [e.model_dump() for e in p.education],
            "projects":      [e.model_dump() for e in p.projects],
            "skills":        p.skills.model_dump(),
            "certifications": p.certifications,
            "personality":   p.personality.model_dump(),
        }
        if sections:
            return {k: v for k, v in all_sections.items() if k in sections}
        return all_sections