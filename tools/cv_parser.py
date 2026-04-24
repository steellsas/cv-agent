import pdfplumber
from pathlib import Path
from langchain_core.messages import HumanMessage
from agents.llm_factory import get_llm
from memory.profile_store import (
    ProfileStore, Experience, Education,
    Project, Skills, PersonalInfo
)
from tools.deduplicator import Deduplicator
import json

CV_EXTRACT_PROMPT = """You are extracting structured career information from a CV/Resume PDF.
Text may have formatting artifacts — ignore them and focus on content.
All output must be in English regardless of input language.

Extract ALL information present in this text chunk.
Be thorough — incomplete data is better than missing data.

IMPORTANT EXTRACTION RULES:
- For each work experience: extract ALL achievements, projects done, skills used, impact and key phrases
- For education: extract ALL entries including online courses, bootcamps, certifications
- For skills: extract from ALL sections — summary, experience descriptions, projects
- Preserve authentic phrases in their original wording

CV Text:
{cv_text}

Return ONLY a JSON object:
{{
    "personal": {{
        "name": "full name or null",
        "email": "email or null",
        "phone": "phone or null",
        "location": "city, country or null",
        "linkedin": "linkedin url or null",
        "github": "github url or null",
        "headline": "professional headline or null"
    }},
    "summary": "full professional summary text or null",
    "experience": [
        {{
            "company": "company name",
            "role": "job title",
            "period": "dates",
            "location": "city or null",
            "description": "main responsibilities",
            "achievements": [
                "specific achievement with context",
                "another concrete achievement"
            ],
            "skills_used": ["skill1", "skill2"],
            "projects": [
                "specific project done in this role"
            ],
            "impact": "business impact or outcome",
            "key_phrases": [
                "authentic phrase from their own description"
            ]
        }}
    ],
    "education": [
        {{
            "institution": "name",
            "degree": "degree or course name",
            "period": "dates or null",
            "field": "field of study or null"
        }}
    ],
    "projects": [
        {{
            "name": "project name",
            "description": "what it does",
            "tech_stack": ["tech1", "tech2"],
            "outcome": "result or metric or null"
        }}
    ],
    "skills": {{
        "technical": ["skills extracted from ALL sections"],
        "familiar": ["mentioned but less prominent"],
        "soft": ["soft skills from summary and descriptions"],
        "tools": ["frameworks, platforms, tools"]
    }},
    "certifications": ["cert1"],
    "languages": ["language1"]
}}

Return ONLY JSON."""

SUMMARY_EXTRACT_PROMPT = """You are extracting career intelligence from a professional summary.
These are the candidate's own words — preserve authentic phrases exactly.

Summary text:
{summary}

Return ONLY a JSON object:
{{
    "tech_skills": {{
        "technical": ["explicitly mentioned tech skills"],
        "tools": ["frameworks, platforms, tools"],
        "familiar": ["mentioned but less prominent"]
    }},
    "personality": {{
        "traits": ["problem solver", "analytical"],
        "work_style": "how they describe their work approach in one sentence",
        "motivation": "what drives them",
        "value_proposition": "what makes them unique"
    }},
    "projects_mentioned": [
        {{
            "name": "project name",
            "description": "what it does",
            "tech_stack": ["tech1"],
            "outcome": "result or metric or null"
        }}
    ],
    "authentic_phrases": [
        "exact memorable phrase in their own words"
    ],
    "positioning": "how they position themselves professionally"
}}

Return ONLY JSON."""


class CVParser:
    def __init__(self, config: dict):
        self.llm = get_llm(config)
        self.store = ProfileStore()
        self.deduplicator = Deduplicator(config)

    def _extract_text(self, pdf_path: str) -> str:
        """Extracts text from CV PDF handling multi-column layouts"""
        text_parts = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                width = page.width
                height = page.height

                full_text = page.extract_text(layout=True) or ""

                try:
                    main_area = page.crop((width * 0.25, 0, width, height))
                    main_text = main_area.extract_text() or ""
                except:
                    main_text = ""

                best = main_text if (
                    len(main_text) > 100 and
                    len(main_text) > len(full_text) * 0.6
                ) else full_text

                text_parts.append(f"--- Page {page_num + 1} ---\n{best}")

        return "\n\n".join(text_parts)

    def _parse_chunk(self, chunk: str) -> dict:
        """Parses a single text chunk with LLM"""
        prompt = CV_EXTRACT_PROMPT.format(cv_text=chunk[:6000])
        response = self.llm.invoke([HumanMessage(content=prompt)])

        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        try:
            return json.loads(raw.strip())
        except Exception as e:
            print(f"  ⚠️ Chunk parse error: {e}")
            return {}

    def _analyze_summary(self, summary: str) -> dict:
        """Extracts extra intelligence from CV summary"""
        if not summary or len(summary) < 50:
            return {}

        prompt = SUMMARY_EXTRACT_PROMPT.format(summary=summary)
        response = self.llm.invoke([HumanMessage(content=prompt)])

        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        try:
            return json.loads(raw.strip())
        except:
            return {}

    def _merge_chunks(self, chunks: list[dict]) -> dict:
        """Merges data from multiple chunks into one profile"""
        merged = {
            "personal": {},
            "summary": "",
            "experience": [],
            "education": [],
            "projects": [],
            "skills": {
                "technical": [],
                "familiar": [],
                "soft": [],
                "tools": []
            },
            "certifications": [],
            "languages": []
        }

        for chunk in chunks:
            # Personal — take first non-empty
            if chunk.get("personal") and not merged["personal"].get("name"):
                merged["personal"] = chunk["personal"]

            # Summary — take longest
            if chunk.get("summary") and len(
                chunk["summary"]) > len(merged["summary"]):
                merged["summary"] = chunk["summary"]

            # Experience — merge avoiding duplicates
            for exp in chunk.get("experience", []) or []:
                if not exp.get("company"):
                    continue
                existing = [
                    e for e in merged["experience"]
                    if e.get("company", "").lower() == exp.get("company", "").lower()
                ]
                if not existing:
                    merged["experience"].append(exp)
                else:
                    # Merge achievements and key_phrases
                    idx = merged["experience"].index(existing[0])
                    for ach in exp.get("achievements", []) or []:
                        if ach and ach not in merged["experience"][idx].get("achievements", []):
                            merged["experience"][idx].setdefault("achievements", []).append(ach)
                    for phrase in exp.get("key_phrases", []) or []:
                        if phrase and phrase not in merged["experience"][idx].get("key_phrases", []):
                            merged["experience"][idx].setdefault("key_phrases", []).append(phrase)
                    for proj in exp.get("projects", []) or []:
                        if proj and proj not in merged["experience"][idx].get("projects", []):
                            merged["experience"][idx].setdefault("projects", []).append(proj)
                    if exp.get("impact") and not merged["experience"][idx].get("impact"):
                        merged["experience"][idx]["impact"] = exp["impact"]

            # Education — merge avoiding duplicates
            for edu in chunk.get("education", []) or []:
                if not edu.get("institution"):
                    continue
                existing = [
                    e for e in merged["education"]
                    if e.get("institution", "").lower() == edu.get("institution", "").lower()
                ]
                if not existing:
                    merged["education"].append(edu)

            # Projects — merge avoiding duplicates
            for proj in chunk.get("projects", []) or []:
                if not proj.get("name"):
                    continue
                existing = [
                    p for p in merged["projects"]
                    if p.get("name", "").lower() == proj.get("name", "").lower()
                ]
                if not existing:
                    merged["projects"].append(proj)

            # Skills — merge all unique
            skills = chunk.get("skills") or {}
            for skill_type in ["technical", "familiar", "soft", "tools"]:
                for skill in skills.get(skill_type, []) or []:
                    if skill and skill.lower() not in [
                        s.lower() for s in merged["skills"][skill_type]
                    ]:
                        merged["skills"][skill_type].append(skill)

            # Certifications
            for cert in chunk.get("certifications", []) or []:
                if cert and cert not in merged["certifications"]:
                    merged["certifications"].append(cert)

            # Languages
            for lang in chunk.get("languages", []) or []:
                if lang and lang not in merged["languages"]:
                    merged["languages"].append(lang)

        return merged

    def _enrich_with_summary(self, data: dict, insights: dict) -> dict:
        """Enriches profile data with summary analysis"""
        if not insights:
            return data

        # Add tech skills from summary
        tech_skills = insights.get("tech_skills", {})
        for skill_type in ["technical", "tools", "familiar"]:
            for skill in tech_skills.get(skill_type, []) or []:
                if skill and skill.lower() not in [
                    s.lower() for s in data["skills"].get(skill_type, [])
                ]:
                    data["skills"].setdefault(skill_type, []).append(skill)

        # Add projects mentioned in summary
        for proj in insights.get("projects_mentioned", []) or []:
            if not proj.get("name"):
                continue
            existing = [
                p for p in data["projects"]
                if p.get("name", "").lower() == proj.get("name", "").lower()
            ]
            if not existing:
                data["projects"].append(proj)

        # Add personality data
        personality = insights.get("personality", {})
        if personality:
            data["personality"] = personality

        # Add authentic phrases
        data["authentic_phrases"] = insights.get("authentic_phrases", [])
        data["positioning"] = insights.get("positioning", "")

        return data

    def _save_to_store(self, data: dict) -> int:
        """Saves extracted CV data to ProfileStore"""
        saved = 0

        # Personal info
        if data.get("personal"):
            p = data["personal"]
            self.store.profile.personal = PersonalInfo(
                name=p.get("name") or "",
                email=p.get("email") or "",
                phone=p.get("phone") or "",
                location=p.get("location") or "",
                linkedin=p.get("linkedin") or "",
                github=p.get("github") or "",
                headline=p.get("headline") or ""
            )
            saved += 1

        # Summary
        if data.get("summary"):
            self.store.profile.summary = data["summary"] or ""
            saved += 1

        # Work experience — with new fields
        for exp in data.get("experience", []) or []:
            if exp.get("company") and exp.get("role"):
                self.store.add_experience(Experience(
                    company=exp.get("company") or "",
                    role=exp.get("role") or "",
                    period=exp.get("period") or "",
                    location=exp.get("location") or "",
                    description=exp.get("description") or "",
                    achievements=[a for a in (exp.get("achievements") or []) if a],
                    skills_used=[s for s in (exp.get("skills_used") or []) if s],
                    projects=[p for p in (exp.get("projects") or []) if p],
                    impact=exp.get("impact") or "",
                    key_phrases=[p for p in (exp.get("key_phrases") or []) if p]
                ))
                saved += 1

        # Education
        for edu in data.get("education", []) or []:
            if edu.get("institution"):
                self.store.add_education(Education(
                    institution=edu.get("institution") or "",
                    degree=edu.get("degree") or "",
                    period=edu.get("period") or "",
                    field=edu.get("field") or ""
                ))
                saved += 1

        # Projects
        for proj in data.get("projects", []) or []:
            if proj.get("name"):
                self.store.add_project(Project(
                    name=proj.get("name") or "",
                    description=proj.get("description") or "",
                    tech_stack=[t for t in (proj.get("tech_stack") or []) if t],
                    outcome=proj.get("outcome") or "",
                    source="cv"
                ))
                saved += 1

        # Skills
        skills = data.get("skills") or {}
        if skills:
            self.store.add_skills(
                technical=[s for s in (skills.get("technical") or []) if s],
                familiar=[s for s in (skills.get("familiar") or []) if s],
                soft=[s for s in (skills.get("soft") or []) if s],
                tools=[s for s in (skills.get("tools") or []) if s]
            )
            saved += 1

        # Certifications
        for cert in data.get("certifications", []) or []:
            if cert and cert not in self.store.profile.certifications:
                self.store.profile.certifications.append(cert)
                saved += 1

        # Personality from summary analysis
        if data.get("personality"):
            pers = data["personality"]
            self.store.update_personality(
                traits=pers.get("traits", []),
                work_style=pers.get("work_style", ""),
                motivation=pers.get("motivation", ""),
                phrases=data.get("authentic_phrases", [])
            )
            saved += 1

        # Positioning as authentic phrase
        if data.get("positioning"):
            self.store.update_personality(
                phrases=[data["positioning"]]
            )

        return saved

    def _display_extracted(self, data: dict):
        """Shows what was extracted"""
        print("\n📋 Extracted from CV:")

        if data.get("personal", {}).get("name"):
            print(f"  👤 Name: {data['personal']['name']}")
        if data.get("personal", {}).get("headline"):
            print(f"  💼 {data['personal']['headline']}")

        exp_list = data.get("experience", [])
        if exp_list:
            print(f"\n  💼 Work experience: {len(exp_list)} positions")
            for exp in exp_list:
                print(f"     • {exp.get('role')} @ {exp.get('company')} ({exp.get('period')})")
                if exp.get("achievements"):
                    print(f"       ↳ {len(exp['achievements'])} achievements")
                if exp.get("impact"):
                    print(f"       ↳ Impact: {exp['impact'][:60]}")

        edu_list = data.get("education", [])
        if edu_list:
            print(f"\n  🎓 Education: {len(edu_list)} entries")
            for edu in edu_list:
                print(f"     • {edu.get('degree')} — {edu.get('institution')}")

        proj_list = data.get("projects", [])
        if proj_list:
            print(f"\n  🚀 Projects: {len(proj_list)} found")
            for proj in proj_list:
                print(f"     • {proj.get('name')}: {proj.get('description', '')[:60]}")

        skills = data.get("skills", {})
        if skills.get("technical"):
            print(f"\n  🛠️  Technical: {', '.join(skills['technical'][:8])}")
        if skills.get("soft"):
            print(f"  🤝 Soft: {', '.join(skills['soft'][:5])}")

        if data.get("authentic_phrases"):
            print(f"\n  💬 Authentic phrases: {len(data['authentic_phrases'])} captured")

        if data.get("personality"):
            pers = data["personality"]
            if pers.get("work_style"):
                print(f"  👤 Work style: {pers['work_style'][:80]}")

    def parse(self, pdf_path: str) -> bool:
        """Main method — parses CV PDF and saves to ProfileStore"""
        path = Path(pdf_path)
        if not path.exists():
            print(f"❌ File not found: {pdf_path}")
            return False

        print(f"\n📄 Reading CV: {path.name}")
        print("🔍 Extracting text...")
        text = self._extract_text(str(path))

        if not text.strip():
            print("❌ Could not extract text from PDF")
            return False

        print(f"✅ Extracted {len(text)} characters")

        # Split into chunks
        pages = text.split("--- Page")
        pages = [p for p in pages if p.strip()]
        chunk_size = 3
        chunks = []
        for i in range(0, len(pages), chunk_size):
            chunk = "--- Page" + "--- Page".join(pages[i:i + chunk_size])
            chunks.append(chunk)

        print(f"🤖 Analyzing {len(pages)} pages in {len(chunks)} chunks...")

        all_data = []
        for i, chunk in enumerate(chunks):
            print(f"  🔍 Chunk {i + 1}/{len(chunks)}...")
            data = self._parse_chunk(chunk)
            if data:
                all_data.append(data)

        if not all_data:
            print("❌ Could not parse CV data")
            return False

        # Merge chunks
        print("🔄 Merging chunks...")
        merged = self._merge_chunks(all_data)

        # Analyze summary for extra insights
        if merged.get("summary"):
            print("🧠 Analyzing summary for extra insights...")
            insights = self._analyze_summary(merged["summary"])
            if insights:
                merged = self._enrich_with_summary(merged, insights)
                print(f"  ✅ Found {len(insights.get('authentic_phrases', []))} authentic phrases")
                if insights.get("personality", {}).get("work_style"):
                    print(f"  ✅ Work style captured")

        # Deduplicate
        print("🧹 Cleaning duplicates...")
        merged = self.deduplicator.clean_profile(merged)

        # Display
        self._display_extracted(merged)

        # Save
        saved = self._save_to_store(merged)
        self.store.add_source("cv_pdf")
        self.store.save()

        print(f"\n✅ Saved {saved} items to profile")
        return True