import pdfplumber
import pandas as pd
from pathlib import Path
from langchain_core.messages import HumanMessage
from agents.llm_factory import get_llm
from memory.vector_store import VectorStore
import json

LINKEDIN_EXTRACT_PROMPT = """You are extracting structured career information from a LinkedIn profile.

Analyze the text below and extract all relevant information. Return ONLY a JSON object.

Return this exact structure:
{{
    "full_name": "person's full name or null",
    "headline": "professional headline or null",
    "summary": "professional summary or null",
    "work_experience": [
        {{
            "company": "company name",
            "role": "job title", 
            "period": "start - end dates",
            "description": "job description"
        }}
    ],
    "education": [
        {{
            "institution": "school/university name",
            "degree": "degree or certificate",
            "period": "start - end dates"
        }}
    ],
    "skills": ["skill1", "skill2"],
    "projects": [
        {{
            "name": "project name",
            "description": "project description"
        }}
    ],
    "certifications": ["cert1", "cert2"],
    "languages": ["language1", "language2"]
}}

LinkedIn data:
{text}

Return ONLY the JSON, no other text."""


class LinkedInParser:
    def __init__(self, config: dict):
        self.llm = get_llm(config)
        self.store = VectorStore(config)

    def _read_csv(self, folder: str) -> str:
        """Reads all useful CSV files from LinkedIn export folder"""
        folder_path = Path(folder)
        text_parts = []

        # Profile.csv
        profile_path = folder_path / "Profile.csv"
        if profile_path.exists():
            df = pd.read_csv(profile_path)
            text_parts.append("=== PROFILE ===")
            text_parts.append(df.to_string(index=False))

        # Recommendations
        for rec_file in ["Recommendations_Given.csv", "Recommendations_Received.csv"]:
            rec_path = folder_path / rec_file
            if rec_path.exists():
                df = pd.read_csv(rec_path)
                text_parts.append(f"\n=== {rec_file.replace('.csv', '').upper()} ===")
                text_parts.append(df.to_string(index=False))

        return "\n".join(text_parts)

    def _read_pdf(self, folder: str) -> str:
        """Reads Profile.pdf from LinkedIn export folder"""
        pdf_path = Path(folder) / "Profile.pdf"
        if not pdf_path.exists():
            return ""

        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()

    def _parse_with_llm(self, text: str) -> dict:
        """Uses LLM to extract structured data"""
        prompt = LINKEDIN_EXTRACT_PROMPT.format(text=text[:6000])
        response = self.llm.invoke([HumanMessage(content=prompt)])

        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        return json.loads(raw.strip())

    def _save_to_store(self, data: dict) -> int:
        """Saves extracted data to Qdrant"""
        saved = 0

        if data.get("summary"):
            self.store.save(
                text=data["summary"],
                category="personality",
                metadata={"source": "linkedin", "type": "summary"}
            )
            saved += 1

        if data.get("headline"):
            self.store.save(
                text=data["headline"],
                category="other",
                metadata={"source": "linkedin", "type": "headline"}
            )
            saved += 1

        for exp in data.get("work_experience", []):
            text = f"{exp.get('role')} at {exp.get('company')} ({exp.get('period')}): {exp.get('description')}"
            self.store.save(
                text=text,
                category="work_experience",
                metadata={
                    "source": "linkedin",
                    "company": exp.get("company"),
                    "role": exp.get("role")
                }
            )
            saved += 1

        for edu in data.get("education", []):
            text = f"{edu.get('degree')} at {edu.get('institution')} ({edu.get('period')})"
            self.store.save(
                text=text,
                category="education",
                metadata={"source": "linkedin"}
            )
            saved += 1

        if data.get("skills"):
            self.store.save(
                text="Skills: " + ", ".join(data["skills"]),
                category="tech_skill",
                metadata={"source": "linkedin"}
            )
            saved += 1

        for proj in data.get("projects", []):
            self.store.save(
                text=f"Project: {proj.get('name')} — {proj.get('description')}",
                category="project",
                metadata={"source": "linkedin"}
            )
            saved += 1

        if data.get("certifications"):
            self.store.save(
                text="Certifications: " + ", ".join(data["certifications"]),
                category="other",
                metadata={"source": "linkedin", "type": "certifications"}
            )
            saved += 1

        return saved

    def parse(self, folder: str = "data/linkedin") -> dict:
        """Main method — reads CSV + PDF and saves to Qdrant"""
        print(f"\n📂 Reading LinkedIn data from: {folder}")

        # Read both sources
        csv_text = self._read_csv(folder)
        pdf_text = self._read_pdf(folder)

        if csv_text:
            print("✅ CSV files read")
        if pdf_text:
            print("✅ PDF file read")

        if not csv_text and not pdf_text:
            print("❌ No LinkedIn data found")
            return {}

        # Combine both sources
        combined = ""
        if csv_text:
            combined += "=== CSV DATA ===\n" + csv_text + "\n\n"
        if pdf_text:
            combined += "=== PDF DATA ===\n" + pdf_text

        # Parse with LLM
        print("🤖 Analyzing with LLM...")
        data = self._parse_with_llm(combined)

        # Save to Qdrant
        saved = self._save_to_store(data)
        print(f"💾 Saved {saved} items to profile")

        # Summary
        if data.get("full_name"):
            print(f"\n👤 Name: {data['full_name']}")
        if data.get("headline"):
            print(f"💼 Headline: {data['headline']}")
        if data.get("work_experience"):
            print(f"🏢 Work experience: {len(data['work_experience'])} positions")
        if data.get("education"):
            print(f"🎓 Education: {len(data['education'])} entries")
        if data.get("skills"):
            print(f"🛠️  Skills: {len(data['skills'])} skills")

        return data