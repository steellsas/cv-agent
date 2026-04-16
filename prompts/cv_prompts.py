ANALYZE_JOB_PROMPT = """You are analyzing a job posting to extract key requirements.

Extract the following from the job posting below and return ONLY a JSON object:
{{
    "job_title": "position title",
    "company": "company name or null",
    "required_skills": ["skill1", "skill2"],
    "preferred_skills": ["skill1", "skill2"],
    "experience_level": "junior / mid / senior",
    "key_responsibilities": ["responsibility1", "responsibility2"],
    "soft_skills": ["skill1", "skill2"],
    "industry": "industry or domain"
}}
IMPORTANT: Always extract and save ALL information in English only, 
regardless of the input language.
Job posting:
{job_posting}

Return ONLY the JSON, no other text."""


GENERATE_SUMMARY_PROMPT = """You are a professional CV writer. Write a compelling personal summary for a CV.

Job requirements:
{job_requirements}

Candidate profile information:
{profile_info}

Guidelines:
- 3-4 sentences maximum
- Highlight most relevant experience and skills for THIS specific job
- Be specific, not generic
- Do not fabricate information — only use what is provided
- Write in {language}
- Tone: professional but natural

Return ONLY the summary text, no labels or extra text."""


GENERATE_EXPERIENCE_PROMPT = """You are a professional CV writer. Write work experience descriptions for a CV.

Job requirements:
{job_requirements}

Candidate work experience:
{experience_info}

Guidelines:
- Focus on achievements relevant to the job posting
- Use action verbs (Developed, Built, Implemented, Led, etc.)
- Be specific and quantify where possible
- Do not fabricate — only use provided information
- Write in {language}
- Format each position as JSON:

Return ONLY a JSON array:
[
    {{
        "company": "company name",
        "role": "job title",
        "period": "dates",
        "bullets": ["achievement 1", "achievement 2", "achievement 3"]
    }}
]"""


GENERATE_SKILLS_PROMPT = """You are a professional CV writer. Select and organize the most relevant skills for this CV.

Job requirements:
{job_requirements}

Candidate skills:
{skills_info}

Return ONLY a JSON object:
{{
    "technical_skills": ["most relevant tech skills for this job"],
    "tools": ["relevant tools and frameworks"],
    "soft_skills": ["relevant soft skills"]
}}"""


GENERATE_PROJECTS_PROMPT = """You are a professional CV writer. Select and describe the most relevant projects for this CV.

Job requirements:
{job_requirements}

Candidate projects:
{projects_info}

Guidelines:
- Select 2-3 most relevant projects
- Emphasize aspects relevant to the job
- Be specific about technologies and outcomes
- Write in {language}

Return ONLY a JSON array:
[
    {{
        "name": "project name",
        "description": "2-3 sentence description emphasizing relevance to job",
        "tech_stack": ["tech1", "tech2"],
        "highlights": "key achievement or metric"
    }}
IMPORTANT: Always extract and save ALL information in English only, 
regardless of the input language.
]"""