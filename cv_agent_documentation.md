# CV Agent — Project Documentation

> **Status:** Planning phase  
> **Version:** 0.1  
> **Language:** English  
> **Interface:** Terminal (initial), Web UI (future)

---

## 1. Project Goal

Build an AI-powered agent that creates **authentic, professional CVs** tailored to specific job postings, based on the user's real data — their history, experience, projects, and personal traits.

The program does **not fabricate experience** but emphasizes the most relevant aspects for each position. It collects as much information as possible about the user and selects the most suitable content for each job application.

---

## 2. Core Problems Solved

- Manual CV creation is time-consuming and hard to tailor per job
- Users often undersell themselves or miss important details
- A single generic CV doesn't work well across different positions
- Interview preparation is disconnected from the actual CV and job posting

---

## 3. Functional Overview

### 3.1 User Profile Building
- Agent conducts a warm, HR-style conversation to collect the user's story
- Collects data from multiple sources:
  - Conversational Q&A (HR-style questions)
  - LinkedIn PDF export (user uploads manually)
  - GitHub profile (via GitHub public API)
  - Old CVs (PDF upload, parsed automatically)
- Analyzes and cross-references information across sources
- Extracts: work experience, projects, education, tech skills, soft skills, personality traits
- Profile is **continuously updated** — new experience or skills can be added at any time
- Not all collected information is used in every CV — the agent selects what's most relevant

### 3.2 CV Generation
- User submits a job posting (text paste or photo/screenshot)
- Agent checks the user profile against the job requirements
- If information is missing → agent immediately asks the user
- Agent prepares draft text sections (summary, experience descriptions, soft skills)
- User reviews each section and can request changes
- Final CV is generated only when all sections are approved
- CV is exported as a **PDF document**
- CV is saved and can be reused or modified later

### 3.3 Language & Style
- CV language is selectable: **English (primary), Lithuanian, others**
- Tone and speaking style can be adjusted to match the user's personality
- The final CV should feel natural and authentic — not generic AI output

### 3.4 Interview Preparation *(later stage)*
- Separate module, launched after CV creation
- Generates likely interview questions based on the job posting and field
- Prepares suggested answers based on the user's own history and experience
- Uses the same user profile data already collected

---

## 4. System Architecture

### 4.1 Agent Structure

```
cv_agent/
│
├── main.py                    # Entry point (terminal interface)
│
├── agents/
│   ├── orchestrator.py        # Main coordinator agent
│   ├── profile_agent.py       # Collects and manages user profile
│   └── cv_agent.py            # Generates CV from job posting
│
├── modules/
│   ├── interview_prep.py      # Interview preparation (separate module)
│   └── pdf_export.py          # CV export to PDF
│
├── tools/
│   ├── linkedin_parser.py     # Parses LinkedIn PDF export
│   ├── github_scraper.py      # Fetches GitHub data via API
│   ├── cv_parser.py           # Parses old CV PDFs
│   └── job_parser.py          # Parses job postings (text or image/OCR)
│
├── memory/
│   ├── vector_store.py        # Qdrant vector DB — user profile storage
│   └── cv_store.py            # Saved CV management
│
├── prompts/
│   ├── profile_prompts.py     # HR conversation and profile collection prompts
│   ├── cv_prompts.py          # CV generation prompts
│   └── interview_prompts.py   # Interview preparation prompts
│
└── config.yaml                # LLM settings, language, behavior config
```

### 4.2 Agent Descriptions

| Agent | Role |
|---|---|
| **Orchestrator** | Coordinates all agents, manages conversation flow, decides what to do next |
| **Profile Agent** | Asks HR-style questions, parses uploaded files, stores info in vector DB |
| **CV Agent** | Reads job posting, retrieves relevant profile info, generates CV sections |
| **Interview Module** | Generates interview questions and suggested answers (separate, later stage) |

### 4.3 Data Flow

```
FIRST TIME SETUP:
User → Orchestrator → Profile Agent
                           ↓
               HR-style conversation
               LinkedIn PDF upload
               GitHub API fetch
               Old CV PDF upload
                           ↓
                     Qdrant Vector DB

EACH NEW CV:
Job Posting → Orchestrator → CV Agent
                                  ↓
                        Check Qdrant profile
                        Missing info? → Ask user
                                  ↓
                        Draft sections → User reviews
                        User requests changes → Agent revises
                                  ↓
                        Final CV → PDF → CV Store

LATER STAGE:
CV + Job Posting → Interview Module → Questions + Answers
```

### 4.4 UI Separation Principle

The UI is fully decoupled from business logic:

```
Terminal UI  ──┐
Web UI       ──┼──→ Orchestrator (logic unchanged)
Mobile UI    ──┘
```

Switching from terminal to web only requires changing the UI layer — all agent logic remains identical.

### 4.5 Agent Extensibility

All agents follow a common base interface, making it easy to add new agents:

```python
class BaseAgent:
    def run(self, input) -> output
    def get_tools(self) -> list
    def get_description(self) -> str
```

New agent = new class implementing `BaseAgent` + register with orchestrator.

---

## 5. Data Storage

### 5.1 Qdrant Vector DB — User Profile

Each piece of user information is stored as a separate chunk with metadata:

```
- Work experience     → { company, role, period, description, source }
- Projects            → { name, tech_stack, description, links, source }
- Education           → { institution, degree, period, source }
- Technical skills    → { skill, level, source }
- Soft skills         → { trait, examples, source }
- Personality summary → { summary, source }
- Source              → linkedin / github / old_cv / conversation
```

The agent performs **semantic search** in the vector DB to find the most relevant information for each job posting.

### 5.2 CV Storage

Saved CVs stored as structured JSON + rendered PDF:

```
cv_store/
├── cv_2024_01_software_engineer.json
├── cv_2024_01_software_engineer.pdf
├── cv_2024_03_data_scientist.json
└── cv_2024_03_data_scientist.pdf
```

### 5.3 Configuration

```yaml
# config.yaml
llm:
  provider: "openai"        # openai / ollama / anthropic
  model: "gpt-4o-mini"      # easily swappable
  temperature: 0.7

language:
  default: "en"             # en / lt

profile:
  vector_db_url: "http://qdrant:6333"
  collection_name: "user_profile"
```

---

## 6. Tech Stack

### 6.1 LLM & Agent Framework

| Tool | Purpose |
|---|---|
| **LangGraph** | Agent flow management, graph-based logic |
| **LangChain** | LLM abstraction, tool integration |
| **LangSmith** | Monitoring, debugging, step visualization |
| **LangGraph Studio** | Visual graph structure viewer (local, free) |
| **GPT-4o / GPT-4o-mini** | Primary LLM (OpenAI) |
| **Ollama** | Local LLM for experimentation and privacy |

### 6.2 Data Storage

| Tool | Purpose |
|---|---|
| **Qdrant** | Vector DB — user profile semantic storage |
| **JSON files** | Saved CVs and configuration |
| **config.yaml** | LLM and app settings |

### 6.3 Agent Tools

| Tool | Purpose |
|---|---|
| **GitHub REST API** | Fetch GitHub repos, projects, contributions |
| **PyMuPDF / pdfplumber** | Parse LinkedIn PDF export and old CVs |
| **Pytesseract / OpenAI Vision** | OCR for job posting screenshots |
| **ReportLab / WeasyPrint** | Export final CV to PDF |

### 6.4 Interface

| Tool | Purpose |
|---|---|
| **Python Terminal** | Initial UI — fast, simple |
| **Rich** | Enhanced terminal output (colors, tables, layout) |
| **Flask / FastAPI** | Future web UI (later stage) |

### 6.5 Infrastructure

| Tool | Purpose |
|---|---|
| **Docker Compose** | Containerizes the app + Qdrant together |
| **uv** | Python package and environment management |
| **python-dotenv** | API key management |
| **Pydantic** | Data validation between agents |

### 6.6 Monitoring

| Tool | Purpose |
|---|---|
| **LangSmith** | Traces every agent step, token usage, latency |
| **LangGraph Studio** | Visualizes graph structure and node connections |

---

## 7. Docker Compose Structure

```yaml
# docker-compose.yml
services:
  cv_agent:
    build: .
    volumes:
      - ./:/app
    env_file: .env
    depends_on:
      - qdrant

  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  qdrant_data:
```

---

## 8. LLM Switching

The LLM provider is configured in one place and applies to all agents:

```python
# llm_factory.py
def get_llm(config):
    if config.provider == "openai":
        return ChatOpenAI(model=config.model)
    elif config.provider == "ollama":
        return ChatOllama(model=config.model)
    elif config.provider == "anthropic":
        return ChatAnthropic(model=config.model)
```

Changing the LLM = change one line in `config.yaml`.

---

## 9. Development Roadmap

### Phase 1 — Core (Terminal)
- [ ] Project setup (uv, Docker, Qdrant)
- [ ] Basic LangGraph orchestrator
- [ ] Profile agent with HR conversation flow
- [ ] LinkedIn PDF parser
- [ ] GitHub API tool
- [ ] Old CV PDF parser
- [ ] Vector DB storage (Qdrant)
- [ ] CV agent with job posting analysis
- [ ] Draft section review loop with user
- [ ] PDF export

### Phase 2 — Polish
- [ ] Language selection (EN / LT)
- [ ] Style and tone adjustment
- [ ] CV versioning and storage
- [ ] LangSmith monitoring integration

### Phase 3 — Interview Prep Module
- [ ] Interview question generation
- [ ] Suggested answers from user profile

### Phase 4 — Web UI (Future)
- [ ] Flask / FastAPI backend
- [ ] Simple web interface
- [ ] Multi-user support

---

## 10. Key Design Principles

1. **Modularity** — every component is independently replaceable
2. **Extensibility** — new agents plug in via base class interface
3. **UI agnostic** — business logic never depends on the interface
4. **No hallucination** — CV content is always grounded in real user data
5. **Privacy first** — all data stored locally or in self-hosted containers
6. **Continuous profile** — user profile grows over time, never reset

---

*Documentation last updated: 2026-04-15*
