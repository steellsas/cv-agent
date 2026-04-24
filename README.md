

# cv-agent
cv-agent/
│
├── main.py
├── config.yaml
│
├── agents/
│   ├── orchestrator.py      ← atnaujinamas
│   ├── llm_factory.py       ← lieka
│   ├── state.py             ← lieka
│   ├── profile_builder.py   ← NAUJAS (Pokalbis 1)
│   ├── cv_planner.py        ← NAUJAS (Pokalbis 2)
│   └── cv_agent.py          ← atnaujinamas
│
├── tools/
│   ├── cv_parser.py         ← NAUJAS (universalus PDF)
│   └── github_scraper.py    ← lieka
│
├── memory/
│   ├── profile_store.py     ← NAUJAS (JSON)
│   └── vector_store.py      ← atnaujinamas (tik semantika)
│
├── prompts/
│   ├── profile_prompts.py   ← atnaujinamas
│   ├── cv_prompts.py        ← atnaujinamas
│   └── interview_prompts.py ← lieka
│
└── tests/                   ← papildomi testai


1. profile_store.py   → JSON pagrindas
2. cv_parser.py       → PDF ištraukimas
3. profile_builder.py → Pokalbis 1
4. cv_planner.py      → Pokalbis 2
5. cv_agent.py        → atnaujinimas
6. orchestrator.py    → atnaujinimas
7. testai             → naujų komponentų testai