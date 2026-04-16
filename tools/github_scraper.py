import os
import requests
from agents.llm_factory import get_llm
from memory.vector_store import VectorStore
from langchain_core.messages import HumanMessage

GITHUB_EXTRACT_PROMPT = """You are extracting career-relevant information from a GitHub repository.

Analyze the repository data below and extract useful professional information.
Return ONLY a JSON object.

Return this exact structure:
{{
    "description": "what this project does in 2-3 sentences",
    "tech_stack": ["technology1", "technology2"],
    "complexity": "simple / medium / complex",
    "highlights": "most impressive aspects of this project or null"
}}

Repository data:
{repo_data}

Return ONLY the JSON, no other text."""


class GitHubScraper:
    def __init__(self, config: dict):
        self.llm = get_llm(config)
        self.store = VectorStore(config)
        self.token = os.getenv("GITHUB_TOKEN")
        self.username = os.getenv("GITHUB_USERNAME")
        self.headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json"
        }

    def _get_repos(self) -> list:
        """Fetches all public repositories"""
        url = f"https://api.github.com/users/{self.username}/repos"
        response = requests.get(url, headers=self.headers, params={"type": "public", "per_page": 50})

        if response.status_code != 200:
            print(f"❌ GitHub API error: {response.status_code}")
            return []

        return response.json()

    def _get_readme(self, repo_name: str) -> str:
        """Fetches README content for a repository"""
        url = f"https://api.github.com/repos/{self.username}/{repo_name}/readme"
        response = requests.get(url, headers=self.headers)

        if response.status_code != 200:
            return ""

        import base64
        content = response.json().get("content", "")
        try:
            return base64.b64decode(content).decode("utf-8")[:2000]
        except:
            return ""

    def _get_languages(self, repo_name: str) -> dict:
        """Fetches programming languages used in repository"""
        url = f"https://api.github.com/repos/{self.username}/{repo_name}/languages"
        response = requests.get(url, headers=self.headers)

        if response.status_code != 200:
            return {}

        return response.json()

    def _analyze_repo(self, repo: dict) -> dict:
        """Uses LLM to analyze repository and extract career info"""
        readme = self._get_readme(repo["name"])
        languages = self._get_languages(repo["name"])

        repo_data = f"""
Name: {repo['name']}
Description: {repo.get('description', 'No description')}
Stars: {repo.get('stargazers_count', 0)}
Languages: {', '.join(languages.keys()) if languages else 'Unknown'}
Topics: {', '.join(repo.get('topics', []))}
README:
{readme[:1500] if readme else 'No README'}
"""

        prompt = GITHUB_EXTRACT_PROMPT.format(repo_data=repo_data)
        response = self.llm.invoke([HumanMessage(content=prompt)])

        import json
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        try:
            return json.loads(raw.strip())
        except:
            return {}

    def _save_repo(self, repo: dict, analysis: dict):
        """Saves repository info to Qdrant"""
        # Save project info
        project_text = (
            f"Project: {repo['name']} — {analysis.get('description', repo.get('description', ''))}"
            f" Tech stack: {', '.join(analysis.get('tech_stack', []))}"
        )
        self.store.save(
            text=project_text,
            category="project",
            metadata={
                "source": "github",
                "name": repo["name"],
                "url": repo.get("html_url", ""),
                "complexity": analysis.get("complexity", "unknown"),
                "stars": repo.get("stargazers_count", 0)
            }
        )

        # Save tech stack separately
        if analysis.get("tech_stack"):
            tech_text = f"GitHub project {repo['name']} uses: {', '.join(analysis['tech_stack'])}"
            self.store.save(
                text=tech_text,
                category="tech_skill",
                metadata={
                    "source": "github",
                    "repo": repo["name"]
                }
            )

        # Save highlights if present
        if analysis.get("highlights"):
            self.store.save(
                text=f"Project highlight — {repo['name']}: {analysis['highlights']}",
                category="other",
                metadata={
                    "source": "github",
                    "type": "highlight",
                    "repo": repo["name"]
                }
            )

    def scrape(self) -> list:
        """Main method — scrapes all public repos and saves to Qdrant"""
        print(f"\n🐙 Fetching GitHub repos for: {self.username}")

        repos = self._get_repos()
        if not repos:
            print("❌ No repositories found")
            return []

        print(f"✅ Found {len(repos)} public repositories")
        results = []

        for repo in repos:
            print(f"\n  📦 Analyzing: {repo['name']}...")
            analysis = self._analyze_repo(repo)
            self._save_repo(repo, analysis)
            results.append({
                "name": repo["name"],
                "analysis": analysis
            })
            print(f"  ✅ {repo['name']} — {analysis.get('complexity', '?')} complexity")

        print(f"\n💾 Saved {len(repos)} repositories to profile")
        return results