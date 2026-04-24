from langchain_core.messages import HumanMessage
from agents.llm_factory import get_llm
import json

# ── Deduplication rules per category ──
CATEGORY_RULES = {
    "education": "Same institution (even if abbreviated differently e.g. KTU = Kaunas University of Technology) AND similar degree or course",
    "experience": "Same company AND overlapping time period",
    "projects": "Same project name OR very similar description (same purpose)",
    "tech_skill": "Same skill with different naming (e.g. JS = JavaScript, ML = Machine Learning)",
    "certification": "Same certification with different formatting or provider name",
    "soft_skill": "Same trait with different wording (e.g. 'communication' = 'communicative')",
    "other": "Clearly identical information"
}

DEDUP_PROMPT = """You are a deduplication specialist for career profile data.
Check if Entry A and Entry B are duplicates.

Category: {category}
Duplication rule for this category: {rule}

Entry A:
{entry_a}

Entry B:
{entry_b}

Return ONLY a JSON object:
{{
    "is_duplicate": true or false,
    "confidence": "high / medium / low",
    "reason": "one sentence explanation",
    "keep": "A or B — which entry has more complete information"
}}

Return ONLY JSON."""

BATCH_DEDUP_PROMPT = """You are a deduplication specialist for career profile data.
Find all duplicates in this list of {category} entries.

Duplication rule: {rule}

Entries:
{entries}

Return ONLY a JSON object:
{{
    "groups": [
        {{
            "keep_index": 0,
            "duplicate_indices": [1, 2],
            "confidence": "high / medium / low",
            "reason": "why these are duplicates"
        }}
    ],
    "unique_indices": [3, 4]
}}

If no duplicates found return:
{{"groups": [], "unique_indices": [0, 1, 2, ...]}}

Return ONLY JSON."""


class Deduplicator:
    def __init__(self, config: dict):
        self.llm = get_llm(config)

    def _call_llm(self, prompt: str) -> dict:
        """Calls LLM and parses JSON response"""
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

    def _format_entry(self, entry) -> str:
        """Formats entry for LLM prompt"""
        if isinstance(entry, dict):
            return json.dumps(entry, indent=2)
        elif hasattr(entry, "model_dump"):
            return json.dumps(entry.model_dump(), indent=2)
        return str(entry)

    def _auto_deduplicate(self, items: list, category: str) -> dict:
        """Uses LLM to find duplicates in batch"""
        if len(items) <= 1:
            return {"kept": items, "removed": [], "uncertain": []}

        rule = CATEGORY_RULES.get(category, CATEGORY_RULES["other"])
        entries_text = "\n\n".join([
            f"[{i}] {self._format_entry(item)}"
            for i, item in enumerate(items)
        ])

        prompt = BATCH_DEDUP_PROMPT.format(
            category=category,
            rule=rule,
            entries=entries_text
        )

        result = self._call_llm(prompt)
        if not result:
            return {"kept": items, "removed": [], "uncertain": []}

        kept = []
        removed = []
        uncertain = []
        processed = set()

        # Process duplicate groups
        for group in result.get("groups", []):
            confidence = group.get("confidence", "low")
            keep_idx = group.get("keep_index", 0)
            dup_indices = group.get("duplicate_indices", [])
            reason = group.get("reason", "")

            if keep_idx < len(items):
                kept.append(items[keep_idx])
                processed.add(keep_idx)

            for idx in dup_indices:
                if idx < len(items):
                    processed.add(idx)
                    if confidence == "high":
                        # Auto remove
                        removed.append({
                            "item": items[idx],
                            "reason": reason,
                            "kept_instead": items[keep_idx]
                        })
                    else:
                        # Ask user
                        uncertain.append({
                            "item": items[idx],
                            "possible_duplicate_of": items[keep_idx],
                            "reason": reason,
                            "confidence": confidence
                        })

        # Add unique items
        for idx in result.get("unique_indices", []):
            if idx < len(items) and idx not in processed:
                kept.append(items[idx])
                processed.add(idx)

        # Safety — add any unprocessed items
        for idx, item in enumerate(items):
            if idx not in processed:
                kept.append(item)

        return {"kept": kept, "removed": removed, "uncertain": uncertain}

    def _resolve_uncertain(self, uncertain: list, category: str) -> tuple[list, list]:
        """Asks user about uncertain duplicates"""
        confirmed_kept = []
        confirmed_removed = []

        for item in uncertain:
            print(f"\n⚠️  Possible duplicate found in [{category}]:")
            print(f"\n  Entry A (keeping):")
            print(f"  {self._format_entry(item['possible_duplicate_of'])[:150]}")
            print(f"\n  Entry B (possible duplicate):")
            print(f"  {self._format_entry(item['item'])[:150]}")
            print(f"\n  Reason: {item['reason']}")
            print(f"  Confidence: {item['confidence']}")

            choice = input("\n  Is this a duplicate? (yes/no/keep both): ").strip().lower()

            if choice in ["yes", "y"]:
                confirmed_removed.append(item["item"])
                print("  ✅ Removed duplicate")
            elif choice in ["no", "n"]:
                confirmed_kept.append(item["item"])
                print("  ✅ Kept both entries")
            else:
                confirmed_kept.append(item["item"])
                print("  ✅ Kept both entries")

        return confirmed_kept, confirmed_removed

    def clean(self, items: list, category: str, silent: bool = False) -> dict:
        """
        Main method — cleans duplicates from a list of items.

        Args:
            items: list of dicts or Pydantic models
            category: education / experience / projects / tech_skill / etc.
            silent: if True, auto-removes all duplicates without asking user

        Returns:
            {
                "kept": [...],      # unique items to keep
                "removed": [...],   # auto-removed duplicates
                "uncertain": [...]  # items user was asked about
            }
        """
        if len(items) <= 1:
            return {"kept": items, "removed": [], "uncertain": []}

        print(f"\n🔍 Checking {len(items)} {category} entries for duplicates...")

        # Auto deduplication
        result = self._auto_deduplicate(items, category)

        auto_removed = result["removed"]
        uncertain = result["uncertain"]

        if auto_removed:
            print(f"  ✅ Auto-removed {len(auto_removed)} obvious duplicate(s)")
            for r in auto_removed:
                reason = r.get("reason", "")
                print(f"     • {reason}")

        # Resolve uncertain with user
        user_kept = []
        user_removed = []

        if uncertain and not silent:
            print(f"\n  ❓ Found {len(uncertain)} uncertain case(s) — your input needed:")
            user_kept, user_removed = self._resolve_uncertain(uncertain, category)
        elif uncertain and silent:
            # In silent mode keep uncertain items
            user_kept = [u["item"] for u in uncertain]

        final_kept = result["kept"] + user_kept
        final_removed = auto_removed + [{"item": i} for i in user_removed]

        print(f"  📊 Result: {len(final_kept)} unique, {len(final_removed)} removed")

        return {
            "kept": final_kept,
            "removed": final_removed,
            "uncertain": uncertain
        }

    def clean_profile(self, profile_data: dict, silent: bool = False) -> dict:
        """
        Cleans entire profile data dict.
        Useful after CV parsing or merging multiple sources.

        Args:
            profile_data: dict with keys: experience, education, projects, etc.
            silent: auto-remove without asking

        Returns:
            cleaned profile_data dict
        """
        category_map = {
            "experience": "experience",
            "education": "education",
            "projects": "projects",
            "certifications": "certification",
            "languages": "other"
        }

        cleaned = profile_data.copy()

        for field, category in category_map.items():
            items = profile_data.get(field, [])
            if items and len(items) > 1:
                result = self.clean(items, category, silent=silent)
                cleaned[field] = result["kept"]

        # Clean skills separately
        skills = profile_data.get("skills", {})
        for skill_type, category in [
            ("technical", "tech_skill"),
            ("soft", "soft_skill"),
            ("tools", "tech_skill"),
            ("familiar", "tech_skill")
        ]:
            skill_items = skills.get(skill_type, [])
            if skill_items and len(skill_items) > 1:
                # Convert to dict for dedup
                skill_dicts = [{"name": s} for s in skill_items]
                result = self.clean(skill_dicts, category, silent=silent)
                cleaned.setdefault("skills", {})[skill_type] = [
                    item["name"] for item in result["kept"]
                ]

        return cleaned