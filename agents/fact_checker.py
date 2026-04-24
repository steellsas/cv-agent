from agents.llm_factory import get_llm
from memory.vector_store import VectorStore, ProfileBlock
from langchain_core.messages import HumanMessage
from qdrant_client.models import Filter, FieldCondition, MatchValue
import json

CATEGORIES = [
    "work_experience",
    "project",
    "education",
    "tech_skill",
    "soft_skill",
    "personality",
    "other"
]

CATEGORY_LABELS = {
    "work_experience": "💼 Work Experience",
    "project":         "🚀 Projects",
    "education":       "🎓 Education",
    "tech_skill":      "🛠️  Technical Skills",
    "soft_skill":      "🤝 Soft Skills",
    "personality":     "👤 Personality",
    "other":           "📌 Other"
}

CONSOLIDATE_PROMPT = """You are consolidating duplicate or overlapping career profile entries.

Below are multiple entries from the same category collected from different sources.
Merge them into clean, non-duplicate blocks. Keep all unique information.
Each block should be a clear, concise English statement.

Entries:
{entries}

Return ONLY a JSON array of merged blocks:
[
    {{
        "text": "clean merged statement",
        "confidence": "high/medium/low",
        "sources": ["linkedin", "github", "conversation"]
    }}
]
Return ONLY JSON."""


class FactChecker:
    def __init__(self, config: dict):
        self.llm = get_llm(config)
        self.store = VectorStore(config)

    def _get_source_icon(self, sources: list) -> str:
        icons = []
        if "linkedin" in sources:
            icons.append("LinkedIn")
        if "github" in sources:
            icons.append("GitHub")
        if "conversation" in sources:
            icons.append("HR chat")
        if "old_cv" in sources:
            icons.append("Old CV")
        return " + ".join(icons) if icons else "unknown"

    def _consolidate_category(self, items: list) -> list:
        """Uses LLM to merge duplicate entries in a category"""
        if len(items) <= 1:
            return items

        entries_text = "\n".join([
            f"[{i+1}] (source: {', '.join(item['source'])}) {item['text']}"
            for i, item in enumerate(items)
        ])

        prompt = CONSOLIDATE_PROMPT.format(entries=entries_text)
        response = self.llm.invoke([HumanMessage(content=prompt)])

        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        try:
            merged = json.loads(raw.strip())
            # Convert back to expected format
            return [
                {
                    "text": m["text"],
                    "source": m.get("sources", []),
                    "confidence": m.get("confidence", "medium"),
                    "user_confirmed": False,
                    "category": items[0]["category"]
                }
                for m in merged
            ]
        except:
            return items

    def _display_category(self, label: str, items: list):
        """Displays a category block nicely"""
        print(f"\n{label}")
        print("─" * 50)
        for i, item in enumerate(items, 1):
            source = self._get_source_icon(item["source"])
            confidence = item.get("confidence", "medium")
            confirmed = "✅" if item.get("user_confirmed") else "○"
            print(f"  {confirmed} [{i}] {item['text'][:80]}")
            print(f"       Source: {source} | Confidence: {confidence}")

    def _review_category(self, category: str, items: list) -> list:
        """Lets user review and confirm/edit a category block"""
        label = CATEGORY_LABELS.get(category, category)
        self._display_category(label, items)

        if not items:
            print("  (no data)")
            add = input("\n  Add information? (yes/no): ").strip().lower()
            if add in ["yes", "y"]:
                new_text = input("  Enter information: ").strip()
                if new_text:
                    self.store.save_block(ProfileBlock(
                        category=category,
                        text=new_text,
                        source=["conversation"],
                        confidence="high",
                        metadata={"type": "user_added"},
                        user_confirmed=True
                    ))
                    print("  ✅ Saved!")
            return items

        print(f"\n  Options:")
        print("  'ok'       — confirm all entries")
        print("  'edit N'   — edit entry number N")
        print("  'delete N' — delete entry number N")
        print("  'add'      — add new entry")
        print("  'skip'     — skip this category")

        while True:
            choice = input("\n  Your choice: ").strip().lower()

            if choice == "ok":
                # Mark all as confirmed
                self._confirm_all(category)
                print(f"  ✅ {label} confirmed!")
                return items

            elif choice == "skip":
                return items

            elif choice == "add":
                new_text = input("  Enter new information: ").strip()
                if new_text:
                    self.store.save_block(ProfileBlock(
                        category=category,
                        text=new_text,
                        source=["conversation"],
                        confidence="high",
                        metadata={"type": "user_added"},
                        user_confirmed=True
                    ))
                    print("  ✅ Added!")
                    items = self.store.get_all(category=category)
                    self._display_category(label, items)

            elif choice.startswith("edit "):
                try:
                    idx = int(choice.split()[1]) - 1
                    if 0 <= idx < len(items):
                        print(f"  Current: {items[idx]['text']}")
                        new_text = input("  New text: ").strip()
                        if new_text:
                            self.store.save_block(ProfileBlock(
                                category=category,
                                text=new_text,
                                source=items[idx]["source"],
                                confidence="high",
                                metadata={"type": "user_edited"},
                                user_confirmed=True
                            ))
                            print("  ✅ Updated!")
                            items = self.store.get_all(category=category)
                            self._display_category(label, items)
                    else:
                        print("  ❌ Invalid number")
                except:
                    print("  ❌ Invalid command")

            elif choice.startswith("delete "):
                try:
                    idx = int(choice.split()[1]) - 1
                    if 0 <= idx < len(items):
                        confirm = input(f"  Delete: '{items[idx]['text'][:50]}'? (yes/no): ").strip().lower()
                        if confirm in ["yes", "y"]:
                            print("  ✅ Marked for removal")
                            items.pop(idx)
                            self._display_category(label, items)
                    else:
                        print("  ❌ Invalid number")
                except:
                    print("  ❌ Invalid command")
            else:
                print("  ❓ Unknown command")

    def _confirm_all(self, category: str):
        """Marks all entries in a category as user_confirmed"""
        items = self.store.get_all(category=category)
        for item in items:
            # Re-save with user_confirmed = True
            self.store.save_block(ProfileBlock(
                category=category,
                text=item["text"],
                source=item["source"],
                confidence=item["confidence"],
                metadata=item.get("metadata", {}),
                user_confirmed=True
            ))

    def run(self):
        """Main fact-checking flow — review all categories"""
        print("\n" + "="*50)
        print("🔍 FACT-CHECKER — Profile Review")
        print("="*50)
        print("Review and confirm your profile data block by block.")
        print("This ensures your CV is based on verified information.\n")

        # Show overall summary first
        summary = self.store.get_profile_summary()
        total = sum(d["count"] for d in summary.values())

        if total == 0:
            print("❌ No profile data found. Please import LinkedIn, GitHub or run HR conversation first.")
            input("\nPress Enter to continue...")
            return

        print(f"📊 Total entries found: {total}")
        for cat, data in summary.items():
            if data["count"] > 0:
                print(f"  {CATEGORY_LABELS.get(cat, cat)}: {data['count']} entries")

        print("\n" + "─"*50)
        proceed = input("\nReview each category? (yes/no): ").strip().lower()
        if proceed not in ["yes", "y"]:
            return

        # Review each category
        for category in CATEGORIES:
            items = self.store.get_all(category=category)

            # Consolidate duplicates first
            if len(items) > 2:
                print(f"\n🔄 Consolidating duplicates in {category}...")
                items = self._consolidate_category(items)

            self._review_category(category, items)

        # Final summary
        print("\n" + "="*50)
        print("✅ PROFILE REVIEW COMPLETE")
        print("="*50)

        summary = self.store.get_profile_summary()
        print("\n📊 Final profile status:")
        for cat, data in summary.items():
            if data["count"] > 0:
                confirmed = data["confirmed"]
                total_cat = data["count"]
                bar = "█" * confirmed + "░" * (total_cat - confirmed)
                print(f"  {CATEGORY_LABELS.get(cat, cat):<25} {bar} {confirmed}/{total_cat} confirmed")

        input("\nPress Enter to continue...")