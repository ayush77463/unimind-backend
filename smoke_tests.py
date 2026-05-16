import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ["UNIMIND_DISABLE_GEMINI"] = "1"
_MODULE_TEMP_DIR = tempfile.TemporaryDirectory()
os.environ["UNIMIND_STORAGE_DIR"] = _MODULE_TEMP_DIR.name

from fastapi.testclient import TestClient

from unimind_memory.api.memory import get_memory_manager, set_memory_manager
from unimind_memory.main import app
from unimind_memory.memory.embedding_service import EmbeddingService
from unimind_memory.memory.memory_manager import MemoryManager
from unimind_memory.services.llm_service import LLMUnavailableError


class FakeLLMService:
    available = True

    def __init__(self):
        self.last_prompt = ""

    def generate_response(self, prompt: str) -> str:
        self.last_prompt = prompt
        return "Mock response using memory."

    def extract_memories(self, messages):
        return [
            {
                "content": "User likes backend tests",
                "category": "preference",
                "importance": 0.8,
            }
        ]

    def summarize(self, messages):
        return None


class TimeoutLLMService:
    available = True

    def generate_response(self, prompt: str) -> str:
        raise TimeoutError("LLM timed out")

    def extract_memories(self, messages):
        raise TimeoutError("LLM extraction timed out")

    def summarize(self, messages):
        raise TimeoutError("LLM summarization timed out")


class UnavailableLLMService:
    available = False

    def generate_response(self, prompt: str) -> str:
        raise LLMUnavailableError("Backend Gemini LLM is not configured")

    def extract_memories(self, messages):
        return []

    def summarize(self, messages):
        return None


class MemoryTestCase(unittest.TestCase):
    def setUp(self):
        self.original_manager = get_memory_manager()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_dir = Path(self.temp_dir.name)
        self.manager = MemoryManager(
            storage_dir=self.storage_dir,
            run_migrations=False,
            llm_service=UnavailableLLMService(),
        )
        set_memory_manager(self.manager)
        self.client = TestClient(app)

    def tearDown(self):
        set_memory_manager(self.original_manager)
        self.temp_dir.cleanup()


class MemoryCoreTests(MemoryTestCase):
    def test_local_hash_embedding_fallback_is_available(self):
        service = EmbeddingService(api_key="")
        result = service.embed("User likes Flutter and Python")
        self.assertEqual(result.provider, "local_hash")
        self.assertEqual(result.vector.shape[0], 384)

    def test_store_and_retrieve_memory_with_root_api(self):
        store = self.client.post(
            "/memory/store",
            json={
                "user_id": "root_user",
                "content": "User prefers concise Python explanations",
                "memory_type": "preference",
                "importance": 0.9,
            },
        )
        self.assertEqual(store.status_code, 200)
        self.assertTrue(store.json()["success"])

        retrieve = self.client.get(
            "/memory/retrieve",
            params={
                "user_id": "root_user",
                "query": "python explanation style",
                "top_k": 5,
            },
        )
        self.assertEqual(retrieve.status_code, 200)
        body = retrieve.json()
        self.assertTrue(body["success"])
        self.assertEqual(body["total_found"], 1)
        self.assertEqual(body["memories"][0]["memory_type"], "preference")

    def test_sqlite_and_faiss_persist_across_manager_reload(self):
        self.manager.store_memory(
            user_id="persist_user",
            content="User is building a Flutter AI assistant",
            memory_type="semantic",
            importance=0.8,
        )

        reloaded = MemoryManager(
            storage_dir=self.storage_dir,
            run_migrations=False,
            llm_service=UnavailableLLMService(),
        )
        results = reloaded.retrieve_memories(
            user_id="persist_user",
            query="Flutter assistant",
            top_k=3,
        )

        self.assertEqual(len(results), 1)
        self.assertIn("Flutter", results[0]["content"])

    def test_duplicate_facts_return_existing_memory(self):
        first_id = self.manager.add_fact(
            "dupe_user",
            "User prefers dark mode",
            "preference",
        )
        second_id = self.manager.add_fact(
            "dupe_user",
            " user prefers dark mode ",
            "preference",
        )
        facts = self.manager.semantic.get_all_facts("dupe_user")

        self.assertEqual(first_id, second_id)
        self.assertEqual(len(facts), 1)

    def test_semantic_score_beats_unrelated_memory(self):
        old_time = (
            datetime.now(timezone.utc) - timedelta(days=120)
        ).isoformat()
        self.manager._store_memory_result(
            user_id="rank_user",
            content="User knows Python and data science",
            memory_type="semantic",
            importance=0.7,
            source="test",
            created_at=old_time,
        )
        self.manager.store_memory(
            user_id="rank_user",
            content="User likes cooking pasta",
            memory_type="semantic",
            importance=0.7,
        )

        results = self.manager.retrieve_memories(
            user_id="rank_user",
            query="python programming skills",
            top_k=2,
        )

        self.assertGreaterEqual(len(results), 2)
        self.assertIn("Python", results[0]["content"])

    def test_legacy_json_migration_imports_existing_memory(self):
        legacy_dir = self.storage_dir / "legacy"
        legacy_dir.mkdir()
        (legacy_dir / "semantic_memory.json").write_text(
            json.dumps(
                {
                    "legacy_user": [
                        {
                            "id": "legacy_fact_1",
                            "fact": "User loves reliable systems",
                            "category": "preference",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        (legacy_dir / "episodic_memory.json").write_text(
            json.dumps(
                {
                    "legacy_user": [
                        {
                            "id": "legacy_episode_1",
                            "summary": "Conversation about memory architecture",
                            "conversation": [],
                            "tags": ["legacy"],
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        (legacy_dir / "short_term_memory.json").write_text(
            json.dumps(
                {
                    "legacy_user": [
                        {
                            "role": "user",
                            "content": "Remember that I like stable APIs",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        migrated = MemoryManager(
            storage_dir=legacy_dir,
            run_migrations=True,
            llm_service=UnavailableLLMService(),
        )

        self.assertEqual(
            migrated.storage.get_message_count("legacy_user"),
            1,
        )
        self.assertEqual(len(migrated.semantic.get_all_facts("legacy_user")), 1)
        self.assertEqual(len(migrated.episodic.get_all_episodes("legacy_user")), 1)

    def test_intelligent_extraction_categories_and_importance(self):
        result = self.manager.add_exchange(
            user_id="extract_user",
            user_message=(
                "My name is Ayush. I prefer dark mode. "
                "I want to build a reliable AI memory project. "
                "I am currently in Patna and going to my hometown Ballia. "
                "I work as a backend developer."
            ),
            assistant_message="Noted.",
            session_id="extract-demo",
            ai_enrich=False,
        )

        facts = self.manager.semantic.get_all_facts("extract_user")
        categories = {fact.get("category"): fact for fact in facts}

        self.assertTrue(result["success"])
        self.assertIn("personal", categories)
        self.assertIn("preference", categories)
        self.assertIn("goal", categories)
        self.assertIn("location", categories)
        self.assertIn("home", categories)
        self.assertGreaterEqual(categories["goal"]["importance"], 0.85)
        self.assertGreaterEqual(categories["personal"]["importance"], 0.85)
        self.assertLess(categories["preference"]["importance"], 0.8)

    def test_recurring_topics_are_stored_after_repetition(self):
        self.manager.add_exchange(
            user_id="topic_user",
            user_message="What is transformers?",
            assistant_message="Transformers are neural network models.",
            session_id="topic-demo",
            ai_enrich=False,
        )
        self.manager.add_exchange(
            user_id="topic_user",
            user_message="Tell me about transformers",
            assistant_message="They use attention.",
            session_id="topic-demo",
            ai_enrich=False,
        )

        facts = self.manager.semantic.get_all_facts("topic_user")
        topics = [fact for fact in facts if fact.get("category") == "recurring_topic"]

        self.assertEqual(len(topics), 1)
        self.assertIn("transformers", topics[0]["content"].lower())
        self.assertGreaterEqual(topics[0]["importance"], 0.6)

    def test_memory_decay_preserves_high_importance_facts(self):
        old_time = (
            datetime.now(timezone.utc) - timedelta(days=420)
        ).isoformat()
        low_id = self.manager.store_memory(
            user_id="decay_user",
            content="User briefly chatted about a temporary event",
            memory_type="semantic",
            importance=0.5,
            metadata={"category": "general"},
            created_at=old_time,
        )
        high_id = self.manager.store_memory(
            user_id="decay_user",
            content="User wants to become an AI engineer",
            memory_type="semantic",
            importance=0.92,
            metadata={"category": "goal"},
            created_at=old_time,
        )

        changed = self.manager.apply_memory_decay(user_id="decay_user")
        low = self.manager.storage.get_memory(low_id)
        high = self.manager.storage.get_memory(high_id)

        self.assertGreaterEqual(changed, 1)
        self.assertLess(low["importance"], 0.5)
        self.assertEqual(high["importance"], 0.92)

    def test_retrieval_debug_deduplicates_and_explains_selection(self):
        self.manager.store_memory(
            user_id="debug_user",
            memory_id="debug_dup_1",
            content="User prefers concise Python explanations",
            memory_type="preference",
            importance=0.7,
            metadata={"category": "preference"},
        )
        self.manager.store_memory(
            user_id="debug_user",
            memory_id="debug_dup_2",
            content="User prefers concise Python explanations",
            memory_type="preference",
            importance=0.7,
            metadata={"category": "preference"},
        )

        results = self.manager.retrieve_memories(
            user_id="debug_user",
            query="python explanations",
            top_k=5,
            debug=True,
        )
        debug = self.manager.retriever.last_debug

        self.assertEqual(len(results), 1)
        self.assertEqual(debug["selected_count"], 1)
        self.assertTrue(any(item["reason"] == "duplicate" for item in debug["dropped"]))
        self.assertIn("semantic_score", debug["selected"][0])

    def test_context_budget_prioritizes_important_memory(self):
        self.manager.store_memory(
            user_id="budget_user",
            content="User wants to complete an AI memory academic project",
            memory_type="semantic",
            importance=0.92,
            metadata={"category": "goal"},
        )
        for index in range(8):
            self.manager.store_memory(
                user_id="budget_user",
                content=(
                    "User had a low priority temporary chat about "
                    f"topic {index} with many filler details that should not dominate context"
                ),
                memory_type="semantic",
                importance=0.3,
                metadata={"category": "general"},
            )

        payload = self.manager.build_context_payload(
            user_id="budget_user",
            query="AI memory academic project",
            max_chars=500,
            debug=True,
        )

        self.assertLessEqual(payload["context_length"], 500)
        self.assertIn("AI memory academic project", payload["context"])
        self.assertIn("context_budget", payload["debug"])

    def test_llm_timeout_falls_back_to_local_extraction_and_summary(self):
        manager = MemoryManager(
            storage_dir=self.storage_dir,
            run_migrations=False,
            llm_service=TimeoutLLMService(),
        )
        result = manager.add_exchange(
            user_id="timeout_user",
            user_message="Remember that I prefer reliable APIs.",
            assistant_message="Saved.",
            session_id="timeout-demo",
            ai_enrich=True,
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["facts_added"], 1)
        self.assertEqual(len(manager.semantic.get_all_facts("timeout_user")), 1)

    def test_embedding_failure_falls_back_to_local_hash(self):
        service = EmbeddingService(api_key="")
        service._gemini = object()
        service._gemini_failed = False

        def fail_embedding(text, task_type):
            raise TimeoutError("fake embedding timeout")

        service._embed_with_gemini = fail_embedding
        result = service.embed("User likes fallback embeddings")

        self.assertEqual(result.provider, "local_hash")
        self.assertEqual(result.vector.shape[0], 384)

    def test_vector_search_failure_rebuilds_once(self):
        class BrokenIndex:
            d = 384
            ntotal = 1

            def search(self, vector, count):
                raise RuntimeError("broken vector index")

        self.manager.store_memory(
            user_id="vector_user",
            content="User prefers resilient retrieval",
            memory_type="preference",
            importance=0.7,
        )
        self.manager.retriever.index = BrokenIndex()

        results = self.manager.retrieve_memories(
            user_id="vector_user",
            query="resilient retrieval",
            top_k=3,
        )

        self.assertEqual(len(results), 1)
        self.assertTrue(
            any("vector_search_failed" in item for item in self.manager.retriever.last_warnings)
        )


class MemoryApiCompatibilityTests(MemoryTestCase):
    def test_health_endpoints_are_available(self):
        for path in ["/health", "/api/v1/health", "/"]:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200)

    def test_flutter_compatible_exchange_context_and_viewer_routes(self):
        response = self.client.post(
            "/api/v1/memory/exchange",
            json={
                "user_id": "flutter_user",
                "user_message": "Remember that I prefer automatic memory.",
                "assistant_message": "Saved.",
                "session_id": "chat-123",
                "tags": ["flutter-chat"],
                "ai_enrich": False,
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["success"])
        self.assertEqual(body["episode_id"], "session_chat-123")
        self.assertEqual(body["facts_added"], 1)

        context = self.client.get(
            "/api/v1/memory/context/flutter_user",
            params={"query": "automatic memory"},
        )
        self.assertEqual(context.status_code, 200)
        self.assertIn("KNOWN FACTS ABOUT USER", context.json()["context"])

        facts = self.client.get("/api/v1/memory/facts/flutter_user")
        episodes = self.client.get("/api/v1/memory/episodes/flutter_user")
        short_term = self.client.get("/api/v1/memory/short-term/flutter_user")
        status = self.client.get("/api/v1/memory/status/flutter_user")

        self.assertEqual(facts.status_code, 200)
        self.assertEqual(episodes.status_code, 200)
        self.assertEqual(short_term.status_code, 200)
        self.assertEqual(status.status_code, 200)
        self.assertEqual(facts.json()["total_facts"], 1)
        self.assertEqual(episodes.json()["total_returned"], 1)
        self.assertEqual(short_term.json()["count"], 2)
        self.assertEqual(status.json()["total_episodes"], 1)

    def test_memory_search_alias_returns_results(self):
        self.client.post(
            "/api/v1/memory/fact",
            json={
                "user_id": "search_user",
                "fact": "User knows Python programming",
                "category": "skill",
            },
        )
        response = self.client.post(
            "/api/v1/memory/search",
            json={
                "user_id": "search_user",
                "query": "programming",
                "top_k": 10,
            },
        )
        debug_response = self.client.post(
            "/api/v1/memory/search",
            json={
                "user_id": "search_user",
                "query": "programming",
                "top_k": 10,
                "debug": True,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["total_found"], 1)
        self.assertEqual(debug_response.status_code, 200)
        self.assertIn("debug", debug_response.json())
        self.assertEqual(debug_response.json()["debug"]["selected_count"], 1)

    def test_default_flutter_responses_omit_debug_payloads(self):
        self.client.post(
            "/api/v1/memory/fact",
            json={
                "user_id": "shape_user",
                "fact": "User prefers lightweight JSON responses",
                "category": "preference",
            },
        )

        context = self.client.get(
            "/api/v1/memory/context/shape_user",
            params={"query": "JSON responses"},
        )
        retrieve = self.client.get(
            "/api/v1/memory/retrieve",
            params={"user_id": "shape_user", "query": "JSON responses"},
        )

        self.assertEqual(context.status_code, 200)
        self.assertEqual(retrieve.status_code, 200)
        self.assertNotIn("debug", context.json())
        self.assertNotIn("warnings", context.json())
        self.assertNotIn("debug", retrieve.json())
        self.assertNotIn("warnings", retrieve.json())

    def test_invalid_message_role_returns_422(self):
        response = self.client.post(
            "/api/v1/memory/message",
            json={
                "user_id": "validation_user",
                "role": "system",
                "content": "Hello",
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_chat_returns_503_when_backend_llm_is_not_configured(self):
        response = self.client.post(
            "/chat",
            json={"user_id": "chat_user", "message": "Hello"},
        )
        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.json()["success"])

    def test_chat_with_mocked_llm_stores_response_and_memory(self):
        fake_llm = FakeLLMService()
        manager = MemoryManager(
            storage_dir=self.storage_dir,
            run_migrations=False,
            llm_service=fake_llm,
        )
        set_memory_manager(manager)

        response = self.client.post(
            "/api/v1/chat",
            json={
                "user_id": "mock_chat_user",
                "message": "My name is Ayush and I prefer stable APIs.",
                "session_id": "demo",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["success"])
        self.assertEqual(body["response"], "Mock response using memory.")
        self.assertIn("MEMORY CONTEXT", fake_llm.last_prompt)
        self.assertGreaterEqual(
            len(manager.semantic.get_all_facts("mock_chat_user")),
            2,
        )
        self.assertEqual(manager.storage.get_message_count("mock_chat_user"), 2)

    def test_chat_stream_returns_sse_delta_and_done(self):
        fake_llm = FakeLLMService()
        manager = MemoryManager(
            storage_dir=self.storage_dir,
            run_migrations=False,
            llm_service=fake_llm,
        )
        set_memory_manager(manager)

        with self.client.stream(
            "POST",
            "/api/v1/chat/stream",
            json={
                "user_id": "stream_user",
                "message": "Remember that I prefer streaming UX.",
                "session_id": "stream-demo",
            },
        ) as response:
            self.assertEqual(response.status_code, 200)
            body = "".join(response.iter_text())

        self.assertIn('"type": "delta"', body)
        self.assertIn('"type": "done"', body)
        self.assertIn('"memory_used"', body)
        self.assertEqual(manager.storage.get_message_count("stream_user"), 2)

    def test_memory_all_returns_lightweight_cards_with_pagination(self):
        first = self.manager.add_fact(
            "vault_user",
            "User prefers compact Flutter responses",
            "preference",
        )
        second = self.manager.add_fact(
            "vault_user",
            "User wants to present UniMind academically",
            "goal",
        )

        response = self.client.get(
            "/api/v1/memory/all/vault_user",
            params={"limit": 1, "offset": 0},
        )
        page_two = self.client.get(
            "/api/v1/memory/all/vault_user",
            params={"limit": 1, "offset": 1},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["success"])
        self.assertEqual(body["total_returned"], 1)
        self.assertEqual(page_two.json()["total_returned"], 1)
        card = body["memories"][0]
        for field in [
            "id",
            "content",
            "summary",
            "memory_type",
            "category",
            "importance",
            "pinned",
            "created_at",
            "updated_at",
            "last_accessed_at",
        ]:
            self.assertIn(field, card)
        self.assertIn(card["id"], {first, second})

    def test_memory_all_filters_by_category_bucket(self):
        self.manager.add_fact(
            "filter_user",
            "User prefers dark mode",
            "preference",
        )
        self.manager.add_fact(
            "filter_user",
            "User wants to learn vector databases",
            "goal",
        )

        response = self.client.get(
            "/api/v1/memory/all/filter_user",
            params={"category": "goals"},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["total_returned"], 1)
        self.assertEqual(body["memories"][0]["bucket"], "goals")

    def test_memory_pin_and_delete_routes(self):
        memory_id = self.manager.add_fact(
            "pin_user",
            "User prefers pinned important memories",
            "preference",
        )

        pinned = self.client.post(
            "/api/v1/memory/pin",
            json={"memory_id": memory_id, "pinned": True},
        )
        self.assertEqual(pinned.status_code, 200)
        self.assertTrue(pinned.json()["memory"]["pinned"])

        unpinned = self.client.post(
            "/api/v1/memory/pin",
            json={"memory_id": memory_id, "pinned": False},
        )
        self.assertEqual(unpinned.status_code, 200)
        self.assertFalse(unpinned.json()["memory"]["pinned"])

        deleted = self.client.delete(f"/api/v1/memory/{memory_id}")
        self.assertEqual(deleted.status_code, 200)
        after = self.client.get("/api/v1/memory/all/pin_user")
        self.assertEqual(after.json()["total_returned"], 0)


if __name__ == "__main__":
    unittest.main()
