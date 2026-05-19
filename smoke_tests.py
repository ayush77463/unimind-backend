import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

os.environ["UNIMIND_DISABLE_GEMINI"] = "1"
os.environ["UNIMIND_FORCE_LOCAL_STORAGE"] = "1"
os.environ["UNIMIND_VECTOR_BACKEND"] = "chroma"
os.environ["UNIMIND_ENABLE_LOCAL_MODELS"] = "0"
os.environ["UNIMIND_ALLOW_MODEL_DOWNLOADS"] = "0"
_MODULE_TEMP_DIR = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
os.environ["UNIMIND_STORAGE_DIR"] = _MODULE_TEMP_DIR.name

from fastapi.testclient import TestClient

from unimind_memory.api.memory import get_memory_manager, set_memory_manager
from unimind_memory.main import app
from unimind_memory.memory.embedding_service import EmbeddingService
from unimind_memory.memory.intelligence import MemoryIntelligenceAnalyzer
from unimind_memory.memory.memory_manager import MemoryManager
from unimind_memory.memory.supabase_retriever import SupabaseRetriever
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
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.storage_dir = Path(self.temp_dir.name)
        self.manager = MemoryManager(
            storage_dir=self.storage_dir,
            run_migrations=False,
            llm_service=UnavailableLLMService(),
        )
        set_memory_manager(self.manager)
        self.client = TestClient(app)

    def tearDown(self):
        close = getattr(self.manager, "close", None)
        if callable(close):
            close()
        set_memory_manager(self.original_manager)
        self.temp_dir.cleanup()


class MemoryCoreTests(MemoryTestCase):
    def test_memory_intelligence_examples(self):
        analyzer = MemoryIntelligenceAnalyzer()

        greeting = analyzer.analyze("Hi", base_importance=0.6)
        ai_interest = analyzer.analyze("I love AI Engineering", base_importance=0.6)
        project = analyzer.analyze(
            "My project uses Flutter and semantic retrieval",
            base_importance=0.6,
        )

        self.assertEqual(greeting.importance_label, "Low")
        self.assertEqual(greeting.category, "General")
        self.assertEqual(greeting.sentiment, "Neutral")
        self.assertEqual(ai_interest.importance_label, "High")
        self.assertEqual(ai_interest.category, "AI")
        self.assertEqual(ai_interest.sentiment, "Excited")
        self.assertEqual(project.importance_label, "High")
        self.assertIn(project.category, {"AI", "Flutter", "Projects"})

    def test_local_hash_embedding_fallback_is_available(self):
        service = EmbeddingService(api_key="")
        result = service.embed("User likes Flutter and Python")
        self.assertEqual(result.provider, "local_hash")
        self.assertEqual(result.vector.shape[0], 384)

    def test_embedding_cache_is_bounded(self):
        service = EmbeddingService(api_key="")
        service.cache_size = 2

        service.embed("alpha cache item")
        service.embed("beta cache item")
        service.embed("gamma cache item")

        self.assertLessEqual(len(service._cache), 2)
        self.assertNotIn(
            ("retrieval_document", "alpha cache item"),
            service._cache,
        )

    def test_sentence_transformer_provider_is_lazy_and_env_gated(self):
        service = EmbeddingService(api_key="")
        service.embedding_provider_preference = "sentence_transformer"
        service.enable_local_models = False

        result = service.embed("User likes lazy model loading")
        diagnostics = service.diagnostics()

        self.assertEqual(result.provider, "local_hash")
        self.assertFalse(diagnostics["local_models_enabled"])
        self.assertFalse(diagnostics["sentence_model_loaded"])

    def test_topic_classifier_uses_semantic_prototypes(self):
        analyzer = MemoryIntelligenceAnalyzer(self.manager.embedding_service)

        result = analyzer.analyze(
            "The project needs transformer embeddings and vector retrieval ranking",
            base_importance=0.7,
        )

        self.assertIn(result.category, {"AI", "Projects", "Programming"})
        self.assertGreater(result.signals["topic_confidence"], 0)
        self.assertIn("sklearn_cosine", result.signals["intelligence_provider"])

    def test_transformer_sentiment_falls_back_without_downloads(self):
        analyzer = MemoryIntelligenceAnalyzer(
            self.manager.embedding_service,
            enable_transformers=True,
        )

        def fail_transformer(text):
            raise RuntimeError("offline model cache missing")

        analyzer._transformer_sentiment = fail_transformer
        result = analyzer.analyze(
            "I am stressed and confused about the deadline",
            base_importance=0.6,
        )

        self.assertEqual(result.sentiment, "Stressed")
        self.assertGreater(result.signals["sentiment_confidence"], 0)
        self.assertTrue(analyzer._sentiment_failed)

    def test_logistic_importance_features_are_recorded(self):
        memory_id = self.manager.store_memory(
            user_id="importance_user",
            content="User wants to build an AI engineering portfolio project",
            memory_type="semantic",
            importance=0.8,
            metadata={"category": "goal"},
        )

        memory = self.manager.storage.get_memory(memory_id)
        signals = memory["metadata"]["intelligence_signals"]

        self.assertEqual(memory["importance_label"], "High")
        self.assertIn("importance_base_importance", signals)
        self.assertIn("importance_features", memory["metadata"])

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
        try:
            results = reloaded.retrieve_memories(
                user_id="persist_user",
                query="Flutter assistant",
                top_k=3,
            )
        finally:
            reloaded.close()

        self.assertEqual(len(results), 1)
        self.assertIn("Flutter", results[0]["content"])

    def test_chroma_retriever_syncs_searches_and_deletes(self):
        memory_id = self.manager.store_memory(
            user_id="chroma_user",
            content="User is testing Chroma vector retrieval with embeddings",
            memory_type="semantic",
            importance=0.8,
        )

        diagnostics = getattr(self.manager.retriever, "diagnostics", lambda: {})()
        results = self.manager.retrieve_memories(
            user_id="chroma_user",
            query="Chroma retrieval embeddings",
            top_k=3,
            debug=True,
        )

        self.assertEqual(diagnostics.get("status"), "ready")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], memory_id)
        self.assertEqual(results[0].get("vector_backend"), "chroma")

        self.assertTrue(self.manager.delete_memory(memory_id))
        after_delete = self.manager.retrieve_memories(
            user_id="chroma_user",
            query="Chroma retrieval embeddings",
            top_k=3,
        )
        self.assertEqual(after_delete, [])

    def test_chroma_falls_back_when_unavailable(self):
        self.manager.store_memory(
            user_id="fallback_user",
            content="User prefers fallback-safe vector retrieval",
            memory_type="preference",
            importance=0.8,
        )
        if not hasattr(self.manager.retriever, "_available"):
            self.skipTest("Chroma retriever not active")

        self.manager.retriever._available = False
        results = self.manager.retrieve_memories(
            user_id="fallback_user",
            query="fallback vector retrieval",
            top_k=3,
        )

        self.assertEqual(len(results), 1)
        self.assertIn("fallback", self.manager.retriever.last_debug.get("vector_backend", ""))

    def test_chroma_reindexes_provider_mismatches_opportunistically(self):
        memory_id = self.manager.store_memory(
            user_id="provider_user",
            content="User studies semantic retrieval provider migration",
            memory_type="semantic",
            importance=0.7,
        )
        self.manager.storage.update_embedding(
            memory_id,
            np.zeros(384, dtype=np.float32),
            "legacy_provider",
        )

        results = self.manager.retrieve_memories(
            user_id="provider_user",
            query="semantic retrieval provider migration",
            top_k=3,
        )
        updated = self.manager.storage.get_memory(memory_id)

        self.assertEqual(results, [])
        self.assertEqual(updated["embedding_provider"], "local_hash")

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

        try:
            self.assertEqual(
                migrated.storage.get_message_count("legacy_user"),
                1,
            )
            self.assertEqual(len(migrated.semantic.get_all_facts("legacy_user")), 1)
            self.assertEqual(len(migrated.episodic.get_all_episodes("legacy_user")), 1)
        finally:
            migrated.close()

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
        self.assertGreaterEqual(high["importance"], 0.92)

    def test_memory_decay_is_throttled_per_user(self):
        old_time = (
            datetime.now(timezone.utc) - timedelta(days=420)
        ).isoformat()
        self.manager.store_memory(
            user_id="throttle_decay_user",
            content="User briefly mentioned a low priority temporary topic",
            memory_type="semantic",
            importance=0.3,
            metadata={"category": "general"},
            created_at=old_time,
        )

        first = self.manager.apply_memory_decay(user_id="throttle_decay_user")
        second = self.manager.apply_memory_decay(user_id="throttle_decay_user")

        self.assertGreaterEqual(first, 1)
        self.assertEqual(second, 0)

    def test_supabase_retriever_falls_back_when_batch_hydration_fails(self):
        class FakeStorage:
            get_count = 0
            accessed: list[str] = []

            def vector_search(self, **kwargs):
                self.vector_kwargs = kwargs
                return [("memory_one", 0.92)]

            def get_memories_by_ids(self, ids):
                raise RuntimeError("batch hydrate failed")

            def get_memory(self, memory_id):
                self.get_count += 1
                return {
                    "id": memory_id,
                    "user_id": "batch_user",
                    "memory_type": "semantic",
                    "content": "User likes resilient Python retrieval",
                    "summary": "User likes resilient Python retrieval",
                    "category": "skill",
                    "importance": 0.8,
                    "embedding_provider": "local_hash",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }

            def mark_memories_accessed(self, ids):
                self.accessed = list(ids)

        storage = FakeStorage()
        retriever = SupabaseRetriever(
            storage=storage,
            embedding_service=EmbeddingService(api_key=""),
        )

        results = retriever.search(
            user_id="batch_user",
            query="python retrieval",
            top_k=1,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(storage.get_count, 1)
        self.assertEqual(storage.accessed, ["memory_one"])
        self.assertTrue(
            any("batch_hydration_failed" in warning for warning in retriever.last_warnings)
        )
        self.assertEqual(storage.vector_kwargs["embedding_provider"], "local_hash")

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
        try:
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
        finally:
            manager.close()

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

    def test_vector_search_failure_uses_guarded_fallback(self):
        self.manager.store_memory(
            user_id="vector_user",
            content="User prefers resilient retrieval",
            memory_type="preference",
            importance=0.7,
        )
        if hasattr(self.manager.retriever, "_query_chroma"):
            self.manager.retriever._query_chroma = lambda **kwargs: None
        elif hasattr(self.manager.retriever, "index"):
            class BrokenIndex:
                d = 384
                ntotal = 1

                def search(self, vector, count):
                    raise RuntimeError("broken vector index")

            self.manager.retriever.index = BrokenIndex()

        results = self.manager.retrieve_memories(
            user_id="vector_user",
            query="resilient retrieval",
            top_k=3,
        )

        self.assertEqual(len(results), 1)
        self.assertTrue(
            any(
                "vector_search_failed" in item or "chroma_query_failed" in item
                for item in self.manager.retriever.last_warnings
            )
        )


class MemoryApiCompatibilityTests(MemoryTestCase):
    def test_health_endpoints_are_available(self):
        for path in ["/health", "/api/v1/health", "/"]:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200)
        health = self.client.get("/health").json()
        self.assertIn("latency_ms", health)
        self.assertIn("vector_backend", health)
        self.assertIn("ai_pipeline", health)
        self.assertIn("chroma", health)

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
        try:
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
        finally:
            manager.close()

    def test_chat_stream_returns_sse_delta_and_done(self):
        fake_llm = FakeLLMService()
        manager = MemoryManager(
            storage_dir=self.storage_dir,
            run_migrations=False,
            llm_service=fake_llm,
        )
        set_memory_manager(manager)
        try:
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
        finally:
            manager.close()

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
            "memory_category",
            "importance",
            "importance_score",
            "importance_label",
            "sentiment",
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

    def test_memory_analytics_returns_distributions(self):
        self.manager.store_memory(
            user_id="analytics_user",
            content="User loves AI Engineering and semantic memory systems",
            memory_type="semantic",
            importance=0.85,
            metadata={"category": "goal"},
        )
        self.manager.store_memory(
            user_id="analytics_user",
            content="User is confused about Flutter state management",
            memory_type="semantic",
            importance=0.65,
            metadata={"category": "study"},
        )

        response = self.client.get("/api/v1/memory/analytics/analytics_user")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["success"])
        self.assertEqual(body["total_memories"], 2)
        self.assertGreaterEqual(body["high_importance_memories"], 1)
        self.assertIn("category_distribution", body)
        self.assertIn("sentiment_distribution", body)
        self.assertTrue(body["most_discussed_topic"])
        self.assertIn("topic_distribution", body)
        self.assertIn("source_distribution", body)
        self.assertIn("embedding_provider_distribution", body)
        self.assertIn("importance_statistics", body)
        self.assertEqual(body["ai_pipeline"], "pandas_numpy_analytics")

    def test_document_upload_returns_langchain_chunks(self):
        text = (
            "UniMind document ingestion should chunk PDFs and notes for semantic retrieval. "
            "The uploaded document discusses Chroma, embeddings, and multimodal memory. "
        ) * 40

        response = self.client.post(
            "/api/v1/document/upload",
            data={"user_id": "doc_user"},
            files={"file": ("notes.txt", text.encode("utf-8"), "text/plain")},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["success"])
        self.assertIn("extracted_text", body)
        self.assertGreater(body["chunk_count"], 1)
        self.assertIn("document_summary", body)
        first_chunk = body["chunks"][0]
        self.assertEqual(first_chunk["metadata"]["source"], "document_upload")
        self.assertEqual(first_chunk["metadata"]["filename"], "notes.txt")

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
