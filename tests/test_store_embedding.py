"""Tests for store_embedding vector-column writes.

Regression: both INSERT branches in MemoryDB.store_embedding had a
column/value count mismatch (7 columns / 6 placeholders, and 6 / 5) because the
`embedding` column had no placeholder of its own — TO_VECTOR(?) consumed the
argument meant for it. Every insert raised <ARGUMENT ERROR> and fell into the
bare `except` that inserts without the vector columns, so emb / emb_384 were
NULL for all 6,493 rows. VECTOR_COSINE against NULL scores 0.0, which silently
killed semantic recall.

These tests write and clean up their own rows. Live los-iris required.
Set SKIP_IRIS_TESTS=true to skip (CI without Docker).
"""

import json
import os
import random
import sys
import uuid

import pytest

SKIP_IRIS = os.environ.get("SKIP_IRIS_TESTS", "false").lower() == "true"
skip_iris = pytest.mark.skipif(SKIP_IRIS, reason="SKIP_IRIS_TESTS=true")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def db():
    from database.db import MemoryDB
    return MemoryDB("drift")


@pytest.fixture
def temp_memory_id(db):
    """A unique memory_id, removed from text_embeddings after the test."""
    mid = f"t{uuid.uuid4().hex[:7]}"
    yield mid
    with db._conn() as conn:
        cur = conn.cursor()
        cur.execute(
            f"DELETE FROM {db._t('text_embeddings')} WHERE memory_id = ?", [mid]
        )


def _vec(dim: int, seed: int) -> list:
    random.seed(seed)
    return [random.gauss(0, 1) for _ in range(dim)]


class TestStoreEmbeddingPopulatesVectorColumns:
    """The 1536-dim emb column must be non-NULL after an insert."""

    @skip_iris
    def test_insert_populates_emb(self, db, temp_memory_id):
        db.store_embedding(temp_memory_id, _vec(1536, 1), preview="hello world",
                           model="test")
        with db._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                f"SELECT embedding, emb FROM {db._t('text_embeddings')} "
                "WHERE memory_id = ?", [temp_memory_id]
            )
            row = cur.fetchone()
        assert row is not None, "row was not inserted at all"
        assert row[0] is not None, "embedding (text) column is NULL"
        assert row[1] is not None, "emb VECTOR column is NULL — insert fell into except"

    @skip_iris
    def test_update_populates_emb(self, db, temp_memory_id):
        """The UPDATE branch (row already exists) must also set emb."""
        db.store_embedding(temp_memory_id, _vec(1536, 2), preview="first", model="test")
        db.store_embedding(temp_memory_id, _vec(1536, 3), preview="second", model="test")
        with db._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                f"SELECT preview, emb FROM {db._t('text_embeddings')} "
                "WHERE memory_id = ?", [temp_memory_id]
            )
            row = cur.fetchone()
        assert row[0] == "second", "update did not overwrite the row"
        assert row[1] is not None, "emb VECTOR column is NULL after update"

    @skip_iris
    def test_insert_without_preview_still_populates_emb(self, db, temp_memory_id):
        """No preview → no emb_384, but emb must still be written."""
        db.store_embedding(temp_memory_id, _vec(1536, 4), preview="", model="test")
        with db._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                f"SELECT emb FROM {db._t('text_embeddings')} WHERE memory_id = ?",
                [temp_memory_id]
            )
            row = cur.fetchone()
        assert row[0] is not None, "emb VECTOR column is NULL on the no-preview path"


class TestA384DimCallerKeepsTheColumnItsVectorFits:
    """A 384-dim caller must not lose BOTH vector columns.

    Both vector columns have a FIXED width — `emb` is `%Library.Vector` LEN 1536,
    `emb_384` is LEN 384 — and IRIS enforces it: a 384-dim value into `emb` raises
    `SQLCODE -104 … failed validation`. Because one `try` wrote BOTH columns and its single
    `except` fell back to a text-only insert, a caller passing a 384-dim `embedding` lost
    `emb_384` as well — even though that value was exactly the right width for it.

    That is not a hypothetical caller. Every consumer in LOS standardised on
    `all-MiniLM-L6-v2` (384-dim): the session indexer, the binding index, cortical patches
    and the cross-referencer. Found from
    `productivity-framework tools/los/tests/e2e/test_cross_session_pipeline.py::
    test_phase2_drift_memory_write_and_embed`, which wrote two memories, got the
    "falling back to text-only" warning, and then could not find either of them by semantic
    search. Written before the fix.
    """

    @skip_iris
    def test_a_384_dim_embedding_populates_emb_384(self, db, temp_memory_id):
        db.store_embedding(temp_memory_id, _vec(384, 11), preview="minilm caller",
                           model="all-MiniLM-L6-v2")
        with db._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                f"SELECT embedding, emb_384 FROM {db._t('text_embeddings')} "
                "WHERE memory_id = ?", [temp_memory_id]
            )
            row = cur.fetchone()
        assert row is not None, "row was not inserted at all"
        assert row[0] is not None, "embedding (text) column is NULL"
        assert row[1] is not None, (
            "emb_384 is NULL — the 384-dim vector fit this column exactly, and was "
            "discarded because the too-wide `emb` write failed in the same statement"
        )

    @skip_iris
    def test_a_384_dim_embedding_is_searchable(self, db, temp_memory_id):
        """The write is only worth anything if `search_embeddings` then finds it."""
        vec = _vec(384, 12)
        db.store_embedding(temp_memory_id, vec, preview="findable minilm row",
                           model="all-MiniLM-L6-v2")
        ids = [r.get("id") for r in db.search_embeddings(vec, limit=5)]
        assert temp_memory_id in ids, (
            f"384-dim row not returned by a 384-dim search; got {ids}"
        )

    @skip_iris
    def test_the_384_update_branch_also_keeps_emb_384(self, db, temp_memory_id):
        """The UPDATE branch has the same all-or-nothing try, so it needs the same guard."""
        db.store_embedding(temp_memory_id, _vec(384, 13), preview="first",
                           model="all-MiniLM-L6-v2")
        db.store_embedding(temp_memory_id, _vec(384, 14), preview="second",
                           model="all-MiniLM-L6-v2")
        with db._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                f"SELECT preview, emb_384 FROM {db._t('text_embeddings')} "
                "WHERE memory_id = ?", [temp_memory_id]
            )
            row = cur.fetchone()
        assert row[0] == "second", "update did not overwrite the row"
        assert row[1] is not None, "emb_384 is NULL after a 384-dim update"

    @skip_iris
    def test_a_1536_dim_caller_is_unaffected(self, db, temp_memory_id):
        """Symmetry check: fixing the narrow caller must not cost the wide one its column."""
        db.store_embedding(temp_memory_id, _vec(1536, 15), preview="qwen caller",
                           model="qwen3-embedding")
        with db._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                f"SELECT emb, emb_384 FROM {db._t('text_embeddings')} "
                "WHERE memory_id = ?", [temp_memory_id]
            )
            row = cur.fetchone()
        assert row[0] is not None, "emb is NULL for a correctly-sized 1536-dim caller"
        assert row[1] is not None, (
            "emb_384 is NULL — it comes from _encode_384(preview), which is independent "
            "of the caller's dimension and must still be written"
        )


class TestStoredEmbeddingIsSearchable:
    """A stored vector must be retrievable via VECTOR_COSINE, not score 0.0."""

    @skip_iris
    def test_vector_cosine_self_similarity(self, db, temp_memory_id):
        vec = _vec(1536, 5)
        db.store_embedding(temp_memory_id, vec, preview="searchable", model="test")
        with db._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                f"SELECT VECTOR_COSINE(emb, TO_VECTOR(?, DOUBLE)) "
                f"FROM {db._t('text_embeddings')} WHERE memory_id = ?",
                [json.dumps(vec), temp_memory_id]
            )
            sim = cur.fetchone()[0]
        assert sim is not None, "VECTOR_COSINE returned NULL — emb was not stored"
        assert float(sim) > 0.99, f"self-similarity should be ~1.0, got {sim}"

    @skip_iris
    def test_search_embeddings_finds_stored_row(self, db, temp_memory_id):
        vec = _vec(1536, 6)
        db.store_embedding(temp_memory_id, vec, preview="findable row", model="test")
        results = db.search_embeddings(vec, limit=5)
        ids = [r.get("id") for r in results]
        assert temp_memory_id in ids, (
            f"stored row not returned by search_embeddings; got {ids}"
        )
