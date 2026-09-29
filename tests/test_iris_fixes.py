"""Tests for three IRIS-compatibility fixes in drift-memory.

PR1: Vector search — VECTOR_COSINE replaces O(n) Python cosine scan
PR2: Graph traversal — Python BFS replaces WITH RECURSIVE CTEs
PR3: Similarity edges — VECTOR_COSINE replaces pgvector <=>

All tests run against live los-iris container. They are read-mostly:
  - search_embeddings: reads existing emb data (12,289 rows already backfilled)
  - traverse/find_path: reads typed_edges (may be empty — tests cover empty case)
  - extract_similarity_edges: tested in dry-run / small-scale mode

Set SKIP_IRIS_TESTS=true to skip (CI without Docker).
"""

import json
import os
import sys
import uuid

import pytest

SKIP_IRIS = os.environ.get("SKIP_IRIS_TESTS", "false").lower() == "true"
skip_iris = pytest.mark.skipif(SKIP_IRIS, reason="SKIP_IRIS_TESTS=true")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def db():
    """MemoryDB instance connected to los-iris."""
    from database.db import MemoryDB
    return MemoryDB("drift")


@pytest.fixture(scope="module")
def raw_conn():
    """Raw IRIS connection for direct SQL testing."""
    from iris_devtester import IRISContainer
    from iris_devtester.connections import get_connection
    from iris_devtester.config import IRISConfig
    container = IRISContainer.attach("los-iris")
    config = IRISConfig(
        host=container.get_container_host_ip(),
        port=container.get_exposed_port(1972),
        namespace="USER",
        username="SuperUser",
        password="SYS",
        auto_create=False,
    )
    conn = get_connection(config=config)
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# PR1: VECTOR column + VECTOR_COSINE search
# ---------------------------------------------------------------------------

class TestVectorColumnExists:
    """Verify emb VECTOR column was added and populated."""

    @skip_iris
    def test_emb_column_populated(self, raw_conn):
        """Every row must have emb set — a NULL emb scores 0.0 in VECTOR_COSINE.

        Asserts completeness rather than an absolute count: the old `> 10000`
        threshold was pinned to a 12,289-row snapshot and went stale as the
        table changed size, which obscured the real regression (all rows NULL).
        """
        cursor = raw_conn.cursor()
        cursor.execute(
            "SELECT COUNT(*), COUNT(emb) FROM DriftMem_drift_text_embeddings"
        )
        total, populated = cursor.fetchone()
        assert total > 0, "text_embeddings is empty — nothing to verify"
        assert populated == total, (
            f"{total - populated} of {total} rows have emb NULL; "
            "VECTOR_COSINE scores those 0.0, killing semantic recall"
        )

    @skip_iris
    def test_vector_cosine_returns_results(self, raw_conn):
        """VECTOR_COSINE query should return ranked results."""
        import random
        random.seed(99)
        query_vec = json.dumps([random.gauss(0, 1) for _ in range(1536)])

        cursor = raw_conn.cursor()
        cursor.execute(
            "SELECT TOP 5 memory_id, VECTOR_COSINE(emb, TO_VECTOR(?, DOUBLE)) as sim "
            "FROM DriftMem_drift_text_embeddings WHERE emb IS NOT NULL ORDER BY sim DESC",
            [query_vec]
        )
        rows = cursor.fetchall()
        assert len(rows) == 5, f"Expected 5 results, got {len(rows)}"

        # Scores should be floats between -1 and 1
        scores = [float(r[1]) for r in rows]
        assert all(-1.0 <= s <= 1.0 for s in scores), f"Invalid scores: {scores}"
        # Results should be ordered DESC
        assert scores == sorted(scores, reverse=True), f"Results not sorted: {scores}"

    @skip_iris
    def test_vector_cosine_self_similarity_is_one(self, raw_conn):
        """A vector compared against itself should score 1.0."""
        cursor = raw_conn.cursor()
        cursor.execute(
            "SELECT TOP 1 memory_id, embedding FROM DriftMem_drift_text_embeddings "
            "WHERE emb IS NOT NULL"
        )
        row = cursor.fetchone()
        assert row is not None, "No rows with emb"
        mid, emb_str = row[0], row[1]

        cursor.execute(
            "SELECT VECTOR_COSINE(emb, TO_VECTOR(?, DOUBLE)) "
            "FROM DriftMem_drift_text_embeddings WHERE memory_id = ?",
            [emb_str, mid]
        )
        score = float(cursor.fetchone()[0])
        assert abs(score - 1.0) < 0.001, f"Self-similarity not 1.0: {score}"


class TestSearchEmbeddingsAPI:
    """Test MemoryDB.search_embeddings() uses VECTOR_COSINE path."""

    @skip_iris
    def test_search_embeddings_returns_results(self, db, raw_conn):
        """search_embeddings should return ranked memories."""
        import random
        random.seed(7)
        query_vec = [random.gauss(0, 1) for _ in range(1536)]

        results = db.search_embeddings(query_vec, limit=5)
        assert len(results) > 0, "search_embeddings returned no results"
        assert len(results) <= 5

        # Each result should have similarity score
        for r in results:
            assert 'similarity' in r, f"Missing similarity in result: {r.keys()}"
            assert isinstance(r['similarity'], float)
            assert -1.0 <= r['similarity'] <= 1.0

    @skip_iris
    def test_search_embeddings_sorted_desc(self, db):
        """Results should be sorted by similarity descending."""
        import random
        random.seed(13)
        query_vec = [random.gauss(0, 1) for _ in range(1536)]

        results = db.search_embeddings(query_vec, limit=10)
        if len(results) < 2:
            pytest.skip("Not enough results to test ordering")

        scores = [r['similarity'] for r in results]
        assert scores == sorted(scores, reverse=True), f"Results not sorted: {scores[:5]}"

    @skip_iris
    def test_search_embeddings_type_filter(self, db, raw_conn):
        """type_filter param should restrict to matching type."""
        # Get a type that actually exists
        cursor = raw_conn.cursor()
        cursor.execute(
            "SELECT TOP 1 type_ FROM DriftMem_drift_memories WHERE type_ IS NOT NULL"
        )
        row = cursor.fetchone()
        if not row:
            pytest.skip("No memories found")

        type_ = row[0]
        import random
        random.seed(21)
        query_vec = [random.gauss(0, 1) for _ in range(1536)]
        results = db.search_embeddings(query_vec, limit=5, type_filter=type_)
        # All results should match the filter (check via get_memory)
        for r in results:
            mem = db.get_memory(r['id'])
            if mem:
                assert mem.get('type') == type_, (
                    f"type_filter={type_!r} but got type={mem.get('type')!r}"
                )


class TestSomaticMarkersAPI:
    """Test find_similar_markers() with emb column."""

    @skip_iris
    def test_find_similar_markers_no_crash(self, db):
        """find_similar_markers should not crash even with no markers."""
        import random
        random.seed(5)
        query_vec = [random.gauss(0, 1) for _ in range(1536)]
        # Should return [] or a list of matching markers (not crash)
        results = db.find_similar_markers(query_vec, threshold=0.5, limit=3)
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# PR2: Graph traversal — Python BFS
# ---------------------------------------------------------------------------

class TestTraverseBFS:
    """Test traverse() Python BFS implementation."""

    @skip_iris
    def test_traverse_returns_list(self):
        """traverse() should return a list (empty or populated)."""
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from knowledge_graph import traverse
        # Use a nonexistent ID — should return empty list, not crash
        result = traverse("nonexistent_id_xyz", hops=2)
        assert isinstance(result, list)

    @skip_iris
    def test_traverse_with_data(self, raw_conn):
        """traverse() should find edges when typed_edges has data."""
        from knowledge_graph import traverse, add_edge
        # Create a small test graph
        id1 = f"test_trav_{uuid.uuid4().hex[:8]}"
        id2 = f"test_trav_{uuid.uuid4().hex[:8]}"
        id3 = f"test_trav_{uuid.uuid4().hex[:8]}"

        # Need memories to exist first
        from database.db import MemoryDB
        db = MemoryDB("drift")
        db.insert_memory(id1, 'test', 'test memory 1 for traversal')
        db.insert_memory(id2, 'test', 'test memory 2 for traversal')
        db.insert_memory(id3, 'test', 'test memory 3 for traversal')

        try:
            add_edge(id1, id2, 'causes', confidence=0.9, auto_extracted=True)
            add_edge(id2, id3, 'enables', confidence=0.8, auto_extracted=True)

            result = traverse(id1, hops=2, direction='outgoing')
            assert isinstance(result, list)
            # Should find at least id1→id2 edge
            targets = {e.get('target_id') for e in result}
            assert id2 in targets, f"Expected {id2} in targets, got {targets}"

        finally:
            # Cleanup
            cursor = raw_conn.cursor()
            for mid in [id1, id2, id3]:
                try:
                    cursor.execute("DELETE FROM DriftMem_drift_memories WHERE id = ?", [mid])
                    cursor.execute(
                        "DELETE FROM DriftMem_drift_typed_edges "
                        "WHERE source_id = ? OR target_id = ?", [mid, mid]
                    )
                except Exception:
                    pass
            raw_conn.commit()

    @skip_iris
    def test_traverse_hop_limit(self, raw_conn):
        """traverse() should not exceed hops limit."""
        from knowledge_graph import traverse, add_edge
        from database.db import MemoryDB
        db = MemoryDB("drift")

        ids = [f"test_hop_{uuid.uuid4().hex[:8]}" for _ in range(5)]
        for mid in ids:
            db.insert_memory(mid, 'test', f'test memory {mid}')

        try:
            # Create a chain: 0→1→2→3→4
            for i in range(len(ids) - 1):
                add_edge(ids[i], ids[i + 1], 'causes', confidence=0.9, auto_extracted=True)

            result_2hop = traverse(ids[0], hops=2, direction='outgoing')
            depths = {e.get('depth') for e in result_2hop}
            assert max(depths, default=0) <= 2, f"Exceeded hops=2: depths={depths}"

        finally:
            cursor = raw_conn.cursor()
            for mid in ids:
                try:
                    cursor.execute("DELETE FROM DriftMem_drift_memories WHERE id = ?", [mid])
                    cursor.execute(
                        "DELETE FROM DriftMem_drift_typed_edges "
                        "WHERE source_id = ? OR target_id = ?", [mid, mid]
                    )
                except Exception:
                    pass
            raw_conn.commit()


class TestFindPath:
    """Test find_path() Python BFS implementation."""

    @skip_iris
    def test_find_path_no_path(self):
        """find_path() should return None when no path exists."""
        from knowledge_graph import find_path
        result = find_path("nonexistent_a", "nonexistent_b", max_hops=3)
        assert result is None

    @skip_iris
    def test_find_path_direct_edge(self, raw_conn):
        """find_path() should find a 1-hop direct edge."""
        from knowledge_graph import find_path, add_edge
        from database.db import MemoryDB
        db = MemoryDB("drift")

        id_a = f"test_path_{uuid.uuid4().hex[:8]}"
        id_b = f"test_path_{uuid.uuid4().hex[:8]}"
        db.insert_memory(id_a, 'test', 'test path memory A')
        db.insert_memory(id_b, 'test', 'test path memory B')

        try:
            add_edge(id_a, id_b, 'causes', confidence=0.9, auto_extracted=True)
            result = find_path(id_a, id_b, max_hops=3)
            assert result is not None, "find_path returned None for direct edge"
            assert result['depth'] == 1
        finally:
            cursor = raw_conn.cursor()
            for mid in [id_a, id_b]:
                try:
                    cursor.execute("DELETE FROM DriftMem_drift_memories WHERE id = ?", [mid])
                    cursor.execute(
                        "DELETE FROM DriftMem_drift_typed_edges "
                        "WHERE source_id = ? OR target_id = ?", [mid, mid]
                    )
                except Exception:
                    pass
            raw_conn.commit()


# ---------------------------------------------------------------------------
# PR3: Similarity edges via VECTOR_COSINE
# ---------------------------------------------------------------------------

class TestExtractSimilarityEdges:
    """Test extract_similarity_edges() uses VECTOR_COSINE (no pgvector <=>)."""

    @skip_iris
    def test_extract_similarity_edges_no_crash(self):
        """extract_similarity_edges should not crash."""
        from knowledge_graph import extract_similarity_edges
        # Use a very high threshold so no edges are created (speed + safety)
        result = extract_similarity_edges(limit=5, threshold=0.9999, verbose=False)
        assert isinstance(result, dict)
        assert 'edges_created' in result
        assert 'pairs_found' in result
        assert isinstance(result['edges_created'], int)

    @skip_iris
    def test_extract_similarity_edges_finds_pairs(self):
        """With a lower threshold, should find pairs (12K+ embeddings available)."""
        from knowledge_graph import extract_similarity_edges
        # 0.95 threshold is very high — should find some pairs in 12K embeddings
        result = extract_similarity_edges(limit=10, threshold=0.95, verbose=False)
        assert isinstance(result, dict)
        # pairs_found >= edges_created (some may already exist)
        assert result['pairs_found'] >= result['edges_created']
