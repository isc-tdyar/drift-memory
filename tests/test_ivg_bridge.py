"""Tests for IVGBridge — drift-memory ↔ Graph_KG dual-write integration.

Verifies that:
- register_node() creates drift: nodes in Graph_KG.nodes/rdf_labels/rdf_props
- register_edge() writes to Graph_KG.rdf_edges with qualifiers JSON
- khop() traverses Graph_KG.rdf_edges natively (no custom SQL BFS)
- ivg_traverse() / ivg_find_path() wrappers return correct edge format
- add_edge() dual-writes to both DriftMem_drift_typed_edges and Graph_KG.rdf_edges
- insert_memory() registers node in Graph_KG automatically

All tests run against live los-iris. SKIP_IRIS_TESTS=true to skip.
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
def raw_conn():
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


@pytest.fixture(scope="module")
def bridge():
    from ivg_bridge import IVGBridge
    return IVGBridge(schema="drift")


@pytest.fixture(scope="module")
def db():
    from database.db import MemoryDB
    return MemoryDB("drift")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unique_id() -> str:
    return f"t{uuid.uuid4().hex[:7]}"


def _cleanup(raw_conn, *memory_ids):
    cur = raw_conn.cursor()
    for mid in memory_ids:
        node_id = f"drift:{mid}"
        try:
            cur.execute("DELETE FROM DriftMem_drift_memories WHERE id = ?", [mid])
        except Exception:
            pass
        try:
            cur.execute("DELETE FROM DriftMem_drift_typed_edges WHERE source_id = ? OR target_id = ?", [mid, mid])
        except Exception:
            pass
        try:
            cur.execute("DELETE FROM Graph_KG.rdf_edges WHERE s = ? OR o_id = ?", [node_id, node_id])
        except Exception:
            pass
        try:
            cur.execute("DELETE FROM Graph_KG.rdf_props WHERE s = ?", [node_id])
        except Exception:
            pass
        try:
            cur.execute("DELETE FROM Graph_KG.rdf_labels WHERE s = ?", [node_id])
        except Exception:
            pass
        try:
            cur.execute("DELETE FROM Graph_KG.nodes WHERE node_id = ?", [node_id])
        except Exception:
            pass
    raw_conn.commit()


# ---------------------------------------------------------------------------
# register_node() tests
# ---------------------------------------------------------------------------

class TestRegisterNode:

    @skip_iris
    def test_register_node_creates_graph_kg_node(self, bridge, raw_conn):
        mid = _unique_id()
        try:
            ok = bridge.register_node(mid, type_="lesson", content="test content")
            assert ok is True, "register_node should return True for new node"

            cur = raw_conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM Graph_KG.nodes WHERE node_id = ?",
                [f"drift:{mid}"]
            )
            assert cur.fetchone()[0] == 1, "Node should exist in Graph_KG.nodes"
        finally:
            _cleanup(raw_conn, mid)

    @skip_iris
    def test_register_node_creates_labels(self, bridge, raw_conn):
        mid = _unique_id()
        try:
            bridge.register_node(mid, type_="core", content="test")

            cur = raw_conn.cursor()
            cur.execute(
                "SELECT label FROM Graph_KG.rdf_labels WHERE s = ? ORDER BY label",
                [f"drift:{mid}"]
            )
            labels = {r[0] for r in cur.fetchall()}
            assert "DriftMemory" in labels, f"DriftMemory label missing: {labels}"
            assert "core" in labels, f"type label 'core' missing: {labels}"
        finally:
            _cleanup(raw_conn, mid)

    @skip_iris
    def test_register_node_creates_props(self, bridge, raw_conn):
        mid = _unique_id()
        content = "unique content for prop test"
        try:
            bridge.register_node(mid, type_="active", content=content)

            cur = raw_conn.cursor()
            cur.execute(
                "SELECT key, val FROM Graph_KG.rdf_props WHERE s = ? ORDER BY key",
                [f"drift:{mid}"]
            )
            props = {r[0]: r[1] for r in cur.fetchall()}
            assert props.get("drift_id") == mid, f"drift_id prop wrong: {props}"
            assert props.get("type") == "active", f"type prop wrong: {props}"
            assert content[:100] in props.get("content_preview", ""), "content_preview missing"
        finally:
            _cleanup(raw_conn, mid)

    @skip_iris
    def test_register_node_idempotent(self, bridge, raw_conn):
        mid = _unique_id()
        try:
            ok1 = bridge.register_node(mid, type_="lesson", content="a")
            ok2 = bridge.register_node(mid, type_="lesson", content="a")  # duplicate
            assert ok1 is True
            assert ok2 is False, "Second register should return False (already exists)"

            cur = raw_conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM Graph_KG.nodes WHERE node_id = ?",
                [f"drift:{mid}"]
            )
            assert cur.fetchone()[0] == 1, "Should still be exactly 1 node"
        finally:
            _cleanup(raw_conn, mid)


# ---------------------------------------------------------------------------
# register_edge() tests
# ---------------------------------------------------------------------------

class TestRegisterEdge:

    @skip_iris
    def test_register_edge_creates_rdf_edge(self, bridge, raw_conn):
        src, tgt = _unique_id(), _unique_id()
        try:
            bridge.register_node(src, type_="lesson", content="src")
            bridge.register_node(tgt, type_="lesson", content="tgt")
            ok = bridge.register_edge(src, tgt, "causes", confidence=0.9)
            assert ok is True

            cur = raw_conn.cursor()
            cur.execute(
                "SELECT qualifiers FROM Graph_KG.rdf_edges WHERE s = ? AND p = ? AND o_id = ?",
                [f"drift:{src}", "causes", f"drift:{tgt}"]
            )
            row = cur.fetchone()
            assert row is not None, "Edge should exist in Graph_KG.rdf_edges"
            q = json.loads(row[0])
            assert abs(q.get("confidence", 0) - 0.9) < 0.001
        finally:
            _cleanup(raw_conn, src, tgt)

    @skip_iris
    def test_register_edge_auto_creates_bare_nodes(self, bridge, raw_conn):
        """Edges with unknown node IDs should auto-register bare nodes."""
        src, tgt = _unique_id(), _unique_id()
        try:
            ok = bridge.register_edge(src, tgt, "enables", confidence=0.7)
            assert ok is True

            cur = raw_conn.cursor()
            for nid in [f"drift:{src}", f"drift:{tgt}"]:
                cur.execute("SELECT COUNT(*) FROM Graph_KG.nodes WHERE node_id = ?", [nid])
                assert cur.fetchone()[0] == 1, f"Bare node {nid} not created"
        finally:
            _cleanup(raw_conn, src, tgt)

    @skip_iris
    def test_register_edge_upserts_qualifiers(self, bridge, raw_conn):
        """Re-registering an edge should update qualifiers."""
        src, tgt = _unique_id(), _unique_id()
        try:
            bridge.register_edge(src, tgt, "supports", confidence=0.5)
            bridge.register_edge(src, tgt, "supports", confidence=0.9)  # upsert

            cur = raw_conn.cursor()
            cur.execute(
                "SELECT qualifiers FROM Graph_KG.rdf_edges WHERE s = ? AND p = ? AND o_id = ?",
                [f"drift:{src}", "supports", f"drift:{tgt}"]
            )
            row = cur.fetchone()
            q = json.loads(row[0])
            assert abs(q.get("confidence", 0) - 0.9) < 0.001, f"Expected 0.9 after upsert: {q}"

            # Only 1 edge row
            cur.execute(
                "SELECT COUNT(*) FROM Graph_KG.rdf_edges WHERE s = ? AND p = ? AND o_id = ?",
                [f"drift:{src}", "supports", f"drift:{tgt}"]
            )
            assert cur.fetchone()[0] == 1
        finally:
            _cleanup(raw_conn, src, tgt)


# ---------------------------------------------------------------------------
# khop() traversal tests
# ---------------------------------------------------------------------------

class TestKhop:

    @skip_iris
    def test_khop_empty_for_isolated_node(self, bridge, raw_conn):
        mid = _unique_id()
        try:
            bridge.register_node(mid, type_="lesson", content="isolated")
            result = bridge.khop(mid, hops=2)
            # Isolated node: no outgoing edges
            assert isinstance(result, dict)
            assert "nodes" in result and "edges" in result
            assert mid in result["nodes"]  # seed node itself
            assert len(result["edges"]) == 0
        finally:
            _cleanup(raw_conn, mid)

    @skip_iris
    def test_khop_finds_direct_edges(self, bridge, raw_conn):
        src, tgt = _unique_id(), _unique_id()
        try:
            bridge.register_node(src, type_="lesson", content="src")
            bridge.register_node(tgt, type_="lesson", content="tgt")
            bridge.register_edge(src, tgt, "causes", confidence=0.9)

            result = bridge.khop(src, hops=1)
            targets = {e.get("o") for e in result["edges"]}
            assert tgt in targets, f"Expected {tgt} in targets: {targets}"
        finally:
            _cleanup(raw_conn, src, tgt)

    @skip_iris
    def test_khop_multi_hop(self, bridge, raw_conn):
        a, b, c = _unique_id(), _unique_id(), _unique_id()
        try:
            for mid in [a, b, c]:
                bridge.register_node(mid, type_="lesson", content=mid)
            bridge.register_edge(a, b, "causes", confidence=0.9)
            bridge.register_edge(b, c, "enables", confidence=0.8)

            result_1hop = bridge.khop(a, hops=1)
            targets_1 = {e.get("o") for e in result_1hop["edges"]}
            assert b in targets_1
            assert c not in targets_1, "c should not be reachable in 1 hop"

            result_2hop = bridge.khop(a, hops=2)
            targets_2 = {e.get("o") for e in result_2hop["edges"]}
            assert b in targets_2
            assert c in targets_2, "c should be reachable in 2 hops"
        finally:
            _cleanup(raw_conn, a, b, c)

    @skip_iris
    def test_khop_edge_format(self, bridge, raw_conn):
        src, tgt = _unique_id(), _unique_id()
        try:
            bridge.register_edge(src, tgt, "resolves", confidence=0.75)
            result = bridge.khop(src, hops=1)
            assert len(result["edges"]) >= 1
            edge = result["edges"][0]
            assert "s" in edge and "p" in edge and "o" in edge
            assert "w" in edge and "step" in edge
            assert isinstance(edge["w"], float)
            assert edge["step"] == 1
        finally:
            _cleanup(raw_conn, src, tgt)


# ---------------------------------------------------------------------------
# ivg_traverse() and ivg_find_path() wrappers
# ---------------------------------------------------------------------------

class TestIVGWrappers:

    @skip_iris
    def test_ivg_traverse_returns_list(self):
        from ivg_bridge import ivg_traverse
        result = ivg_traverse("nonexistent_xyz_abc", hops=2)
        assert isinstance(result, list)

    @skip_iris
    def test_ivg_traverse_finds_edges(self, bridge, raw_conn):
        from ivg_bridge import ivg_traverse
        src, tgt = _unique_id(), _unique_id()
        try:
            bridge.register_edge(src, tgt, "causes", confidence=0.85)
            result = ivg_traverse(src, hops=1)
            assert isinstance(result, list)
            targets = {e.get("target_id") for e in result}
            assert tgt in targets
        finally:
            _cleanup(raw_conn, src, tgt)

    @skip_iris
    def test_ivg_find_path_returns_none_for_no_path(self):
        from ivg_bridge import ivg_find_path
        result = ivg_find_path("nonexistent_a", "nonexistent_b", max_hops=3)
        assert result is None

    @skip_iris
    def test_ivg_find_path_direct_edge(self, bridge, raw_conn):
        from ivg_bridge import ivg_find_path
        src, tgt = _unique_id(), _unique_id()
        try:
            bridge.register_edge(src, tgt, "causes", confidence=0.9)
            result = ivg_find_path(src, tgt, max_hops=3)
            assert result is not None, "Should find direct path"
            assert result["depth"] == 1
        finally:
            _cleanup(raw_conn, src, tgt)


# ---------------------------------------------------------------------------
# Dual-write integration: add_edge() → Graph_KG, insert_memory() → Graph_KG
# ---------------------------------------------------------------------------

class TestDualWrite:

    @skip_iris
    def test_add_edge_dual_writes_to_graph_kg(self, raw_conn):
        """add_edge() should write to both typed_edges AND Graph_KG.rdf_edges."""
        from knowledge_graph import add_edge
        from database.db import MemoryDB
        db = MemoryDB("drift")
        src, tgt = _unique_id(), _unique_id()

        try:
            db.insert_memory(src, "lesson", "src for dual-write test")
            db.insert_memory(tgt, "lesson", "tgt for dual-write test")
            add_edge(src, tgt, "causes", confidence=0.88, auto_extracted=True)

            cur = raw_conn.cursor()
            # Verify DriftMem table
            cur.execute(
                "SELECT COUNT(*) FROM DriftMem_drift_typed_edges "
                "WHERE source_id = ? AND target_id = ? AND relationship = 'causes'",
                [src, tgt]
            )
            assert cur.fetchone()[0] == 1, "Edge missing from DriftMem_drift_typed_edges"

            # Verify Graph_KG table
            cur.execute(
                "SELECT qualifiers FROM Graph_KG.rdf_edges WHERE s = ? AND p = 'causes' AND o_id = ?",
                [f"drift:{src}", f"drift:{tgt}"]
            )
            row = cur.fetchone()
            assert row is not None, "Edge missing from Graph_KG.rdf_edges"
            q = json.loads(row[0])
            assert abs(q.get("confidence", 0) - 0.88) < 0.01
        finally:
            _cleanup(raw_conn, src, tgt)

    @skip_iris
    def test_insert_memory_registers_in_graph_kg(self, raw_conn):
        """insert_memory() should create a node in Graph_KG.nodes."""
        from database.db import MemoryDB
        db = MemoryDB("drift")
        mid = _unique_id()

        try:
            db.insert_memory(mid, "lesson", "test memory for graph_kg registration")

            cur = raw_conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM Graph_KG.nodes WHERE node_id = ?",
                [f"drift:{mid}"]
            )
            assert cur.fetchone()[0] == 1, "Memory node not registered in Graph_KG.nodes"
        finally:
            _cleanup(raw_conn, mid)
            # Also cleanup memories table
            try:
                conn2 = raw_conn
                cur2 = conn2.cursor()
                cur2.execute("DELETE FROM DriftMem_drift_memories WHERE id = ?", [mid])
                conn2.commit()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Fix 4: register_cooccurrence_edge() + upsert_edge() dual-write
# ---------------------------------------------------------------------------

class TestCooccurrenceEdge:

    @skip_iris
    def test_register_cooccurrence_edge_creates_rdf_edge(self, bridge, raw_conn):
        """register_cooccurrence_edge() should create a co_occurs_with edge in rdf_edges."""
        a, b = _unique_id(), _unique_id()
        try:
            ok = bridge.register_cooccurrence_edge(a, b, belief=0.75)
            assert ok is True

            cur = raw_conn.cursor()
            cur.execute(
                "SELECT qualifiers FROM Graph_KG.rdf_edges WHERE s = ? AND p = 'co_occurs_with' AND o_id = ?",
                [f"drift:{a}", f"drift:{b}"]
            )
            row = cur.fetchone()
            assert row is not None, "co_occurs_with edge missing from Graph_KG.rdf_edges"
            q = json.loads(row[0])
            assert abs(q.get("confidence", 0) - 0.75) < 0.01, f"Expected belief 0.75: {q}"
        finally:
            _cleanup(raw_conn, a, b)

    @skip_iris
    def test_register_cooccurrence_edge_auto_creates_nodes(self, bridge, raw_conn):
        """register_cooccurrence_edge() should auto-register both nodes."""
        a, b = _unique_id(), _unique_id()
        try:
            bridge.register_cooccurrence_edge(a, b, belief=0.5)

            cur = raw_conn.cursor()
            for nid in [f"drift:{a}", f"drift:{b}"]:
                cur.execute("SELECT COUNT(*) FROM Graph_KG.nodes WHERE node_id = ?", [nid])
                assert cur.fetchone()[0] == 1, f"Node {nid} not auto-created"
        finally:
            _cleanup(raw_conn, a, b)

    @skip_iris
    def test_upsert_edge_dual_writes_cooccurrence(self, raw_conn, monkeypatch):
        """upsert_edge() should dual-write co_occurs_with to Graph_KG.rdf_edges.

        The dual-write is opt-in (DRIFT_IVG_BRIDGE=1, see CLAUDE.md); without the flag
        this asserted a write the code is designed not to make.
        """
        monkeypatch.setenv("DRIFT_IVG_BRIDGE", "1")
        from database.db import MemoryDB
        db = MemoryDB("drift")
        a, b = _unique_id(), _unique_id()

        try:
            db.insert_memory(a, "lesson", "cooccur src")
            db.insert_memory(b, "lesson", "cooccur tgt")
            db.upsert_edge(a, b, belief=0.65)

            cur = raw_conn.cursor()
            # Canonical order (a < b)
            canon_a, canon_b = (a, b) if a < b else (b, a)
            cur.execute(
                "SELECT qualifiers FROM Graph_KG.rdf_edges "
                "WHERE s = ? AND p = 'co_occurs_with' AND o_id = ?",
                [f"drift:{canon_a}", f"drift:{canon_b}"]
            )
            row = cur.fetchone()
            assert row is not None, "upsert_edge did not dual-write to Graph_KG.rdf_edges"
            q = json.loads(row[0])
            assert abs(q.get("confidence", 0) - 0.65) < 0.01
        finally:
            _cleanup(raw_conn, a, b)
            try:
                cur2 = raw_conn.cursor()
                cur2.execute("DELETE FROM DriftMem_drift_memories WHERE id = ?", [a])
                cur2.execute("DELETE FROM DriftMem_drift_memories WHERE id = ?", [b])
                cur2.execute(
                    "DELETE FROM DriftMem_drift_edges WHERE (id1 = ? AND id2 = ?) OR (id1 = ? AND id2 = ?)",
                    [a, b, b, a]
                )
                raw_conn.commit()
            except Exception:
                pass

    @skip_iris
    def test_cooccurrence_upsert_updates_belief(self, bridge, raw_conn):
        """Re-calling register_cooccurrence_edge should update the belief (upsert)."""
        a, b = _unique_id(), _unique_id()
        try:
            bridge.register_cooccurrence_edge(a, b, belief=0.3)
            bridge.register_cooccurrence_edge(a, b, belief=0.8)  # upsert

            cur = raw_conn.cursor()
            cur.execute(
                "SELECT qualifiers FROM Graph_KG.rdf_edges WHERE s = ? AND p = 'co_occurs_with' AND o_id = ?",
                [f"drift:{a}", f"drift:{b}"]
            )
            row = cur.fetchone()
            q = json.loads(row[0])
            assert abs(q.get("confidence", 0) - 0.8) < 0.01, f"Expected 0.8 after upsert: {q}"

            # Exactly one edge row
            cur.execute(
                "SELECT COUNT(*) FROM Graph_KG.rdf_edges WHERE s = ? AND p = 'co_occurs_with' AND o_id = ?",
                [f"drift:{a}", f"drift:{b}"]
            )
            assert cur.fetchone()[0] == 1
        finally:
            _cleanup(raw_conn, a, b)

    @skip_iris
    def test_cooccurrence_edges_visible_in_khop(self, bridge, raw_conn):
        """co_occurs_with edges registered by register_cooccurrence_edge should be traversable."""
        a, b = _unique_id(), _unique_id()
        try:
            bridge.register_cooccurrence_edge(a, b, belief=0.7)
            result = bridge.khop(a, hops=1)
            targets = {e.get("o") for e in result["edges"]}
            assert b in targets, f"Expected {b} reachable via co_occurs_with: {targets}"
        finally:
            _cleanup(raw_conn, a, b)


# ---------------------------------------------------------------------------
# Fix 5: Behavioral fields in register_node()
# ---------------------------------------------------------------------------

class TestBehavioralFields:

    @skip_iris
    def test_register_node_with_behavioral_fields(self, bridge, raw_conn):
        """register_node() should store behavioral fields in rdf_props."""
        mid = _unique_id()
        try:
            bridge.register_node(
                mid, type_="lesson", content="behavioral test",
                emotional_weight=0.8,
                q_value=0.5,
                importance=0.9,
                freshness=0.7,
                memory_tier="hot",
                recall_count=3,
            )

            cur = raw_conn.cursor()
            cur.execute(
                "SELECT key, val FROM Graph_KG.rdf_props WHERE s = ? ORDER BY key",
                [f"drift:{mid}"]
            )
            props = {r[0]: r[1] for r in cur.fetchall()}

            assert abs(float(props.get("emotional_weight", -1)) - 0.8) < 0.01
            assert abs(float(props.get("q_value", -1)) - 0.5) < 0.01
            assert abs(float(props.get("importance", -1)) - 0.9) < 0.01
            assert abs(float(props.get("freshness", -1)) - 0.7) < 0.01
            assert props.get("memory_tier") == "hot"
            assert int(props.get("recall_count", -1)) == 3
        finally:
            _cleanup(raw_conn, mid)

    @skip_iris
    def test_register_node_without_behavioral_fields_succeeds(self, bridge, raw_conn):
        """register_node() without behavioral fields should work (fields optional)."""
        mid = _unique_id()
        try:
            ok = bridge.register_node(mid, type_="lesson", content="no behavioral fields")
            assert ok is True

            cur = raw_conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM Graph_KG.nodes WHERE node_id = ?",
                [f"drift:{mid}"]
            )
            assert cur.fetchone()[0] == 1

            # Behavioral fields should not be present
            cur.execute(
                "SELECT key FROM Graph_KG.rdf_props WHERE s = ? AND key IN "
                "('emotional_weight', 'q_value', 'importance', 'freshness', 'memory_tier', 'recall_count')",
                [f"drift:{mid}"]
            )
            behavioral_keys = [r[0] for r in cur.fetchall()]
            assert len(behavioral_keys) == 0, f"Unexpected behavioral props: {behavioral_keys}"
        finally:
            _cleanup(raw_conn, mid)

    @skip_iris
    def test_register_node_partial_behavioral_fields(self, bridge, raw_conn):
        """Providing only some behavioral fields should store only those."""
        mid = _unique_id()
        try:
            bridge.register_node(mid, type_="core", content="partial test",
                                 importance=0.95, memory_tier="warm")

            cur = raw_conn.cursor()
            cur.execute(
                "SELECT key, val FROM Graph_KG.rdf_props WHERE s = ? AND key IN "
                "('importance', 'memory_tier', 'emotional_weight', 'q_value')",
                [f"drift:{mid}"]
            )
            props = {r[0]: r[1] for r in cur.fetchall()}
            assert "importance" in props
            assert "memory_tier" in props
            assert "emotional_weight" not in props
            assert "q_value" not in props
        finally:
            _cleanup(raw_conn, mid)
