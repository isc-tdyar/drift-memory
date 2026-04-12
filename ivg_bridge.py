#!/usr/bin/env python3
"""
IVG Bridge — Mirror drift-memory nodes and typed edges into Graph_KG.

When a memory is created, its node is registered in Graph_KG.nodes/rdf_labels/rdf_props.
When a typed edge is added, it is mirrored into Graph_KG.rdf_edges.

This feeds the ^KG global (populated by Graph.KG.Traversal.BuildKG) so that
iris-vector-graph's khop(), variable-length Cypher, PPR, and BFSFastJson all
work natively over drift-memory graphs — no custom BFS SQL needed.

Node ID convention: drift:{memory_id}
  e.g. memory 'vmylw4ao' → node 'drift:vmylw4ao'

Edge predicate: the relationship string as-is (e.g. 'causes', 'enables')
Qualifiers JSON: {"confidence": 0.9, "evidence": "...", "auto_extracted": true}

Usage:
    from ivg_bridge import IVGBridge
    bridge = IVGBridge()

    # On memory creation
    bridge.register_node(memory_id, type_='lesson', content='...')

    # On typed edge creation
    bridge.register_edge(source_id, target_id, relationship='causes', confidence=0.9)

    # Traverse using native IVG khop()
    result = bridge.khop('vmylw4ao', hops=2)

    # Variable-length Cypher traversal
    results = bridge.cypher_traverse('vmylw4ao', relationship='causes', max_hops=3)

    # Backfill existing memories + edges
    bridge.backfill()
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Node prefix — namespaces drift-memory IDs within Graph_KG to avoid collision
_DRIFT_PREFIX = "drift:"


def _drift_node_id(memory_id: str) -> str:
    """Convert drift-memory ID to Graph_KG node ID."""
    return f"{_DRIFT_PREFIX}{memory_id}"


def _strip_prefix(node_id: str) -> str:
    """Strip drift: prefix from Graph_KG node ID to get memory_id."""
    if node_id.startswith(_DRIFT_PREFIX):
        return node_id[len(_DRIFT_PREFIX):]
    return node_id


def _get_conn():
    import os
    import iris as _iris
    host     = os.environ.get("IRIS_HOST",      "localhost")
    port     = int(os.environ.get("IRIS_PORT",  "11982"))
    ns       = os.environ.get("IRIS_NAMESPACE", "USER")
    user     = os.environ.get("IRIS_USERNAME",  "SuperUser")
    password = os.environ.get("IRIS_PASSWORD",  "SYS")
    return _iris.connect(host, port, ns, user, password)


class IVGBridge:
    """Mirror drift-memory into Graph_KG for native IVG traversal.

    Keeps a lazy-initialized IRISGraphEngine for khop() and Cypher calls.
    All write operations (register_node, register_edge) use raw IRIS SQL
    for minimal overhead and no IVG schema re-init cost on every call.
    """

    def __init__(self, schema: str = "drift"):
        self.schema = schema
        self._engine = None

    # ------------------------------------------------------------------
    # Lazy engine (for khop / Cypher only)
    # ------------------------------------------------------------------

    def _get_engine(self):
        if self._engine is None:
            from iris_vector_graph import IRISGraphEngine
            from iris_vector_graph.cypher import set_schema_prefix
            set_schema_prefix("Graph_KG")
            conn = _get_conn()
            self._engine = IRISGraphEngine(conn, embedding_dimension=384)
        return self._engine

    # ------------------------------------------------------------------
    # Node registration — mirrors insert_memory()
    # ------------------------------------------------------------------

    def register_node(self, memory_id: str, type_: str = "unknown",
                      content: str = "", conn=None,
                      emotional_weight: float = None,
                      q_value: float = None,
                      importance: float = None,
                      freshness: float = None,
                      memory_tier: str = None,
                      recall_count: int = None) -> bool:
        """Register a drift-memory node in Graph_KG.

        Creates: Graph_KG.nodes, rdf_labels (DriftMemory + type_),
                 rdf_props (drift_schema, drift_id, type, content_preview,
                            + optional behavioral fields for PPR weighting)

        Args:
            memory_id: The 8-char drift-memory ID.
            type_: Memory type (e.g. 'lesson', 'core', 'active').
            content: Memory content (first 200 chars stored as content_preview).
            conn: Optional existing IRIS connection (avoids reconnect on bulk ops).
            emotional_weight: Cached copy from memories.emotional_weight (PPR seed).
            q_value: Cached copy from memories.q_value (RL signal for PPR).
            importance: Cached copy from memories.importance.
            freshness: Recency score (0–1, higher = more recent).
            memory_tier: Tier string (e.g. 'hot', 'warm', 'cold').
            recall_count: How many times this memory has been recalled.

        Returns:
            True if registered, False if already existed.
        """
        node_id = _drift_node_id(memory_id)
        content_preview = str(content)[:200] if content else ""
        owns_conn = conn is None

        if owns_conn:
            conn = _get_conn()

        try:
            cur = conn.cursor()

            # 1. nodes — INSERT IF NOT EXISTS
            cur.execute(
                "SELECT COUNT(*) FROM Graph_KG.nodes WHERE node_id = ?",
                [node_id]
            )
            if cur.fetchone()[0] > 0:
                return False  # Already registered

            cur.execute(
                "INSERT INTO Graph_KG.nodes (node_id) VALUES (?)",
                [node_id]
            )

            # 2. rdf_labels — DriftMemory (generic) + specific type
            for label in ("DriftMemory", type_):
                cur.execute(
                    "INSERT INTO Graph_KG.rdf_labels (s, label) "
                    "SELECT ?, ? WHERE NOT EXISTS "
                    "(SELECT 1 FROM Graph_KG.rdf_labels WHERE s = ? AND label = ?)",
                    [node_id, label, node_id, label]
                )

            # 3. rdf_props — key/val properties
            props = {
                "id": node_id,
                "drift_id": memory_id,
                "drift_schema": self.schema,
                "type": type_,
                "content_preview": content_preview,
            }
            # Behavioral fields (cached copies for PPR weighting — memories table is authoritative)
            if emotional_weight is not None:
                props["emotional_weight"] = str(emotional_weight)
            if q_value is not None:
                props["q_value"] = str(q_value)
            if importance is not None:
                props["importance"] = str(importance)
            if freshness is not None:
                props["freshness"] = str(freshness)
            if memory_tier is not None:
                props["memory_tier"] = str(memory_tier)
            if recall_count is not None:
                props["recall_count"] = str(recall_count)

            for key, val in props.items():
                cur.execute(
                    "INSERT INTO Graph_KG.rdf_props (s, key, val) "
                    "SELECT ?, ?, ? WHERE NOT EXISTS "
                    "(SELECT 1 FROM Graph_KG.rdf_props WHERE s = ? AND key = ?)",
                    [node_id, key, str(val), node_id, key]
                )

            if owns_conn:
                conn.commit()
            return True

        except Exception as e:
            logger.warning(f"IVGBridge.register_node({memory_id}): {e}")
            if owns_conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            return False
        finally:
            if owns_conn:
                try:
                    conn.close()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Edge registration — mirrors add_edge()
    # ------------------------------------------------------------------

    def register_edge(self, source_id: str, target_id: str,
                      relationship: str, confidence: float = 0.8,
                      evidence: str = None, auto_extracted: bool = False,
                      conn=None) -> bool:
        """Mirror a typed edge into Graph_KG.rdf_edges.

        The edge predicate is the relationship string directly (e.g. 'causes').
        Qualifiers JSON carries confidence, evidence, auto_extracted.

        If source or target nodes don't exist in Graph_KG yet, they are
        auto-registered as bare DriftMemory nodes (no content).

        Args:
            source_id: Source drift-memory ID (no prefix).
            target_id: Target drift-memory ID (no prefix).
            relationship: Typed relationship (e.g. 'causes', 'enables').
            confidence: Edge confidence score [0, 1].
            evidence: Optional evidence string.
            auto_extracted: Whether edge was auto-extracted vs. manual.
            conn: Optional existing IRIS connection.

        Returns:
            True on success.
        """
        src_node = _drift_node_id(source_id)
        tgt_node = _drift_node_id(target_id)
        qualifiers = json.dumps({
            "confidence": confidence,
            "evidence": evidence,
            "auto_extracted": auto_extracted,
        })
        owns_conn = conn is None

        if owns_conn:
            conn = _get_conn()

        try:
            cur = conn.cursor()

            # Auto-register nodes if not present (bare — no content lookup)
            for nid in (src_node, tgt_node):
                cur.execute(
                    "SELECT COUNT(*) FROM Graph_KG.nodes WHERE node_id = ?", [nid]
                )
                if cur.fetchone()[0] == 0:
                    cur.execute(
                        "INSERT INTO Graph_KG.nodes (node_id) VALUES (?)", [nid]
                    )
                    cur.execute(
                        "INSERT INTO Graph_KG.rdf_labels (s, label) VALUES (?, ?)",
                        [nid, "DriftMemory"]
                    )

            # Upsert edge: check u_spo unique constraint (s, p, o_id)
            cur.execute(
                "SELECT COUNT(*) FROM Graph_KG.rdf_edges "
                "WHERE s = ? AND p = ? AND o_id = ?",
                [src_node, relationship, tgt_node]
            )
            if cur.fetchone()[0] > 0:
                cur.execute(
                    "UPDATE Graph_KG.rdf_edges SET qualifiers = ? "
                    "WHERE s = ? AND p = ? AND o_id = ?",
                    [qualifiers, src_node, relationship, tgt_node]
                )
            else:
                cur.execute(
                    "INSERT INTO Graph_KG.rdf_edges (s, p, o_id, qualifiers) "
                    "VALUES (?, ?, ?, ?)",
                    [src_node, relationship, tgt_node, qualifiers]
                )

            if owns_conn:
                conn.commit()
            return True

        except Exception as e:
            logger.warning(f"IVGBridge.register_edge({source_id}->{target_id}): {e}")
            if owns_conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            return False
        finally:
            if owns_conn:
                try:
                    conn.close()
                except Exception:
                    pass

    def register_cooccurrence_edge(self, id1: str, id2: str,
                                    belief: float = 0.5,
                                    conn=None) -> bool:
        """Mirror a drift_edges Hebbian co-occurrence pair into Graph_KG.rdf_edges.

        Predicate is 'co_occurs_with'; belief maps to qualifiers.confidence.
        The edge is bidirectional in the co-occurrence sense but stored as a
        directed edge id1→id2 (consistent with rdf_edges convention).
        Auto-registers bare nodes for both IDs if not already present.

        Args:
            id1: First memory ID (no prefix).
            id2: Second memory ID (no prefix).
            belief: Hebbian belief strength (0–1). Higher = stronger co-occurrence.
            conn: Optional existing IRIS connection.

        Returns:
            True if edge was inserted/updated, False on error.
        """
        return self.register_edge(
            id1, id2,
            relationship="co_occurs_with",
            confidence=float(belief),
            evidence=None,
            auto_extracted=True,
            conn=conn,
        )

    # ------------------------------------------------------------------
    # Native IVG traversal — khop() populates from ^KG global
    # ------------------------------------------------------------------

    def rebuild_kg(self) -> bool:
        """Rebuild the ^KG ObjectScript global from Graph_KG.rdf_edges.

        Requires Graph.KG.Traversal ObjectScript class to be deployed.
        In los-iris (Community Edition), ObjectScript may not be deployed —
        kg_NEIGHBORHOOD_EXPANSION (pure SQL) is used instead and doesn't need ^KG.
        This method is a no-op when ObjectScript isn't available.
        """
        try:
            engine = self._get_engine()
            if not engine.capabilities.objectscript_deployed:
                logger.debug("IVGBridge.rebuild_kg: ObjectScript not deployed — skipping (SQL path is used)")
                return True  # Not an error — SQL path works without ^KG
            iris_obj = engine._iris_obj()
            iris_obj.classMethodValue("Graph.KG.Traversal", "BuildKG")
            logger.info("IVGBridge: ^KG rebuilt")
            return True
        except Exception as e:
            logger.debug(f"IVGBridge.rebuild_kg: {e}")
            return False

    def khop(self, memory_id: str, hops: int = 2,
             max_nodes: int = 500) -> Dict[str, Any]:
        """k-hop neighborhood expansion via IVG's kg_NEIGHBORHOOD_EXPANSION.

        Uses IVG's native SQL-based neighborhood expansion over Graph_KG.rdf_edges —
        one batched IN query per hop, no ObjectScript required.

        When Graph.KG.Traversal is deployed (^KG populated), IVG's khop() /
        BFSFastJson is used instead (faster, ObjectScript-accelerated). The
        engine.khop() call handles the Arno/BFSFast/SQL fallback chain automatically.

        Returns:
            {'nodes': [memory_id, ...], 'edges': [{s, p, o, confidence, depth}, ...]}
            Node IDs are raw drift-memory IDs (drift: prefix stripped).
        """
        node_id = _drift_node_id(memory_id)
        engine = self._get_engine()

        # Try native khop() first (uses BFSFastJson if ObjectScript deployed)
        if engine.capabilities.objectscript_deployed:
            result = engine.khop(node_id, hops=hops, max_nodes=max_nodes)
            if result.get('nodes') or result.get('edges'):
                result['nodes'] = [_strip_prefix(n) for n in result.get('nodes', [])]
                result['edges'] = [
                    {**e, 's': _strip_prefix(e.get('s', '')), 'o': _strip_prefix(e.get('o', ''))}
                    for e in result.get('edges', [])
                ]
                return result

        # SQL-based BFS via kg_NEIGHBORHOOD_EXPANSION (works without ObjectScript)
        return self._khop_sql(node_id, hops=hops, max_nodes=max_nodes)

    def _khop_sql(self, node_id: str, hops: int, max_nodes: int) -> Dict[str, Any]:
        """Pure-SQL BFS over Graph_KG.rdf_edges — IVG kg_NEIGHBORHOOD_EXPANSION pattern.

        One IN-list query per hop over the entire frontier. Confidence extracted
        from qualifiers JSON via %INLIST / substring (no JSON_TABLE required).
        This is the same pattern as IVG's Python BFS but operates on Graph_KG.rdf_edges
        directly so all drift-memory edges registered via register_edge() are visible.
        """
        conn = _get_conn()
        try:
            cur = conn.cursor()

            visited_nodes = {node_id}
            frontier = [node_id]
            all_edges: List[Dict] = []
            seen_edge_keys: set = set()

            for depth in range(1, hops + 1):
                if not frontier:
                    break

                placeholders = ','.join(['?' for _ in frontier])
                cur.execute(
                    f"SELECT s, p, o_id, qualifiers "
                    f"FROM Graph_KG.rdf_edges "
                    f"WHERE s IN ({placeholders})",
                    frontier
                )
                rows = cur.fetchall()

                next_frontier: List[str] = []
                for row in rows:
                    src, pred, tgt, quals = row[0], row[1], row[2], row[3]
                    # Extract confidence from qualifiers JSON (best effort)
                    conf = 0.8
                    if quals:
                        try:
                            import json as _json
                            q = _json.loads(quals)
                            conf = float(q.get('confidence', 0.8))
                        except Exception:
                            pass

                    edge_key = (src, pred, tgt)
                    if edge_key in seen_edge_keys:
                        continue
                    seen_edge_keys.add(edge_key)

                    all_edges.append({
                        's': _strip_prefix(src),
                        'p': pred,
                        'o': _strip_prefix(tgt),
                        'w': conf,
                        'step': depth,
                    })

                    if tgt not in visited_nodes and len(visited_nodes) < max_nodes:
                        visited_nodes.add(tgt)
                        next_frontier.append(tgt)

                frontier = next_frontier

            nodes = list({_strip_prefix(n) for n in visited_nodes})
            return {'nodes': nodes, 'edges': all_edges}
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def cypher_traverse(self, memory_id: str, relationship: str = None,
                        max_hops: int = 3) -> List[Dict]:
        """Variable-length Cypher traversal over drift-memory graph.

        Uses IVG's execute_cypher() with *..N variable-length paths.
        Requires nodes to be in Graph_KG (registered via register_node).

        Args:
            memory_id: Starting drift-memory ID (no prefix).
            relationship: Relationship type to follow (None = any).
            max_hops: Maximum traversal depth.

        Returns:
            List of dicts with memory_id, relationship, depth.
        """
        node_id = _drift_node_id(memory_id)
        engine = self._get_engine()

        rel_clause = f"[:{relationship}*1..{max_hops}]" if relationship else f"[*1..{max_hops}]"
        cypher = f"MATCH (a:DriftMemory)-{rel_clause}->(b:DriftMemory) WHERE a.id = '{node_id}' RETURN b.drift_id, b.type"

        try:
            result = engine.execute_cypher(cypher)
            rows = result.get('rows', [])
            return [
                {'memory_id': _strip_prefix(r.get('b.drift_id', '')),
                 'type': r.get('b.type', '')}
                for r in rows
            ]
        except Exception as e:
            logger.warning(f"IVGBridge.cypher_traverse: {e}")
            return []

    # ------------------------------------------------------------------
    # Backfill — mirror all existing drift-memory data into Graph_KG
    # ------------------------------------------------------------------

    def backfill(self, batch_size: int = 200, dry_run: bool = False,
                 verbose: bool = True) -> Dict[str, int]:
        """Backfill all existing drift-memory nodes and edges into Graph_KG.

        Reads DriftMem_drift_memories and DriftMem_drift_typed_edges,
        inserts missing nodes/edges into Graph_KG, then rebuilds ^KG.

        Args:
            batch_size: Rows to process per batch (nodes and edges separately).
            dry_run: If True, count but don't write.
            verbose: Print progress.

        Returns:
            {'nodes_added': N, 'edges_added': M, 'nodes_skipped': K, 'edges_skipped': J}
        """
        from database.db import MemoryDB
        db = MemoryDB(self.schema)

        stats = {
            'nodes_added': 0, 'nodes_skipped': 0,
            'edges_added': 0, 'edges_skipped': 0,
            'cooccur_added': 0, 'cooccur_skipped': 0,
        }

        if dry_run:
            with db._conn() as conn:
                cur = conn.cursor()
                cur.execute(f"SELECT COUNT(*) FROM DriftMem_{self.schema}_memories")
                stats['nodes_total'] = cur.fetchone()[0]
                cur.execute(f"SELECT COUNT(*) FROM DriftMem_{self.schema}_typed_edges")
                stats['edges_total'] = cur.fetchone()[0]
                try:
                    cur.execute(f"SELECT COUNT(*) FROM DriftMem_{self.schema}_edges")
                    stats['cooccur_total'] = cur.fetchone()[0]
                except Exception:
                    stats['cooccur_total'] = 0
            if verbose:
                print(f"[dry-run] Would process {stats.get('nodes_total', '?')} memories, "
                      f"{stats.get('edges_total', '?')} typed_edges, "
                      f"{stats.get('cooccur_total', '?')} drift_edges")
            return stats

        # --- Backfill nodes ---
        raw_conn = _get_conn()
        try:
            with db._conn() as drift_conn:
                cur = drift_conn.cursor()
                cur.execute(
                    f"SELECT COUNT(*) FROM DriftMem_{self.schema}_memories"
                )
                total_memories = cur.fetchone()[0]
                if verbose:
                    print(f"Backfilling {total_memories} memories into Graph_KG.nodes ...")

                # IRIS doesn't support TOP + OFFSET together; use cursor-based pagination
                # via last_id tracking (id is VARCHAR, alphabetical order)
                last_id = ""
                while True:
                    cur.execute(
                        f"SELECT TOP ? id, type_, content FROM DriftMem_{self.schema}_memories "
                        f"WHERE id > ? ORDER BY id",
                        [batch_size, last_id]
                    )
                    rows = cur.fetchall()
                    if not rows:
                        break

                    for memory_id, type_, content in rows:
                        added = self.register_node(
                            str(memory_id),
                            type_=str(type_ or 'unknown'),
                            content=str(content or ''),
                            conn=raw_conn,
                        )
                        if added:
                            stats['nodes_added'] += 1
                        else:
                            stats['nodes_skipped'] += 1

                    last_id = str(rows[-1][0])
                    raw_conn.commit()
                    if verbose:
                        done = stats['nodes_added'] + stats['nodes_skipped']
                        print(f"  nodes: {done}/{total_memories} "
                              f"(+{stats['nodes_added']} new, "
                              f"={stats['nodes_skipped']} existing)", end='\r')

                    if len(rows) < batch_size:
                        break

            if verbose:
                print(f"\n  nodes done: +{stats['nodes_added']} new, "
                      f"={stats['nodes_skipped']} already in Graph_KG")

            # --- Backfill edges ---
            with db._conn() as drift_conn:
                cur = drift_conn.cursor()
                cur.execute(
                    f"SELECT COUNT(*) FROM DriftMem_{self.schema}_typed_edges"
                )
                total_edges = cur.fetchone()[0]
                if verbose:
                    print(f"Backfilling {total_edges} typed_edges into Graph_KG.rdf_edges ...")

                # Cursor-based pagination: track last (source_id, target_id, relationship)
                last_src, last_tgt, last_rel = "", "", ""
                while True:
                    cur.execute(
                        f"SELECT TOP ? source_id, target_id, relationship, confidence, evidence, auto_extracted "
                        f"FROM DriftMem_{self.schema}_typed_edges "
                        f"WHERE (source_id > ? OR (source_id = ? AND target_id > ?) "
                        f"OR (source_id = ? AND target_id = ? AND relationship > ?)) "
                        f"ORDER BY source_id, target_id, relationship",
                        [batch_size, last_src, last_src, last_tgt, last_src, last_tgt, last_rel]
                    )
                    rows = cur.fetchall()
                    if not rows:
                        break

                    for row in rows:
                        src, tgt, rel, conf, ev, auto = row
                        added = self.register_edge(
                            str(src), str(tgt), str(rel),
                            confidence=float(conf or 0.8),
                            evidence=str(ev) if ev else None,
                            auto_extracted=bool(auto),
                            conn=raw_conn,
                        )
                        if added:
                            stats['edges_added'] += 1
                        else:
                            stats['edges_skipped'] += 1

                    last_src, last_tgt, last_rel = str(rows[-1][0]), str(rows[-1][1]), str(rows[-1][2])
                    raw_conn.commit()
                    if verbose:
                        done = stats['edges_added'] + stats['edges_skipped']
                        print(f"  edges: {done}/{total_edges} "
                              f"(+{stats['edges_added']} new, "
                              f"={stats['edges_skipped']} existing)", end='\r')

                    if len(rows) < batch_size:
                        break

            if verbose:
                print(f"\n  edges done: +{stats['edges_added']} new, "
                      f"={stats['edges_skipped']} already in Graph_KG")

            # --- Backfill co-occurrence edges (drift_edges Hebbian graph) ---
            try:
                with db._conn() as drift_conn:
                    cur = drift_conn.cursor()
                    cur.execute(
                        f"SELECT COUNT(*) FROM DriftMem_{self.schema}_edges"
                    )
                    total_cooccur = cur.fetchone()[0]
                    if verbose:
                        print(f"Backfilling {total_cooccur} drift_edges (Hebbian co-occurrence) "
                              f"into Graph_KG.rdf_edges ...")

                    # Cursor-based pagination on (id1, id2)
                    last_id1, last_id2 = "", ""
                    while True:
                        cur.execute(
                            f"SELECT TOP ? id1, id2, belief "
                            f"FROM DriftMem_{self.schema}_edges "
                            f"WHERE (id1 > ? OR (id1 = ? AND id2 > ?)) "
                            f"ORDER BY id1, id2",
                            [batch_size, last_id1, last_id1, last_id2]
                        )
                        rows = cur.fetchall()
                        if not rows:
                            break

                        for row in rows:
                            id1, id2, belief = row
                            added = self.register_cooccurrence_edge(
                                str(id1), str(id2),
                                belief=float(belief or 0.5),
                                conn=raw_conn,
                            )
                            if added:
                                stats['cooccur_added'] += 1
                            else:
                                stats['cooccur_skipped'] += 1

                        last_id1, last_id2 = str(rows[-1][0]), str(rows[-1][1])
                        raw_conn.commit()
                        if verbose:
                            done = stats['cooccur_added'] + stats['cooccur_skipped']
                            print(f"  co-occur: {done}/{total_cooccur} "
                                  f"(+{stats['cooccur_added']} new, "
                                  f"={stats['cooccur_skipped']} existing)", end='\r')

                        if len(rows) < batch_size:
                            break

                if verbose:
                    print(f"\n  co-occur done: +{stats['cooccur_added']} new, "
                          f"={stats['cooccur_skipped']} already in Graph_KG")
            except Exception as e:
                # drift_edges table may not exist in all schemas — non-fatal
                if verbose:
                    print(f"\n  co-occur backfill skipped: {e}")
                logger.debug(f"IVGBridge.backfill drift_edges: {e}")

            # --- Rebuild ^KG global ---
            if verbose:
                print("Rebuilding ^KG global for khop()/BFSFastJson ...")
            self.rebuild_kg()
            if verbose:
                print("Done. IVG traversal is now live over drift-memory graph.")

        finally:
            try:
                raw_conn.close()
            except Exception:
                pass

        return stats


# ---------------------------------------------------------------------------
# Module-level singleton — auto-instantiated for drift: + other schemas
# ---------------------------------------------------------------------------

_bridge: Optional[IVGBridge] = None


def get_bridge(schema: str = "drift") -> IVGBridge:
    """Get (or create) the module-level IVGBridge for a schema."""
    global _bridge
    if _bridge is None or _bridge.schema != schema:
        _bridge = IVGBridge(schema=schema)
    return _bridge


# ---------------------------------------------------------------------------
# Drop-in replacements for traverse() and find_path() — IVG native versions
# ---------------------------------------------------------------------------

def ivg_traverse(start_id: str, relationship: str = None,
                 hops: int = 2, direction: str = 'outgoing',
                 min_confidence: float = 0.3) -> List[Dict]:
    """IVG-native multi-hop traversal using khop() / BFSFastJson.

    Drop-in replacement for knowledge_graph.traverse() that routes through
    the ^KG global instead of custom SQL BFS.

    Returns same format as traverse(): list of edge dicts with 'depth' field.
    Node IDs are raw drift-memory IDs (prefix stripped).
    """
    bridge = get_bridge()
    result = bridge.khop(start_id, hops=hops)
    edges = result.get('edges', [])

    # Filter by relationship and min_confidence if specified
    out = []
    for e in edges:
        # Edge dict from BFSFastJson: {s, p, o, w (weight/confidence), step}
        rel = e.get('p', '')
        weight = float(e.get('w', 1.0) or 1.0)
        depth = int(e.get('step', 0) or 0)

        if relationship and rel != relationship:
            continue
        if weight < min_confidence:
            continue
        if direction == 'outgoing' and e.get('s') != start_id:
            continue
        if direction == 'incoming' and e.get('o') != start_id:
            continue

        out.append({
            'source_id': e.get('s', ''),
            'target_id': e.get('o', ''),
            'relationship': rel,
            'confidence': weight,
            'depth': depth,
        })

    out.sort(key=lambda e: (e.get('depth', 0), -float(e.get('confidence', 0))))
    return out


def ivg_find_path(id1: str, id2: str, max_hops: int = 5) -> Optional[Dict]:
    """IVG-native path finding using khop() neighborhood expansion.

    Drop-in replacement for knowledge_graph.find_path().
    Returns {'depth': N, 'edges': [...]} or None.
    """
    bridge = get_bridge()
    # Expand from id1 up to max_hops; check if id2 appears in neighborhood
    result = bridge.khop(id1, hops=max_hops)
    nodes = set(result.get('nodes', []))
    edges = result.get('edges', [])

    if id2 not in nodes:
        return None

    # Build adjacency for path reconstruction
    adj: Dict[str, List[Dict]] = {}
    for e in edges:
        src = e.get('s', '')
        if src not in adj:
            adj[src] = []
        adj[src].append(e)

    # BFS path reconstruction using already-fetched edges
    from collections import deque
    queue: deque = deque([[id1]])
    visited = {id1}

    while queue:
        path = queue.popleft()
        current = path[-1]
        for e in adj.get(current, []):
            nxt = e.get('o', '')
            edge_info = {
                'source_id': e.get('s', ''),
                'target_id': nxt,
                'relationship': e.get('p', ''),
                'confidence': float(e.get('w', 1.0) or 1.0),
            }
            new_path = path + [edge_info]
            if nxt == id2:
                edge_path = [x for x in new_path if isinstance(x, dict)]
                return {'depth': len(edge_path), 'edges': edge_path}
            if nxt not in visited:
                visited.add(nxt)
                queue.append(new_path)

    return None


# ---------------------------------------------------------------------------
# CLI — backfill and verify
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import logging

    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(
        description="IVG Bridge — mirror drift-memory into Graph_KG for native traversal"
    )
    sub = parser.add_subparsers(dest="cmd")

    bp = sub.add_parser("backfill", help="Mirror all existing drift-memory data into Graph_KG")
    bp.add_argument("--schema", default="drift")
    bp.add_argument("--batch-size", type=int, default=200)
    bp.add_argument("--dry-run", action="store_true")

    vp = sub.add_parser("verify", help="Run a khop query to verify the bridge is working")
    vp.add_argument("--schema", default="drift")
    vp.add_argument("--id", dest="memory_id", help="Drift memory ID to query")
    vp.add_argument("--hops", type=int, default=2)

    sp = sub.add_parser("stats", help="Show Graph_KG drift node/edge counts")
    sp.add_argument("--schema", default="drift")

    args = parser.parse_args()

    if args.cmd == "backfill":
        bridge = IVGBridge(schema=args.schema)
        stats = bridge.backfill(batch_size=args.batch_size, dry_run=args.dry_run)
        print(f"\nBackfill complete: {stats}")

    elif args.cmd == "verify":
        bridge = IVGBridge(schema=args.schema)
        if args.memory_id:
            result = bridge.khop(args.memory_id, hops=args.hops)
            print(f"khop({args.memory_id!r}, hops={args.hops}): "
                  f"{len(result['nodes'])} nodes, {len(result['edges'])} edges")
            for e in result['edges'][:10]:
                print(f"  {e.get('s')} --[{e.get('p')}]--> {e.get('o')}  (step={e.get('step')})")
        else:
            conn = _get_conn()
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM Graph_KG.nodes WHERE node_id LIKE 'drift:%'")
            n_nodes = cur.fetchone()[0]
            cur.execute(
                "SELECT COUNT(*) FROM Graph_KG.rdf_edges e "
                "JOIN Graph_KG.nodes n ON n.node_id = e.s WHERE n.node_id LIKE 'drift:%'"
            )
            n_edges = cur.fetchone()[0]
            conn.close()
            print(f"Graph_KG drift nodes: {n_nodes}")
            print(f"Graph_KG drift edges: {n_edges}")

    elif args.cmd == "stats":
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM Graph_KG.nodes WHERE node_id LIKE 'drift:%'")
        print(f"drift: nodes in Graph_KG: {cur.fetchone()[0]}")
        cur.execute(
            "SELECT COUNT(*) FROM Graph_KG.rdf_edges e "
            "JOIN Graph_KG.nodes n ON n.node_id = e.s WHERE n.node_id LIKE 'drift:%'"
        )
        print(f"drift: edges in Graph_KG: {cur.fetchone()[0]}")
        conn.close()

    else:
        parser.print_help()
