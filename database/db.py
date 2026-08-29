"""
Database Abstraction Layer — IRIS backend for drift-memory.

Drop-in replacement for cogmem's PostgreSQL/pgvector MemoryDB.
Uses iris-vector-graph + iris-devtester. Schema: DriftMem_<agent>.

Tables live in the IRIS USER namespace under the prefix DriftMem_<SCHEMA>_*
  e.g. schema='drift' → DriftMem_drift_memories, DriftMem_drift_text_embeddings, ...

Embeddings use Graph_KG.kg_NodeEmbeddings via iris-vector-graph for
vector search (Python cosine fallback when stored proc not installed).

Usage:
    from database.db import MemoryDB
    db = MemoryDB('drift')
    memory = db.get_memory('abc12345')
"""

import atexit
import fcntl
import json
import math
import os
import signal
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# IRIS connection — env-var driven, defaults to los-iris (localhost:11972).
# Uses a persistent module-level connection reused across calls in a process.
# conn.close() was observed to hang on some intersystems-irispython/macOS
# combos, so we close via atexit with a SIGALRM timeout to avoid blocking
# while still releasing the IRIS license unit on process exit.
# ---------------------------------------------------------------------------

_persistent_conn = None


def _close_on_exit():
    global _persistent_conn
    if _persistent_conn is None:
        return
    conn = _persistent_conn
    _persistent_conn = None
    # Use SIGALRM to cap the close() call at 2s — prevents zombie license units
    # without risking an indefinite hang at shutdown.
    def _timeout_handler(signum, frame):
        raise TimeoutError("conn.close() timed out")
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(2)
    try:
        conn.close()
    except Exception:
        pass
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


atexit.register(_close_on_exit)


def _get_connection():
    global _persistent_conn
    if _persistent_conn is not None:
        try:
            cur = _persistent_conn.cursor()
            cur.execute("SELECT 1")
            cur.fetchone()
            return _persistent_conn
        except Exception:
            _persistent_conn = None

    host     = os.environ.get("IRIS_HOST",      "localhost")
    port     = int(os.environ.get("IRIS_PORT",  "11972"))
    ns       = os.environ.get("IRIS_NAMESPACE", "USER")
    user     = os.environ.get("IRIS_USERNAME",  "SuperUser")
    password = os.environ.get("IRIS_PASSWORD",  "SYS")
    # iris.dbapi.connect() returns a Connection whose cursors are iris.dbapi.Cursor,
    # which includes the fetchall/fetchone chunked fallback for the irispython 5.3.0+
    # read-ahead buffer bug (DP-445872 / "Character stream length mismatch").
    import iris.dbapi as _dbapi
    _persistent_conn = _dbapi.connect(
        hostname=host, port=port, namespace=ns, username=user, password=password
    )
    return _persistent_conn


@contextmanager
def get_conn():
    conn = _get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ts(val) -> Optional[str]:
    """Coerce datetime/str/None → ISO string or None."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.isoformat()
    return str(val)


def _cosine_similarity(a: list, b: list) -> float:
    """Python cosine similarity between two equal-length vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


_minilm_encoder = None


def _get_minilm_encoder():
    """Lazy-load all-MiniLM-L6-v2 (384-dim). Returns None on ImportError."""
    global _minilm_encoder
    if _minilm_encoder is None:
        try:
            from sentence_transformers import SentenceTransformer
            _minilm_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            _minilm_encoder = False  # sentinel: import failed, don't retry
    return _minilm_encoder if _minilm_encoder is not False else None


#: Vector column per embedding dimension. Widths are fixed in the DDL (`_EMBEDDINGS_SQL`)
#: and IRIS validates them, so the caller's dimension decides the column — assuming one
#: column meant a 384-dim caller hit `SQLCODE -104` on every write.
_VECTOR_COLUMN_BY_DIM = {1536: 'emb', 384: 'emb_384'}

_vector_fallback_warned = False


def _warn_vector_fallback(exc: Exception):
    """Warn once per process when a vector-column write falls back to text-only.

    The fallback is legitimate on instances without the emb/emb_384 VECTOR
    columns, but a *permanent* failure here silently NULLs every vector and
    VECTOR_COSINE then scores all rows 0.0 — i.e. semantic recall dies quietly.
    A bare `except` once hid exactly that for 6,493 rows. Keep this audible.
    """
    global _vector_fallback_warned
    if _vector_fallback_warned:
        return
    _vector_fallback_warned = True
    import sys as _sys
    print(
        f"[drift-memory] WARNING: vector-column write failed, falling back to "
        f"text-only embedding storage. Semantic search will not use "
        f"VECTOR_COSINE until this is fixed. Cause: {type(exc).__name__}: {exc}",
        file=_sys.stderr,
    )


def _encode_384(text: str) -> Optional[list]:
    """Encode text to 384-dim using all-MiniLM-L6-v2. Returns None on failure."""
    enc = _get_minilm_encoder()
    if enc is None or not text:
        return None
    try:
        vec = enc.encode(text, show_progress_bar=False)
        return vec.tolist()
    except Exception:
        return None


def _json_dumps(v) -> str:
    if v is None:
        return '{}'
    if isinstance(v, str):
        return v
    return json.dumps(v)


def _json_loads(v):
    if v is None:
        return {}
    if isinstance(v, (dict, list)):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return {}
    return {}


def _list_to_csv(lst) -> str:
    """Encode Python list as comma-separated string for IRIS storage."""
    if not lst:
        return ''
    return ','.join(str(x) for x in lst)


def _csv_to_list(s) -> list:
    """Decode comma-separated string back to list."""
    if not s:
        return []
    return [x for x in s.split(',') if x]


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS DriftMem_{schema}_memories (
    id VARCHAR(32) NOT NULL,
    type_ VARCHAR(10) NOT NULL,
    content LONGVARCHAR NOT NULL,
    created VARCHAR(64),
    last_recalled VARCHAR(64),
    recall_count INTEGER DEFAULT 0,
    sessions_since_recall INTEGER DEFAULT 0,
    emotional_weight DOUBLE DEFAULT 0.5,
    tags VARCHAR(2000),
    event_time VARCHAR(64),
    entities LONGVARCHAR,
    caused_by VARCHAR(2000),
    leads_to VARCHAR(2000),
    source_ LONGVARCHAR,
    retrieval_outcomes LONGVARCHAR,
    retrieval_success_rate DOUBLE,
    topic_context VARCHAR(2000),
    contact_context VARCHAR(2000),
    platform_context VARCHAR(2000),
    extra_metadata LONGVARCHAR,
    importance DOUBLE DEFAULT 0.5,
    freshness DOUBLE DEFAULT 1.0,
    memory_tier VARCHAR(20),
    q_value DOUBLE DEFAULT 0.5,
    PRIMARY KEY (id)
)
"""

_EMBEDDINGS_SQL = """
CREATE TABLE IF NOT EXISTS DriftMem_{schema}_text_embeddings (
    memory_id VARCHAR(32) NOT NULL,
    embedding LONGVARCHAR NOT NULL,
    preview VARCHAR(500),
    model VARCHAR(100),
    indexed_at VARCHAR(64),
    emb VECTOR(DOUBLE, 1536),
    emb_384 VECTOR(DOUBLE, 384),
    PRIMARY KEY (memory_id)
)
"""

# Note: IRIS does not support explicit HNSW index creation via SQL DDL in this version.
# VECTOR_COSINE queries use IRIS's built-in approximate nearest neighbor scan which is
# significantly faster than full-table Python cosine (server-side vectorized ops).
_EMBEDDINGS_HNSW_IDX = None  # placeholder — not used

_EDGES_SQL = """
CREATE TABLE IF NOT EXISTS DriftMem_{schema}_edges (
    id1 VARCHAR(32) NOT NULL,
    id2 VARCHAR(32) NOT NULL,
    belief DOUBLE DEFAULT 0,
    first_formed VARCHAR(64),
    last_updated VARCHAR(64),
    platform_context LONGVARCHAR,
    activity_context LONGVARCHAR,
    topic_context LONGVARCHAR,
    PRIMARY KEY (id1, id2)
)
"""

_SOMATIC_SQL = """
CREATE TABLE IF NOT EXISTS DriftMem_{schema}_somatic_markers (
    situation_hash VARCHAR(128) NOT NULL,
    features_text LONGVARCHAR,
    embedding LONGVARCHAR,
    valence DOUBLE DEFAULT 0.0,
    confidence DOUBLE DEFAULT 0.0,
    count_ INTEGER DEFAULT 0,
    category VARCHAR(50),
    last_activated VARCHAR(64),
    emb VECTOR(DOUBLE, 1536),
    PRIMARY KEY (situation_hash)
)
"""

_SOMATIC_HNSW_IDX = None  # placeholder — not used

_SESSIONS_SQL = """
CREATE TABLE IF NOT EXISTS DriftMem_{schema}_sessions (
    id INTEGER NOT NULL,
    started VARCHAR(64),
    ended VARCHAR(64),
    is_active INTEGER DEFAULT 1,
    PRIMARY KEY (id)
)
"""

_LESSONS_SQL = """
CREATE TABLE IF NOT EXISTS DriftMem_{schema}_lessons (
    id VARCHAR(64) NOT NULL,
    category VARCHAR(50),
    lesson LONGVARCHAR NOT NULL,
    evidence LONGVARCHAR,
    source_ VARCHAR(50),
    confidence DOUBLE DEFAULT 0.7,
    created VARCHAR(64),
    applied_count INTEGER DEFAULT 0,
    last_applied VARCHAR(64),
    superseded_by VARCHAR(64),
    PRIMARY KEY (id)
)
"""

_REJECTIONS_SQL = """
CREATE TABLE IF NOT EXISTS DriftMem_{schema}_rejections (
    id INTEGER NOT NULL,
    timestamp_ VARCHAR(64),
    category VARCHAR(30),
    reason LONGVARCHAR,
    target_ VARCHAR(500),
    context_ LONGVARCHAR,
    tags VARCHAR(2000),
    source_ VARCHAR(50),
    PRIMARY KEY (id)
)
"""

_VITALS_SQL = """
CREATE TABLE IF NOT EXISTS DriftMem_{schema}_vitals (
    id INTEGER NOT NULL,
    timestamp_ VARCHAR(64),
    metrics LONGVARCHAR,
    PRIMARY KEY (id)
)
"""

_ATTESTATIONS_SQL = """
CREATE TABLE IF NOT EXISTS DriftMem_{schema}_attestations (
    id INTEGER NOT NULL,
    timestamp_ VARCHAR(64),
    type_ VARCHAR(50),
    hash_ VARCHAR(128),
    data LONGVARCHAR,
    PRIMARY KEY (id)
)
"""

_KV_SQL = """
CREATE TABLE IF NOT EXISTS DriftMem_{schema}_kv (
    key_ VARCHAR(200) NOT NULL,
    value_ LONGVARCHAR,
    updated_at VARCHAR(64),
    PRIMARY KEY (key_)
)
"""

_TYPED_EDGES_SQL = """
CREATE TABLE IF NOT EXISTS DriftMem_{schema}_typed_edges (
    source_id VARCHAR(32) NOT NULL,
    target_id VARCHAR(32) NOT NULL,
    relationship VARCHAR(50) NOT NULL,
    confidence DOUBLE DEFAULT 0.8,
    evidence LONGVARCHAR,
    auto_extracted INTEGER DEFAULT 0,
    created VARCHAR(64),
    PRIMARY KEY (source_id, target_id, relationship)
)
"""

_ALL_SCHEMAS = [
    _SCHEMA_SQL, _EMBEDDINGS_SQL, _EDGES_SQL, _SOMATIC_SQL,
    _SESSIONS_SQL, _LESSONS_SQL, _REJECTIONS_SQL, _VITALS_SQL,
    _ATTESTATIONS_SQL, _KV_SQL, _TYPED_EDGES_SQL,
]

_HNSW_INDEXES = []  # IRIS does not support explicit HNSW DDL in this version

# ALTER TABLE statements to add emb column to existing tables (idempotent — fails silently if exists)
_MIGRATION_SQL = [
    "ALTER TABLE DriftMem_{schema}_text_embeddings ADD emb VECTOR(DOUBLE, 1536)",
    "ALTER TABLE DriftMem_{schema}_somatic_markers ADD emb VECTOR(DOUBLE, 1536)",
    # 384-dim all-MiniLM-L6-v2 column — used for IVG bridge + local-only search
    "ALTER TABLE DriftMem_{schema}_text_embeddings ADD emb_384 VECTOR(DOUBLE, 384)",
]

_initialized_schemas: set = set()


def _ensure_schema(conn, schema: str):
    """Create all DriftMem tables for this schema if they don't exist."""
    if schema in _initialized_schemas:
        return
    cursor = conn.cursor()
    for ddl in _ALL_SCHEMAS:
        sql = ddl.replace('{schema}', schema)
        try:
            cursor.execute(sql)
        except Exception:
            pass  # Already exists or non-fatal — IRIS DDL is idempotent in practice
    # Migrate existing tables: add emb VECTOR column if not present
    for ddl in _MIGRATION_SQL:
        sql = ddl.replace('{schema}', schema)
        try:
            cursor.execute(sql)
        except Exception:
            pass  # Column already exists — silently ignore
    # Create HNSW indexes — may be slow first time, silent if already exist
    for ddl in _HNSW_INDEXES:
        sql = ddl.replace('{schema}', schema)
        try:
            cursor.execute(sql)
        except Exception:
            pass
    try:
        conn.commit()
    except Exception:
        pass
    _initialized_schemas.add(schema)


def _next_int_id(conn, table: str) -> int:
    """Get next integer ID for auto-increment tables."""
    cursor = conn.cursor()
    cursor.execute(f"SELECT MAX(id) FROM {table}")
    row = cursor.fetchone()
    val = row[0] if row and row[0] is not None else 0
    return int(val) + 1


# ---------------------------------------------------------------------------
# Row → dict helpers
# ---------------------------------------------------------------------------

def _memory_row_to_dict(row, columns) -> dict:
    d = dict(zip(columns, row))
    # Normalise field names (IRIS uppercases column names)
    result = {}
    col_map = {
        'TYPE_': 'type', 'SOURCE_': 'source', 'Q_VALUE': 'q_value',
    }
    for k, v in d.items():
        k_lower = k.lower()
        mapped = col_map.get(k.upper(), k_lower)
        result[mapped] = v

    # Deserialise JSON fields
    for field in ('entities', 'extra_metadata', 'retrieval_outcomes', 'source'):
        if field in result and isinstance(result[field], str):
            result[field] = _json_loads(result[field])

    # Deserialise array fields
    for field in ('tags', 'caused_by', 'leads_to', 'topic_context',
                  'contact_context', 'platform_context'):
        if field in result and isinstance(result[field], str):
            result[field] = _csv_to_list(result[field])

    # Parse timestamps to datetime objects (best effort)
    for field in ('created', 'last_recalled', 'event_time'):
        if field in result and result[field]:
            try:
                result[field] = datetime.fromisoformat(result[field])
            except Exception:
                pass

    return result


def _cursor_columns(cursor) -> list:
    """Get lowercase column names from cursor description."""
    if cursor.description:
        return [d[0].lower() for d in cursor.description]
    return []


# ---------------------------------------------------------------------------
# psycopg2 compatibility shim — lets raw SQL callers in semantic_search.py
# use db._conn() with cursor_factory=psycopg2.extras.RealDictCursor style
# ---------------------------------------------------------------------------

class _IRISCursorCompat:
    """Wraps IRIS cursor to mimic psycopg2 RealDictCursor when needed.

    Handles:
    - cursor_factory kwarg (ignored, always returns dict rows when _dict_mode=True)
    - %s → ? param substitution (PostgreSQL → IRIS)
    - __enter__ / __exit__ for 'with conn.cursor() as cur:' pattern
    - fetchall() returns list[dict] in dict mode
    """
    def __init__(self, cursor, dict_mode: bool = False):
        self._cur = cursor
        self._dict_mode = dict_mode
        self._columns: list = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    @property
    def description(self):
        return self._cur.description

    def execute(self, sql: str, params=None):
        # Translate %s → ? for IRIS
        sql = sql.replace('%s', '?')
        if params is None:
            self._cur.execute(sql)
        else:
            self._cur.execute(sql, list(params) if not isinstance(params, list) else params)
        if self._cur.description:
            self._columns = [d[0].lower() for d in self._cur.description]

    def fetchone(self):
        row = self._cur.fetchone()
        if row is None:
            return None
        if self._dict_mode:
            return dict(zip(self._columns, row))
        return row

    def fetchall(self):
        rows = self._cur.fetchall()
        if self._dict_mode:
            return [dict(zip(self._columns, row)) for row in rows]
        return rows

    def __iter__(self):
        for row in self.fetchall():
            yield row


class _IRISConnCompat:
    """Wraps IRIS connection to mimic psycopg2 connection interface.

    Supports: conn.cursor(), conn.cursor(cursor_factory=RealDictCursor),
              with conn.cursor() as cur:, conn.commit(), conn.rollback()
    """
    def __init__(self, conn):
        self._conn = conn

    def cursor(self, cursor_factory=None):
        raw = self._conn.cursor()
        # Treat any cursor_factory as dict mode (drift-memory only uses RealDictCursor)
        dict_mode = cursor_factory is not None
        return _IRISCursorCompat(raw, dict_mode=dict_mode)

    def commit(self):
        self._conn.commit()

    def rollback(self):
        try:
            self._conn.rollback()
        except Exception:
            pass

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# ---------------------------------------------------------------------------
# MemoryDB — drop-in replacement for cogmem's PostgreSQL MemoryDB
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Write-through file log — dual-write architecture (training wheels)
#
# Every write to IRIS also appends a JSONL record to a flat file under
# DRIFT_FILE_LOG_DIR. This gives us a crash-proof audit trail: if IRIS is
# wiped, replay the log to reconstruct. Once IRIS proves stable for 60 days,
# remove this layer (the flag is in db_adapter.py / env var below).
#
# Format: one JSON object per line, fields: op, schema, table, ts, **payload
# File per table: <dir>/<schema>_memories.jsonl, <schema>_edges.jsonl, etc.
#
# Enable:  export DRIFT_FILE_LOG=1  (or set in .env)
# Disable: unset DRIFT_FILE_LOG     (or set DRIFT_FILE_LOG=0)
# Dir:     DRIFT_FILE_LOG_DIR (default: ~/.local/share/drift-memory-log/)
# ---------------------------------------------------------------------------

_FILE_LOG_ENABLED = os.environ.get("DRIFT_FILE_LOG", "1") not in ("0", "false", "")
_FILE_LOG_DIR = Path(os.environ.get(
    "DRIFT_FILE_LOG_DIR",
    Path.home() / ".local" / "share" / "drift-memory-log"
))

# KV keys to exclude from the file log — ephemeral runtime state that changes
# on every memory store call. No recovery value; logging them bloats kv.jsonl
# to GB-scale. Only non-ephemeral KV writes are worth persisting.
_KV_LOG_SKIP_PREFIXES = (".cognitive_", ".affect_", ".source_reliability")


class _FileLog:
    """Append-only JSONL write-through log, one file per (schema, table)."""

    def __init__(self, schema: str):
        self._schema = schema
        self._dir = _FILE_LOG_DIR / schema
        if _FILE_LOG_ENABLED:
            self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, table: str) -> Path:
        return self._dir / f"{table}.jsonl"

    def write(self, op: str, table: str, payload: dict):
        """Append one record. Silently no-ops if file logging is disabled or fails."""
        if not _FILE_LOG_ENABLED:
            return
        record = {"op": op, "schema": self._schema, "table": table,
                  "ts": datetime.now(timezone.utc).isoformat(), **payload}
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        path = self._path(table)
        try:
            with open(path, "a", encoding="utf-8") as f:
                # Advisory lock so concurrent writes don't interleave lines
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    f.write(line)
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
        except Exception:
            pass  # Never block the caller — log failure is silent


class MemoryDB:
    """
    IRIS-backed memory database. Schema-isolated per agent.

    Tables: DriftMem_<schema>_memories, _text_embeddings, _edges,
            _somatic_markers, _sessions, _lessons, _rejections,
            _vitals, _attestations, _kv

    Write-through: every mutation also appends to a JSONL file log under
    ~/.local/share/drift-memory-log/<schema>/ (controlled by DRIFT_FILE_LOG env var).
    """

    def __init__(self, schema: str = 'drift', config=None):
        self.schema = schema
        self._config = config  # unused, kept for API compat
        self._flog = _FileLog(schema)

    def _t(self, name: str) -> str:
        return f"DriftMem_{self.schema}_{name}"

    def _table(self, name: str) -> str:
        """Alias for _t() — cogmem raw SQL callers use db._table('memories')."""
        return self._t(name)

    @contextmanager
    def _conn(self):
        """Yield an IRIS connection wrapped with psycopg2 compat shim."""
        with get_conn() as conn:
            _ensure_schema(conn, self.schema)
            yield _IRISConnCompat(conn)

    # -----------------------------------------------------------------------
    # MEMORIES — CRUD
    # -----------------------------------------------------------------------

    def get_memory(self, memory_id: str) -> Optional[dict]:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(f"SELECT * FROM {self._t('memories')} WHERE id = ?", [memory_id])
            row = cur.fetchone()
            if not row:
                return None
            return _memory_row_to_dict(row, _cursor_columns(cur))

    def insert_memory(self, memory_id: str, type_: str, content: str,
                      tags: list = None, entities: dict = None,
                      emotional_weight: float = 0.5,
                      topic_context: list = None, contact_context: list = None,
                      platform_context: list = None, extra_metadata: dict = None,
                      created: datetime = None, source: str = None, **kwargs) -> dict:
        now = _ts(created) or _now_iso()
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(f"""
                INSERT INTO {self._t('memories')}
                (id, type_, content, created, emotional_weight, tags, entities,
                 topic_context, contact_context, platform_context, extra_metadata, source_)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                memory_id, type_, content, now, emotional_weight,
                _list_to_csv(tags),
                _json_dumps(entities),
                _list_to_csv(topic_context),
                _list_to_csv(contact_context),
                _list_to_csv(platform_context),
                _json_dumps(extra_metadata),
                source,
            ])
        self._flog.write("insert", "memories", {
            "id": memory_id, "type_": type_, "content": content, "created": now,
            "emotional_weight": emotional_weight, "tags": tags, "entities": entities,
            "topic_context": topic_context, "contact_context": contact_context,
            "platform_context": platform_context, "extra_metadata": extra_metadata,
            "source_": source,
        })
        # Mirror into Graph_KG for native IVG traversal (khop, Cypher, PPR)
        try:
            from ivg_bridge import get_bridge
            get_bridge(self.schema).register_node(memory_id, type_=type_, content=content)
        except Exception:
            pass  # Bridge is best-effort — never block memory writes
        return self.get_memory(memory_id) or {'id': memory_id}

    def update_memory(self, memory_id: str, **fields) -> Optional[dict]:
        if not fields:
            return self.get_memory(memory_id)

        set_parts = []
        values = []
        json_fields = {'entities', 'retrieval_outcomes', 'source', 'extra_metadata'}
        array_fields = {'tags', 'caused_by', 'leads_to', 'topic_context',
                        'contact_context', 'platform_context'}
        col_remap = {'type': 'type_', 'source': 'source_'}

        for key, val in fields.items():
            col = col_remap.get(key, key)
            if key in json_fields:
                set_parts.append(f"{col} = ?")
                values.append(_json_dumps(val))
            elif key in array_fields:
                set_parts.append(f"{col} = ?")
                values.append(_list_to_csv(val) if isinstance(val, list) else val)
            else:
                set_parts.append(f"{col} = ?")
                values.append(_ts(val) if isinstance(val, datetime) else val)

        values.append(memory_id)
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                f"UPDATE {self._t('memories')} SET {', '.join(set_parts)} WHERE id = ?",
                values
            )
        self._flog.write("update", "memories", {"id": memory_id, **fields})
        return self.get_memory(memory_id)

    def delete_memory(self, memory_id: str) -> bool:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(f"DELETE FROM {self._t('memories')} WHERE id = ?", [memory_id])
        self._flog.write("delete", "memories", {"id": memory_id})
        return True  # IRIS doesn't expose rowcount reliably here

    def recall_memory(self, memory_id: str, session_id: int = None,
                      source: str = 'manual') -> Optional[dict]:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(f"""
                UPDATE {self._t('memories')}
                SET recall_count = recall_count + 1,
                    last_recalled = ?,
                    sessions_since_recall = 0
                WHERE id = ?
            """, [_now_iso(), memory_id])
        return self.get_memory(memory_id)

    def list_memories(self, type_: str = None, tags: list = None,
                      limit: int = 100, offset: int = 0) -> list:
        conditions = []
        where_values = []
        if type_:
            conditions.append("type_ = ?")
            where_values.append(type_)
        # Tag filter: substring match on CSV tags field (IRIS SQL uses LIKE, not %CONTAINS)
        if tags:
            for tag in tags:
                conditions.append("tags LIKE ?")
                where_values.append(f"%{tag}%")

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        # SELECT TOP ? must be the first parameter — WHERE params follow
        values = [limit + offset] + where_values

        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                f"SELECT TOP ? * FROM {self._t('memories')} {where} ORDER BY created DESC",
                values
            )
            rows = cur.fetchall()
            cols = _cursor_columns(cur)
            results = [_memory_row_to_dict(r, cols) for r in rows]
            return results[offset:]

    def count_memories(self, type_: str = None) -> int:
        with self._conn() as conn:
            cur = conn.cursor()
            if type_:
                cur.execute(f"SELECT COUNT(*) FROM {self._t('memories')} WHERE type_ = ?", [type_])
            else:
                cur.execute(f"SELECT COUNT(*) FROM {self._t('memories')}")
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def search_fulltext(self, query: str, limit: int = 10) -> list:
        """Full-text search via LIKE on content (IRIS doesn't have tsvector)."""
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                f"SELECT TOP ? * FROM {self._t('memories')} WHERE content LIKE ?",
                [limit, f"%{query}%"]
            )
            rows = cur.fetchall()
            cols = _cursor_columns(cur)
            return [_memory_row_to_dict(r, cols) for r in rows]

    def find_by_entity(self, entity_type: str, entity_name: str, limit: int = 50) -> list:
        """Find memories whose entities JSON contains the entity name."""
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                f"SELECT TOP ? * FROM {self._t('memories')} WHERE entities LIKE ?",
                [limit, f"%{entity_name}%"]
            )
            rows = cur.fetchall()
            cols = _cursor_columns(cur)
            return [_memory_row_to_dict(r, cols) for r in rows]

    # -----------------------------------------------------------------------
    # Q-VALUES
    # -----------------------------------------------------------------------

    def get_q_values(self, memory_ids: list) -> dict:
        if not memory_ids:
            return {}
        result = {mid: 0.5 for mid in memory_ids}
        with self._conn() as conn:
            cur = conn.cursor()
            for mid in memory_ids:
                cur.execute(
                    f"SELECT q_value FROM {self._t('memories')} WHERE id = ?", [mid]
                )
                row = cur.fetchone()
                if row and row[0] is not None:
                    result[mid] = float(row[0])
        return result

    # -----------------------------------------------------------------------
    # EMBEDDINGS — vector search
    # -----------------------------------------------------------------------

    def store_embedding(self, memory_id: str, embedding: list,
                        preview: str = '', model: str = 'unknown'):
        """Upsert the text embedding plus whichever VECTOR columns the values fit.

        Both vector columns are FIXED width (`emb` VECTOR(DOUBLE, 1536), `emb_384`
        VECTOR(DOUBLE, 384)) and IRIS enforces it — a 384-dim value into `emb` raises
        `SQLCODE -104 … failed validation`. So the caller's `embedding` is routed to the
        column matching its dimension rather than assumed to be the 1536-dim one: every
        consumer in LOS is `all-MiniLM-L6-v2` at 384 dims, and those writes were failing.

        The columns are also written INDEPENDENTLY. Previously one `try` wrote both and a
        single `except` fell back to text-only, so one bad width discarded the other
        column's perfectly valid value: a 384-dim caller lost `emb_384` as well as `emb`,
        landed with no vectors at all, and could not be found by semantic search
        afterwards. Whatever fits is now kept.
        """
        emb_str = json.dumps(embedding)
        preview_str = preview[:500] if preview else ''
        now = _now_iso()

        # Route the caller's vector to the column of its width. An unrecognised dimension
        # gets no vector column — better than a guaranteed -104 — but the text `embedding`
        # column still holds it, so nothing is lost outright.
        vectors = {}
        if len(embedding) in _VECTOR_COLUMN_BY_DIM:
            vectors[_VECTOR_COLUMN_BY_DIM[len(embedding)]] = emb_str

        # 384-dim all-MiniLM embedding of the preview text (best-effort). Independent of the
        # caller's dimension: for a 1536-dim caller this is the only thing populating
        # emb_384, and for a 384-dim caller it simply agrees with the routed value.
        emb_384_vec = _encode_384(preview_str) if preview_str else None
        if emb_384_vec:
            vectors.setdefault('emb_384', json.dumps(emb_384_vec))

        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                f"SELECT memory_id FROM {self._t('text_embeddings')} WHERE memory_id = ?",
                [memory_id]
            )
            exists = cur.fetchone()
            if exists:
                cur.execute(f"""
                    UPDATE {self._t('text_embeddings')}
                    SET embedding = ?, preview = ?, model = ?, indexed_at = ?
                    WHERE memory_id = ?
                """, [emb_str, preview_str, model, now, memory_id])
            else:
                cur.execute(f"""
                    INSERT INTO {self._t('text_embeddings')}
                    (memory_id, embedding, preview, model, indexed_at)
                    VALUES (?, ?, ?, ?, ?)
                """, [memory_id, emb_str, preview_str, model, now])

            # One statement per vector column, so a width mismatch or a missing column on
            # an older instance costs only that column.
            for column, value in vectors.items():
                try:
                    cur.execute(f"""
                        UPDATE {self._t('text_embeddings')}
                        SET {column} = TO_VECTOR(?, DOUBLE)
                        WHERE memory_id = ?
                    """, [value, memory_id])
                except Exception as exc:
                    _warn_vector_fallback(exc)
        self._flog.write("upsert", "text_embeddings", {
            "memory_id": memory_id, "preview": preview[:500] if preview else '',
            "model": model, "embedding_len": len(embedding),
            # Store embedding itself — large but needed for full reconstruction
            "embedding": embedding,
        })

    def search_embeddings(self, query_embedding: list, limit: int = 5,
                          type_filter: str = None) -> list:
        """Cosine similarity search using IRIS VECTOR_COSINE.

        Prefers 384-dim emb_384 column (all-MiniLM-L6-v2, local, no external deps)
        when query is 384-dim. Falls back to 1536-dim emb column, then Python scan.
        """
        dim = len(query_embedding)
        query_vec_str = json.dumps(query_embedding)
        # Choose vector column by query dimension
        emb_col = "emb_384" if dim == 384 else "emb"
        emb_not_null = f"{emb_col} IS NOT NULL"

        with self._conn() as conn:
            cur = conn.cursor()
            try:
                # Fast path: VECTOR_COSINE — no JOIN (avoids %qaqpre singletonGroups
                # compile crash on migrated IRIS instances; type_filter applied in
                # Python after fetching memory details).
                cur.execute(f"""
                    SELECT TOP ? e.memory_id,
                           VECTOR_COSINE(e.{emb_col}, TO_VECTOR(?, DOUBLE)) AS similarity,
                           e.preview
                    FROM {self._t('text_embeddings')} e
                    WHERE e.{emb_not_null}
                    ORDER BY similarity DESC
                """, [limit * 4 if type_filter else limit, query_vec_str])

                rows = cur.fetchall()
                if rows:
                    results = []
                    for row in rows:
                        memory_id, similarity, preview = row[0], row[1], row[2]
                        try:
                            similarity = float(similarity)
                        except (TypeError, ValueError):
                            similarity = 0.0
                        mem = self.get_memory(memory_id) or {'id': memory_id}
                        if type_filter and mem.get('type_') != type_filter:
                            continue
                        mem['preview'] = preview or ''
                        mem['similarity'] = similarity
                        results.append(mem)
                        if len(results) >= limit:
                            break
                    if results:
                        return results

                # emb column empty — fall through to Python scan
            except Exception:
                pass  # VECTOR_COSINE failed — fall through to Python fallback

            # Python fallback: full-table scan — no JOIN (same %qaqpre guard);
            # fetch embeddings then memories separately.
            cur.execute(f"""
                SELECT e.memory_id, e.preview, e.embedding
                FROM {self._t('text_embeddings')} e
            """)
            emb_rows = cur.fetchall()
            if type_filter:
                # Fetch all memory IDs matching type in a second query (no JOIN)
                cur.execute(f"""
                    SELECT id FROM {self._t('memories')} WHERE type_ = ?
                """, [type_filter])
                allowed_ids = {r[0] for r in cur.fetchall()}
            else:
                allowed_ids = None

        scored = []
        for row in emb_rows:
            memory_id, preview, emb_raw = row[0], row[1], row[2]
            if allowed_ids is not None and memory_id not in allowed_ids:
                continue
            try:
                emb = json.loads(emb_raw) if isinstance(emb_raw, str) else emb_raw
            except Exception:
                continue
            sim = _cosine_similarity(query_embedding, emb)
            mem = self.get_memory(memory_id) or {'id': memory_id}
            mem['preview'] = preview or ''
            mem['similarity'] = sim
            scored.append(mem)

        scored.sort(key=lambda x: x['similarity'], reverse=True)
        return scored[:limit]

    # -----------------------------------------------------------------------
    # EDGES (co-occurrence graph)
    # -----------------------------------------------------------------------

    def _canon(self, id1, id2):
        return (id1, id2) if id1 < id2 else (id2, id1)

    def get_edge(self, id1: str, id2: str) -> Optional[dict]:
        a, b = self._canon(id1, id2)
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                f"SELECT * FROM {self._t('edges')} WHERE id1 = ? AND id2 = ?", [a, b]
            )
            row = cur.fetchone()
            if not row:
                return None
            cols = _cursor_columns(cur)
            d = dict(zip(cols, row))
            for f in ('platform_context', 'activity_context', 'topic_context'):
                if f in d:
                    d[f] = _json_loads(d[f])
            return d

    def upsert_edge(self, id1: str, id2: str, belief: float,
                    platform_context: dict = None, activity_context: dict = None,
                    topic_context: dict = None, **kwargs) -> dict:
        a, b = self._canon(id1, id2)
        existing = self.get_edge(a, b)
        now = _now_iso()
        with self._conn() as conn:
            cur = conn.cursor()
            if existing:
                cur.execute(f"""
                    UPDATE {self._t('edges')}
                    SET belief = ?, last_updated = ?,
                        platform_context = ?, activity_context = ?, topic_context = ?
                    WHERE id1 = ? AND id2 = ?
                """, [
                    belief, now,
                    _json_dumps(platform_context), _json_dumps(activity_context),
                    _json_dumps(topic_context), a, b,
                ])
            else:
                cur.execute(f"""
                    INSERT INTO {self._t('edges')}
                    (id1, id2, belief, first_formed, last_updated,
                     platform_context, activity_context, topic_context)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    a, b, belief, now, now,
                    _json_dumps(platform_context), _json_dumps(activity_context),
                    _json_dumps(topic_context),
                ])
        self._flog.write("upsert", "edges", {
            "id1": a, "id2": b, "belief": belief,
            "platform_context": platform_context,
            "activity_context": activity_context,
            "topic_context": topic_context,
        })
        if os.environ.get("DRIFT_IVG_BRIDGE", "0") == "1":
            try:
                from ivg_bridge import get_bridge
                get_bridge(self.schema).register_cooccurrence_edge(a, b, belief=belief)
            except Exception:
                pass
        return {"id1": a, "id2": b, "belief": belief}

    def add_observation(self, id1: str, id2: str, source_type: str,
                        session_id: str = None, agent: str = None,
                        platform: str = None, activity: str = None,
                        weight: float = 1.0, trust_tier: str = 'self',
                        **kwargs) -> dict:
        """Boost edge belief by weight (simplified vs Postgres FK approach)."""
        a, b = self._canon(id1, id2)
        edge = self.get_edge(a, b)
        new_belief = (edge['belief'] if edge else 0.0) + weight
        return self.upsert_edge(a, b, new_belief)

    def batch_decay_edges(self, decay_rate: float, exclude_pairs: list = None) -> int:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                f"UPDATE {self._t('edges')} SET belief = belief * ? WHERE belief > 0",
                [1.0 - decay_rate]
            )
        return 0  # rowcount not reliable

    def prune_weak_edges(self, threshold: float = 0.01) -> int:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                f"DELETE FROM {self._t('edges')} WHERE belief < ?", [threshold]
            )
        return 0

    def get_neighbors(self, memory_id: str, min_belief: float = 0.0,
                      limit: int = 50) -> list:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(f"""
                SELECT TOP ? * FROM {self._t('edges')}
                WHERE (id1 = ? OR id2 = ?) AND belief >= ?
                ORDER BY belief DESC
            """, [limit, memory_id, memory_id, min_belief])
            rows = cur.fetchall()
            cols = _cursor_columns(cur)
            results = []
            for row in rows:
                d = dict(zip(cols, row))
                for f in ('platform_context', 'activity_context', 'topic_context'):
                    if f in d:
                        d[f] = _json_loads(d[f])
                results.append(d)
            return results

    def edge_stats(self) -> dict:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(f"""
                SELECT COUNT(*), SUM(belief), AVG(belief)
                FROM {self._t('edges')}
            """)
            row = cur.fetchone()
            total, total_b, avg_b = row if row else (0, 0, 0)
            cur.execute(
                f"SELECT COUNT(*) FROM {self._t('edges')} WHERE belief >= 3.0"
            )
            strong = cur.fetchone()[0]
            return {
                'total_edges': int(total or 0),
                'total_belief': float(total_b or 0),
                'avg_belief': float(avg_b or 0),
                'strong_links': int(strong or 0),
            }

    # -----------------------------------------------------------------------
    # SESSIONS
    # -----------------------------------------------------------------------

    def start_session(self) -> int:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                f"UPDATE {self._t('sessions')} SET ended = ?, is_active = 0 WHERE is_active = 1",
                [_now_iso()]
            )
            sid = _next_int_id(conn, self._t('sessions'))
            cur.execute(
                f"INSERT INTO {self._t('sessions')} (id, started, is_active) VALUES (?, ?, 1)",
                [sid, _now_iso()]
            )
        return sid

    def end_session(self, session_id: int = None):
        with self._conn() as conn:
            cur = conn.cursor()
            if session_id:
                cur.execute(
                    f"UPDATE {self._t('sessions')} SET ended = ?, is_active = 0 WHERE id = ?",
                    [_now_iso(), session_id]
                )
            else:
                cur.execute(
                    f"UPDATE {self._t('sessions')} SET ended = ?, is_active = 0 WHERE is_active = 1",
                    [_now_iso()]
                )

    def get_active_session(self) -> Optional[int]:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                f"SELECT TOP 1 id FROM {self._t('sessions')} WHERE is_active = 1 ORDER BY started DESC"
            )
            row = cur.fetchone()
            return int(row[0]) if row else None

    # -----------------------------------------------------------------------
    # REJECTIONS
    # -----------------------------------------------------------------------

    def log_rejection(self, category: str, reason: str, target: str = None,
                      context: str = None, tags: list = None, source: str = None) -> dict:
        with self._conn() as conn:
            rid = _next_int_id(conn, self._t('rejections'))
            cur = conn.cursor()
            cur.execute(f"""
                INSERT INTO {self._t('rejections')}
                (id, timestamp_, category, reason, target_, context_, tags, source_)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [rid, _now_iso(), category, reason, target, context,
                  _list_to_csv(tags), source])
        return {'id': rid, 'category': category, 'reason': reason}

    # -----------------------------------------------------------------------
    # LESSONS
    # -----------------------------------------------------------------------

    def get_lessons(self, category: str = None) -> list:
        with self._conn() as conn:
            cur = conn.cursor()
            if category:
                cur.execute(
                    f"SELECT * FROM {self._t('lessons')} WHERE category = ?", [category]
                )
            else:
                cur.execute(f"SELECT * FROM {self._t('lessons')}")
            rows = cur.fetchall()
            cols = _cursor_columns(cur)
            results = []
            for row in rows:
                d = dict(zip(cols, row))
                d['id'] = d.pop('id', d.get('id'))
                results.append(d)
            return results

    def add_lesson(self, lesson_id: str, category: str, lesson: str,
                   evidence: str = None, source: str = 'manual',
                   confidence: float = 0.7) -> dict:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                f"SELECT id FROM {self._t('lessons')} WHERE id = ?", [lesson_id]
            )
            exists = cur.fetchone()
            if exists:
                cur.execute(f"""
                    UPDATE {self._t('lessons')}
                    SET lesson = ?, evidence = ?, confidence = ?
                    WHERE id = ?
                """, [lesson, evidence, confidence, lesson_id])
            else:
                cur.execute(f"""
                    INSERT INTO {self._t('lessons')}
                    (id, category, lesson, evidence, source_, confidence, created)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, [lesson_id, category, lesson, evidence, source, confidence, _now_iso()])
        self._flog.write("upsert", "lessons", {
            "id": lesson_id, "category": category, "lesson": lesson,
            "evidence": evidence, "source": source, "confidence": confidence,
        })
        return {'id': lesson_id, 'lesson': lesson, 'category': category}

    # -----------------------------------------------------------------------
    # VITALS
    # -----------------------------------------------------------------------

    def record_vitals(self, metrics: dict):
        with self._conn() as conn:
            vid = _next_int_id(conn, self._t('vitals'))
            cur = conn.cursor()
            cur.execute(
                f"INSERT INTO {self._t('vitals')} (id, timestamp_, metrics) VALUES (?, ?, ?)",
                [vid, _now_iso(), _json_dumps(metrics)]
            )

    # -----------------------------------------------------------------------
    # ATTESTATIONS
    # -----------------------------------------------------------------------

    def store_attestation(self, type_: str, hash_: str, data: dict):
        with self._conn() as conn:
            aid = _next_int_id(conn, self._t('attestations'))
            cur = conn.cursor()
            cur.execute(f"""
                INSERT INTO {self._t('attestations')} (id, timestamp_, type_, hash_, data)
                VALUES (?, ?, ?, ?, ?)
            """, [aid, _now_iso(), type_, hash_, _json_dumps(data)])

    # -----------------------------------------------------------------------
    # SOMATIC MARKERS
    # -----------------------------------------------------------------------

    def upsert_somatic_marker(self, situation_hash: str, features_text: str,
                               embedding: list = None, valence: float = 0.0,
                               confidence: float = 0.0, count: int = 0,
                               category: str = 'general', last_activated: str = None):
        emb_str = json.dumps(embedding) if embedding else None
        emb_vec_str = json.dumps(embedding) if embedding else None
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                f"SELECT situation_hash FROM {self._t('somatic_markers')} WHERE situation_hash = ?",
                [situation_hash]
            )
            exists = cur.fetchone()
            if exists:
                if emb_vec_str:
                    try:
                        cur.execute(f"""
                            UPDATE {self._t('somatic_markers')}
                            SET features_text = ?, embedding = ?, emb = TO_VECTOR(?, DOUBLE),
                                valence = ?, confidence = ?, count_ = ?, category = ?, last_activated = ?
                            WHERE situation_hash = ?
                        """, [features_text, emb_str, emb_vec_str, valence, confidence, count,
                              category, last_activated or _now_iso(), situation_hash])
                    except Exception:
                        cur.execute(f"""
                            UPDATE {self._t('somatic_markers')}
                            SET features_text = ?, embedding = ?, valence = ?,
                                confidence = ?, count_ = ?, category = ?, last_activated = ?
                            WHERE situation_hash = ?
                        """, [features_text, emb_str, valence, confidence, count,
                              category, last_activated or _now_iso(), situation_hash])
                else:
                    cur.execute(f"""
                        UPDATE {self._t('somatic_markers')}
                        SET features_text = ?, valence = ?,
                            confidence = ?, count_ = ?, category = ?, last_activated = ?
                        WHERE situation_hash = ?
                    """, [features_text, valence, confidence, count,
                          category, last_activated or _now_iso(), situation_hash])
            else:
                if emb_vec_str:
                    try:
                        cur.execute(f"""
                            INSERT INTO {self._t('somatic_markers')}
                            (situation_hash, features_text, embedding, emb, valence,
                             confidence, count_, category, last_activated)
                            VALUES (?, ?, ?, TO_VECTOR(?, DOUBLE), ?, ?, ?, ?, ?)
                        """, [situation_hash, features_text, emb_str, emb_vec_str, valence,
                              confidence, count, category, last_activated or _now_iso()])
                    except Exception:
                        cur.execute(f"""
                            INSERT INTO {self._t('somatic_markers')}
                            (situation_hash, features_text, embedding, valence,
                             confidence, count_, category, last_activated)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, [situation_hash, features_text, emb_str, valence,
                              confidence, count, category, last_activated or _now_iso()])
                else:
                    cur.execute(f"""
                        INSERT INTO {self._t('somatic_markers')}
                        (situation_hash, features_text, embedding, valence,
                         confidence, count_, category, last_activated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, [situation_hash, features_text, emb_str, valence,
                          confidence, count, category, last_activated or _now_iso()])
        self._flog.write("upsert", "somatic_markers", {
            "situation_hash": situation_hash, "features_text": features_text,
            "valence": valence, "confidence": confidence, "count": count,
            "category": category, "last_activated": last_activated or _now_iso(),
        })

    def find_similar_markers(self, embedding: list, threshold: float = 0.70,
                              limit: int = 3) -> list:
        """Find somatic markers similar to embedding using VECTOR_COSINE (HNSW indexed).

        Falls back to Python cosine scan when emb column is not yet populated.
        """
        query_vec_str = json.dumps(embedding)

        with self._conn() as conn:
            cur = conn.cursor()
            try:
                # Fast path: VECTOR_COSINE against HNSW-indexed emb column
                cur.execute(f"""
                    SELECT TOP ? situation_hash, features_text, valence, confidence,
                           count_, category, last_activated,
                           VECTOR_COSINE(emb, TO_VECTOR(?, DOUBLE)) AS similarity
                    FROM {self._t('somatic_markers')}
                    WHERE emb IS NOT NULL
                      AND VECTOR_COSINE(emb, TO_VECTOR(?, DOUBLE)) >= ?
                    ORDER BY similarity DESC
                """, [limit, query_vec_str, query_vec_str, threshold])
                rows = cur.fetchall()
                cols = _cursor_columns(cur)
                if rows:
                    results = []
                    for row in rows:
                        d = dict(zip(cols, row))
                        d['count'] = d.pop('count_', 0)
                        try:
                            d['similarity'] = float(d.get('similarity', 0))
                        except (TypeError, ValueError):
                            d['similarity'] = 0.0
                        results.append(d)
                    return results
            except Exception:
                pass  # Fall through to Python fallback

            # Python fallback: full scan
            cur.execute(
                f"SELECT * FROM {self._t('somatic_markers')} WHERE embedding IS NOT NULL"
            )
            rows = cur.fetchall()
            cols = _cursor_columns(cur)

        scored = []
        for row in rows:
            d = dict(zip(cols, row))
            emb_raw = d.get('embedding')
            if not emb_raw:
                continue
            try:
                emb = json.loads(emb_raw)
            except Exception:
                continue
            sim = _cosine_similarity(embedding, emb)
            if sim >= threshold:
                d['similarity'] = sim
                d['count'] = d.pop('count_', 0)
                scored.append(d)

        scored.sort(key=lambda x: x['similarity'], reverse=True)
        return scored[:limit]

    def load_all_somatic_markers(self) -> dict:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(f"SELECT * FROM {self._t('somatic_markers')}")
            rows = cur.fetchall()
            cols = _cursor_columns(cur)
        result = {}
        for row in rows:
            d = dict(zip(cols, row))
            h = d.pop('situation_hash', '')
            d.pop('embedding', None)  # exclude for load_all (not needed)
            d['count'] = d.pop('count_', 0)
            result[h] = d
        return result

    def count_somatic_markers(self) -> int:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(f"SELECT COUNT(*) FROM {self._t('somatic_markers')}")
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def delete_somatic_marker(self, situation_hash: str) -> bool:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                f"DELETE FROM {self._t('somatic_markers')} WHERE situation_hash = ?",
                [situation_hash]
            )
        return True

    # -----------------------------------------------------------------------
    # KEY-VALUE STORE
    # -----------------------------------------------------------------------

    def kv_get(self, key: str) -> Optional[dict]:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                f"SELECT value_ FROM {self._t('kv')} WHERE key_ = ?", [key]
            )
            row = cur.fetchone()
            return _json_loads(row[0]) if row and row[0] else None

    def kv_set(self, key: str, value):
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                f"SELECT key_ FROM {self._t('kv')} WHERE key_ = ?", [key]
            )
            exists = cur.fetchone()
            if exists:
                cur.execute(
                    f"UPDATE {self._t('kv')} SET value_ = ?, updated_at = ? WHERE key_ = ?",
                    [_json_dumps(value), _now_iso(), key]
                )
            else:
                cur.execute(
                    f"INSERT INTO {self._t('kv')} (key_, value_, updated_at) VALUES (?, ?, ?)",
                    [key, _json_dumps(value), _now_iso()]
                )
        if not any(key.startswith(p) for p in _KV_LOG_SKIP_PREFIXES):
            self._flog.write("upsert", "kv", {"key": key, "value": value})

    def kv_set_batch(self, items: dict):
        for k, v in items.items():
            self.kv_set(k, v)

    def kv_delete(self, key: str) -> bool:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(f"DELETE FROM {self._t('kv')} WHERE key_ = ?", [key])
        return True

    def kv_get_prefix(self, prefix: str) -> dict:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                f"SELECT key_, value_ FROM {self._t('kv')} WHERE key_ LIKE ?",
                [prefix + '%']
            )
            rows = cur.fetchall()
        return {row[0]: _json_loads(row[1]) for row in rows}

    # -----------------------------------------------------------------------
    # Stubs for less-critical methods (avoid import errors)
    # -----------------------------------------------------------------------

    def get_retrieved_ids(self, session_id: int) -> list:
        return []

    def log_retrieval(self, session_id: int, memory_id: str, source: str = 'semantic'):
        pass

    def store_image_embedding(self, *args, **kwargs):
        pass

    def search_image_embeddings(self, *args, **kwargs) -> list:
        return []

    def store_context_graph(self, *args, **kwargs):
        pass

    def get_context_graph(self, *args, **kwargs) -> Optional[dict]:
        return None

    def log_explanation(self, *args, **kwargs):
        pass

    def upsert_typed_edge(self, *args, **kwargs):
        pass

    def get_typed_edges(self, *args, **kwargs) -> list:
        return []

    def log_session_event(self, *args, **kwargs):
        pass

    def get_session_events(self, *args, **kwargs) -> list:
        return []

    def record_fingerprint(self, *args, **kwargs):
        pass

    def record_decay(self, *args, **kwargs):
        pass

    def get_swarm_messages(self, *args, **kwargs) -> list:
        return []

    def post_swarm_message(self, *args, **kwargs):
        pass
