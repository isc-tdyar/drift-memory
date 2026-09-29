#!/usr/bin/env python3
"""
drift-memory File Log Replay — restore IRIS from JSONL write-through log.

Run from ~/ws/drift-memory/ or with PYTHONPATH=~/ws/drift-memory.

Usage:
    python3 database/replay.py [--schema drift] [--dry-run] [--since YYYY-MM-DD]
    python3 database/replay.py --verify   # check log health without restoring

Run from ~/ws/drift-memory/.

This is the disaster recovery entry point. When IRIS is wiped:
  1. docker compose up -d                     # fresh IRIS
  2. python3 database/replay.py --dry-run     # audit what will be restored
  3. python3 database/replay.py               # restore everything
  4. python3 database/replay.py --verify      # confirm counts match
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Allow running as `python3 database/replay.py` from the drift-memory root
sys.path.insert(0, str(Path(__file__).parent.parent))

LOG_DIR = Path.home() / ".local" / "share" / "drift-memory-log"
PYTHON = "/opt/homebrew/Caskroom/miniconda/base/bin/python3"


def iter_log(schema: str, table: str, since: str | None = None):
    """Yield parsed records from a table's JSONL log, oldest first."""
    path = LOG_DIR / schema / f"{table}.jsonl"
    if not path.exists():
        return
    since_ts = None
    if since:
        since_ts = datetime.fromisoformat(since).replace(tzinfo=timezone.utc)

    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [WARN] {table}.jsonl line {lineno}: parse error ({e})", file=sys.stderr)
                continue
            if since_ts:
                try:
                    rec_ts = datetime.fromisoformat(rec.get("ts", "")).replace(tzinfo=timezone.utc)
                    if rec_ts < since_ts:
                        continue
                except Exception:
                    pass
            yield rec


def verify(schema: str):
    """Print log stats — count ops per table."""
    print(f"\n=== File log verification — schema: {schema} ===")
    log_dir = LOG_DIR / schema
    if not log_dir.exists():
        print(f"[WARN] No log directory at {log_dir}")
        return

    for jf in sorted(log_dir.glob("*.jsonl")):
        table = jf.stem
        counts = defaultdict(int)
        for rec in iter_log(schema, table):
            counts[rec.get("op", "?")] += 1
        total = sum(counts.values())
        breakdown = ", ".join(f"{op}={n}" for op, n in sorted(counts.items()))
        size_kb = jf.stat().st_size // 1024
        print(f"  {table:25s}  {total:6d} records  ({breakdown})  [{size_kb}KB]")


def restore_memories(db, schema: str, dry_run: bool, since: str | None) -> int:
    """Replay memories log into MemoryDB. Returns count restored."""
    count = 0
    skipped = 0
    for rec in iter_log(schema, "memories", since):
        op = rec.get("op")
        mem_id = rec.get("id")
        if not mem_id:
            continue

        if dry_run:
            print(f"  [DRY] {op} memory {mem_id}: {str(rec.get('content',''))[:60]}")
            count += 1
            continue

        try:
            if op == "insert":
                # Skip if already exists (idempotent)
                existing = db.get_memory(mem_id)
                if existing:
                    skipped += 1
                    continue
                db.insert_memory(
                    mem_id,
                    type_=rec.get("type_", "active"),
                    content=rec.get("content", ""),
                    tags=rec.get("tags") or [],
                    entities=rec.get("entities") or {},
                    emotional_weight=rec.get("emotional_weight", 0.5),
                    topic_context=rec.get("topic_context"),
                    contact_context=rec.get("contact_context"),
                    platform_context=rec.get("platform_context"),
                    extra_metadata=rec.get("extra_metadata"),
                )
                count += 1
            elif op == "update":
                fields = {k: v for k, v in rec.items()
                          if k not in ("op", "schema", "table", "ts", "id")}
                if fields:
                    db.update_memory(mem_id, **fields)
                count += 1
            elif op == "delete":
                db.delete_memory(mem_id)
                count += 1
        except Exception as e:
            print(f"  [WARN] {op} {mem_id}: {e}", file=sys.stderr)

    print(f"  memories: {count} replayed, {skipped} skipped (already exist)")
    return count


def restore_embeddings(db, schema: str, dry_run: bool, since: str | None) -> int:
    count = 0
    for rec in iter_log(schema, "text_embeddings", since):
        mem_id = rec.get("memory_id")
        embedding = rec.get("embedding")
        if not mem_id or not embedding:
            continue
        if dry_run:
            print(f"  [DRY] upsert embedding {mem_id} ({len(embedding)}d)")
            count += 1
            continue
        try:
            db.store_embedding(mem_id, embedding,
                               preview=rec.get("preview", ""),
                               model=rec.get("model", "unknown"))
            count += 1
        except Exception as e:
            print(f"  [WARN] embedding {mem_id}: {e}", file=sys.stderr)
    print(f"  embeddings: {count} replayed")
    return count


def restore_edges(db, schema: str, dry_run: bool, since: str | None) -> int:
    count = 0
    for rec in iter_log(schema, "edges", since):
        id1, id2 = rec.get("id1"), rec.get("id2")
        belief = rec.get("belief", 0.0)
        if not id1 or not id2:
            continue
        if dry_run:
            print(f"  [DRY] upsert edge {id1} ↔ {id2} (belief={belief:.2f})")
            count += 1
            continue
        try:
            db.upsert_edge(id1, id2, belief,
                           platform_context=rec.get("platform_context"),
                           activity_context=rec.get("activity_context"),
                           topic_context=rec.get("topic_context"))
            count += 1
        except Exception as e:
            print(f"  [WARN] edge {id1}/{id2}: {e}", file=sys.stderr)
    print(f"  edges: {count} replayed")
    return count


def restore_somatic_markers(db, schema: str, dry_run: bool, since: str | None) -> int:
    count = 0
    for rec in iter_log(schema, "somatic_markers", since):
        h = rec.get("situation_hash")
        if not h:
            continue
        if dry_run:
            print(f"  [DRY] upsert somatic marker {h[:16]}")
            count += 1
            continue
        try:
            db.upsert_somatic_marker(
                h,
                features_text=rec.get("features_text", ""),
                valence=rec.get("valence", 0.0),
                confidence=rec.get("confidence", 0.0),
                count=rec.get("count", 0),
                category=rec.get("category", "general"),
                last_activated=rec.get("last_activated"),
            )
            count += 1
        except Exception as e:
            print(f"  [WARN] somatic marker {h}: {e}", file=sys.stderr)
    print(f"  somatic_markers: {count} replayed")
    return count


def restore_lessons(db, schema: str, dry_run: bool, since: str | None) -> int:
    count = 0
    for rec in iter_log(schema, "lessons", since):
        lid = rec.get("id")
        if not lid:
            continue
        if dry_run:
            print(f"  [DRY] upsert lesson {lid}")
            count += 1
            continue
        try:
            db.add_lesson(lid,
                          category=rec.get("category", "general"),
                          lesson=rec.get("lesson", ""),
                          evidence=rec.get("evidence"),
                          source=rec.get("source", "manual"),
                          confidence=rec.get("confidence", 0.7))
            count += 1
        except Exception as e:
            print(f"  [WARN] lesson {lid}: {e}", file=sys.stderr)
    print(f"  lessons: {count} replayed")
    return count


def restore_kv(db, schema: str, dry_run: bool, since: str | None) -> int:
    count = 0
    # KV: only keep the LAST value per key (replay is ordered oldest→newest)
    final_kv = {}
    for rec in iter_log(schema, "kv", since):
        key = rec.get("key")
        if key:
            final_kv[key] = rec.get("value")
    for key, value in final_kv.items():
        if dry_run:
            print(f"  [DRY] kv set {key}")
            count += 1
            continue
        try:
            db.kv_set(key, value)
            count += 1
        except Exception as e:
            print(f"  [WARN] kv {key}: {e}", file=sys.stderr)
    print(f"  kv: {count} keys replayed ({len(final_kv)} unique)")
    return count


def main():
    parser = argparse.ArgumentParser(description="Replay drift-memory file log into IRIS")
    parser.add_argument("--schema", default="drift", help="Memory schema (default: drift)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be replayed")
    parser.add_argument("--verify", action="store_true", help="Show log stats and exit")
    parser.add_argument("--since", default=None, help="Only replay records since YYYY-MM-DD")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip embedding replay (fast but no vector search)")
    args = parser.parse_args()

    if args.verify:
        verify(args.schema)
        return

    print(f"\n=== drift-memory file log replay — schema: {args.schema} ===")
    if args.dry_run:
        print("  DRY RUN — no writes to IRIS")
    if args.since:
        print(f"  Since: {args.since}")

    from database.db import MemoryDB
    db = MemoryDB(args.schema)

    total = 0
    total += restore_memories(db, args.schema, args.dry_run, args.since)
    if not args.skip_embeddings:
        total += restore_embeddings(db, args.schema, args.dry_run, args.since)
    total += restore_edges(db, args.schema, args.dry_run, args.since)
    total += restore_somatic_markers(db, args.schema, args.dry_run, args.since)
    total += restore_lessons(db, args.schema, args.dry_run, args.since)
    total += restore_kv(db, args.schema, args.dry_run, args.since)

    print(f"\nReplay complete: {total} total operations")
    if args.dry_run:
        print("(dry run — re-run without --dry-run to apply)")


if __name__ == "__main__":
    main()
