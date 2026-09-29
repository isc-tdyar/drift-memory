#!/usr/bin/env python3
"""Migration: Backfill DriftMem_drift_text_embeddings.emb VECTOR column.

Reads existing LONGVARCHAR embedding JSON strings and writes them to the new
VECTOR(DOUBLE, 1536) column via TO_VECTOR(). Creates the column if it doesn't
already exist.

IRIS VECTOR_COSINE server-side scan is significantly faster than the prior
full-table Python cosine loop (12,289 rows × 1536-dim deserialization).

Run once:
    python3 migrations/migrate_emb_vector.py [--schema drift] [--batch-size 200] [--dry-run]
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_conn():
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
    return get_connection(config=config)


def ensure_vector_column(conn, schema: str) -> bool:
    """Add emb VECTOR column if not present. Returns True if column was added."""
    cursor = conn.cursor()
    try:
        cursor.execute(
            f"ALTER TABLE DriftMem_{schema}_text_embeddings ADD emb VECTOR(DOUBLE, 1536)"
        )
        conn.commit()
        print(f"  Added emb VECTOR(DOUBLE, 1536) column to DriftMem_{schema}_text_embeddings")
        return True
    except Exception as e:
        if 'already exists' in str(e).lower() or 'duplicate' in str(e).lower():
            print(f"  emb column already exists")
            return False
        print(f"  ALTER failed: {e}")
        return False
    finally:
        cursor.close()


def backfill(conn, schema: str, batch_size: int = 200, dry_run: bool = False) -> int:
    """Backfill emb from embedding LONGVARCHAR. Returns count updated."""
    table = f"DriftMem_{schema}_text_embeddings"
    cursor = conn.cursor()

    # Count rows needing backfill
    cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE emb IS NULL AND embedding IS NOT NULL")
    total = cursor.fetchone()[0]
    print(f"  {total} rows need backfill")

    if dry_run or total == 0:
        cursor.close()
        return total if dry_run else 0

    updated = 0
    errors = 0

    while True:
        # Fetch a batch of rows without emb set
        cursor.execute(
            f"SELECT TOP ? memory_id, embedding FROM {table} "
            f"WHERE emb IS NULL AND embedding IS NOT NULL",
            [batch_size]
        )
        rows = cursor.fetchall()
        if not rows:
            break

        batch_updated = 0
        for row in rows:
            memory_id, emb_str = row[0], row[1]
            try:
                # Validate JSON before sending to TO_VECTOR
                json.loads(emb_str)
                cursor.execute(
                    f"UPDATE {table} SET emb = TO_VECTOR(?, DOUBLE) WHERE memory_id = ?",
                    [emb_str, memory_id]
                )
                batch_updated += 1
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"    Error on {memory_id}: {e}")

        conn.commit()
        updated += batch_updated
        print(f"  ... {updated}/{total} backfilled ({errors} errors)", end='\r')

        if batch_updated < batch_size:
            break

    print(f"\n  Backfill complete: {updated} updated, {errors} errors")
    return updated


def verify(conn, schema: str):
    """Quick smoke test: run VECTOR_COSINE query and print top result."""
    import json as _json
    cursor = conn.cursor()
    # Use a zero vector as query — any result shows the path works
    test_vec = _json.dumps([0.0] * 1536)
    cursor.execute(
        f"SELECT TOP 1 memory_id, VECTOR_COSINE(emb, TO_VECTOR(?, DOUBLE)) as sim "
        f"FROM DriftMem_{schema}_text_embeddings WHERE emb IS NOT NULL ORDER BY sim DESC",
        [test_vec]
    )
    row = cursor.fetchone()
    if row:
        print(f"  Verification OK: memory_id={row[0]}, sim={row[1]:.4f}")
    else:
        print("  Verification: no rows with emb set yet")
    cursor.close()


def main():
    parser = argparse.ArgumentParser(description="Backfill emb VECTOR column")
    parser.add_argument("--schema", default="drift")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"Migrating DriftMem_{args.schema}_text_embeddings.emb")
    if args.dry_run:
        print("  (dry-run mode — no writes)")

    t0 = time.time()
    conn = get_conn()

    ensure_vector_column(conn, args.schema)
    count = backfill(conn, args.schema, batch_size=args.batch_size, dry_run=args.dry_run)
    if not args.dry_run:
        verify(conn, args.schema)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s — {count} rows {'would be ' if args.dry_run else ''}updated")
    conn.close()


if __name__ == "__main__":
    main()
