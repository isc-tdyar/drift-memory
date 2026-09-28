"""At exit, store_memory's background threads finish before the connection closes.

Measured 2026-09-28 on mac-studio: a caller that exited while `_kg_extract_bg` was
inside a Bedrock call SIGSEGV'd in finalization — vault-ingestion exited -11 on 41
of 204 runs after its work completed. database/db.py now joins those threads in an
atexit handler that runs before `_close_on_exit` (atexit is LIFO), bounded so a hung
call cannot hold the process.

Each case is a real process: exit ordering cannot be observed in-process.
"""

import os
import subprocess
import sys
import textwrap
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(body):
    code = textwrap.dedent("""
        import sys, threading, time
        sys.path.insert(0, %r)
        import database.db as db

        class FakeConn:
            def close(self):
                print("closed", flush=True)

        db._persistent_conn = FakeConn()

        def _kg_extract_bg(delay):
            time.sleep(delay)
            print("extracted", flush=True)
    """ % ROOT) + textwrap.dedent(body)
    t0 = time.monotonic()
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    return r, time.monotonic() - t0


def test_background_work_finishes_before_the_connection_closes():
    r, _ = _run("threading.Thread(target=_kg_extract_bg, args=(0.5,), daemon=True).start()\n")
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["extracted", "closed"]


def test_the_wait_is_bounded():
    r, elapsed = _run("""
        db.BACKGROUND_JOIN_S = 0.3
        threading.Thread(target=_kg_extract_bg, args=(30,), daemon=True).start()
    """)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["closed"]
    assert elapsed < 10


def test_unrelated_daemon_threads_are_not_waited_for():
    r, elapsed = _run("""
        def idle():
            time.sleep(30)
        threading.Thread(target=idle, daemon=True).start()
    """)
    assert r.returncode == 0, r.stderr
    assert elapsed < 10


def test_every_background_target_in_memory_store_is_joined():
    """The join matches threads by target name, so a renamed or added target in
    memory_store.py must be listed too, or it silently stops being waited for."""
    import ast
    import database.db as db

    tree = ast.parse(open(os.path.join(ROOT, "memory_store.py")).read())
    started = {
        kw.value.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "Thread"
        for kw in node.keywords
        if kw.arg == "target" and isinstance(kw.value, ast.Name)
    }
    assert started, "found no Thread(target=...) in memory_store.py"
    assert started <= set(db.BACKGROUND_TARGETS)
