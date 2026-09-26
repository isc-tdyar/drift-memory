"""Every drift-memory connection dials the same IRIS when IRIS_PORT is unset.

ivg_bridge defaulted to 11982 while database/db.py defaulted to 11972, so with IRIS_PORT
unset (a hook's environment, a bare cron) memories were written to one port and their
Graph_KG mirror dialed another — which no registry entry assigns, so the mirror failed
and the memory write reported success. No live IRIS needed: connect() is intercepted.
"""

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _dialed(fn, env):
    seen = {}

    def fake_connect(**kw):
        seen.update(kw)
        raise ConnectionError("intercepted")

    with patch.dict(os.environ, env, clear=False), patch("iris.dbapi.connect", fake_connect):
        for k in ("IRIS_PORT", "IRIS_HOST"):
            if k not in env:
                os.environ.pop(k, None)
        try:
            fn()
        except ConnectionError:
            pass
    return seen


def _bridge_conn():
    import ivg_bridge
    ivg_bridge._get_conn()


def _db_conn():
    import database.db as db
    db._persistent_conn = None
    db._get_connection()


def test_bridge_and_store_default_to_the_same_port():
    bridge, store = _dialed(_bridge_conn, {}), _dialed(_db_conn, {})
    assert bridge["port"] == store["port"] == 11972, (bridge, store)


def test_bridge_honours_iris_port_like_the_store():
    env = {"IRIS_PORT": "11973"}
    assert _dialed(_bridge_conn, env)["port"] == _dialed(_db_conn, env)["port"] == 11973
