import os, sqlite3, time
from contextlib import contextmanager

DB_PATH = os.getenv("ENTITLEMENTS_DB_PATH", "/var/data/entitlements.db")

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS entitlements (
            email TEXT PRIMARY KEY,
            plan TEXT NOT NULL,
            updated_at INTEGER NOT NULL,
            source TEXT,
            order_id TEXT,
            subscription_id TEXT
        )
        """)

@contextmanager
def db():
    con = sqlite3.connect(DB_PATH)
    try:
        yield con
        con.commit()
    finally:
        con.close()

def set_plan(email: str, plan: str, source: str = None, order_id: str = None, subscription_id: str = None):
    now = int(time.time())
    with db() as con:
        con.execute("""
        INSERT INTO entitlements(email, plan, updated_at, source, order_id, subscription_id)
        VALUES(?,?,?,?,?,?)
        ON CONFLICT(email) DO UPDATE SET
            plan=excluded.plan,
            updated_at=excluded.updated_at,
            source=excluded.source,
            order_id=excluded.order_id,
            subscription_id=excluded.subscription_id
        """, (email.lower(), plan, now, source, order_id, subscription_id))

def get_plan(email: str) -> str | None:
    with db() as con:
        cur = con.execute("SELECT plan FROM entitlements WHERE email=?", (email.lower(),))
        row = cur.fetchone()
        return row[0] if row else None
