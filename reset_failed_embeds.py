import sqlite3, pathlib, os
home = pathlib.Path(os.environ['USERPROFILE']) / '.hubris' / 'global' / 'hubris.db'
conn = sqlite3.connect(str(home))
cur = conn.execute(
    "UPDATE memory_actions SET status='pending', attempts=0 WHERE status='failed' AND action_type='reconcile_link'"
)
conn.commit()
print(f'Reset {cur.rowcount} rows to pending')
conn.close()
