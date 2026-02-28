import psycopg
import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
try:
    conn = psycopg.connect(host='localhost', port=5433, dbname='bakery_rag', user='postgres', password='secret')
    cur = conn.cursor()
    cur.execute('SELECT COUNT(*) FROM embeddings')
    count = cur.fetchone()[0]
    print(f"Rows in embeddings: {count}")
    if count > 0:
        cur.execute("SELECT id, id_document, LEFT(texte_fragment, 80) FROM embeddings LIMIT 3")
        for row in cur.fetchall():
            print(f"  id={row[0]}, doc={row[1]}, text={row[2]}...")
    cur.close()
    conn.close()
except Exception as e:
    print(f"ERROR: {e}")
