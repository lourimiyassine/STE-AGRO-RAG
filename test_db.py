import psycopg
import sys

# Windows UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

try:
    print("Attempting connection to localhost on port 5433...")
    conn = psycopg.connect(host='localhost', port=5433, dbname='bakery_rag', user='postgres', password='secret')
    print("SUCCESS: localhost:5433 (PostgreSQL Docker container is reachable!)")
    conn.close()
except Exception as e:
    print(f"FAILED: {type(e).__name__} - {str(e)}")
