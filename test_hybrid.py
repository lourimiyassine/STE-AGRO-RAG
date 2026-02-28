import psycopg
from src.db import get_connection

conn = get_connection()
cur = conn.cursor()

question = "Dose d'utilisation de l'acide ascorbique"
# Build a simple OR query from the question
words = [w for w in question.replace("'", " ").split() if len(w) > 3]
or_query = " | ".join(words)
print("OR Query:", or_query)

cur.execute("""
    SELECT texte_fragment, 
           ts_rank(to_tsvector('french', texte_fragment), to_tsquery('french', %s)) as r 
    FROM embeddings 
    WHERE ts_rank(to_tsvector('french', texte_fragment), to_tsquery('french', %s)) > 0
    ORDER BY r DESC LIMIT 3
""", (or_query, or_query))

print("Results for Acid:")
for res in cur.fetchall():
    print(res[1], res[0][:50])

question2 = "trouver des enzymes pour la panification"
words2 = [w for w in question2.replace("'", " ").split() if len(w) > 3]
or_query2 = " | ".join(words2)
print("\nOR Query 2:", or_query2)

cur.execute("""
    SELECT texte_fragment, 
           ts_rank(to_tsvector('french', texte_fragment), to_tsquery('french', %s)) as r 
    FROM embeddings 
    WHERE ts_rank(to_tsvector('french', texte_fragment), to_tsquery('french', %s)) > 0
    ORDER BY r DESC LIMIT 3
""", (or_query2, or_query2))

for res in cur.fetchall():
    print(res[1], res[0][:50])

cur.close()
conn.close()
