import psycopg2

conn = psycopg2.connect('postgresql://godseye_user:godseye_pass@localhost:5433/godseye')
cur = conn.cursor()

# Get actual column names from features_fused
cur.execute("""
    SELECT column_name
    FROM information_schema.columns
    WHERE table_name = 'features_fused'
    ORDER BY ordinal_position
""")
cols = [row[0] for row in cur.fetchall()]
print(f"Total columns: {len(cols)}")
print("Columns:")
for i, col in enumerate(cols):
    print(f"  [{i}] {col}")

# Also show a sample row
cur.execute("SELECT * FROM features_fused LIMIT 1")
row = cur.fetchone()
if row:
    print(f"\nSample row (first 5 values): {row[:5]}")

conn.close()