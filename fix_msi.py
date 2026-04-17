import psycopg2

conn = psycopg2.connect('postgresql://godseye_user:godseye_pass@localhost:5433/godseye')
conn.rollback()
cur = conn.cursor()
cur.execute('DROP TABLE IF EXISTS features_msi CASCADE')
conn.commit()
print('Dropped features_msi — ready for MSIExtractor to recreate with full schema')
conn.close()