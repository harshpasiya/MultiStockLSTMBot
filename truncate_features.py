import psycopg2

conn = psycopg2.connect('postgresql://godseye_user:godseye_pass@localhost:5433/godseye')
conn.rollback()
cur = conn.cursor()

tables = [
    'features_trend',
    'features_msi',
    'features_fii_dii',
    'features_sentiment',
    'features_volatility',
    'features_correlation',
    'features_fused',
    'correlation_matrix',
]

for t in tables:
    try:
        cur.execute(f'TRUNCATE TABLE {t} CASCADE')
        print(f'Truncated {t}')
    except Exception as e:
        conn.rollback()
        print(f'Skipped {t}: {e}')

conn.commit()
print('All feature tables truncated cleanly.')
conn.close()