import wfdb

try:
    print("Fetching PTB database record list...")
    records = wfdb.get_record_list('ptbdb')
    print(f"Found {len(records)} records.")
    print("First 10 records:")
    for r in records[:10]:
        print(r)
except Exception as e:
    print(f"Error: {e}")
