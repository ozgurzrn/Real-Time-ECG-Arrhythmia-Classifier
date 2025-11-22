import wfdb
import os

def download_mitdb(dl_dir='data/raw'):
    """
    Downloads the MIT-BIH Arrhythmia Database from PhysioNet.
    """
    if not os.path.exists(dl_dir):
        os.makedirs(dl_dir)
    
    print(f"Downloading MIT-BIH Arrhythmia Database to {dl_dir}...")
    try:
        wfdb.dl_database('mitdb', dl_dir=dl_dir)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading database: {e}")

if __name__ == "__main__":
    download_mitdb()
