import os
import urllib.request
import pandas as pd

def download_dataset():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    target_csv = os.path.join(data_dir, "spam.csv")
    sms_csv = os.path.join(data_dir, "SMS.csv")
    
    if os.path.exists(target_csv) and os.path.exists(sms_csv):
        print(f"Dataset already exists at: {target_csv}")
        return target_csv

    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    print(f"Downloading SMS Spam dataset from {url}...")
    
    tsv_path = os.path.join(data_dir, "sms.tsv")
    urllib.request.urlretrieve(url, tsv_path)
    
    # Read TSV (label \t message)
    df = pd.read_csv(tsv_path, sep='\t', header=None, names=['v1', 'v2'])
    
    # Save as CSV with utf-8 encoding
    df.to_csv(target_csv, index=False, encoding='utf-8')
    df.to_csv(sms_csv, index=False, encoding='utf-8')
    
    # Cleanup temp tsv
    if os.path.exists(tsv_path):
        os.remove(tsv_path)
        
    print(f"Successfully saved SMS Spam dataset ({df.shape[0]} rows) to {target_csv} & {sms_csv}")
    return target_csv

if __name__ == "__main__":
    download_dataset()
