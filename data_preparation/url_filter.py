import pandas as pd
import requests
from fake_useragent import UserAgent
from tqdm import tqdm
import random
import time
import psycopg2

DB_PARAMS = {
    "dbname": YOUR_DATABASE_NAME,
    "user": PG_USERNAME,
    "password": PG_PASSWORD,
    "host": "localhost",
    "port": 5433,
}

drop_sql = '''
DELETE FROM amazon_products
WHERE asin = ANY(%s)
'''
    
def is_valid_url(url):
    # ua = UserAgent()
    headers = {"User-Agent": ua.chrome}
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            # Check for error page indicators in the HTML
            if (
                "Sorry! We couldn’t find that page" in response.text or
                "https://images-na.ssl-images-amazon.com/images/G/01/error/en_US/title._TTD_.png" in response.text
            ):
                return False
            return True
        else:
            return False
    except requests.exceptions.RequestException as e:
        print(f"Error accessing {url}: {e}")
        return False

if __name__ == "__main__":

    input_file = "amazon_products.csv"  
    output_file = "invalid.csv"     
    df = pd.read_csv(input_file)

    rows_invalid = []

    ua = UserAgent()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Checking URLs"):
        url = row["productURL"]
        # print(url, end=': ')
        if not is_valid_url(url):
            rows_invalid.append(row)
        # time.sleep(random.random())

    df_invalid = pd.DataFrame(rows_invalid)
    df_invalid.to_csv(output_file, index=False)
    print(f"Saved valid rows to {output_file}, total len: {len(rows_invalid)}")
    
    # Start droping
    df = pd.read_csv("invalid.csv")
    asin_list = df['asin'].tolist()
    
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        with conn.cursor() as cur:
            cur.execute(drop_sql, (asin_list,))
            conn.commit()
            print(f"{cur.rowcount} rows deleted.")


    except Exception as e:
        print("Error:", e)
        conn.rollback()
    finally:
        conn.close()