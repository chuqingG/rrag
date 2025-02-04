import pandas as pd
import psycopg2
import os
import PIL
from PIL import Image
import torch
import requests
from tqdm import tqdm
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

DB_PARAMS = {
    "dbname": YOUR_DATABASE_NAME,
    "user": PG_USERNAME,
    "password": PG_PASSWORD,
    "host": "localhost",
    "port": 5433,
}
add_vector_extension_sql = '''
CREATE EXTENSION IF NOT EXISTS vector
'''

create_table_sql = """
CREATE TABLE IF NOT EXISTS amazon_products (
    asin TEXT PRIMARY KEY,                -- Product ID (unique)
    title TEXT,                           -- Title of the product
    product_url TEXT,                     -- URL of the product
    stars REAL,                           -- Product rating (floating-point number)
    reviews INTEGER,                      -- Number of reviews
    price REAL,                           -- Buy now price (floating-point number)
    list_price REAL,                      -- Original price before discount
    category_id INTEGER,                  -- Category ID
    is_best_seller BOOLEAN,                -- Whether the product is a bestseller
    img_embedding vector(1024)            -- Image Embedding
);
"""

create_index_sql = """
CREATE INDEX img_index ON amazon_products USING ivfflat (img_embedding) WITH (lists = 100);
"""

def insert_data_to_postgres(df, conn, table_name="amazon_products"):
    with conn.cursor() as cur:
        for _, row in df.iterrows():
            insert_sql = f"""
            INSERT INTO {table_name} (asin, title, product_url, stars, reviews, price, list_price, category_id, is_best_seller, img_embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (asin) DO NOTHING;
            """
            cur.execute(
                insert_sql,
                (
                    row["asin"],
                    row["title"],
                    row["productURL"],
                    row["stars"],
                    row["reviews"],
                    row["price"],
                    row["listPrice"],
                    row["category_id"],
                    row["isBestSeller"],
                    row['img_embedding'].tolist() if row['img_embedding'] is not None else None,
                ),
            )
        conn.commit()
        
def getEmbeddingVector(inputs, model, device):
    with torch.no_grad():
        embedding = model(inputs)
    if device == 'cpu':
        for key, value in embedding.items():
            vec = value.reshape(-1)
            vec = vec.numpy()
            return(vec)
    else:
        for key, value in embedding.items():
            vec = value.reshape(-1)
            vec = vec.cpu().numpy()
            return(vec)

def dataToEmbedding(dataIn,dtype, model, device):
    if dtype == 'image':
        data_path = [dataIn]
        inputs = {
        ModalityType.VISION: data.load_and_transform_vision_data(data_path, device)
        }
    elif dtype == 'text':
        txt = [dataIn]
        inputs = {
        ModalityType.TEXT: data.load_and_transform_text(txt, device)
        }
    vec = getEmbeddingVector(inputs, model, device)
    return(vec)

def add_image_embedding(asin, model, device):
    img_path = "/ssd_root/gao688/amazon_img/" + asin + '.jpg'
    if not os.path.exists(img_path):
        print(f"File not found: {asin}.jpg")
        return None  
    vec = dataToEmbedding(img_path, 'image', model, device)
    return vec

if __name__ == "__main__":
    # The whole dataset need ~24 hours, we can use a subset for test
    N = 100000
    df = pd.read_csv("amazon_products.csv", nrows=N)
    # download images
    output_dir = "/ssd_root/gao688/amazon_img"  
    os.makedirs(output_dir, exist_ok=True)
    
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_url = row.get("imgUrl")  # Adjust if column name is different
        asin = row.get("asin")         # Use a unique identifier like 'asin'
        if pd.notna(image_url):
            try:
                response = requests.get(image_url, stream=True, timeout=10)
                if response.status_code == 200:
                    image_path = os.path.join(output_dir, f"{asin}.jpg")
                    with open(image_path, "wb") as f:
                        for chunk in response.iter_content(1024):
                            f.write(chunk)
            except Exception as e:
                print(f"Failed to download {image_url}: {e}")
            
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)
    
    df['img_embedding'] = df['asin'].apply(lambda asin: add_image_embedding(asin, model, device))

    # insert to postgresql
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        with conn.cursor() as cur:
            cur.execute(add_vector_extension_sql)
            cur.execute(create_table_sql)
            cur.execute(create_index_sql)
            conn.commit()

        insert_data_to_postgres(df, conn)
        print("Data inserted successfully.")

    except Exception as e:
        print("Error:", e)
    