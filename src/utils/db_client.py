import os
from supabase import create_client, Client, ClientOptions
from dotenv import load_dotenv
import pandas as pd
import numpy as np

load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

if not url or not key:
    raise ValueError("Supabase credentials not found. Check your .env file.")

if not url.endswith('/'):
    url += '/'


# =========================
# CLIENT
# =========================
def get_supabase_client() -> Client:
    return create_client(
        url,
        key,
        options=ClientOptions(
            postgrest_client_timeout=600,
            storage_client_timeout=600
        )
    )


# =========================
# 🔥 GLOBAL CLEANER (IMPORTANT FIX FOR SHAP)
# =========================
def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # remove string artifacts like "[7.9E1]"
    df = df.replace(r'[\[\]]', '', regex=True)

    # convert everything possible to numeric
    for col in df.columns:
        if col != "timestamp":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# =========================
# UPLOAD FEATURES (FIXED)
# =========================
def push_features_to_store(df, table_name="feature_store", chunk_size=250):

    supabase = get_supabase_client()

    data = df.reset_index()

    # timestamp handling
    if "date" in data.columns:
        data["timestamp"] = pd.to_datetime(data["date"], errors="coerce").astype(str)
        data = data.drop(columns=["date"])

    # 🔥 FIX 1: CLEAN BEFORE UPLOAD
    data = clean_numeric(data)

    # replace NaN → None (Supabase safe)
    data = data.where(pd.notnull(data), None)

    total_rows = len(data)
    num_batches = (total_rows // chunk_size) + (1 if total_rows % chunk_size > 0 else 0)

    for i in range(0, total_rows, chunk_size):
        chunk = data.iloc[i:i + chunk_size]

        try:
            print(f"📤 Uploading batch {i//chunk_size + 1}/{num_batches}...")

            supabase.table(table_name).upsert(
                chunk.to_dict(orient="records")
            ).execute()

        except Exception as e:
            print(f"❌ Error: {e}")
            return

    print(f"✅ {total_rows} rows uploaded to {table_name}")


# =========================
# FETCH FEATURES (FIXED FOR SHAP)
# =========================
def fetch_training_data(table_name="feature_store"):

    supabase = get_supabase_client()

    response = (
        supabase.table(table_name)
        .select("*")
        .order("timestamp", desc=True)
        .limit(5000)
        .execute()
    )

    if not hasattr(response, "data") or not response.data:
        print("⚠️ No data found in Feature Store")
        return pd.DataFrame()

    df = pd.DataFrame(response.data)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

    # 🔥 FIX 2: CLEAN AFTER FETCH (CRITICAL FOR SHAP)
    df = clean_numeric(df)

    # final safety cast
    df = df.astype("float32", errors="ignore")

    return df


# =========================
# MODEL STORAGE (UNCHANGED)
# =========================
def upload_model_to_registry(file_path, file_name, bucket="model-registry"):

    supabase = get_supabase_client()

    try:
        with open(file_path, "rb") as f:
            supabase.storage.from_(bucket).upload(
                path=file_name,
                file=f.read(),
                file_options={
                    "cache-control": "3600",
                    "upsert": "true"
                }
            )

        print(f"✅ {file_name} saved")

    except Exception as e:
        print(f"❌ Error uploading {file_name}: {e}")


def download_model_from_registry(file_name, bucket="model-registry"):

    supabase = get_supabase_client()

    local_path = os.path.join("models", file_name)
    os.makedirs("models", exist_ok=True)

    try:
        response = supabase.storage.from_(bucket).download(file_name)

        with open(local_path, "wb") as f:
            f.write(response)

        print(f"✅ Downloaded {file_name}")
        return local_path

    except Exception as e:
        print(f"❌ Error downloading {file_name}: {e}")

        if os.path.exists(local_path):
            os.remove(local_path)

        return None