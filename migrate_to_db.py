import pandas as pd
import toml
from sqlalchemy import create_engine

secrets = toml.load(".streamlit/secrets.toml")
DB_URL = secrets["DATABASE_URL"]

engine = create_engine(DB_URL)

df = pd.read_csv("services.csv")
df.to_sql("services", engine, if_exists="replace", index=False)

print(f"Migration done: {len(df)} rows loaded into 'services' table")
