# utils.py
import pandas as pd
from sqlalchemy import create_engine
import pickle

def insert_to_sql(df: pd.DataFrame):
    try:
        engine = create_engine("mysql+mysqlconnector://root:14072003@localhost:3306/Obesity")
        df.to_sql("Obesity_Data", con=engine, if_exists="append", index=False)
    except Exception as e:
        print(f"❌ Error inserting to SQL: {e}")

def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def convert_height_to_meters(feet: int, inches: int) -> float:
    return round(((feet * 12) + inches) * 0.0254, 2)

