import sqlite3
import pandas as pd

conn = sqlite3.connect("predictions.db")
df = pd.read_sql("SELECT * FROM predictions", conn)

print("Total predictions logged:", df.shape[0])
print(df[['age', 'time_in_hospital', 'probability', 'prediction', 'timestamp']].head())

conn.close()