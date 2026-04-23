"""
Run once: python training/generate_data.py
Generates 500 clean healthcare + 500 financial records.
"""
import pandas as pd
import numpy as np
from faker import Faker
import random
import os

fake = Faker()
Faker.seed(42)
random.seed(42)
np.random.seed(42)

DEPT_MAP = {
    1: "Cardiology", 2: "Neurology", 3: "Oncology",
    4: "Pediatrics", 5: "Orthopedics", 6: "Radiology",
    7: "Emergency",  8: "Surgery",    9: "Psychiatry", 10: "General",
}

def generate_healthcare(n=500) -> pd.DataFrame:
    records = []
    for i in range(n):
        dept_id = random.randint(1, 10)
        birth_year = random.randint(1940, 2005)
        age = 2024 - birth_year
        records.append({
            "patient_id":     i + 1,
            "name":           fake.name(),
            "age":            age,
            "birth_year":     birth_year,
            "email":          fake.email(),
            "phone":          fake.numerify("##########"),
            "diagnosis":      random.choice(["Hypertension", "Diabetes", "Fracture",
                                             "Infection", "Migraine", "Asthma"]),
            "department_id":  dept_id,
            "department_name":DEPT_MAP[dept_id],
            "admission_date": fake.date_between("-2y", "today").strftime("%Y-%m-%d"),
        })
    return pd.DataFrame(records)

def generate_financial(n=500) -> pd.DataFrame:
    records = []
    for i in range(n):
        records.append({
            "transaction_id":  i + 1,
            "account_id":      random.randint(1000, 9999),
            "amount":          round(random.uniform(0.01, 10000), 2),
            "currency":        random.choice(["USD", "EUR", "GBP", "INR"]),
            "transaction_date":fake.date_between("-1y", "today").strftime("%Y-%m-%d"),
            "merchant":        fake.company(),
            "category":        random.choice(["Food", "Travel", "Shopping",
                                              "Utilities", "Healthcare", "Entertainment"]),
            "status":          random.choice(["completed", "pending", "failed"]),
        })
    return pd.DataFrame(records)

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    
    hc = generate_healthcare(500)
    hc.to_csv("data/healthcare_clean.csv", index=False)
    print(f"Healthcare: {len(hc)} rows -> data/healthcare_clean.csv")
    
    fin = generate_financial(500)
    fin.to_csv("data/financial_clean.csv", index=False)
    print(f"Financial:  {len(fin)} rows -> data/financial_clean.csv")
    
    print("Done. These are your ground truth files. Never corrupt these originals.")
