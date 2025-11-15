import numpy as np
import pandas as pd

np.random.seed(42)
n = 1000
age = np.random.randint(18, 60, n)
tenure_months = np.random.randint(1, 120, n)
job_level = np.random.choice([1, 2, 3, 4], n, p=[0.4, 0.3, 0.2, 0.1])
department = np.random.choice(["Sales", "Support", "IT", "HR", "Finance", "Operations"], n,
                              p=[0.2, 0.2, 0.2, 0.1, 0.15, 0.15])
salary_k = np.round(np.random.normal(6 + job_level*2, 1.2, n), 2)
satisfaction = np.clip(np.random.normal(0.6, 0.15, n), 0, 1)
training_hours = np.clip(np.random.normal(20, 8, n), 0, 60)
remote = np.random.choice([0, 1], n, p=[0.6, 0.4])
promoted_last_2y = np.random.choice([0, 1], n, p=[0.85, 0.15])
overtime_hours_m = np.clip(np.random.normal(10, 5, n), 0, 40)
performance_score = np.clip(np.random.normal(0.65, 0.18, n), 0, 1)

logit = (
    -2.0 + 0.02 * (age - 35) - 0.03 * (tenure_months - 24)
    - 2.5 * satisfaction + 0.15 * overtime_hours_m - 0.8 * remote
    - 0.6 * promoted_last_2y + 0.2 * (1 / job_level) - 0.05 * training_hours
)
prob = 1 / (1 + np.exp(-logit))
turnover = (np.random.rand(n) < prob).astype(int)

df = pd.DataFrame({
    "age": age, "tenure_months": tenure_months, "job_level": job_level,
    "department": department, "salary_k": salary_k, "satisfaction": satisfaction,
    "training_hours": training_hours, "remote": remote, "promoted_last_2y": promoted_last_2y,
    "overtime_hours_m": overtime_hours_m, "performance_score": performance_score,
    "turnover": turnover
})
df.to_csv("data/rh_turnover_dataset.csv", index=False)
print("OK:", df.shape, "turnover rate:", df['turnover'].mean())
