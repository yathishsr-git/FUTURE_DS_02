"""
Generate a realistic synthetic SaaS subscription dataset for
Customer Retention & Churn Analysis.

Business context: "StreamLearn" - an online course subscription platform.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

N = 5000
today = datetime(2026, 7, 1)

# Signup dates spread over last 24 months (creates cohorts)
signup_offsets = np.random.randint(0, 730, N)
signup_dates = [today - timedelta(days=int(d)) for d in signup_offsets]

plans = np.random.choice(
    ["Basic", "Standard", "Premium"], N, p=[0.45, 0.35, 0.20]
)
plan_price = {"Basic": 9.99, "Standard": 19.99, "Premium": 39.99}

regions = np.random.choice(
    ["North America", "Europe", "Asia Pacific", "Latin America"],
    N, p=[0.40, 0.30, 0.20, 0.10]
)

acquisition_channel = np.random.choice(
    ["Organic Search", "Paid Ads", "Referral", "Social Media", "Partnership"],
    N, p=[0.30, 0.25, 0.20, 0.15, 0.10]
)

age_group = np.random.choice(
    ["18-24", "25-34", "35-44", "45-54", "55+"], N,
    p=[0.15, 0.35, 0.25, 0.15, 0.10]
)

# Engagement metrics (drive churn probability)
avg_sessions_per_week = np.round(np.random.gamma(2.0, 1.3, N), 1)
courses_completed = np.random.poisson(2.2, N)
support_tickets = np.random.poisson(0.6, N)
used_mobile_app = np.random.choice([1, 0], N, p=[0.55, 0.45])
autopay_enabled = np.random.choice([1, 0], N, p=[0.7, 0.3])

# --- Build churn probability from realistic drivers ---
base = 0.10
churn_logit = np.zeros(N)

# Plan effect: cheaper plans churn more
plan_effect = np.where(plans == "Basic", 0.35, np.where(plans == "Standard", 0.05, -0.35))
# Engagement effect: low engagement -> higher churn
engagement_effect = -0.28 * (avg_sessions_per_week - avg_sessions_per_week.mean())
completion_effect = -0.18 * (courses_completed - courses_completed.mean())
support_effect = 0.22 * (support_tickets - support_tickets.mean())
mobile_effect = np.where(used_mobile_app == 1, -0.25, 0.10)
autopay_effect = np.where(autopay_enabled == 1, -0.30, 0.30)
tenure_months_so_far = signup_offsets / 30.44
tenure_effect = -0.015 * tenure_months_so_far  # longer-tenured customers churn less (survivorship)
channel_effect = np.where(acquisition_channel == "Paid Ads", 0.15,
                  np.where(acquisition_channel == "Referral", -0.15, 0.0))

logit = (
    np.log(base / (1 - base))
    + plan_effect + engagement_effect + completion_effect
    + support_effect + mobile_effect + autopay_effect
    + tenure_effect + channel_effect
    + np.random.normal(0, 0.4, N)
)
churn_prob = 1 / (1 + np.exp(-logit))
churned = np.random.binomial(1, np.clip(churn_prob, 0.02, 0.95))

# Tenure: if churned, tenure is shorter than time since signup; if active, tenure = time since signup
max_possible_tenure = np.maximum(tenure_months_so_far, 0.5)
tenure_months = np.where(
    churned == 1,
    np.round(np.clip(np.random.beta(1.5, 3.5, N) * max_possible_tenure, 0.5, None), 1),
    np.round(max_possible_tenure, 1)
)
tenure_months = np.minimum(tenure_months, max_possible_tenure)

churn_reason = np.array(["Still Active"] * N, dtype=object)
reason_pool = ["Too Expensive", "Not Enough Content Use", "Found Alternative",
               "Poor Customer Support", "Technical Issues", "No Longer Needed"]
reason_weights_by_plan = {
    "Basic": [0.35, 0.20, 0.15, 0.10, 0.10, 0.10],
    "Standard": [0.20, 0.25, 0.20, 0.15, 0.10, 0.10],
    "Premium": [0.10, 0.20, 0.25, 0.20, 0.10, 0.15],
}
for i in range(N):
    if churned[i] == 1:
        w = reason_weights_by_plan[plans[i]]
        churn_reason[i] = np.random.choice(reason_pool, p=w)

monthly_charges = np.array([plan_price[p] for p in plans])
# Small realistic noise/discounts
monthly_charges = np.round(monthly_charges * np.random.uniform(0.95, 1.0, N), 2)

df = pd.DataFrame({
    "customer_id": [f"CUST{i:05d}" for i in range(1, N + 1)],
    "signup_date": [d.strftime("%Y-%m-%d") for d in signup_dates],
    "signup_month": [d.strftime("%Y-%m") for d in signup_dates],
    "plan": plans,
    "monthly_charges": monthly_charges,
    "region": regions,
    "acquisition_channel": acquisition_channel,
    "age_group": age_group,
    "avg_sessions_per_week": avg_sessions_per_week,
    "courses_completed": courses_completed,
    "support_tickets": support_tickets,
    "used_mobile_app": used_mobile_app,
    "autopay_enabled": autopay_enabled,
    "tenure_months": tenure_months,
    "churned": churned,
    "churn_reason": churn_reason,
})

df["lifetime_value"] = np.round(df["monthly_charges"] * df["tenure_months"], 2)

df.to_csv("/home/claude/churn/subscription_data.csv", index=False)
print(df.shape)
print(df["churned"].mean())
print(df.head())
