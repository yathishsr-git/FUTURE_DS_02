import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})

df = pd.read_csv("/home/claude/churn/subscription_data.csv", parse_dates=["signup_date"])
CHARTS = "/home/claude/churn/charts"

# ---------- 1. Overall churn ----------
overall_churn = df["churned"].mean()
print(f"Overall churn rate: {overall_churn:.1%}")

# ---------- 2. Churn by plan ----------
churn_by_plan = df.groupby("plan")["churned"].mean().reindex(["Basic", "Standard", "Premium"])
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(churn_by_plan.index, churn_by_plan.values, color=["#e07a5f", "#3d5a80", "#81b29a"])
ax.set_title("Churn Rate by Subscription Plan")
ax.set_ylabel("Churn Rate")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
for b, v in zip(bars, churn_by_plan.values):
    ax.text(b.get_x() + b.get_width()/2, v + 0.003, f"{v:.1%}", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig(f"{CHARTS}/churn_by_plan.png", dpi=150)
plt.close()

# ---------- 3. Churn by region ----------
churn_by_region = df.groupby("region")["churned"].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(6.5, 4))
bars = ax.barh(churn_by_region.index[::-1], churn_by_region.values[::-1], color="#3d5a80")
ax.set_title("Churn Rate by Region")
ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
for b, v in zip(bars, churn_by_region.values[::-1]):
    ax.text(v + 0.002, b.get_y() + b.get_height()/2, f"{v:.1%}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{CHARTS}/churn_by_region.png", dpi=150)
plt.close()

# ---------- 4. Churn by acquisition channel ----------
churn_by_channel = df.groupby("acquisition_channel")["churned"].mean().sort_values(ascending=False)

# ---------- 5. Engagement vs churn ----------
df["engagement_bucket"] = pd.cut(
    df["avg_sessions_per_week"], bins=[-0.1, 1, 2, 4, 100],
    labels=["<1/wk", "1-2/wk", "2-4/wk", "4+/wk"]
)
churn_by_engagement = df.groupby("engagement_bucket", observed=True)["churned"].mean()
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(churn_by_engagement.index.astype(str), churn_by_engagement.values, color="#e07a5f")
ax.set_title("Churn Rate by Weekly Engagement")
ax.set_xlabel("Average sessions per week")
ax.set_ylabel("Churn Rate")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
for b, v in zip(bars, churn_by_engagement.values):
    ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.1%}", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig(f"{CHARTS}/churn_by_engagement.png", dpi=150)
plt.close()

# ---------- 6. Support tickets vs churn ----------
df["support_bucket"] = pd.cut(df["support_tickets"], bins=[-0.1, 0, 1, 2, 100],
                               labels=["0", "1", "2", "3+"])
churn_by_support = df.groupby("support_bucket", observed=True)["churned"].mean()

# ---------- 7. Autopay & mobile app effect ----------
churn_by_autopay = df.groupby("autopay_enabled")["churned"].mean()
churn_by_mobile = df.groupby("used_mobile_app")["churned"].mean()

# ---------- 8. Tenure distribution among churned customers ----------
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.hist(df.loc[df.churned == 1, "tenure_months"], bins=20, color="#e07a5f", edgecolor="white")
ax.set_title("Tenure at Time of Churn (months)")
ax.set_xlabel("Months as a subscriber")
ax.set_ylabel("Number of churned customers")
plt.tight_layout()
plt.savefig(f"{CHARTS}/tenure_at_churn.png", dpi=150)
plt.close()

median_tenure_churned = df.loc[df.churned == 1, "tenure_months"].median()
avg_tenure_active = df.loc[df.churned == 0, "tenure_months"].mean()

# ---------- 9. Cohort retention analysis ----------
# For each signup cohort (month), compute what % are still active as of "today",
# bucketed by months-since-signup -> this approximates a retention curve.
df["months_since_signup"] = ((pd.Timestamp("2026-07-01") - df["signup_date"]).dt.days / 30.44)
df["age_bucket"] = df["months_since_signup"].astype(int)

# Retention curve: of customers who are AT LEAST X months old, what % are still active (not churned)
max_month = 20
retention_curve = []
for m in range(0, max_month + 1):
    cohort = df[df["months_since_signup"] >= m]
    if len(cohort) == 0:
        continue
    # still active means: churned==0, OR churned==1 but their tenure_months > m (they survived past month m)
    survived = ((cohort["churned"] == 0) | (cohort["tenure_months"] > m)).mean()
    retention_curve.append((m, survived))

rc_df = pd.DataFrame(retention_curve, columns=["month", "retention_rate"])
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(rc_df["month"], rc_df["retention_rate"] * 100, marker="o", color="#3d5a80", linewidth=2)
ax.set_title("Customer Retention Curve (% still subscribed by month N)")
ax.set_xlabel("Months since signup")
ax.set_ylabel("% of customers retained")
ax.set_ylim(0, 100)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{CHARTS}/retention_curve.png", dpi=150)
plt.close()

# ---------- 10. Cohort heatmap (signup month x retention by age) ----------
cohort_months = sorted(df["signup_month"].unique())[-12:]  # last 12 cohorts with enough data
heat_data = []
for cm in cohort_months:
    cohort = df[df["signup_month"] == cm]
    row = []
    for m in range(0, 9):
        eligible = cohort[cohort["months_since_signup"] >= m]
        if len(eligible) == 0:
            row.append(np.nan)
            continue
        survived = ((eligible["churned"] == 0) | (eligible["tenure_months"] > m)).mean()
        row.append(survived * 100)
    heat_data.append(row)

heat_df = pd.DataFrame(heat_data, index=cohort_months, columns=[f"M{m}" for m in range(0, 9)])
fig, ax = plt.subplots(figsize=(9, 6))
im = ax.imshow(heat_df.values, cmap="RdYlGn", vmin=50, vmax=100, aspect="auto")
ax.set_xticks(range(len(heat_df.columns)))
ax.set_xticklabels(heat_df.columns)
ax.set_yticks(range(len(heat_df.index)))
ax.set_yticklabels(heat_df.index)
ax.set_title("Cohort Retention Heatmap (% retained by months since signup)")
for i in range(heat_df.shape[0]):
    for j in range(heat_df.shape[1]):
        v = heat_df.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                     color="black" if v > 70 else "white")
plt.colorbar(im, ax=ax, label="% Retained")
plt.tight_layout()
plt.savefig(f"{CHARTS}/cohort_heatmap.png", dpi=150)
plt.close()

# ---------- 11. Churn reasons ----------
churn_reasons = df.loc[df.churned == 1, "churn_reason"].value_counts(normalize=True).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(6.5, 4))
bars = ax.barh(churn_reasons.index[::-1], (churn_reasons.values[::-1] * 100), color="#81b29a")
ax.set_title("Stated Reasons for Churn")
ax.set_xlabel("% of churned customers")
for b, v in zip(bars, churn_reasons.values[::-1] * 100):
    ax.text(v + 0.5, b.get_y() + b.get_height()/2, f"{v:.0f}%", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{CHARTS}/churn_reasons.png", dpi=150)
plt.close()

# ---------- 12. Logistic regression for churn drivers ----------
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

feat_df = df.copy()
plan_dummies = pd.get_dummies(feat_df["plan"], prefix="plan", drop_first=True)
channel_dummies = pd.get_dummies(feat_df["acquisition_channel"], prefix="ch", drop_first=True)
X = pd.concat([
    feat_df[["avg_sessions_per_week", "courses_completed", "support_tickets",
             "used_mobile_app", "autopay_enabled", "monthly_charges"]],
    plan_dummies, channel_dummies
], axis=1)
y = feat_df["churned"]

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)
coefs = pd.Series(model.coef_[0], index=X.columns).sort_values()

fig, ax = plt.subplots(figsize=(7, 5.5))
colors = ["#e07a5f" if c > 0 else "#3d5a80" for c in coefs.values]
ax.barh(coefs.index, coefs.values, color=colors)
ax.set_title("Churn Drivers (logistic regression coefficients)\nRed = increases churn risk, Blue = reduces churn risk")
ax.axvline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.savefig(f"{CHARTS}/churn_drivers.png", dpi=150)
plt.close()

# ---------- Save summary stats for report ----------
summary = {
    "overall_churn_rate": overall_churn,
    "churn_by_plan": churn_by_plan.to_dict(),
    "churn_by_region": churn_by_region.to_dict(),
    "churn_by_channel": churn_by_channel.to_dict(),
    "churn_by_engagement": churn_by_engagement.to_dict(),
    "churn_by_support": churn_by_support.to_dict(),
    "churn_by_autopay": churn_by_autopay.to_dict(),
    "churn_by_mobile": churn_by_mobile.to_dict(),
    "median_tenure_at_churn": median_tenure_churned,
    "avg_tenure_active": avg_tenure_active,
    "churn_reasons": churn_reasons.to_dict(),
    "top_risk_drivers": coefs.sort_values(ascending=False).head(4).to_dict(),
    "top_protective_drivers": coefs.sort_values().head(4).to_dict(),
    "avg_ltv_active": df.loc[df.churned==0, "lifetime_value"].mean(),
    "avg_ltv_churned": df.loc[df.churned==1, "lifetime_value"].mean(),
}

import json
with open("/home/claude/churn/summary_stats.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

print(json.dumps(summary, indent=2, default=str))
