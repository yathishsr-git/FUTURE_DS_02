StreamLearn — Customer Retention & Churn Analysis
Data Science & Analytics — Task 2 (2026), Future Interns
📌 Overview
This project analyzes subscriber behavior for StreamLearn, a simulated online course subscription platform, to understand why customers churn and what drives them to stay. The goal was to move beyond descriptive charts and produce actionable, business-ready insights — the kind a product manager or founder could act on directly.
❓ Problem Statement
Why are customers leaving the platform?
Which customer segments are most likely to churn?
How long do customers typically stay active before churning?
What specific actions would reduce churn and improve retention?
🗂️ Dataset
A synthetic dataset of 5,000 subscriber records was generated to reflect realistic SaaS/subscription patterns (modeled on public churn datasets such as the Telco Customer Churn dataset on Kaggle). Each record includes:
signup_date, plan (Basic/Standard/Premium), region, acquisition_channel, avg_sessions_per_week, courses_completed, support_tickets, used_mobile_app, autopay_enabled, tenure_months, churned, churn_reason, lifetime_value
The data generation logic (01_generate_data.py) deliberately encodes realistic churn drivers — low engagement, entry-level plans, no autopay — so the downstream analysis mirrors patterns seen in real subscription businesses.
🛠️ Approach & Tools
Python (pandas, NumPy, Matplotlib, scikit-learn) for data generation, churn-rate analysis, cohort survival analysis, and a logistic regression to identify churn drivers.
Excel for a business-friendly workbook with live formulas (COUNTIF/AVERAGEIF) reproducing churn rates by segment, plus a cohort retention table.
Word for the final stakeholder-facing report — executive summary, findings, and recommendations.
📊 Key Insights
Overall churn rate: 10.6%, with the median churned customer leaving after just 2.5 months.
Engagement is the strongest churn predictor — customers with 4+ sessions/week churn at 6.5%, vs. 14.0% for those under 1 session/week.
The first 90 days are the highest-risk window; retention flattens significantly after month 3.
Basic-plan subscribers churn ~63% more than Premium subscribers (12.9% vs. 7.9%).
Autopay and mobile app adoption are strong protective factors — both are levers the business can influence at onboarding.
Active subscribers carry ~4x the lifetime value of churned ones ($234 vs. $58 average LTV).
✅ Recommendations
Build a structured onboarding sequence targeting the month 0–3 danger zone.
Trigger re-engagement flows when weekly sessions drop below 1/week early in the lifecycle.
Test pricing/upgrade incentives for engaged Basic-plan users.
Drive mobile app and autopay adoption at signup.
Proactively flag customers with 2+ support tickets for a customer-success check-in.
📁 Repository Contents
File
Description
StreamLearn_Churn_Report.docx
Final stakeholder report — findings, visuals, and recommendations
StreamLearn_Churn_Analysis.xlsx
Excel workbook with raw data, live churn-rate formulas, and cohort retention table
subscription_data.csv
Underlying dataset (5,000 rows)
01_generate_data.py
Script to generate the synthetic subscriber dataset
02_analysis.py
Script for churn analysis, cohort retention, and driver modeling
🚀 How to Reproduce
Bash
⚠️ Disclaimer
All data in this project is synthetically generated for demonstration purposes and does not represent a real company or real customers.
Built as part of the Future Interns Data Science & Analytics Program.
