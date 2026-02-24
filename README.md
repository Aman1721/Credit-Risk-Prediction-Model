Credit Risk Prediction Model

Banks and lenders make thousands of loan decisions every day, and getting those wrong is expensive. This project builds a machine learning pipeline that looks at an applicant's financial profile and estimates the probability they'll default on a loan — then uses that probability to make an actual approve/reject call.
The whole thing is broken into clean, separate modules so it's easy to follow, modify, or drop into a larger codebase.




What it does?
Takes raw loan application data (credit score, income, debt-to-income ratio, employment history, etc.), runs it through feature engineering, trains three different classifiers, compares them, and picks the best one. From there it simulates different decision thresholds to find the cutoff that makes the most business sense — not just the one that looks best on paper.
At the end you can pass in any new applicant and get back a default probability, a risk band, and a decision.


Getting started:

install dependencies (only needs to happen once)
pip install -r requirements.txt

# run the full pipeline
python main.py

# run tests
python tests/test_pipeline.py