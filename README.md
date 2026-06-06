# Amazon Beauty Product Recommendation System: Deliverable 2

This folder contains the individual Project 2 submission for hyperparameter
tuning, final model evaluation, ethical considerations, and real-world
recommendations using the Kaggle Amazon Beauty product ratings dataset.

Dataset:

https://www.kaggle.com/datasets/satrapankti/amazon-beauty-product-recommendation

## Files

- `amazon_beauty_recommendation_model_tuning.py` - Tunes and evaluates a Surprise SVD product recommender.
- `generate_amazon_beauty_recommendation_report.py` - Generates the APA-style Word report.
- `Amazon_Beauty_Recommendation_Deliverable2_Report.docx` - Report generated after running the report script.
- `Amazon_Beauty_Recommendation_sample_20000.csv` - Reproducible 20,000-rating sample used for this project.
- `Amazon_Beauty_Recommendation_dataset.csv` - Optional full Kaggle dataset; not required for submission.
- `outputs/amazon_beauty_final_model_summary.csv` - Final tuned model metrics.
- `outputs/amazon_beauty_grid_search_results.csv` - GridSearchCV results.
- `outputs/amazon_beauty_prediction_comparison.csv` - Actual vs. predicted test ratings.
- `outputs/amazon_beauty_prediction_sample.csv` - Small prediction sample.
- `outputs/amazon_beauty_top_10_recommendations.csv` - Example top-10 product recommendations.
- `outputs/amazon_beauty_product_type_profile.csv` - Product category counts used in the modeled sample.

## Dataset Setup

The submitted project uses this reproducible sample:

```text
project2/Amazon_Beauty_Recommendation_sample_20000.csv
```

The expected columns are:

```text
UserId, ProductId, ProductType, Rating, Timestamp, URL
```

The full Kaggle file is optional because it is large. If you want to recreate
the sample from the raw dataset, keep the full file as:

```text
project2/Amazon_Beauty_Recommendation_dataset.csv
```

## Run

```bash
.venv/bin/python amazon_beauty_recommendation_model_tuning.py
.venv/bin/python generate_amazon_beauty_recommendation_report.py
```

The model script uses a default sample of `20000` ratings from the large Kaggle
dataset so GridSearchCV remains practical for a class project. To change the
sample size:

```bash
.venv/bin/python amazon_beauty_recommendation_model_tuning.py --max-ratings 300000
```

## Assignment Alignment

The workflow uses:

- `GridSearchCV`
- `SVD`
- `n_factors = [50, 100]`
- `n_epochs = [20, 40]`
- `lr_all = [0.002, 0.005]`
- RMSE and MAE final evaluation
- Actual vs. predicted rating exports
- Ethical discussion focused on product, review, popularity, privacy, and fairness bias
