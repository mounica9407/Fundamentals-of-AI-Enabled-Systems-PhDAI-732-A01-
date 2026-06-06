# Dataset File

Download the Kaggle dataset:

https://www.kaggle.com/datasets/satrapankti/amazon-beauty-product-recommendation

The current project uses the sampled dataset file at the project root:

```text
project2/Amazon_Beauty_Recommendation_sample_20000.csv
```

Expected columns:

```text
UserId, ProductId, ProductType, Rating, Timestamp, URL
```

After adding the file, run:

```bash
cd "/Users/mounica/Documents/tech files/Cumberlands_PHDAI_732/project2"
.venv/bin/python amazon_beauty_recommendation_model_tuning.py
.venv/bin/python generate_amazon_beauty_recommendation_report.py
```
