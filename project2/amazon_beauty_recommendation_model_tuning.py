#!/usr/bin/env python3
"""
Amazon Beauty product recommendation workflow for Deliverable 2.

Expected Kaggle dataset:
https://www.kaggle.com/datasets/satrapankti/amazon-beauty-product-recommendation

Place the downloaded ratings file at:
    Amazon_Beauty_Recommendation_sample_20000.csv

The script tunes a Surprise SVD collaborative-filtering model with GridSearchCV,
evaluates the optimized model with RMSE and MAE, and exports tables for the
APA report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import GridSearchCV, train_test_split


RANDOM_STATE = 42
DATA_PATH = Path("Amazon_Beauty_Recommendation_sample_20000.csv")
OUTPUT_DIR = Path("outputs")
MAX_RATINGS_DEFAULT = 20_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune and evaluate an Amazon Beauty product recommender."
    )
    parser.add_argument(
        "--data-path",
        default=str(DATA_PATH),
        help="Path to the Amazon Beauty recommendation CSV.",
    )
    parser.add_argument(
        "--max-ratings",
        type=int,
        default=MAX_RATINGS_DEFAULT,
        help=(
            "Maximum ratings to use for tuning/evaluation. The project samples "
            "from the large Kaggle file so GridSearchCV remains practical."
        ),
    )
    parser.add_argument(
        "--min-user-ratings",
        type=int,
        default=1,
        help="Keep users with at least this many ratings before sampling.",
    )
    parser.add_argument(
        "--min-product-ratings",
        type=int,
        default=1,
        help="Keep products with at least this many ratings before sampling.",
    )
    return parser.parse_args()


def load_amazon_beauty_ratings(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing dataset file: {path}\n\n"
            "Download the Kaggle dataset from:\n"
            "https://www.kaggle.com/datasets/satrapankti/amazon-beauty-product-recommendation\n\n"
            "Then place Amazon_Beauty_Recommendation_sample_20000.csv in project2/ and rerun:\n"
            ".venv/bin/python amazon_beauty_recommendation_model_tuning.py\n"
        )

    ratings = pd.read_csv(path)
    expected = {"UserId", "ProductId", "Rating", "Timestamp"}
    if not expected.issubset(ratings.columns):
        ratings = pd.read_csv(
            path,
            header=None,
            names=["UserId", "ProductId", "Rating", "Timestamp"],
        )

    keep_columns = [
        column
        for column in ["UserId", "ProductId", "ProductType", "Rating", "Timestamp", "URL"]
        if column in ratings.columns
    ]
    ratings = ratings[keep_columns].copy()
    ratings["Rating"] = pd.to_numeric(ratings["Rating"], errors="coerce")
    ratings = ratings.dropna(subset=["UserId", "ProductId", "Rating"])
    ratings = ratings[(ratings["Rating"] >= 1) & (ratings["Rating"] <= 5)]
    return ratings


def filter_and_sample(
    ratings: pd.DataFrame,
    max_ratings: int,
    min_user_ratings: int,
    min_product_ratings: int,
) -> pd.DataFrame:
    user_counts = ratings["UserId"].value_counts()
    product_counts = ratings["ProductId"].value_counts()
    filtered = ratings[
        ratings["UserId"].isin(user_counts[user_counts >= min_user_ratings].index)
        & ratings["ProductId"].isin(product_counts[product_counts >= min_product_ratings].index)
    ].copy()

    if len(filtered) > max_ratings:
        filtered = filtered.sample(n=max_ratings, random_state=RANDOM_STATE)

    return filtered.reset_index(drop=True)


def to_surprise_dataset(ratings: pd.DataFrame) -> Dataset:
    reader = Reader(rating_scale=(1, 5))
    surprise_frame = ratings[["UserId", "ProductId", "Rating"]]
    return Dataset.load_from_df(surprise_frame, reader)


def prediction_rows(predictions):
    for pred in predictions:
        yield {
            "user_id": pred.uid,
            "product_id": pred.iid,
            "actual_rating": pred.r_ui,
            "predicted_rating": round(pred.est, 3),
            "absolute_error": round(abs(pred.r_ui - pred.est), 3),
        }


def product_metadata(ratings: pd.DataFrame) -> pd.DataFrame:
    metadata_columns = [
        column for column in ["ProductId", "ProductType", "URL"] if column in ratings.columns
    ]
    if metadata_columns == ["ProductId"]:
        return pd.DataFrame({"ProductId": ratings["ProductId"].unique()})
    return ratings[metadata_columns].drop_duplicates(subset=["ProductId"])


def add_product_metadata(frame: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    if metadata.empty:
        return frame
    return frame.merge(
        metadata,
        how="left",
        left_on="product_id",
        right_on="ProductId",
    ).drop(columns=["ProductId"], errors="ignore")


def build_top_n_recommendations(best_model, trainset, user_raw_id: str, n: int = 10) -> pd.DataFrame:
    anti_testset = trainset.build_anti_testset()
    user_items = [row for row in anti_testset if row[0] == user_raw_id]
    predictions = best_model.test(user_items)
    top_predictions = sorted(predictions, key=lambda pred: pred.est, reverse=True)[:n]

    return pd.DataFrame(
        {
            "user_id": pred.uid,
            "product_id": pred.iid,
            "estimated_rating": round(pred.est, 3),
        }
        for pred in top_predictions
    )


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    OUTPUT_DIR.mkdir(exist_ok=True)

    raw_ratings = load_amazon_beauty_ratings(data_path)
    metadata = product_metadata(raw_ratings)
    ratings = filter_and_sample(
        raw_ratings,
        max_ratings=args.max_ratings,
        min_user_ratings=args.min_user_ratings,
        min_product_ratings=args.min_product_ratings,
    )

    data_surprise = to_surprise_dataset(ratings)
    trainset, testset = train_test_split(
        data_surprise,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    param_grid = {
        "n_factors": [50, 100],
        "n_epochs": [20, 40],
        "lr_all": [0.002, 0.005],
        "random_state": [RANDOM_STATE],
    }

    grid_search = GridSearchCV(SVD, param_grid, measures=["rmse"], cv=3, n_jobs=1)
    grid_search.fit(data_surprise)

    best_params = grid_search.best_params["rmse"]
    best_model = SVD(**best_params)
    best_model.fit(trainset)

    predictions = best_model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)

    cv_results = pd.DataFrame(grid_search.cv_results)
    cv_results.to_csv(OUTPUT_DIR / "amazon_beauty_grid_search_results.csv", index=False)

    comparison = pd.DataFrame(prediction_rows(predictions))
    comparison = add_product_metadata(comparison, metadata)
    comparison.sort_values("absolute_error", ascending=False).to_csv(
        OUTPUT_DIR / "amazon_beauty_prediction_comparison.csv",
        index=False,
    )
    comparison.head(25).to_csv(
        OUTPUT_DIR / "amazon_beauty_prediction_sample.csv",
        index=False,
    )

    sample_user = ratings["UserId"].value_counts().index[0]
    top_n = build_top_n_recommendations(best_model, trainset, user_raw_id=sample_user)
    top_n = add_product_metadata(top_n, metadata)
    top_n.to_csv(OUTPUT_DIR / "amazon_beauty_top_10_recommendations.csv", index=False)

    if "ProductType" in ratings.columns:
        ratings["ProductType"].value_counts().rename_axis("product_type").reset_index(
            name="rating_count"
        ).to_csv(OUTPUT_DIR / "amazon_beauty_product_type_profile.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "dataset": "Amazon Beauty Product Ratings",
                "source": "Kaggle: satrapankti/amazon-beauty-product-recommendation",
                "algorithm": "SVD matrix factorization",
                "raw_ratings": len(raw_ratings),
                "modeled_ratings": len(ratings),
                "unique_users": ratings["UserId"].nunique(),
                "unique_products": ratings["ProductId"].nunique(),
                "unique_product_types": (
                    ratings["ProductType"].nunique() if "ProductType" in ratings.columns else ""
                ),
                "train_ratings": trainset.n_ratings,
                "test_ratings": len(testset),
                "best_n_factors": best_params["n_factors"],
                "best_n_epochs": best_params["n_epochs"],
                "best_lr_all": best_params["lr_all"],
                "test_rmse": round(rmse, 4),
                "test_mae": round(mae, 4),
            }
        ]
    )
    summary.to_csv(OUTPUT_DIR / "amazon_beauty_final_model_summary.csv", index=False)

    print("Amazon Beauty recommendation workflow complete.")
    print(summary.to_string(index=False))
    print(f"\nSaved outputs to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
