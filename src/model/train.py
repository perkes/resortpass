"""
Training script for the ResortPass hotel ranking model.

This script loads search and clickthrough data, engineers features,
and trains a LightGBM learning-to-rank model. The trained model and
associated artifacts are saved for later use in prediction.
"""

import logging
import os
import pickle
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, ndcg_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- Logger Setup ---
LOGS_DIR = 'logs'
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler(os.path.join(LOGS_DIR, 'train.log'), mode='w')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def _safe_transform(encoder, series):
    """
    Safely transform a series with a pre-fitted LabelEncoder.
    Unseen values are mapped to -1.
    """
    class_map = {cls: i for i, cls in enumerate(encoder.classes_)}
    return series.map(class_map).fillna(-1).astype(int)


def _calculate_user_stats(df):
    """Calculates user behavior statistics."""
    user_stats = df.groupby('user_uuid').agg({
        'total_time_spent': ['mean', 'std', 'count'],
        'used_pictures_carousel': 'mean',
        'saved': 'mean',
        'read_reviews': 'mean',
        'scrolled_to_bottom': 'mean',
        'converted': 'mean'
    }).reset_index()
    user_stats.columns = ['user_uuid'] + [
        f'user_{col[0]}_{col[1]}' if col[1] else f'user_{col[0]}'
        for col in user_stats.columns[1:]
    ]
    return user_stats


def _calculate_user_search_prefs(df):
    """Calculates user search preferences statistics."""
    feature_pref_cols = [
        'feature_hot_tub', 'feature_outdoor_pool', 'feature_rooftop_pool',
        'feature_infinity_pool', 'feature_indoor_pool', 'feature_kiddie_pool',
        'feature_waterpark', 'feature_cabana', 'feature_daybed',
        'feature_splash_pad', 'feature_lazy_river', 'feature_water_slide'
    ]
    amenity_pref_cols = [
        'amenities_beach_access', 'amenities_all_inclusive', 'amenities_free_parking',
        'amenities_luggage_storage', 'amenities_airport_shuttle',
        'amenities_cruise_port_shuttle', 'amenities_gym',
        'amenities_wheelchair_accessible', 'amenities_showers', 'amenities_lockers'
    ]
    vibe_pref_cols = [
        'vibes_family_friendly', 'vibes_party', 'vibes_serene', 'vibes_luxe',
        'vibes_trendy'
    ]
    other_pref_cols = ['top_rated', 'hotel_class_score']

    agg_prefs = {
        col: 'mean' for col in (
            feature_pref_cols + amenity_pref_cols + vibe_pref_cols + other_pref_cols
        )
    }
    user_search_prefs = df.groupby('user_uuid').agg(agg_prefs).reset_index()
    user_search_prefs.columns = ['user_uuid'] + [
        f'user_pref_{col}' for col in user_search_prefs.columns[1:]
    ]
    return user_search_prefs


def _calculate_hotel_stats(df):
    """Calculates hotel statistics."""
    hotel_stats = df.groupby('hotel_id').agg({
        'total_time_spent': ['mean', 'std', 'count'], 'used_pictures_carousel': 'mean',
        'saved': 'mean', 'read_reviews': 'mean', 'scrolled_to_bottom': 'mean',
        'converted': 'mean', 'review_score': 'mean'
    }).reset_index()
    hotel_stats.columns = ['hotel_id'] + [
        f'hotel_{col[0]}_{col[1]}' if col[1] else f'hotel_{col[0]}'
        for col in hotel_stats.columns[1:]
    ]
    return hotel_stats


def load_and_prepare_data():
    """Load and prepare the data for ranking model training"""

    # Load the data
    searches_df = pd.read_csv('./data/searches.csv')
    clickthroughs_df = pd.read_csv('./data/clickthroughs.csv')

    logger.info("Searches shape: %s", searches_df.shape)
    logger.info("Clickthroughs shape: %s", clickthroughs_df.shape)

    # Join the datasets on search_id
    merged_df = clickthroughs_df.merge(
        searches_df, on='search_id', how='inner', suffixes=('', '_search')
    )
    logger.info("Merged data shape: %s", merged_df.shape)

    # Use the user_uuid from clickthroughs (primary) and drop the one from searches if they exist
    if 'user_uuid_search' in merged_df.columns:
        merged_df = merged_df.drop('user_uuid_search', axis=1)

    return merged_df


def _create_time_features(df):
    """Creates time-based features."""
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = pd.to_datetime(df['date'])
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['days_until_trip'] = (df['date'] - df['timestamp'].dt.normalize()).dt.days
    return df


def _fill_na_with_train_means(df, artifacts):
    """Fills missing values in the dataframe with means from the training set."""
    # Fill with training set means
    for col, mean_val in artifacts['user_stats_means'].items():
        df.loc[:, col] = df[col].fillna(mean_val)
    for col, mean_val in artifacts['user_search_prefs_means'].items():
        df.loc[:, col] = df[col].fillna(mean_val)
    for col, mean_val in artifacts['hotel_stats_means'].items():
        df.loc[:, col] = df[col].fillna(mean_val)
    return df


def engineer_features(train_df, test_df):
    """
    Engineer features for the ranking model
    """
    # Create copies to avoid SettingWithCopyWarning
    train_df = train_df.copy()
    test_df = test_df.copy()

    # --- 1. Time-based features ---
    train_df = _create_time_features(train_df)
    test_df = _create_time_features(test_df)

    # --- 2. Calculate aggregations from training data ---
    artifacts = {
        'user_stats': _calculate_user_stats(train_df),
        'user_search_prefs': _calculate_user_search_prefs(train_df),
        'hotel_stats': _calculate_hotel_stats(train_df)
    }

    # --- 3. Merge aggregated features onto both train and test sets ---
    train_df = train_df.merge(artifacts['user_stats'], on='user_uuid', how='left')
    train_df = train_df.merge(artifacts['user_search_prefs'], on='user_uuid', how='left')
    train_df = train_df.merge(artifacts['hotel_stats'], on='hotel_id', how='left')

    test_df = test_df.merge(artifacts['user_stats'], on='user_uuid', how='left')
    test_df = test_df.merge(artifacts['user_search_prefs'], on='user_uuid', how='left')
    test_df = test_df.merge(artifacts['hotel_stats'], on='hotel_id', how='left')

    # --- 4. Fill missing values created by the merge ---
    artifacts['user_stats_means'] = artifacts['user_stats'].drop(columns=['user_uuid']).mean()
    artifacts['user_search_prefs_means'] = artifacts['user_search_prefs'].drop(
        columns=['user_uuid']
    ).mean()
    artifacts['hotel_stats_means'] = artifacts['hotel_stats'].drop(
        columns=['hotel_id']
    ).mean()

    train_df = _fill_na_with_train_means(train_df, artifacts)
    test_df = _fill_na_with_train_means(test_df, artifacts)

    # --- 5. Categorical Encoding ---
    artifacts['le_type'] = LabelEncoder().fit(train_df['type'])
    artifacts['le_city'] = LabelEncoder().fit(train_df['city'])
    artifacts['le_state'] = LabelEncoder().fit(train_df['state'])

    for df in [train_df, test_df]:
        df['type_encoded'] = _safe_transform(artifacts['le_type'], df['type'])
        df['city_encoded'] = _safe_transform(artifacts['le_city'], df['city'])
        df['state_encoded'] = _safe_transform(artifacts['le_state'], df['state'])

    # --- 6. Final Feature Creation ---
    artifacts['base_feature_cols'] = [
        col for col in train_df.columns if col.startswith('feature_') and 'user_pref' not in col
    ]
    artifacts['base_amenity_cols'] = [
        col for col in train_df.columns if col.startswith('amenities_') and 'user_pref' not in col
    ]
    artifacts['base_vibe_cols'] = [
        col for col in train_df.columns if col.startswith('vibes_') and 'user_pref' not in col
    ]

    for df in [train_df, test_df]:
        df['has_review'] = df['review_score'].notna().astype(int)
        df['review_score_filled'] = df['review_score'].fillna(0)
        df['reviewed_beach'] = (df['product_reviewed'] == 'beach').astype(int)
        df['reviewed_pool'] = (df['product_reviewed'] == 'pool').astype(int)
        df['reviewed_other'] = (df['product_reviewed'] == 'other').astype(int)

        user_pref_feature_cols = [f'user_pref_{f}' for f in artifacts['base_feature_cols']]
        user_pref_amenity_cols = [f'user_pref_{a}' for a in artifacts['base_amenity_cols']]
        user_pref_vibe_cols = [f'user_pref_{v}' for v in artifacts['base_vibe_cols']]

        df['feature_match_score'] = (
            df[artifacts['base_feature_cols']].values * df[user_pref_feature_cols].values
        ).sum(axis=1)
        df['amenity_match_score'] = (
            df[artifacts['base_amenity_cols']].values * df[user_pref_amenity_cols].values
        ).sum(axis=1)
        df['vibe_match_score'] = (
            df[artifacts['base_vibe_cols']].values * df[user_pref_vibe_cols].values
        ).sum(axis=1)

        df['time_spent_log'] = np.log1p(df['total_time_spent'])
        df['time_vs_user_avg'] = df['total_time_spent'] / (
            df['user_total_time_spent_mean'] + 1
        )
        df['time_vs_hotel_avg'] = df['total_time_spent'] / (
            df['hotel_total_time_spent_mean'] + 1
        )

    return train_df, test_df, artifacts


def prepare_ranking_data(df):
    """Prepare data in the format required for LightGBM ranking"""

    # Define feature columns (exclude target and ID columns)
    exclude_cols = ['search_id', 'user_uuid', 'hotel_id', 'timestamp', 'date',
                   'converted', 'review_date', 'product_reviewed',
                   'city', 'state', 'type']  # Exclude original categorical columns

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # This assumes the dataframe is sorted by search_id for grouping to work correctly
    df = df.sort_values('search_id').reset_index(drop=True)

    # Prepare features and target
    X = df[feature_cols].fillna(0)
    y = df['converted'].astype(int)

    # Calculate group sizes for LightGBM, grouping by search_id so that the model
    # can compare between the hotels evaluated by a user in a single search
    group_sizes = df.groupby('search_id').size().tolist()

    return X, y, group_sizes, feature_cols


def train_ranking_model(train_data_tuple, test_data_tuple):
    """Train LightGBM ranking model"""

    X_train, y_train, train_group_sizes = train_data_tuple
    X_test, y_test, test_group_sizes = test_data_tuple

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train, group=train_group_sizes)
    test_data = lgb.Dataset(
        X_test, label=y_test, group=test_group_sizes, reference=train_data
    )

    # LightGBM parameters for ranking
    with open('./config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    params = config['model_params']

    # Train the model
    model = lgb.train(
        params,
        train_data,
        valid_sets=[test_data],
        num_boost_round=1000, # max number of boosting iterations
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    return model


def _calculate_ndcg_scores(y_true, y_pred, group_sizes):
    """Helper to calculate NDCG scores for each query group."""
    ndcg_scores = []
    start_idx = 0
    for group_size in group_sizes:
        end_idx = start_idx + group_size
        if group_size > 1:
            y_true_group = y_true[start_idx:end_idx]
            y_pred_group = y_pred[start_idx:end_idx]
            ndcg = ndcg_score(
                np.array(y_true_group).reshape(1, -1),
                np.array(y_pred_group).reshape(1, -1)
            )
            ndcg_scores.append(ndcg)
        start_idx = end_idx
    return ndcg_scores


def evaluate_model(lgbm_model, X_test, y_test, test_group_sizes):
    """Evaluate the ranking model"""

    # Get predictions
    y_pred = lgbm_model.predict(X_test)

    # Calculate NDCG
    ndcg_scores = _calculate_ndcg_scores(y_test, y_pred, test_group_sizes)
    avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0

    # Overall metrics
    auc = roc_auc_score(y_test, y_pred)

    # Binary prediction for accuracy
    y_pred_binary = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred_binary)

    logger.info("\nModel Evaluation Results:")
    logger.info("Average NDCG: %.4f", avg_ndcg)
    logger.info("AUC: %.4f", auc)
    logger.info("Accuracy: %.4f", accuracy)
    logger.info("Number of test queries: %d", len(test_group_sizes))

    return {
        'ndcg': avg_ndcg,
        'auc': auc,
        'accuracy': accuracy,
        'ndcg_scores': ndcg_scores
    }


def save_model_and_artifacts(lgbm_model, feature_cols, artifacts):
    """Save the trained model and all data artifacts needed for prediction."""

    # Create model directory if it doesn't exist
    os.makedirs('./models', exist_ok=True)

    # Save model
    lgbm_model.save_model('./models/hotel_ranking_model.txt')

    # Save all artifacts in one pickle file
    with open('./models/ranking_artifacts.pkl', 'wb') as f:
        # Combine feature_cols into the artifacts dict for saving
        artifacts_to_save = artifacts.copy()
        artifacts_to_save['feature_cols'] = feature_cols
        pickle.dump(artifacts_to_save, f)

    logger.info("Model and artifacts saved to ./models/")


def _load_and_split_data():
    """Loads and splits the data for training and testing."""
    df = load_and_prepare_data()
    search_ids = df['search_id'].unique()
    train_ids, test_ids = train_test_split(
        search_ids, test_size=0.2, random_state=42
    )
    train_df = df[df['search_id'].isin(train_ids)].copy()
    test_df = df[df['search_id'].isin(test_ids)].copy()
    return train_df, test_df


def main():
    """Main training pipeline"""

    logger.info("Loading and preparing data...")
    train_df, test_df = _load_and_split_data()

    logger.info("Training set size: %d, Test set size: %d", len(train_df), len(test_df))

    logger.info("Engineering features...")
    train_df, test_df, artifacts = engineer_features(train_df, test_df)

    logger.info("Preparing ranking data...")
    X_train, y_train, train_group_sizes, feature_cols = prepare_ranking_data(train_df)
    X_test, y_test, test_group_sizes, _ = prepare_ranking_data(test_df)

    # Align columns - crucial for prediction
    X_test = X_test[feature_cols]

    logger.info("Dataset shape (train): %s", X_train.shape)
    logger.info("Number of features: %d", len(feature_cols))
    logger.info("Number of query groups (train): %d", len(train_group_sizes))

    logger.info("Training ranking model...")
    train_data_tuple = (X_train, y_train, train_group_sizes)
    test_data_tuple = (X_test, y_test, test_group_sizes)
    model = train_ranking_model(train_data_tuple, test_data_tuple)

    logger.info("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test, test_group_sizes)
    logger.info("\nReturned Evaluation Metrics:")
    # Printing the metrics dictionary
    logger.info("{")
    logger.info("  'ndcg': %.4f,", metrics['ndcg'])
    logger.info("  'auc': %.4f,", metrics['auc'])
    logger.info("  'accuracy': %.4f,", metrics['accuracy'])
    logger.info("}")

    logger.info("Saving model and artifacts...")
    save_model_and_artifacts(model, feature_cols, artifacts)

    # Feature importance
    feature_importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)

    logger.info("\nTop 20 Most Important Features:")
    logger.info("%s", feature_importance_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
