"""
Training script for the ResortPass hotel ranking model.

This script loads search and clickthrough data, engineers features,
and trains a LightGBM learning-to-rank model. The trained model and
associated artifacts are saved for later use in prediction.
"""
import json
import logging
import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import LabelEncoder

from src.model.features import (
    create_time_features,
    apply_categorical_encoding, identify_base_columns, create_derived_features,
    fill_missing_with_defaults, ensure_feature_columns,
    calculate_temporal_stats
)

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


def _calculate_entity_stats(df, entity_col, prefix):
    """
    Calculates entity-level statistics (e.g., for user or hotel).
    Generic function to compute statistics for a given entity (user or hotel).
    """
    agg_dict = {
        'total_time_spent': ['mean', 'std', 'count'],
        'used_pictures_carousel': 'mean',
        'saved': 'mean',
        'read_reviews': 'mean',
        'scrolled_to_bottom': 'mean',
        'converted': 'mean'
    }
    # Add review_score for hotels only
    if entity_col == 'hotel_id':
        agg_dict['review_score'] = ['mean', 'std']

    stats = df.groupby(entity_col).agg(agg_dict).reset_index()

    # Create new column names with the given prefix
    stats.columns = [entity_col] + [
        f'{prefix}_{col[0]}_{col[1]}' if col[1] else f'{prefix}_{col[0]}'
        for col in stats.columns[1:]
    ]
    return stats


def _calculate_user_search_prefs(df):
    """Calculates user search preferences statistics."""
    # These are the columns created by the merge in load_and_prepare_data
    feature_pref_cols = [
        'feature_hot_tub_search', 'feature_outdoor_pool_search', 'feature_rooftop_pool_search',
        'feature_infinity_pool_search', 'feature_indoor_pool_search', 'feature_kiddie_pool_search',
        'feature_waterpark_search', 'feature_cabana_search', 'feature_daybed_search',
        'feature_splash_pad_search', 'feature_lazy_river_search', 'feature_water_slide_search'
    ]
    amenity_pref_cols = [
        'amenities_beach_access_search', 'amenities_all_inclusive_search', 'amenities_free_parking_search',
        'amenities_luggage_storage_search', 'amenities_airport_shuttle_search',
        'amenities_cruise_port_shuttle_search', 'amenities_gym_search',
        'amenities_wheelchair_accessible_search', 'amenities_showers_search', 'amenities_lockers_search'
    ]
    vibe_pref_cols = [
        'vibes_family_friendly_search', 'vibes_party_search', 'vibes_serene_search', 'vibes_luxe_search',
        'vibes_trendy_search'
    ]
    other_pref_cols = ['top_rated_search', 'hotel_class_score_search']

    # Check which of these columns actually exist in the dataframe to avoid KeyErrors
    all_pref_cols = feature_pref_cols + amenity_pref_cols + vibe_pref_cols + other_pref_cols
    existing_cols = [col for col in all_pref_cols if col in df.columns]

    agg_prefs = {col: 'mean' for col in existing_cols}

    if not agg_prefs:
        return pd.DataFrame(columns=['user_uuid'])

    user_search_prefs = df.groupby('user_uuid').agg(agg_prefs).reset_index()

    # Rename columns to the format 'user_pref_...'
    rename_dict = {
        col: f'user_pref_{col.replace("_search", "")}'
        for col in user_search_prefs.columns if col != 'user_uuid'
    }
    user_search_prefs = user_search_prefs.rename(columns=rename_dict)

    return user_search_prefs


def _calculate_user_hotel_stats(df):
    """Calculates user-hotel interaction statistics."""
    user_hotel_stats = df.groupby(['user_uuid', 'hotel_id']).agg(
        user_hotel_conversions=('converted', 'sum'),
        user_hotel_views=('search_id', 'count'),
        user_hotel_saves=('saved', 'sum'),
        user_hotel_read_reviews=('read_reviews', 'sum'),
        user_hotel_carousel=('used_pictures_carousel', 'sum'),
        user_hotel_scrolled=('scrolled_to_bottom', 'sum'),
        user_hotel_time_spent=('total_time_spent', 'sum')
    ).reset_index()

    user_hotel_stats['user_hotel_conversion_rate'] = (
        user_hotel_stats['user_hotel_conversions'] / user_hotel_stats['user_hotel_views']
    )

    return user_hotel_stats


def load_and_prepare_data():
    """Load and prepare the data for ranking model training"""

    # Load the data
    searches_df = pd.read_csv('./data/searches.csv')
    clickthroughs_df = pd.read_csv('./data/clickthroughs.csv')
    hotels_df = pd.read_csv('./data/hotels.csv')

    logger.info("Searches shape: %s", searches_df.shape)
    logger.info("Clickthroughs shape: %s", clickthroughs_df.shape)
    logger.info("Hotels shape: %s", hotels_df.shape)

    # Rename all columns in searches_df except the join key 'search_id'
    # to avoid any conflicts during merges.
    search_rename_map = {
        col: f'{col}_search' for col in searches_df.columns if col != 'search_id'
    }
    searches_df = searches_df.rename(columns=search_rename_map)

    # Join clickthroughs with the renamed searches dataframe
    merged_df = clickthroughs_df.merge(searches_df, on='search_id', how='inner')

    # Now, join with the ground-truth hotel data from hotels.csv
    # There will be no column conflicts here.
    merged_df = merged_df.merge(hotels_df, on='hotel_id', how='left')

    logger.info("Merged data shape: %s", merged_df.shape)

    # We don't need the user_uuid and timestamp from the search data,
    # as we are using the ones from the clickthrough data.
    if 'user_uuid_search' in merged_df.columns:
        merged_df = merged_df.drop(columns=['user_uuid_search'])
    if 'timestamp_search' in merged_df.columns:
        merged_df = merged_df.drop(columns=['timestamp_search'])

    # Convert timestamp to datetime
    merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'])

    return merged_df


def engineer_features(train_df, test_df):
    """
    Engineer features for the ranking model
    """
    # Store original split information to reassemble later
    train_df = train_df.copy().reset_index(drop=True)
    test_df = test_df.copy().reset_index(drop=True)
    train_df['_original_split'] = 'train'
    test_df['_original_split'] = 'test'
    
    # Combine datasets for temporal feature calculation
    # This ensures test data gets proper historical context from training data
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    logger.info("Combined dataset shape for temporal features: %s", combined_df.shape)

    # --- 1. Time-based features ---
    combined_df = create_time_features(combined_df, timestamp_col='timestamp', date_col='date_search')

    # --- 2. Calculate temporal statistics using the unified function ---
    # This processes the entire timeline and ensures each row only sees past data
    combined_df = calculate_temporal_stats(combined_df)
    
    # --- 3. Split back into train and test sets ---
    train_df_processed = combined_df[combined_df['_original_split'] == 'train'].copy()
    test_df_processed = combined_df[combined_df['_original_split'] == 'test'].copy()
    
    # Remove the split indicator column
    train_df_processed = train_df_processed.drop(columns=['_original_split'])
    test_df_processed = test_df_processed.drop(columns=['_original_split'])
    
    logger.info("Processed train shape: %s", train_df_processed.shape)
    logger.info("Processed test shape: %s", test_df_processed.shape)
    
    # --- 4. Keep aggregated stats from training data for prediction fallbacks ---
    artifacts = {
        'user_stats': _calculate_entity_stats(train_df_processed, 'user_uuid', 'user'),
        'user_search_prefs': _calculate_user_search_prefs(train_df_processed),
        'hotel_stats': _calculate_entity_stats(train_df_processed, 'hotel_id', 'hotel'),
        'user_hotel_stats': _calculate_user_hotel_stats(train_df_processed)
    }

    # --- 5. Fill any remaining missing values ---
    train_df_processed = fill_missing_with_defaults(train_df_processed)
    test_df_processed = fill_missing_with_defaults(test_df_processed)

    # --- 6. Keep these for prediction fallbacks ---
    artifacts['user_stats_means'] = artifacts['user_stats'].drop(columns=['user_uuid']).mean()
    artifacts['user_search_prefs_means'] = artifacts['user_search_prefs'].drop(
        columns=['user_uuid']
    ).mean()
    artifacts['hotel_stats_means'] = artifacts['hotel_stats'].drop(
        columns=['hotel_id']
    ).mean()

    # --- 7. Categorical Encoding ---
    artifacts['encoders'] = {
        'type': LabelEncoder().fit(train_df_processed['type_search']),
        'city': LabelEncoder().fit(train_df_processed['city_search']),
        'state': LabelEncoder().fit(train_df_processed['state_search'])
    }

    train_df_processed = apply_categorical_encoding(train_df_processed, artifacts['encoders'])
    test_df_processed = apply_categorical_encoding(test_df_processed, artifacts['encoders'])

    # --- 8. Final Feature Creation ---
    base_cols_dict = identify_base_columns(train_df_processed)
    artifacts['base_feature_cols'] = base_cols_dict['feature_cols']
    artifacts['base_amenity_cols'] = base_cols_dict['amenity_cols']
    artifacts['base_vibe_cols'] = base_cols_dict['vibe_cols']

    train_df_processed = create_derived_features(train_df_processed, base_cols_dict)
    test_df_processed = create_derived_features(test_df_processed, base_cols_dict)

    # --- 9. Define final feature columns ---
    feature_cols = [
        # Time features
        'hour_of_day', 'day_of_week', 'days_until_trip',
        # Encoded categorical features
        'type_encoded', 'city_encoded', 'state_encoded',
        # User stats
        'user_total_time_spent_mean', 'user_total_time_spent_std', 'user_total_time_spent_count',
        'user_used_pictures_carousel_mean', 'user_saved_mean', 'user_read_reviews_mean',
        'user_scrolled_to_bottom_mean', 'user_converted_mean',
        # User search preferences
        'user_pref_feature_hot_tub', 'user_pref_feature_outdoor_pool',
        'user_pref_feature_rooftop_pool', 'user_pref_feature_infinity_pool',
        'user_pref_feature_indoor_pool', 'user_pref_feature_kiddie_pool',
        'user_pref_feature_waterpark', 'user_pref_feature_cabana',
        'user_pref_feature_daybed', 'user_pref_feature_splash_pad',
        'user_pref_feature_lazy_river', 'user_pref_feature_water_slide',
        'user_pref_amenities_beach_access', 'user_pref_amenities_all_inclusive',
        'user_pref_amenities_free_parking', 'user_pref_amenities_luggage_storage',
        'user_pref_amenities_airport_shuttle', 'user_pref_amenities_cruise_port_shuttle',
        'user_pref_amenities_gym', 'user_pref_amenities_wheelchair_accessible',
        'user_pref_amenities_showers', 'user_pref_amenities_lockers',
        'user_pref_vibes_family_friendly', 'user_pref_vibes_party',
        'user_pref_vibes_serene', 'user_pref_vibes_luxe', 'user_pref_vibes_trendy',
        'user_pref_top_rated', 'user_pref_hotel_class_score',
        # Hotel stats
        'hotel_total_time_spent_mean', 'hotel_total_time_spent_std', 'hotel_total_time_spent_count',
        'hotel_used_pictures_carousel_mean', 'hotel_saved_mean', 'hotel_read_reviews_mean',
        'hotel_scrolled_to_bottom_mean', 'hotel_converted_mean', 'hotel_review_score_mean',
        'hotel_review_score_std',
        # User-Hotel interaction stats
        'user_hotel_views', 'user_hotel_saves', 'user_hotel_read_reviews',
        'user_hotel_carousel', 'user_hotel_scrolled', 'user_hotel_time_spent',
        'user_hotel_conversion_rate',
        # Match scores
        'feature_match_score', 'amenity_match_score', 'vibe_match_score', 'overall_match_score',
        'class_score_diff', 'class_score_match', 'class_score_exact', 'top_rated_match',
        # Other features
        'has_review', 'review_score_filled', 'reviewed_beach', 'reviewed_pool', 'reviewed_other',
        'time_vs_user_avg', 'time_vs_hotel_avg'
    ]

    # Add base features that are also in the final list
    feature_cols.extend(artifacts['base_feature_cols'])
    feature_cols.extend(artifacts['base_amenity_cols'])
    feature_cols.extend(artifacts['base_vibe_cols'])
    feature_cols.append('top_rated')
    feature_cols.append('hotel_class_score')

    # Ensure all feature columns are present in both datasets using helper function
    train_df_processed = ensure_feature_columns(train_df_processed, feature_cols)
    test_df_processed = ensure_feature_columns(test_df_processed, feature_cols)
    
    # Store the final feature columns
    artifacts['feature_cols'] = feature_cols

    return train_df_processed, test_df_processed, artifacts


def prepare_ranking_data(df):
    """Prepare data for LightGBM ranking model"""

    # Define feature columns (exclude target, ID columns, and ALL search filter columns)
    exclude_cols = ['search_id', 'user_uuid', 'hotel_id', 'timestamp', 'date', 'date_search',
                   'converted', 'review_date', 'product_reviewed']
    
    # Exclude ALL search filter columns (anything ending with '_search')
    # The current search filters are only being used to produce candidate hotels
    # and for feature engineering
    search_filter_cols = [col for col in df.columns if col.endswith('_search')]
    exclude_cols.extend(search_filter_cols)

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
        num_boost_round=params['num_boost_round'], # max number of boosting iterations
        callbacks=[lgb.early_stopping(stopping_rounds=params['stopping_rounds'], verbose=False)]
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

    logger.info("\nModel Evaluation Results:")
    logger.info("Average NDCG: %.4f", avg_ndcg)
    logger.info("Number of test queries: %d", len(test_group_sizes))

    return {
        'ndcg': avg_ndcg,
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
    """Loads and splits the data for training and testing using a temporal split."""
    df = load_and_prepare_data()

    # Ensure timestamp is present for sorting
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Get the first timestamp for each search_id
    search_time_df = df.groupby('search_id')['timestamp'].min().sort_values().reset_index()

    # Split search_ids based on time
    split_point = int(len(search_time_df) * 0.8)
    train_ids = search_time_df.loc[:split_point, 'search_id'].unique()
    test_ids = search_time_df.loc[split_point:, 'search_id'].unique()

    train_df = df[df['search_id'].isin(train_ids)].copy()
    test_df = df[df['search_id'].isin(test_ids)].copy()

    logger.info(
        "Data split temporally: Train includes searches up to %s, Test includes searches after %s",
        train_df['timestamp'].max(),
        test_df['timestamp'].min()
    )

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
    evaluate_model(model, X_test, y_test, test_group_sizes)

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
