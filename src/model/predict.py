"""
Prediction script for the ResortPass hotel ranking model.

This script defines the `HotelRankingPredictor` class, which loads a pre-trained
LightGBM model and associated artifacts to predict hotel rankings for a given user
and search query.
"""
import json
import logging
import os
import pickle

import lightgbm as lgb
import pandas as pd

from src.model.features import (
    create_time_features,
    apply_categorical_encoding, create_derived_features,
    ensure_feature_columns,
    add_user_features, add_hotel_features, add_user_hotel_features
)

pd.set_option('display.max_rows', None)

# --- Logger Setup ---
LOGS_DIR = 'logs'
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

logger = logging.getLogger('predict')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler(os.path.join(LOGS_DIR, 'predict.log'), mode='w')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class HotelRankingPredictor:
    """
    A class to predict hotel rankings for users based on their search preferences
    and historical behavior patterns.
    """

    def __init__(self, model_path='./models/hotel_ranking_model.txt',
                 artifacts_path='./models/ranking_artifacts.pkl'):
        """
        Initialize the predictor with trained model and artifacts.

        Args:
            model_path: Path to the trained LightGBM model.
            artifacts_path: Path to the saved artifacts (encoders, features, stats).
        """
        self.model = lgb.Booster(model_file=model_path)

        with open(artifacts_path, 'rb') as f:
            artifacts = pickle.load(f)

        self.feature_cols = artifacts['feature_cols']
        self.encoders = artifacts['encoders']
        self.user_stats_means = artifacts['user_stats_means']
        self.user_search_prefs_means = artifacts['user_search_prefs_means']
        self.hotel_stats_means = artifacts['hotel_stats_means']
        self.stats = {
            'user': artifacts['user_stats'],
            'user_prefs': artifacts['user_search_prefs'],
            'hotel': artifacts['hotel_stats'],
            'user_hotel': artifacts.get('user_hotel_stats', pd.DataFrame())
        }
        self.base_feature_cols = artifacts.get('base_feature_cols', [])
        self.base_amenity_cols = artifacts.get('base_amenity_cols', [])
        self.base_vibe_cols = artifacts.get('base_vibe_cols', [])

        # Load the ground-truth hotel data
        try:
            self.hotels_df = pd.read_csv('./data/hotels.csv')
        except FileNotFoundError:
            logger.error("hotels.csv not found. Please generate it first.")
            self.hotels_df = pd.DataFrame()


    def build_prediction_set(self, search_preferences, num_hotels=20):
        """
        Build a DataFrame for prediction by combining candidate hotel IDs with search preferences.

        This method filters hotels based on search criteria to get valid candidates,
        then prepares them for ranking.

        Args:
            search_preferences: Dict with the user's search criteria.
            num_hotels: The number of candidate hotels to generate for ranking.

        Returns:
            A DataFrame where each row represents a hotel to be scored.
        """
        if self.hotels_df.empty:
            logger.error("Cannot build prediction set without hotel data.")
            return pd.DataFrame()

        # Filter candidate hotels based on search criteria
        candidate_hotels = self.hotels_df.copy()
        
        # Apply basic filters that would typically be done in a search/retrieval system
        # Note: In a real system, this would be more sophisticated with geographic filters,
        # availability checks, etc.
        
        # Filter by minimum hotel class if specified
        if search_preferences.get('hotel_class_score'):
            candidate_hotels = candidate_hotels[
                candidate_hotels['hotel_class_score'] >= search_preferences['hotel_class_score']
            ]
        
        # Filter by top_rated if specifically requested
        if search_preferences.get('top_rated', False):
            candidate_hotels = candidate_hotels[
                candidate_hotels['top_rated'] == True
            ]
        
        # Filter by specific features if requested (example: if searching for pools)
        search_type = search_preferences.get('type', 'ALL')
        if search_type == 'POOL':
            # For pool searches, prefer hotels with pool-related features
            pool_features = ['feature_outdoor_pool', 'feature_indoor_pool', 'feature_rooftop_pool', 'feature_infinity_pool']
            pool_mask = candidate_hotels[pool_features].any(axis=1)
            if pool_mask.any():  # Only filter if there are hotels with pools
                candidate_hotels = candidate_hotels[pool_mask]
        elif search_type == 'SPA':
            # For spa searches, prefer hotels with hot tubs or serene vibes
            spa_mask = (candidate_hotels['feature_hot_tub'] == True) | (candidate_hotels['vibes_serene'] == True)
            if spa_mask.any():
                candidate_hotels = candidate_hotels[spa_mask]
        
        # Sample if we have more candidates than requested
        if len(candidate_hotels) > num_hotels:
            candidate_hotels = candidate_hotels.sample(n=num_hotels, replace=False)
        elif len(candidate_hotels) == 0:
            # Fallback: if no hotels match, take top-rated hotels
            logger.warning("No hotels match search criteria, falling back to top-rated hotels")
            candidate_hotels = self.hotels_df.nlargest(num_hotels, 'hotel_class_score')

        # Create a DataFrame from the user's search preferences and suffix the columns
        # These will be used for feature engineering but NOT as ranking features
        search_df = pd.DataFrame([search_preferences]).add_suffix('_search')

        # Cross join the candidate hotels with search preferences
        candidates_df = pd.merge(candidate_hotels, search_df, how='cross')

        return candidates_df


    def _add_dummy_behavior_features(self, df):
        """Add dummy behavioral columns expected by the model."""
        behavior_cols = [
            'total_time_spent', 'used_pictures_carousel', 'saved',
            'read_reviews', 'scrolled_to_bottom', 'review_score',
            'review_date', 'product_reviewed'
        ]

        # Use helper function to ensure columns exist
        df = ensure_feature_columns(df, behavior_cols)
        return df


    def engineer_features_for_prediction(self, candidates_df, user_uuid):
        """
        Engineer features for the prediction set.
        This mirrors the feature engineering pipeline from training.
        """
        df = candidates_df.copy()

        # Add user_uuid to the dataframe for merging features
        df['user_uuid'] = user_uuid

        # --- 1. Add dummy behavioral columns ---
        df = self._add_dummy_behavior_features(df)

        # --- 2. Time-based features ---
        df = create_time_features(df, date_col='date_search')

        # --- 3. Add user, hotel, and user-hotel aggregated features ---
        user_artifacts = {
            'user_stats_means': self.user_stats_means,
            'user_search_prefs_means': self.user_search_prefs_means
        }
        hotel_artifacts = {'hotel_stats_means': self.hotel_stats_means}
        
        df = add_user_features(df, self.stats['user'], self.stats['user_prefs'], 
                              artifacts=user_artifacts, user_uuid=user_uuid)
        df = add_hotel_features(df, self.stats['hotel'], artifacts=hotel_artifacts)
        df = add_user_hotel_features(df, self.stats['user_hotel'], user_uuid=user_uuid)

        # --- 4. Categorical Encoding ---
        # Use the encoders fitted on the training data
        df = apply_categorical_encoding(df, self.encoders)

        # --- 5. Final Feature Creation ---
        base_cols_dict = {
            'feature_cols': self.base_feature_cols,
            'amenity_cols': self.base_amenity_cols,
            'vibe_cols': self.base_vibe_cols
        }
        df = create_derived_features(df, base_cols_dict)

        # Ensure all feature columns are present, filling missing ones with 0
        df = ensure_feature_columns(df, self.feature_cols)

        return df[self.feature_cols]



    def predict_rankings(self, search_preferences, user_uuid, num_hotels=20):
        """
        Predict hotel rankings for a user.

        Args:
            search_preferences: Dict with search preferences.
            user_uuid: User identifier.
            num_hotels: Number of hotels to rank.

        Returns:
            DataFrame with hotels ranked by conversion probability.
        """
        candidates = self.build_prediction_set(search_preferences, num_hotels)
        features_df = self.engineer_features_for_prediction(candidates, user_uuid)

        # Align columns and predict
        X = features_df.reindex(columns=self.feature_cols, fill_value=0)
        scores = self.model.predict(X)

        results = pd.DataFrame({
            'hotel_id': candidates['hotel_id'],
            'conversion_score': scores
        })

        results = results.sort_values('conversion_score', ascending=False).reset_index(drop=True)
        results['rank'] = results.index + 1

        return results


def main():
    """
    Main function to run a demo prediction with random user and city from actual data.
    """
    # Load predictor
    predictor = HotelRankingPredictor()

    with open('./config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Load actual data to get random user and city
    try:
        users_df = pd.read_csv('./data/users.csv')
        searches_df = pd.read_csv('./data/searches.csv')
    except FileNotFoundError:
        logger.error("Could not load users.csv or searches.csv")
        return

    # Select a random user from the actual data
    random_user = users_df.sample(n=1).iloc[0]
    user_uuid = random_user['user_uuid']
    
    # Select a random search to get realistic search criteria
    random_search = searches_df.sample(n=1).iloc[0]
    
    logger.info("Selected random user: %s", user_uuid)
    logger.info("Selected random location: %s, %s", random_search['city'], random_search['state'])
    
    # Create search preferences using the config template but with random search data
    search_preferences = config['search_preferences'].copy()
    search_preferences['city'] = random_search['city']
    search_preferences['state'] = random_search['state']
    
    logger.info("Search preferences:")
    logger.info("  Type: %s", search_preferences['type'])
    logger.info("  Location: %s, %s", search_preferences['city'], search_preferences['state'])
    logger.info("  Hotel class: %s", search_preferences['hotel_class_score'])
    logger.info("  Date: %s", search_preferences['date'])

    num_hotels = config.get('num_hotels', 20)

    # Get rankings
    rankings = predictor.predict_rankings(search_preferences, user_uuid, num_hotels=num_hotels)

    logger.info("\nHotel Rankings for User %s:", user_uuid)
    logger.info(rankings)

if __name__ == '__main__':
    main()
