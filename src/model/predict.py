"""
Prediction script for the ResortPass hotel ranking model.

This script defines the `HotelRankingPredictor` class, which loads a pre-trained
LightGBM model and associated artifacts to predict hotel rankings for a given user
and search query.
"""
import logging
import os
import pickle
import json
import pandas as pd
import numpy as np
import lightgbm as lgb

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
            self.encoders = {
                'type': artifacts['le_type'],
                'city': artifacts['le_city'],
                'state': artifacts['le_state']
            }
            self.feature_cols = artifacts['feature_cols']
            self.base_feature_cols = {
                'features': artifacts.get('base_feature_cols', []),
                'amenities': artifacts.get('base_amenity_cols', []),
                'vibes': artifacts.get('base_vibe_cols', [])
            }
            self.stats = {
                'user': artifacts['user_stats'],
                'user_prefs': artifacts['user_search_prefs'],
                'hotel': artifacts['hotel_stats']
            }

    def build_prediction_set(self, search_preferences, num_hotels=20):
        """
        Build a DataFrame for prediction by combining candidate hotel IDs with search preferences.

        Note: In a production system, this would involve a more sophisticated candidate
        retrieval stage. Here, we randomly sample from all known hotels.

        Args:
            search_preferences: Dict with the user's search criteria.
            num_hotels: The number of candidate hotels to generate for ranking.

        Returns:
            A DataFrame where each row represents a hotel to be scored.
        """
        available_hotels = self.stats['hotel']['hotel_id'].unique()
        num_to_sample = min(num_hotels, len(available_hotels))

        if len(available_hotels) > 0:
            selected_hotels = np.random.choice(available_hotels, size=num_to_sample, replace=False)
        else:
            selected_hotels = []

        candidates = []
        for hotel_id in selected_hotels:
            candidate = search_preferences.copy()
            candidate['hotel_id'] = hotel_id
            candidates.append(candidate)

        return pd.DataFrame(candidates)

    def _add_time_features(self, df):
        """Add time-based features to the DataFrame."""
        current_time = pd.Timestamp.now()
        df['timestamp'] = current_time
        df['hour_of_day'] = current_time.hour
        df['day_of_week'] = current_time.dayofweek
        df['days_until_trip'] = (df['date'] - current_time.normalize()).dt.days
        return df

    def _add_user_features(self, df, user_uuid):
        """Add user-specific features to the DataFrame."""
        user_stats_filtered = self.stats['user'][
            self.stats['user']['user_uuid'] == user_uuid
        ]
        if not user_stats_filtered.empty:
            for col in user_stats_filtered.columns:
                if col != 'user_uuid':
                    df[col] = user_stats_filtered[col].iloc[0]
        else:
            for col in self.stats['user'].columns:
                if col != 'user_uuid':
                    df[col] = self.stats['user'][col].mean() if col.endswith('_mean') else 0

        user_prefs_filtered = self.stats['user_prefs'][
            self.stats['user_prefs']['user_uuid'] == user_uuid
        ]
        if not user_prefs_filtered.empty:
            for col in user_prefs_filtered.columns:
                if col != 'user_uuid':
                    df[col] = user_prefs_filtered[col].iloc[0]
        else:
            for col in self.stats['user_prefs'].columns:
                if col != 'user_uuid':
                    df[col] = self.stats['user_prefs'][col].mean()
        return df

    def _add_hotel_features(self, df):
        """Add hotel statistics to the DataFrame."""
        df = df.merge(self.stats['hotel'], on='hotel_id', how='left')
        for col in self.stats['hotel'].columns:
            if col != 'hotel_id' and col in df.columns:
                df[col] = df[col].fillna(self.stats['hotel'][col].mean())
        return df

    def _add_dummy_behavior_features(self, df):
        """Add dummy behavior columns required by the model."""
        behavior_cols = [
            'total_time_spent', 'used_pictures_carousel', 'saved',
            'read_reviews', 'scrolled_to_bottom', 'review_score',
            'review_date', 'product_reviewed'
        ]
        boolean_behavior_cols = [
            'used_pictures_carousel', 'saved', 'read_reviews', 'scrolled_to_bottom'
        ]
        for col in behavior_cols:
            if col not in df.columns:
                if col == 'total_time_spent':
                    df[col] = 120.0  # Average time
                elif col in boolean_behavior_cols:
                    df[col] = False
                else:
                    df[col] = None
        return df

    def _calculate_match_scores(self, df):
        """Calculate feature, amenity, and vibe match scores."""
        for key, cols in self.base_feature_cols.items():
            user_pref_cols = [f'user_pref_{c}' for c in cols]
            match_score = (df[cols].values * df[user_pref_cols].values).sum(axis=1)
            df[f'{key}_match_score'] = match_score
        return df

    def engineer_features_for_prediction(self, candidates_df, user_uuid):
        """
        Engineer features for prediction candidates.

        Args:
            candidates_df: DataFrame with candidate hotels and search preferences.
            user_uuid: User identifier.

        Returns:
            DataFrame with engineered features.
        """
        df = candidates_df.copy()
        df['user_uuid'] = user_uuid
        df['search_id'] = 'PRED_001'
        df['date'] = pd.to_datetime(df['date'])

        df = self._add_time_features(df)
        df = self._add_user_features(df, user_uuid)
        df = self._add_hotel_features(df)
        df = self._add_dummy_behavior_features(df)

        # Ensure 'review_score' is numeric to prevent dtype errors with LightGBM.
        # This converts dummy 'None' values to NaN, making the column float type.
        df['review_score'] = pd.to_numeric(df['review_score'], errors='coerce')

        # Encode categorical variables
        df['type_encoded'] = self._safe_transform(self.encoders['type'], df['type'])
        df['city_encoded'] = self._safe_transform(self.encoders['city'], df['city'])
        df['state_encoded'] = self._safe_transform(self.encoders['state'], df['state'])

        # Final feature creation
        df['has_review'] = df['review_score'].notna().astype(int)
        df['review_score_filled'] = df['review_score'].fillna(0)
        df['reviewed_beach'] = (df['product_reviewed'] == 'beach').astype(int)
        df['reviewed_pool'] = (df['product_reviewed'] == 'pool').astype(int)
        df['reviewed_other'] = (df['product_reviewed'] == 'other').astype(int)

        df = self._calculate_match_scores(df)

        df['time_spent_log'] = np.log1p(df['total_time_spent'])
        user_avg = df.get('user_total_time_spent_mean', pd.Series([120]))[0]
        hotel_avg = df.get('hotel_total_time_spent_mean', 120)
        df['time_vs_user_avg'] = df['total_time_spent'] / (user_avg + 1)
        df['time_vs_hotel_avg'] = df['total_time_spent'] / (hotel_avg + 1)

        return df

    def _safe_transform(self, encoder, values):
        """
        Safely transform values with a pre-fitted LabelEncoder, handling unseen
        categories by assigning them a specific value (-1).
        """
        class_map = {cls: i for i, cls in enumerate(encoder.classes_)}
        return values.map(class_map).fillna(-1).astype(int)

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
    Main function to run a demo prediction.
    """
    # Example Usage
    predictor = HotelRankingPredictor()

    with open('./config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Example search preferences
    search_preferences = config['search_preferences']

    # Example user
    user_uuid = config['user_uuid']  # A user from the historical data
    num_hotels = config.get('num_hotels', 100)

    # Get rankings
    rankings = predictor.predict_rankings(search_preferences, user_uuid, num_hotels=num_hotels)

    logger.info("Hotel Rankings for User:")
    logger.info(rankings)


if __name__ == '__main__':
    main()
