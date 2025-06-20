"""
Shared feature engineering functions for the ResortPass hotel ranking model.
"""
import pandas as pd


def safe_transform(encoder, series):
    """
    Safely transform a series with a pre-fitted LabelEncoder.
    Unseen values are mapped to -1.
    
    Args:
        encoder: Fitted LabelEncoder object.
        series: Pandas Series to transform.
    
    Returns:
        pd.Series: Transformed series with unseen values mapped to -1.
    """
    class_map = {cls: i for i, cls in enumerate(encoder.classes_)}
    return series.map(class_map).fillna(-1).astype(int)


def apply_categorical_encoding(df, encoders):
    """
    Apply categorical encoding to dataframe using fitted encoders.
    
    Args:
        df (pd.DataFrame): DataFrame to encode.
        encoders (dict): Dictionary of fitted encoders.
    
    Returns:
        pd.DataFrame: DataFrame with encoded categorical columns.
    """
    df = df.copy()
    df['type_encoded'] = safe_transform(encoders['type'], df['type_search'])
    df['city_encoded'] = safe_transform(encoders['city'], df['city_search'])
    df['state_encoded'] = safe_transform(encoders['state'], df['state_search'])
    return df


def identify_base_columns(df):
    """
    Identify base feature, amenity, and vibe columns from a dataframe.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze.
    
    Returns:
        dict: Dictionary with 'feature_cols', 'amenity_cols', 'vibe_cols'.
    """
    base_feature_cols = [
        col for col in df.columns if col.startswith('feature_') and not col.endswith('_search')
    ]
    base_amenity_cols = [
        col for col in df.columns if col.startswith('amenities_') and not col.endswith('_search')
    ]
    base_vibe_cols = [
        col for col in df.columns if col.startswith('vibes_') and not col.endswith('_search')
    ]
    
    return {
        'feature_cols': base_feature_cols,
        'amenity_cols': base_amenity_cols,
        'vibe_cols': base_vibe_cols
    }


def create_derived_features(df, base_cols_dict):
    """
    Create derived features like review flags and time ratios.
    
    Args:
        df (pd.DataFrame): DataFrame to add features to.
        base_cols_dict (dict): Dictionary with base column lists.
    
    Returns:
        pd.DataFrame: DataFrame with added derived features.
    """
    df = df.copy()
    
    # Review-related features
    df['has_review'] = df['review_score'].notna().astype(int) if 'review_score' in df.columns else 0
    df['review_score_filled'] = df['review_score'].fillna(0) if 'review_score' in df.columns else 0
    
    # Product review categories
    if 'product_reviewed' in df.columns:
        df['reviewed_beach'] = (df['product_reviewed'] == 'beach').astype(int)
        df['reviewed_pool'] = (df['product_reviewed'] == 'pool').astype(int)
        df['reviewed_other'] = (df['product_reviewed'] == 'other').astype(int)
    else:
        df['reviewed_beach'] = 0
        df['reviewed_pool'] = 0
        df['reviewed_other'] = 0
    
    # Calculate match scores
    df = calculate_match_scores(
        df,
        base_cols_dict['feature_cols'],
        base_cols_dict['amenity_cols'],
        base_cols_dict['vibe_cols']
    )
    
    # Time ratio features
    if 'total_time_spent' in df.columns and 'user_total_time_spent_mean' in df.columns:
        df['time_vs_user_avg'] = df['total_time_spent'] / (df['user_total_time_spent_mean'] + 1)
    else:
        df['time_vs_user_avg'] = 0
        
    if 'total_time_spent' in df.columns and 'hotel_total_time_spent_mean' in df.columns:
        df['time_vs_hotel_avg'] = df['total_time_spent'] / (df['hotel_total_time_spent_mean'] + 1)
    else:
        df['time_vs_hotel_avg'] = 0
    
    return df


def fill_missing_with_defaults(df, user_hotel_interaction_cols=None):
    """
    Fill missing values for user-hotel interaction columns with 0.
    
    Args:
        df (pd.DataFrame): DataFrame to fill.
        user_hotel_interaction_cols (list, optional): List of interaction columns.
    
    Returns:
        pd.DataFrame: DataFrame with filled missing values.
    """
    if user_hotel_interaction_cols is None:
        user_hotel_interaction_cols = [
            'user_hotel_views', 'user_hotel_saves', 'user_hotel_read_reviews',
            'user_hotel_carousel', 'user_hotel_scrolled', 'user_hotel_time_spent',
            'user_hotel_conversion_rate'
        ]
    
    df = df.copy()
    for col in user_hotel_interaction_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    return df


def fill_na_with_train_means(df, artifacts):
    """
    Fill missing values in the dataframe with means from the training set.
    
    NOTE: This function is for PREDICTION ONLY and should not be used during training
    as it can cause target leakage. For training, use the temporal functions instead.
    
    Args:
        df (pd.DataFrame): DataFrame to fill.
        artifacts (dict): Dictionary containing training means.
    
    Returns:
        pd.DataFrame: DataFrame with filled missing values.
    """
    df = df.copy()
    
    # Fill with training set means
    for col, mean_val in artifacts.get('user_stats_means', {}).items():
        if col in df.columns:
            df.loc[:, col] = df[col].fillna(mean_val)
            
    for col, mean_val in artifacts.get('user_search_prefs_means', {}).items():
        if col in df.columns:
            df.loc[:, col] = df[col].fillna(mean_val)
            
    for col, mean_val in artifacts.get('hotel_stats_means', {}).items():
        if col in df.columns:
            df.loc[:, col] = df[col].fillna(mean_val)
    
    return df


def ensure_feature_columns(df, feature_cols):
    """
    Ensure all required feature columns are present, filling missing ones with 0.
    
    Args:
        df (pd.DataFrame): DataFrame to check.
        feature_cols (list): List of required feature columns.
    
    Returns:
        pd.DataFrame: DataFrame with all required columns present.
    """
    df = df.copy()
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    return df


def create_time_features(df, timestamp_col=None, date_col='date_search'):
    """
    Create time-based features from timestamp and date columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        timestamp_col (str, optional): Name of the timestamp column for training.
                                     If None, current time is used for prediction.
        date_col (str, optional): Name of the search date column.

    Returns:
        pd.DataFrame: DataFrame with added time features.
    """
    if timestamp_col and timestamp_col in df.columns:
        timestamp = pd.to_datetime(df[timestamp_col])
        normalized_timestamp = timestamp.dt.normalize()
        hour = timestamp.dt.hour
        dayofweek = timestamp.dt.dayofweek
    else:
        timestamp = pd.Timestamp.now()
        normalized_timestamp = timestamp.normalize()
        hour = timestamp.hour
        dayofweek = timestamp.dayofweek

    df['timestamp'] = timestamp
    df['date'] = pd.to_datetime(df[date_col])
    df['hour_of_day'] = hour
    df['day_of_week'] = dayofweek
    df['days_until_trip'] = (df['date'] - normalized_timestamp).dt.days
    return df


def calculate_match_scores(df, feature_cols, amenity_cols, vibe_cols):
    """
    Calculate scores based on how well hotel features match user's historical preferences.

    Args:
        df (pd.DataFrame): Input DataFrame.
        feature_cols (list): List of hotel feature columns.
        amenity_cols (list): List of hotel amenity columns.
        vibe_cols (list): List of hotel vibe columns.

    Returns:
        pd.DataFrame: DataFrame with added match score features.
    """
    # Feature matching: weight hotel features by user's historical preferences
    df['feature_match_score'] = 0
    for feature_col in feature_cols:
        # Use aggregated user preferences instead of current search filters
        user_pref_col = f'user_pref_{feature_col}'
        if user_pref_col in df.columns:
            # Weight the hotel feature by the user's historical preference for it
            df['feature_match_score'] += (df[user_pref_col] * df[feature_col]).fillna(0)

    # Amenity matching
    df['amenity_match_score'] = 0
    for amenity_col in amenity_cols:
        user_pref_col = f'user_pref_{amenity_col}'
        if user_pref_col in df.columns:
            df['amenity_match_score'] += (df[user_pref_col] * df[amenity_col]).fillna(0)

    # Vibe matching
    df['vibe_match_score'] = 0
    for vibe_col in vibe_cols:
        user_pref_col = f'user_pref_{vibe_col}'
        if user_pref_col in df.columns:
            df['vibe_match_score'] += (df[user_pref_col] * df[vibe_col]).fillna(0)

    # Overall match score
    df['overall_match_score'] = df['feature_match_score'] + df['amenity_match_score'] + df['vibe_match_score']

    # Hotel class match - use historical preference
    if 'user_pref_hotel_class_score' in df.columns:
        df['class_score_diff'] = df['hotel_class_score'] - df['user_pref_hotel_class_score']
        df['class_score_match'] = (df['hotel_class_score'] >= df['user_pref_hotel_class_score']).astype(int)
        df['class_score_exact'] = (df['hotel_class_score'] == df['user_pref_hotel_class_score']).astype(int)
    else:
        df['class_score_diff'] = 0
        df['class_score_match'] = 0
        df['class_score_exact'] = 0

    # Top rated match - use historical preference
    df['top_rated_match'] = 0
    if 'user_pref_top_rated' in df.columns and 'top_rated' in df.columns:
        df['top_rated_match'] = (df['user_pref_top_rated'] * df['top_rated']).fillna(0)

    return df


def add_user_features(df, user_stats, user_search_prefs, artifacts, user_uuid):
    """
    Add user-specific features to the DataFrame for prediction.
    
    NOTE: This function is for PREDICTION ONLY and should not be used during training
    as it can cause target leakage. For training, use calculate_temporal_stats instead.
    
    Args:
        df (pd.DataFrame): DataFrame to add features to.
        user_stats (pd.DataFrame): User statistics DataFrame.
        user_search_prefs (pd.DataFrame): User search preferences DataFrame.
        artifacts (dict): Training artifacts containing means for filling NA values.
        user_uuid (str): Specific user UUID for prediction.
    
    Returns:
        pd.DataFrame: DataFrame with added user features.
    """
    df = df.copy()
    
    # Filter to specific user
    user_stats_filtered = user_stats[user_stats['user_uuid'] == user_uuid]
    user_prefs_filtered = user_search_prefs[user_search_prefs['user_uuid'] == user_uuid]
    
    # Add user statistics
    if not user_stats_filtered.empty:
        for col in user_stats_filtered.columns:
            if col != 'user_uuid':
                df[col] = user_stats_filtered[col].iloc[0]
    
    # Add user search preferences
    if not user_prefs_filtered.empty:
        for col in user_prefs_filtered.columns:
            if col != 'user_uuid':
                df[col] = user_prefs_filtered[col].iloc[0]
    
    # Fill missing values with training means
    df = fill_na_with_train_means(df, artifacts)
    
    return df


def add_hotel_features(df, hotel_stats, artifacts=None):
    """
    Add hotel statistics to the DataFrame. 
    NOTE: This function is used for PREDICTION ONLY.
    
    Args:
        df (pd.DataFrame): DataFrame to add features to.
        hotel_stats (pd.DataFrame): Hotel statistics DataFrame.
        artifacts (dict, optional): Training artifacts containing means for filling NA values.
    
    Returns:
        pd.DataFrame: DataFrame with added hotel features.
    """
    df = df.copy()
    df = df.merge(hotel_stats, on='hotel_id', how='left')
    
    # Fill missing values with training means if artifacts provided
    if artifacts is not None:
        df = fill_na_with_train_means(df, artifacts)
    
    return df


def calculate_temporal_stats(df):
    """
    Calculate all temporal statistics using pandas rolling functions to avoid target leakage.
    Each row only sees data from before its timestamp.
    
    Args:
        df (pd.DataFrame): DataFrame with timestamp column and interaction data.
    
    Returns:
        pd.DataFrame: DataFrame with all temporal statistics columns added.
    """
    # Sort by timestamp to ensure proper temporal ordering
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    
    # Identify numeric search preference columns for user preferences
    exclude_patterns = ['city_search', 'state_search', 'type_search', 'date_search', 'user_uuid_search', 'timestamp_search']
    search_pref_cols = [col for col in df_sorted.columns if col.endswith('_search') and col not in exclude_patterns]
    
    # Filter to numeric search preference columns
    numeric_search_cols = []
    for col in search_pref_cols:
        try:
            sample_val = df_sorted[col].iloc[0]
            float(sample_val)
            numeric_search_cols.append(col)
        except (ValueError, TypeError):
            continue
    
    # Define all statistics configurations including user search preferences
    stats_configs = {
        # User-hotel interaction stats (cumulative sums)
        'user_hotel': {
            'groupby': ['user_uuid', 'hotel_id'],
            'stats': {
                'user_hotel_views': ('search_id', 'count'),  # Special case: just count rows
                'user_hotel_saves': ('saved', 'sum'),
                'user_hotel_read_reviews': ('read_reviews', 'sum'),
                'user_hotel_carousel': ('used_pictures_carousel', 'sum'),
                'user_hotel_scrolled': ('scrolled_to_bottom', 'sum'),
                'user_hotel_time_spent': ('total_time_spent', 'sum'),
                'user_hotel_conversions': ('converted', 'sum')
            }
        },
        # User stats (expanding means/std)
        'user': {
            'groupby': ['user_uuid'],
            'stats': {
                'user_total_time_spent_mean': ('total_time_spent', 'mean'),
                'user_total_time_spent_std': ('total_time_spent', 'std'),
                'user_total_time_spent_count': ('total_time_spent', 'count'),
                'user_used_pictures_carousel_mean': ('used_pictures_carousel', 'mean'),
                'user_saved_mean': ('saved', 'mean'),
                'user_read_reviews_mean': ('read_reviews', 'mean'),
                'user_scrolled_to_bottom_mean': ('scrolled_to_bottom', 'mean'),
                'user_converted_mean': ('converted', 'mean')
            }
        },
        # Hotel stats (expanding means/std)
        'hotel': {
            'groupby': ['hotel_id'],
            'stats': {
                'hotel_total_time_spent_mean': ('total_time_spent', 'mean'),
                'hotel_total_time_spent_std': ('total_time_spent', 'std'),
                'hotel_total_time_spent_count': ('total_time_spent', 'count'),
                'hotel_used_pictures_carousel_mean': ('used_pictures_carousel', 'mean'),
                'hotel_saved_mean': ('saved', 'mean'),
                'hotel_read_reviews_mean': ('read_reviews', 'mean'),
                'hotel_scrolled_to_bottom_mean': ('scrolled_to_bottom', 'mean'),
                'hotel_converted_mean': ('converted', 'mean'),
                'hotel_review_score_mean': ('review_score', 'mean'),
                'hotel_review_score_std': ('review_score', 'std')
            }
        },
        'user_search_prefs': {
            'groupby': ['user_uuid'],
            'stats': {
                f'user_pref_{search_col.replace("_search", "")}': (search_col, 'mean')
                for search_col in numeric_search_cols
            }
        }
    }
    
    # Apply all statistics using the unified function
    for config in stats_configs.values():
        df_sorted = _apply_temporal_stats(df_sorted, config['groupby'], config['stats'])
    
    # Calculate conversion rate for user-hotel interactions
    df_sorted['user_hotel_conversion_rate'] = df_sorted.apply(
        lambda row: row['user_hotel_conversions'] / row['user_hotel_views'] if row['user_hotel_views'] > 0 else 0, 
        axis=1
    )
    
    return df_sorted


def _apply_temporal_stats(df, groupby_cols, stats_dict):
    """
    Apply temporal statistics for a given grouping and set of statistics.
    
    Args:
        df (pd.DataFrame): DataFrame to process
        groupby_cols (list): Columns to group by
        stats_dict (dict): Dictionary of {output_col: (source_col, agg_type)}
    
    Returns:
        pd.DataFrame: DataFrame with temporal statistics added
    """
    grouped = df.groupby(groupby_cols)
    
    for output_col, (source_col, agg_type) in stats_dict.items():
        if agg_type == 'count':
            # Special case: cumcount for counting previous interactions
            df[output_col] = grouped.cumcount()
        elif agg_type == 'sum':
            # Cumulative sum, shifted to exclude current row
            df[output_col] = grouped[source_col].transform(
                lambda x: x.cumsum().shift(1).fillna(0)
            )
        elif agg_type == 'mean':
            # Expanding mean, shifted to exclude current row
            df[output_col] = grouped[source_col].transform(
                lambda x: x.expanding().mean().shift(1).fillna(0)
            )
        elif agg_type == 'std':
            # Expanding standard deviation, shifted to exclude current row
            df[output_col] = grouped[source_col].transform(
                lambda x: x.expanding().std().shift(1).fillna(0)
            )
    
    return df


def add_user_hotel_features(df, user_hotel_stats, user_uuid):
    """
    Add user-hotel interaction features.
    
    NOTE: This function is for PREDICTION ONLY and should not be used during training
    as it can cause target leakage. For training, use calculate_temporal_stats instead.
    
    Args:
        df (pd.DataFrame): DataFrame to add features to.
        user_hotel_stats (pd.DataFrame): User-hotel interaction statistics DataFrame.
        user_uuid (str): Specific user UUID for prediction (required).
    
    Returns:
        pd.DataFrame: DataFrame with added user-hotel interaction features.
    
    Raises:
        ValueError: If user_uuid is None.
    """
    if user_uuid is None:
        raise ValueError("user_uuid is required for prediction. This function should not be used for training.")
    
    df = df.copy()
    
    if not user_hotel_stats.empty:
        # For prediction: filter to specific user's interactions
        user_hotel_interactions = user_hotel_stats[
            user_hotel_stats['user_uuid'] == user_uuid
        ]
        df = df.merge(user_hotel_interactions, on=['user_uuid', 'hotel_id'], how='left')
    
    # Fill NaNs for hotels the user hasn't interacted with
    df = fill_missing_with_defaults(df)
    
    return df
