"""
Generates synthetic clickthrough data for the ResortPass model.
"""
import argparse
import json
import random
import pandas as pd
import numpy as np

# Load the searches and users data
try:
    searches_df = pd.read_csv('./data/searches.csv')
    users_df = pd.read_csv('./data/users.csv')
    hotels_df = pd.read_csv('./data/hotels.csv')
except FileNotFoundError:
    print("Make sure you have generated the searches, users, and hotels data first.")
    exit()


def random_bool(prob=0.3):
    """Return a random boolean with a given probability of being True."""
    return random.random() < prob


def _calculate_preference_score(prefix, weight, search_info, hotel, hotel_feature_cols):
    """Helper function to calculate score for a given preference type."""
    score = 0
    matches = 0
    total_prefs = 0
    
    for col in hotel_feature_cols:
        if col.startswith(prefix):
            search_col = f"{col}_search"
            if search_col in search_info:
                total_prefs += 1
                if search_info[search_col] and hotel[col]:
                    score += weight
                    matches += 1
                elif not search_info[search_col] and not hotel[col]:
                    score += weight * 0.3
    
    return score, matches, total_prefs


def calculate_hotel_match_score(search_info, hotel, hotel_feature_cols):
    """Calculate how well a hotel matches the search preferences with enhanced variance."""
    total_score = 0
    total_weights = 0
    
    # Feature preferences (weight: 4 - increased importance)
    feature_weight = 4
    feature_score, feature_matches, feature_total = _calculate_preference_score(
        'feature_', feature_weight, search_info, hotel, hotel_feature_cols
    )
    total_score += feature_score
    total_weights += feature_weight * feature_total
    
    # Amenity preferences (weight: 3 - increased)
    amenity_weight = 3
    amenity_score, amenity_matches, amenity_total = _calculate_preference_score(
        'amenities_', amenity_weight, search_info, hotel, hotel_feature_cols
    )
    total_score += amenity_score
    total_weights += amenity_weight * amenity_total
    
    # Vibe preferences (weight: 3 - increased)
    vibe_weight = 3
    vibe_score, vibe_matches, vibe_total = _calculate_preference_score(
        'vibes_', vibe_weight, search_info, hotel, hotel_feature_cols
    )
    total_score += vibe_score
    total_weights += vibe_weight * vibe_total
    
    # Hotel class matching (very important - weight: 5)
    class_weight = 5
    hotel_class = hotel.get('hotel_class_score', 3)
    search_class = search_info.get('hotel_class_score_search', 3)
    
    if hotel_class >= search_class:
        # Meeting or exceeding class expectations
        class_bonus = class_weight * (1.0 + (hotel_class - search_class) * 0.2)
        total_score += class_bonus
    else:
        # Below class expectations - penalty
        class_penalty = class_weight * 0.5 * (search_class - hotel_class)
        total_score -= class_penalty
    total_weights += class_weight
    
    # Location/type bonus (weight: 2)
    type_weight = 2
    if search_info.get('type_search') == hotel.get('type'):
        total_score += type_weight
    total_weights += type_weight
    
    # Top rated bonus (weight: 2)
    top_rated_weight = 2
    if search_info.get('top_rated_search', False) and hotel.get('top_rated', False):
        total_score += top_rated_weight
    total_weights += top_rated_weight
    
    # Calculate match percentage with more variance
    if total_weights > 0:
        match_percentage = total_score / total_weights
        # Add quality multipliers based on actual match counts
        feature_quality = feature_matches / max(feature_total, 1) if feature_total > 0 else 0.5
        amenity_quality = amenity_matches / max(amenity_total, 1) if amenity_total > 0 else 0.5
        vibe_quality = vibe_matches / max(vibe_total, 1) if vibe_total > 0 else 0.5
        
        # Weighted average of quality scores
        overall_quality = (feature_quality * 0.4 + amenity_quality * 0.3 + vibe_quality * 0.3)
        
        # Combine base match with quality multiplier
        final_score = match_percentage * (0.7 + overall_quality * 0.3)
        return min(1.0, max(0.0, final_score))
    return 0.5


def calculate_conversion_probability(search_info, hotel, match_score):
    """Calculate conversion probability with reduced match dependency."""
    base_conversion = 0.01  # base rate

    # Increase match score contribution to make it more important
    match_contribution = match_score * 0.25

    # Add random noise to break perfect correlation
    random_factor = random.uniform(-0.01, 0.01)  # ±1% random variation

    # Hotel class bonus - make this more important than pure match
    class_bonus = 0
    if 'hotel_class_score' in hotel:
        hotel_class = hotel['hotel_class_score']
        search_class = search_info.get('hotel_class_score_search', 3)
        if hotel_class >= search_class:
            class_bonus = 0.05 + (hotel_class - search_class) * 0.03  # Significant bonus
        else:
            class_bonus = -0.10 * (search_class - hotel_class)  # Penalty for lower class

    # Top-rated bonus
    top_rated_bonus = 0.08 if hotel.get('top_rated', False) else 0

    type_bonus = 0
    if 'type_search' in search_info:
        search_type = search_info['type_search']
        if search_type == 'SPA':
            type_bonus = 0.12 
        elif search_type == 'POOL':
            type_bonus = 0.08 
        elif search_type == 'ALL':
            type_bonus = 0.05

    total_prob = (base_conversion + match_contribution + class_bonus +
                  top_rated_bonus + type_bonus + random_factor)

    return min(0.8, max(0.01, total_prob))  # Cap between 1% and 80%


def generate_clickthroughs(num_rows, search_ids, hotels_df):
    """Generate a given number of clickthrough records."""
    clickthroughs = []
    
    # Get preference columns (without _search suffix since we'll map them)
    hotel_feature_cols = [col for col in hotels_df.columns if col.startswith(('feature_', 'amenities_', 'vibes_'))]
    
    for _ in range(num_rows):
        # 1. Pick a random search and get its details
        search_id = np.random.choice(search_ids)
        search_info = searches_df[searches_df['search_id'] == search_id].iloc[0]

        # 2. Calculate match scores for all hotels and use weighted selection
        hotel_scores = []
        for _, hotel in hotels_df.iterrows():
            match_score = calculate_hotel_match_score(search_info, hotel, hotel_feature_cols)
            hotel_scores.append((hotel['hotel_id'], match_score))
        
        # 3. Use weighted selection - higher match scores have higher probability of being clicked
        hotel_ids, scores = zip(*hotel_scores)
        # Convert scores to probabilities (add small base probability for diversity)
        probabilities = np.array(scores) + 0.1  # Base 10% chance even for poor matches
        probabilities = probabilities / probabilities.sum()
        
        hotel_id = np.random.choice(hotel_ids, p=probabilities)
        selected_hotel = hotels_df[hotels_df['hotel_id'] == hotel_id].iloc[0]
        
        # 4. Calculate conversion probability based on multiple factors
        match_score = dict(hotel_scores)[hotel_id]
        conversion_prob = calculate_conversion_probability(search_info, selected_hotel, match_score)
        converted = random.random() < conversion_prob

        # 5. Generate behavioral data with NO correlation to conversion
        # We want behavioral features to be informative but not perfect predictors
        # This forces the model to learn from match-based features
        
        # Time spent: Add much more noise, NO conversion bias
        base_time = 45 + (match_score * 150)  # 45-195 seconds base
        time_spent = int(base_time * random.uniform(0.5, 1.5))
        time_spent += random.randint(-30, 30)
        time_spent = max(5, min(700, time_spent))  # Cap between 5-700 seconds
        
        # Carousel usage: No conversion signal, only match signal
        carousel_prob = 0.3 + (match_score * 0.5)
        used_carousel = random_bool(carousel_prob)
        
        # Saving: No conversion signal
        save_prob = 0.05 + (match_score * 0.3)
        saved = random_bool(save_prob)
        
        # Reading reviews: Base more on match quality than conversion
        review_prob = 0.2 + (match_score * 0.6)
        read_reviews = random_bool(review_prob)
        
        # Scrolling to bottom: Similar reduction in conversion bias
        scroll_prob = 0.4 + (match_score * 0.4)
        scrolled_to_bottom = random_bool(scroll_prob)

        # 6. Generate review data (more likely for converters)
        review_score = None
        product_reviewed = None
        # Review is now independent of conversion, but more likely for engaged users
        if random_bool(0.1 + match_score * 0.2):
            # Better matches get slightly better review scores
            base_review = 3.2 + (match_score * 1.5)
            review_score = round(max(2.5, min(5.0, base_review + random.uniform(-0.8, 0.8))), 1)
            
            # Product reviewed based on hotel features
            products = ['other']  # Default
            if hotel.get('amenities_beach_access', False):
                products.append('beach')
            if any(hotel.get(f'feature_{p}', False) for p in ['outdoor_pool', 'indoor_pool', 'rooftop_pool']):
                products.append('pool')
            product_reviewed = random.choice(products)

        clickthrough = {
            'user_uuid': search_info['user_uuid'],
            'search_id': search_id,
            'timestamp': pd.to_datetime(search_info['date']) + pd.to_timedelta(np.random.randint(1, 60), unit='m'),
            'hotel_id': hotel_id,
            'total_time_spent': time_spent,
            'used_pictures_carousel': used_carousel,
            'saved': saved,
            'read_reviews': read_reviews,
            'scrolled_to_bottom': scrolled_to_bottom,
            'converted': converted,
            'review_score': review_score,
            'product_reviewed': product_reviewed
        }
        clickthroughs.append(clickthrough)

    # Convert to DataFrame and save to CSV
    clickthroughs_df = pd.DataFrame(clickthroughs)
    clickthroughs_df.to_csv('./data/clickthroughs.csv', index=False)
    print(f"Generated {num_rows} clickthrough records and saved to ./data/clickthroughs.csv")


def main():
    """Main function to generate clickthrough data."""
    with open('./config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    parser = argparse.ArgumentParser(description="Generate synthetic clickthrough data.")
    parser.add_argument("--num-rows", type=int, default=config['num_clickthroughs'],
                        help="Number of clickthrough records to generate.")
    args = parser.parse_args()

    search_ids = searches_df['search_id'].unique()

    generate_clickthroughs(args.num_rows, search_ids, hotels_df)


if __name__ == '__main__':
    main()
