"""
Generates synthetic search data for the ResortPass model.
"""
import random
import argparse
import json
from faker import Faker
import pandas as pd

# Initialize Faker with US locale for realistic cities/states
fake = Faker('en_US')

# Define possible values
search_types = ['ALL', 'POOL', 'SPA', 'DAY_ROOM']
hotel_classes = [3, 4, 5]


def generate_user_preferences():
    """Generate personalized preference probabilities for a user."""
    # Define base probabilities and variation ranges
    base_prefs = {
        'feature_hot_tub': 0.3,
        'feature_outdoor_pool': 0.4,
        'feature_rooftop_pool': 0.2,
        'feature_infinity_pool': 0.2,
        'feature_indoor_pool': 0.3,
        'feature_kiddie_pool': 0.25,
        'feature_waterpark': 0.15,
        'feature_cabana': 0.2,
        'feature_daybed': 0.2,
        'feature_splash_pad': 0.15,
        'feature_lazy_river': 0.1,
        'feature_water_slide': 0.15,
        'amenities_beach_access': 0.3,
        'amenities_all_inclusive': 0.2,
        'amenities_free_parking': 0.4,
        'amenities_luggage_storage': 0.3,
        'amenities_airport_shuttle': 0.25,
        'amenities_cruise_port_shuttle': 0.15,
        'amenities_gym': 0.3,
        'amenities_wheelchair_accessible': 0.3,
        'amenities_showers': 0.3,
        'amenities_lockers': 0.3,
        'top_rated': 0.4,
        'vibes_family_friendly': 0.3,
        'vibes_party': 0.2,
        'vibes_serene': 0.3,
        'vibes_luxe': 0.25,
        'vibes_trendy': 0.25
    }
    
    # Generate personalized preferences with variation
    user_prefs = {}
    for pref, base_prob in base_prefs.items():
        # Add random variation to base probability (±30% variation)
        variation = random.uniform(-0.3, 0.3)
        personal_prob = base_prob * (1 + variation)
        # Keep probabilities within reasonable bounds
        user_prefs[pref] = max(0, min(0.95, personal_prob))
    
    return user_prefs


def generate_users(num_users):
    """Generate users with unique UUIDs and personalized preferences."""
    users = []
    for _ in range(num_users):
        user = {
            'user_uuid': fake.uuid4(),
            'preferences': generate_user_preferences()
        }
        users.append(user)
    return users


def random_bool(prob=0.3):
    """Return a random boolean with a given probability of being True."""
    return random.random() < prob


def generate_searches(num_rows, users):
    """Generate a given number of search records."""
    # Create a lookup dictionary for user preferences
    user_prefs_lookup = {user['user_uuid']: user['preferences'] for user in users}
    user_uuids = [user['user_uuid'] for user in users]
    
    # Generate searches data
    searches = []
    for i in range(num_rows):
        # Random date within the next 90 days
        search_date = fake.date_between(start_date='today', end_date='+90d').strftime('%Y-%m-%d')

        # Random US city and state
        city = fake.city()
        state = fake.state_abbr()
        
        # Select a user and get their preferences
        selected_user_uuid = random.choice(user_uuids)
        user_preferences = user_prefs_lookup[selected_user_uuid]

        # Create search record using user's personal preferences
        search = {
            'search_id': f'SEARCH_{i+1:06d}',
            'user_uuid': selected_user_uuid,
            'city': city,
            'state': state,
            'date': search_date,
            'type': random.choice(search_types),
            'hotel_class_score': random.choice(hotel_classes),
            # Use user's personal preferences for Bernoulli variables
            'feature_hot_tub': random_bool(user_preferences['feature_hot_tub']),
            'feature_outdoor_pool': random_bool(user_preferences['feature_outdoor_pool']),
            'feature_rooftop_pool': random_bool(user_preferences['feature_rooftop_pool']),
            'feature_infinity_pool': random_bool(user_preferences['feature_infinity_pool']),
            'feature_indoor_pool': random_bool(user_preferences['feature_indoor_pool']),
            'feature_kiddie_pool': random_bool(user_preferences['feature_kiddie_pool']),
            'feature_waterpark': random_bool(user_preferences['feature_waterpark']),
            'feature_cabana': random_bool(user_preferences['feature_cabana']),
            'feature_daybed': random_bool(user_preferences['feature_daybed']),
            'feature_splash_pad': random_bool(user_preferences['feature_splash_pad']),
            'feature_lazy_river': random_bool(user_preferences['feature_lazy_river']),
            'feature_water_slide': random_bool(user_preferences['feature_water_slide']),
            'amenities_beach_access': random_bool(user_preferences['amenities_beach_access']),
            'amenities_all_inclusive': random_bool(user_preferences['amenities_all_inclusive']),
            'amenities_free_parking': random_bool(user_preferences['amenities_free_parking']),
            'amenities_luggage_storage': random_bool(user_preferences['amenities_luggage_storage']),
            'amenities_airport_shuttle': random_bool(user_preferences['amenities_airport_shuttle']),
            'amenities_cruise_port_shuttle': random_bool(user_preferences['amenities_cruise_port_shuttle']),
            'amenities_gym': random_bool(user_preferences['amenities_gym']),
            'amenities_wheelchair_accessible': random_bool(user_preferences['amenities_wheelchair_accessible']),
            'amenities_showers': random_bool(user_preferences['amenities_showers']),
            'amenities_lockers': random_bool(user_preferences['amenities_lockers']),
            'top_rated': random_bool(user_preferences['top_rated']),
            'vibes_family_friendly': random_bool(user_preferences['vibes_family_friendly']),
            'vibes_party': random_bool(user_preferences['vibes_party']),
            'vibes_serene': random_bool(user_preferences['vibes_serene']),
            'vibes_luxe': random_bool(user_preferences['vibes_luxe']),
            'vibes_trendy': random_bool(user_preferences['vibes_trendy'])
        }
        searches.append(search)

    # Convert to DataFrame and save to CSV
    searches_df = pd.DataFrame(searches)
    searches_df.to_csv('./data/searches.csv', index=False)
    print(f"Generated {num_rows} search records and saved to ./data/searches.csv")

    # Save user data with their preferences to users.csv
    users_data = []
    for user in users:
        user_row = {'user_uuid': user['user_uuid']}
        # Add all preference probabilities to the user data
        for pref_name, pref_prob in user['preferences'].items():
            user_row[f'{pref_name}_preference'] = round(pref_prob, 3)
        users_data.append(user_row)
    
    users_df = pd.DataFrame(users_data)
    users_df.to_csv('./data/users.csv', index=False)
    print(f"Generated {len(users_df)} unique user records with personalized preferences and saved to ./data/users.csv")


def main():
    """Main function to generate search data."""
    with open('./config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    parser = argparse.ArgumentParser(description="Generate synthetic search data.")
    parser.add_argument("--num-rows", type=int, default=config['num_searches'],
                        help="Number of search records to generate.")
    parser.add_argument("--num-users", type=int, default=config['num_users'],
                        help="Number of unique users to generate.")
    args = parser.parse_args()

    users = generate_users(args.num_users)
    generate_searches(args.num_rows, users)


if __name__ == '__main__':
    main()
