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


def random_bool(prob=0.3):
    """Return a random boolean with a given probability of being True."""
    return random.random() < prob


def generate_searches(num_rows):
    """Generate a given number of search records."""
    # Generate searches data
    searches = []
    for i in range(num_rows):
        # Random date within the next 90 days
        search_date = fake.date_between(start_date='today', end_date='+90d').strftime('%Y-%m-%d')

        # Random US city and state
        city = fake.city()
        state = fake.state_abbr()

        # Create search record
        search = {
            'search_id': f'SEARCH_{i+1:06d}',
            'user_uuid': fake.uuid4(),
            'city': city,
            'state': state,
            'date': search_date,
            'type': random.choice(search_types),
            'hotel_class_score': random.choice(hotel_classes),
            'feature_hot_tub': random_bool(),
            'feature_outdoor_pool': random_bool(0.4),
            'feature_rooftop_pool': random_bool(0.2),
            'feature_infinity_pool': random_bool(0.2),
            'feature_indoor_pool': random_bool(0.3),
            'feature_kiddie_pool': random_bool(0.25),
            'feature_waterpark': random_bool(0.15),
            'feature_cabana': random_bool(0.2),
            'feature_daybed': random_bool(0.2),
            'feature_splash_pad': random_bool(0.15),
            'feature_lazy_river': random_bool(0.1),
            'feature_water_slide': random_bool(0.15),
            'amenities_beach_access': random_bool(0.3),
            'amenities_all_inclusive': random_bool(0.2),
            'amenities_free_parking': random_bool(0.4),
            'amenities_luggage_storage': random_bool(0.3),
            'amenities_airport_shuttle': random_bool(0.25),
            'amenities_cruise_port_shuttle': random_bool(0.15),
            'amenities_gym': random_bool(0.3),
            'amenities_wheelchair_accessible': random_bool(0.3),
            'amenities_showers': random_bool(0.3),
            'amenities_lockers': random_bool(0.3),
            'top_rated': random_bool(0.4),
            'vibes_family_friendly': random_bool(0.3),
            'vibes_party': random_bool(0.2),
            'vibes_serene': random_bool(0.3),
            'vibes_luxe': random_bool(0.25),
            'vibes_trendy': random_bool(0.25)
        }
        searches.append(search)

    # Convert to DataFrame and save to CSV
    searches_df = pd.DataFrame(searches)
    searches_df.to_csv('./data/searches.csv', index=False)
    print(f"Generated {num_rows} search records and saved to ./data/searches.csv")

    # Save unique users to a users.csv file
    users_df = searches_df[['user_uuid']].drop_duplicates().reset_index(drop=True)
    users_df.to_csv('./data/users.csv', index=False)
    print(f"Generated {len(users_df)} unique user records and saved to ./data/users.csv")


def main():
    """Main function to generate search data."""
    with open('./config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    parser = argparse.ArgumentParser(description="Generate synthetic search data.")
    parser.add_argument("--num-rows", type=int, default=config['num_searches'],
                        help="Number of search records to generate.")
    args = parser.parse_args()
    generate_searches(args.num_rows)


if __name__ == '__main__':
    main()
