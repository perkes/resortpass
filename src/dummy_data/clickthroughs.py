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
except FileNotFoundError:
    print("Make sure you have generated the searches and users data first.")
    exit()


def random_bool(prob=0.3):
    """Return a random boolean with a given probability of being True."""
    return random.random() < prob


def generate_clickthroughs(num_rows, user_uuids, search_ids):
    """Generate a given number of clickthrough records."""
    # Generate clickthroughs data
    clickthroughs = []
    for _ in range(num_rows):
        review_score = None
        product_reviewed = None
        if random_bool(0.2):  # 20% chance of having a review
            review_score = round(random.uniform(3.0, 5.0), 1)
            product_reviewed = random.choice(['beach', 'pool', 'other'])

        clickthrough = {
            'user_uuid': np.random.choice(user_uuids),
            'search_id': np.random.choice(search_ids),
            'timestamp': pd.to_datetime(np.random.choice(pd.to_datetime(searches_df['date']))) + pd.to_timedelta(np.random.randint(1, 60), unit='m'),
            'hotel_id': np.random.randint(1, 100),
            'total_time_spent': random.randint(10, 600),
            'used_pictures_carousel': random_bool(0.5),
            'saved': random_bool(0.2),
            'read_reviews': random_bool(0.4),
            'scrolled_to_bottom': random_bool(0.6),
            'converted': random_bool(0.1),
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

    user_uuids = users_df['user_uuid'].unique()
    search_ids = searches_df['search_id'].unique()

    generate_clickthroughs(args.num_rows, user_uuids, search_ids)


if __name__ == '__main__':
    main()
