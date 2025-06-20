"""
Generates synthetic hotel data for the ResortPass model.
"""
import random
import argparse
import json
import pandas as pd

def random_bool(prob=0.3):
    """Return a random boolean with a given probability of being True."""
    return random.random() < prob

def generate_hotels(num_hotels):
    """Generate a given number of hotel records."""
    hotels = []
    for i in range(1, num_hotels + 1):
        hotel = {
            'hotel_id': i,
            'hotel_class_score': random.choice([3, 4, 5]),
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
        hotels.append(hotel)

    hotels_df = pd.DataFrame(hotels)
    hotels_df.to_csv('./data/hotels.csv', index=False)
    print(f"Generated {num_hotels} hotel records and saved to ./data/hotels.csv")

def main():
    """Main function to generate hotel data."""
    with open('./config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    parser = argparse.ArgumentParser(description="Generate synthetic hotel data.")
    parser.add_argument("--num-hotels", type=int, default=config['num_hotels'],
                        help="Number of hotel records to generate.")
    args = parser.parse_args()
    generate_hotels(args.num_hotels)

if __name__ == '__main__':
    main()
