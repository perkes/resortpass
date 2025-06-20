# ResortPass Hotel Ranking Model: Technical Deep Dive

## Introduction

The goal of this project is to evolve the ResortPass hotel ranking system from its current manual, static state into a dynamic, personalized engine powered by machine learning. The existing approach, where hotels are ranked uniformly for all users based on internal criteria like sales performance, has significant limitations. It fails to cater to individual user preferences, potentially leading to a frustrating user experience and missed conversion opportunities.

This document provides a detailed overview of the strategy, modeling approach, and technical implementation of a new learn-to-rank system designed to address these challenges. It expands upon the concepts presented in the project slides, offering a deeper look into the data, feature engineering, and production considerations.

## 1. Data: The Foundation of Personalization

To build a model that understands user intent, we first need the right data. The core idea is to capture user search preferences and their subsequent interactions with hotel listings. For this project, we conceptualized and generated two primary datasets.

### Data Generation

The scripts in `src/dummy_data/` generate synthetic but (somewhat) realistic data that simulates user behavior on the ResortPass platform.

-   **`searches.csv`**: Represents a user's search query. Each row is a unique search session, containing:
    *   `search_id`: A unique identifier for the search session.
    *   `user_uuid`: A unique identifier for the user.
    *   `date` and `location`: The basic parameters of the search.
    *   Search Preferences: A rich set of boolean flags indicating what the user is looking for, such as `feature_hot_tub`, `amenities_beach_access`, or `vibes_family_friendly`. These are crucial for understanding explicit user intent.

-   **`clickthroughs.csv`**: Represents a user's interaction with a hotel listing during a search session. This behavioral data is the key to unlocking implicit preferences. It includes:
    *   `search_id` and `user_uuid`: To link the interaction back to a search and a user.
    *   `hotel_id`: The hotel that was interacted with.
    *   `converted`: The target variable. A boolean flag indicating if the user booked the hotel.
    *   Engagement Metrics: Fields like `total_time_spent`, `used_pictures_carousel`, `saved`, and `read_reviews` provide strong signals about a user's level of interest in a property.

## 2. Modeling Approach: A Learn-to-Rank System

Instead of predicting a simple "click" or "no-click" outcome (classification), the goal is to determine the *optimal order* of hotels for a given search. This is a classic ranking problem, and we've chosen a **learn-to-rank (LTR)** approach.

### Why LightGBM?

We selected **LightGBM**, a gradient-boosting framework, as our modeling tool for several reasons:
*   **Ranking-Optimized:** It has a built-in `LGBMRanker` that is specifically designed for LTR tasks. It uses a group-wise objective function (LambdaRank) that directly optimizes for ranking metrics like NDCG.
*   **Performance:** It is known for its high speed and efficiency, making it suitable for real-time prediction scenarios.
*   **Feature Handling:** It handles a mix of numerical and categorical features gracefully and is robust to missing values.

### Target Variable: Conversion

The model is trained to predict the probability of **conversion**. We chose conversion over a metric like revenue for several key reasons:
*   **User-Centric:** A conversion is a direct indicator of user satisfaction and intent fulfillment. The user found what they were looking for.
*   **Stability:** Revenue can be easily skewed by promotions or a few high-priced bookings, which might not reflect true user preference. Optimizing for conversions leads to more sustainable long-term growth.
*   **Interpretability:** It is easier to debug and understand a model that is optimizing for a clear action (a booking) rather than a fluctuating monetary value.

### Model Architecture

The model is trained on groups of historical data, where each group corresponds to a unique `search_id`. This allows it to learn from the context of a full search session. When a user performs a new search, a list of candidate hotels is generated based on the search criteria (e.g., location, availability). The model then scores and ranks these candidates, ordering them based on the predicted likelihood of conversion for that specific user. The goal is not to predict an absolute probability, but to determine the optimal relative ordering of the hotels.

## 3. Feature Engineering: Understanding the User

The raw data is transformed into a rich set of features to feed the model. This is where the personalization happens. The feature engineering strategy is detailed in `src/model/train.py` and can be broken down into four main categories:

*   **User-Based Features:** These capture the long-term global behavior and preferences of a specific user, aggregated from all their past activities.
    *   *Examples:* `user_conversion_rate` (how likely is this user to book in general?), `user_avg_time_spent`, and their preferences for specific amenities, features, and "vibes" based on their search history.

*   **Hotel-Based Features:** These capture the overall popularity and quality of a hotel, aggregated from all user interactions.
    *   *Examples:* `hotel_conversion_rate` (how often is this hotel booked?), `hotel_avg_review_score`, and its average engagement metrics.

*   **Interaction Features:** These dynamic features capture the specific match between a user's current search and a hotel's attributes, as well as their detailed past interactions with it.
    *   *Examples:* A `match_score` that calculates how well a hotel's features align with the user's stated preferences. We also track a user's specific history with each hotel, including `user_has_booked_hotel`, `user_hotel_saves` (number of times saved), and `user_hotel_read_reviews` (number of times reviews were read).

*   **Time-Based Features:** These capture temporal patterns in user behavior.
    *   *Examples:* `search_hour_of_day`, `search_day_of_week`, and `booking_lead_time` (the number of days between the search and the travel date).

## 4. Success Metrics and Experimentation

To evaluate the model and validate its effectiveness, we use both offline and online testing strategies.

### Offline Evaluation

During training, we use a **temporal data split**, using older searches for training and more recent ones for testing. This prevents data leakage and simulates a real-world scenario where we predict future behavior. The key success metric is:

*   **NDCG (Normalized Discounted Cumulative Gain):** It evaluates the quality of the ranking by giving higher scores to lists that place relevant items (converted hotels) closer to the top. It's the industry standard for ranking evaluation.

### Online Experimentation: A/B Testing

The ultimate test of the model is its performance in a live environment. We would propose a controlled **A/B test**:
*   **Control Group:** A segment of users who continue to see the old, manually curated hotel rankings.
*   **Treatment Group:** A segment of users who see the new, personalized rankings from the ML model.

We would then track key business KPIs for both groups to measure the impact of the new system. The primary success metric would be an increase in the overall **conversion rate**. Secondary metrics would include **click-through rate (CTR)**, **user session length**, and **long-term user retention**. We would also monitor **average booking value** to ensure the new model doesn't inadvertently favor lower-priced options at the expense of overall revenue.

## 5. Challenges and Production Considerations

Deploying a machine learning model into a production environment comes with a unique set of challenges, from data pipelines to system architecture and potential biases.

### Common Challenges & Mitigations

*   **The Cold Start Problem:** How do we rank hotels for a brand-new user or a newly listed hotel?
    *   **Mitigation:** For new users, we fall back to a simpler, popularity-based model using global feature averages. For new hotels, they can be temporarily boosted in rankings to gather initial interaction data.

*   **Data Freshness and Scalability:** The system requires a robust data pipeline that can process user interactions and update features in near real-time.
    *   **Mitigation:** A queue-based system for incremental feature updates, processed in small batches, ensures that the features used for ranking are always fresh without overwhelming the system.

*   **Model Drift:** User preferences and hotel inventory change over time. The model's performance will degrade if it's not regularly retrained.
    *   **Mitigation:** The model should be retrained on a regular schedule (e.g., weekly) using the latest data to adapt to these changing patterns.

*   **Seasonality:** User travel patterns can vary significantly depending on the time of year.
    *   **Mitigation:** The model should be trained on at least one full year of data to capture these seasonal effects.

*   **Power Users:** A small percentage of users might generate a disproportionately large amount of revenue and interaction data.
    *   **Mitigation:** We should monitor the model's performance across different user segments. If a power-user bias is detected, we could consider training separate models for different user tiers.

### Production Architecture

For a production system, latency and availability are critical. The prediction script (`src/model/predict.py`) outlines a basic pipeline, but a robust production architecture would be more sophisticated.

*   **Cache Strategy:** To ensure a fast response time (<50ms p95 latency), we would implement a **caching layer** using Redis. This cache would store pre-computed ranking scores for user-hotel pairs.
    *   **Memory Optimization:** To manage the cache size, scores can be quantized to 8-bit integers, and hotel IDs can be compressed.
    *   **Hotel ID Compression:** Permanent hotel IDs can be mapped to transient Huffman codes, which are updated periodically based on popularity. This can yield a 3-4x compression ratio.

*   **Real-Time Updates:** User features are updated incrementally via a queue-based system. Actions are processed in batches (e.g., every 100-1000 actions) to re-score affected users without requiring a model retrain.

*   **Fallback Strategy:** A multi-layered fallback system ensures a graceful degradation of service.
    *   If a user's ranked list is not in the cache (a cache miss), we invoke the real-time scoring pipeline.
    *   For new users, we use global feature averages to provide a sensible default ranking.
    *   The system is designed for no service interruption during cache updates.

*   **Performance Targets:**
    *   **Cached Requests:** <50ms p95 latency.
    *   **Cache Misses (Real-time scoring):** <500ms p95 latency.
    *   **Cache Hit Rate:** ~98%+ for active users.

## 6. Future Work

The current model provides a strong foundation, but there are many opportunities for future enhancement.

*   **Advanced Feature Engineering:** Incorporate user and hotel embeddings or graph-based features to capture more nuanced relationships.
*   **Richer Data Signals:** Integrate real-time external data streams, such as local events, google trends, weather forecasts, or social media trends, to provide more context-aware rankings.
*   **Multi-Objective Optimization:** Evolve the model to optimize for a combination of goals, such as balancing conversion rate with revenue, profit margin, or hotel partner satisfaction.
*   **Deep Learning Models:** Experiment with more complex neural ranking models to potentially capture non-linear patterns that the gradient boosting model might miss.
*   **Explainability:** Add a layer to the UI that explains *why* certain hotels are being recommended (e.g., "Popular with families," "Great match for your preferences," "You've saved this hotel before"). This builds user trust and provides transparency.
*   **Continuous Improvement:** Implement a robust A/B testing framework to allow for the continuous, data-driven optimization of the ranking model and its underlying features.
