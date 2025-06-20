# ResortPass Hotel Ranking Model

This project implements a learning-to-rank model for the ResortPass search engine. The model predicts the likelihood of a user converting (booking) a hotel based on search criteria, user behavior, and hotel features.

For a comprehensive overview of the solution, see:
- **[Technical Deep Dive](NOTA_PRAEVIA.md)** - Detailed technical documentation
- **[Project Presentation](https://docs.google.com/presentation/d/1jOxpOtSjWqTI71hj9bgUcKmeWFokljzXknoSF2eIlBs/edit?slide=id.g36a2e215c8d_0_1301#slide=id.g36a2e215c8d_0_1301)** - Visual overview and key concepts

## Prerequisites

Before you begin, ensure you have the following installed:

*   **Python (3.8+):** [Download Python](https://www.python.org/downloads/)
*   **virtualenv:** A tool to create isolated Python environments.
    ```bash
    pip3 install virtualenv
    ```

### Platform-Specific Requirements

**macOS:**
- **Homebrew:** Required for installing OpenMP (needed by LightGBM)
  ```bash
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  ```

**Ubuntu/Linux:**
- **build-essential and OpenMP:** Will be installed automatically by the setup script
  ```bash
  sudo apt-get update
  sudo apt-get install -y build-essential libgomp1
  ```

## Setup

1.  **Make Scripts Executable:**

    Before running the setup scripts, you need to make them executable. Open your terminal and run the following command from the project root:

    ```bash
    chmod +x setup.sh generate_data.sh train_test.sh clean.sh
    ```

2.  **Run the Setup Script:**

    The `setup.sh` script will:
    - Install system dependencies (OpenMP for LightGBM) based on your platform
    - Create a Python virtual environment 
    - Install the required Python dependencies from `requirements.txt`
    
    **Note:** On macOS, this requires Homebrew to be installed first.

    ```bash
    ./setup.sh
    ```

    This will create the following directories:
    *   `venv/`: The Python virtual environment.
    *   `data/`: Contains the generated `searches.csv`, `users.csv`, `hotels.csv` and 
        `clickthroughs.csv` files.
    *   `models/`: Stores the trained model and artifacts.
    *   `logs/`: Stores log files from the training and prediction scripts.

3. **Generate Dummy Data**

    The `generate_data.sh` script will create the following files in the data directory 
    that will be used during the model's training:
    * `clickthroughs.csv`: All details related to user interactions with hotel pages
        after clicking the hotel link. Linked to searches through the search_id FK.
    * `hotels.csv`: Contains all features, amenities, vibes and details related to hotels.
    * `searches.csv`: Contains all the details related to a search, location, date, filters, etc.
    * `users.csv`: All generated user ids.

To call the script:
```bash
./generate_data.sh
```

Relevant parameters for this script in config.json:
* `num_searches`
* `num_clickthroughs`
* `num_users`
* `num_hotels`

## Training and Evaluation

To train the model and evaluate its performance, run the `train_test.sh` script:

```bash
./train_test.sh
```

This script will:
1.  Execute the training script (`src/model/train.py`).
2.  Save the trained model to `models/hotel_ranking_model.txt` and artifacts to `models/ranking_artifacts.pkl`.
3.  Execute the prediction script (`src/model/predict.py`) to generate a sample ranking.

## Configuration

The `config.json` file allows you to configure various aspects of the project without modifying the code.

*   `num_searches`, `num_clickthroughs`: Control the amount of dummy data to generate.
*   `model_params`: Parameters for the LightGBM ranking model.
*   `search_preferences`, `user_uuid`, `num_hotels`: Used by the prediction script (`src/model/predict.py`) to generate rankings for a specific user and search query.

You can modify these values to experiment with different scenarios.

## Logging

The scripts generate logs to help you monitor their execution and debug issues.

*   `logs/train.log`: Contains logs from the training process.
*   `logs/predict.log`: Contains logs from the prediction process, including the final ranked list of hotels.

You can view the logs using any text editor or the `cat` command:

```bash
cat logs/train.log
```

## Cleaning Up

The `clean.sh` script removes all generated directories (`data`, `logs`, `models`, `venv`). This is useful for starting with a clean slate.

```bash
./clean.sh
```

**Warning:** This will permanently delete the contents of these directories.
