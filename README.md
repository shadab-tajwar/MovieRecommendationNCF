Movie Recommendation System using NCF

[](https://www.google.com/search?q=https://github.com/shadab-tajwar/MovieRecommendationNCF/blob/main/LICENSE)
[](https://www.python.org/)
[](https://keras.io/)

A deep learning-based recommendation system that utilizes **Neural Collaborative Filtering (NCF)** and the **MovieLens 1M dataset** to predict movie ratings and generate personalized recommendations.

Project Overview

This project implements a recommendation engine using a Neural Collaborative Filtering architecture built with Keras (TensorFlow). It covers the full machine learning lifecycle, from data acquisition and preprocessing to model training, evaluation, and recommendation generation.

The notebook explores two primary approaches:

1.  **Base NCF Model:** Pure collaborative filtering using user and movie embeddings.
2.  **Hybrid NCF Model:** Incorporating **side information** (user demographics and movie genres) alongside embeddings for potentially better performance.

Installation

To set up the project locally, you will need Python 3.8+ and the necessary libraries.

Prerequisites

  * Python (3.8 or newer)
  * **Git**

Steps

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/shadab-tajwar/MovieRecommendationNCF.git
    cd MovieRecommendationNCF
    ```

2.  **Install Dependencies**
    It's recommended to use a virtual environment.

    ```bash
    # Create and activate environment (optional)
    # python -m venv venv
    # source venv/bin/activate

    # Install required packages (assuming common ML libraries)
    pip install pandas numpy scikit-learn tensorflow keras
    ```

Dataset Setup and Preparation

The project requires the **MovieLens 1M dataset**. The notebook automatically handles downloading and preprocessing.

1.  **Download and Unzip Data:**
    The following commands are executed within the notebook to download and prepare the files:

    ```bash
    !wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
    !unzip ml-1m.zip
    ```

2.  **Preprocessing Steps:**

      * Converts `ratings.dat`, `movies.dat`, and `users.dat` into clean CSV files (`ratings.csv`, `movies.csv`).
      * **Normalizes ratings** from the original $1-5$ scale to a $0.0-1.0$ scale using the formula: $(R - 1) / 4.0$.
      * Encodes `userId` and `movieId` to $0$-based integer indices for embedding layers.

| Metric | Value |
| :--- | :--- |
| **Number of Users** | 6040 |
| **Number of Movies** | 3706 |
| **Total Ratings** | 1,000,209 |

Usage and Results

### 1\. Base NCF Model

This model uses only learned user and movie embeddings to predict the normalized rating.

**Architecture Summary:**

  * **Inputs:** `user_input` and `movie_input`.
  * **Embeddings:** `user_embedding` (size 64) and `movie_embedding` (size 64).
  * **MLP:** Concatenates flattened embeddings, followed by Dense layers (256, 128, 64) with **Batch Normalization** and **Dropout** (0.5).
  * **Loss:** Mean Squared Error (MSE).

**Training Performance (10 Epochs):**

| Metric | Training Loss | Validation Loss | Validation MAE |
| :--- | :--- | :--- | :--- |
| **Final Epoch (10)** | \~0.0439 | **0.0480** | **0.1721** |

**Evaluation (RMSE):**
$$\text{RMSE} = 0.21907$$

### 2\. Hybrid NCF Model

This model extends the Base NCF by integrating side features: **gender** and **normalized age** for the user, and **one-hot encoded genres** for the movie.

**Architecture Summary:**

  * The model concatenates the user embedding with the user features (gender, age), and the movie embedding with the movie features (genres).
  * These combined vectors are then concatenated and fed into a smaller MLP (128, 64) for the final prediction.

**Training Performance (10 Epochs - Hybrid Model):**

| Metric | Training Loss | Validation Loss | Validation MAE |
| :--- | :--- | :--- | :--- |
| **Final Epoch (10)** | \~0.0446 | **0.0497** | **0.1759** |

**Recommendation Example (User 1 - Hybrid Model):**

| Movie Title | Genres | Predicted Rating (0-1) |
| :--- | :--- | :--- |
| Oscar and Lucinda (1997) | Drama|Romance | 0.89 |
| Bandits (1997) | Drama | 0.88 |
| Feast of July (1995) | Drama | 0.88 |
| Assassins (1995) | Thriller | 0.88 |
| City of Lost Children, The (1995) | Adventure|Sci-Fi | 0.87 |

Contributing

This project is licensed under the Creative Commons Zero v1.0 Universal (CC0) Public Domain Dedication. Feel free to fork the repository, open issues, or submit pull requests with improvements or fixes.

License

This repository is licensed under the **Creative Commons Zero v1.0 Universal (CC0)** Public Domain Dedication. You are free to use, modify, and distribute the work without any restrictions.

Contact

**Project Link:** [https://github.com/shadab-tajwar/MovieRecommendationNCF](https://github.com/shadab-tajwar/MovieRecommendationNCF)
**Maintainer:** shadab-tajwar

-----

Would you like to explore the specifics of the Hybrid NCF architecture in more detail, or perhaps generate a plot of the training history?
