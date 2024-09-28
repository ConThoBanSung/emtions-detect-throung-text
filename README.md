# Twitter Sentiment Analysis using Logistic Regression

## Overview

This project aims to perform sentiment analysis on Twitter data using a Logistic Regression model. The model is trained to classify tweets into two categories: positive and negative sentiments. The project includes scripts for training the model, evaluating its performance, and visualizing results.

## Features

- Data preprocessing using stemming and TF-IDF vectorization
- Logistic Regression for sentiment classification
- Evaluation metrics including accuracy, precision, recall, and F1 score
- Visualization of confusion matrix
- Saving trained models and metrics for future use

## Requirements

To run this project, you'll need the following libraries:

- `numpy`
- `pandas`
- `scikit-learn`
- `nltk`
- `joblib`
- `matplotlib`
- `seaborn`

You can install the required libraries using pip:

```bash
pip install numpy pandas scikit-learn nltk joblib matplotlib seaborn
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/twitter-sentiment-analysis.git
   cd twitter-sentiment-analysis
   ```

2. Download the dataset:
   - Ensure you have the Twitter dataset file named `training.1600000.processed.noemoticon.csv` placed in the project directory.

## Usage

### Training the Model

Run the following command to train the Logistic Regression model:

```bash
python train_model.py
```

This will preprocess the data, train the model, and save the trained model and vectorizer as `logistic_regression_model.joblib` and `tfidf_vectorizer.joblib`.

### Evaluating the Model

After training the model, you can evaluate its performance by running:

```bash
python evaluate_model.py
```

This script will load the saved model, preprocess the test data, and output evaluation metrics to the console. It will also save a CSV file named `model_metrics.csv` containing the metrics.

## Results

The evaluation will produce the following metrics:
- **Accuracy:** The proportion of true results among the total number of cases examined.
- **Precision:** The ratio of correctly predicted positive observations to the total predicted positives.
- **Recall:** The ratio of correctly predicted positive observations to all actual positives.
- **F1 Score:** The weighted average of Precision and Recall.

  ![image](https://github.com/user-attachments/assets/0c583a00-b622-4ec8-8948-6c7d56765711)

  ![image](https://github.com/user-attachments/assets/5bc14137-5b3d-473c-92d1-0a6581ae60a8)

  ![image](https://github.com/user-attachments/assets/fc33162f-d075-48c1-84b0-1f7f6e4f60ab)

  ![image](https://github.com/user-attachments/assets/c871a3b1-8aa7-439a-b95c-8f20314872f0)

  ![image](https://github.com/user-attachments/assets/22e63019-23ac-4438-9673-856c782125d9)

  






A confusion matrix will also be displayed to visualize the model's performance.

## Contributing

Contributions are welcome! If you have suggestions for improvements or features, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the NLTK library for providing the stopwords and PorterStemmer functionalities.
- The dataset used in this project is sourced from [Sentiment140](http://help.sentiment140.com/for-students/).
