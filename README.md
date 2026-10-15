# 📰 Fake News Detection System

A Machine Learning-based web application that predicts whether a news article is **Fake News or Real News** using Natural Language Processing (NLP) and Logistic Regression.

The project uses Flask to provide a user-friendly web interface where users can enter a news article and view the model's prediction along with a confidence score.

## ✨ Features

* 📰 Fake and real news classification.
* 🤖 Machine learning-based prediction.
* 🧠 Text processing using TF-IDF Vectorization.
* 🌐 Interactive web interface built with Flask.
* 🎨 User-friendly design using HTML and CSS.
* 📊 Displays prediction results and model confidence scores.
* 🔄 Separate model training script for training the classifier.

## 🛠️ Technologies Used

| Technology          | Purpose                               |
| ------------------- | ------------------------------------- |
| Python              | Core programming language             |
| Flask               | Web application framework             |
| Pandas              | Dataset loading and processing        |
| Scikit-learn        | Machine learning and model evaluation |
| TF-IDF Vectorizer   | Converts text into numerical features |
| Logistic Regression | Classifies news articles              |
| HTML                | Web page structure                    |
| CSS                 | Web page styling                      |
| Pickle              | Saves and loads trained model objects |

## 📂 Project Structure

```text
Fake-News-Detection-System/
│
├── app.py
├── train_model.py
├── requirements.txt
│
├── dataset/
│   ├── Fake.csv
│   └── True.csv
│
├── model/
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── README.md
└── LICENSE
```

**Note:** The filenames shown above are examples. Make sure they match your actual project files.

## 📥 Dataset Download

This project uses the Fake and Real News Dataset to train the machine learning model.

**Download the dataset from Kaggle:**

👉 [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

### How to Set Up the Dataset

1. Open the Kaggle dataset link.
2. Download the dataset ZIP file.
3. Extract the downloaded ZIP file.
4. Locate `Fake.csv` and `True.csv`.
5. Place both files inside the `dataset` folder.

Your dataset directory should look like this:

```text
dataset/
├── Fake.csv
└── True.csv
```

**Important:** If the dataset files are not included in this GitHub repository, users must download them separately before training the model.

## ⚙️ Installation and Setup

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
```

Replace `<your-repository-url>` with your actual GitHub repository URL.

### Step 2: Navigate to the Project Directory

```bash
cd Fake-News-Detection-System
```

### Step 3: Create a Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

### Step 4: Install Dependencies

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

### Step 5: Download the Dataset

Download the dataset from Kaggle and place `Fake.csv` and `True.csv` inside the `dataset` folder.

### Step 6: Train the Machine Learning Model

Run the training script:

```bash
python train_model.py
```

This script prepares the dataset, extracts text features, trains the classifier, evaluates its performance, and saves the trained model files.

**Note:** The saved model filenames and output directory depend on the implementation in `train_model.py`.

### Step 7: Run the Flask Application

Start the web application:

```bash
python app.py
```

### Step 8: Open the Application

Open your browser and visit:

**http://127.0.0.1:5000/**

Enter a news article into the input field and submit it to see the prediction.

## 🔍 How It Works

The system follows these steps:

1. **Dataset Loading:** Loads fake and real news articles from the CSV files.
2. **Data Preprocessing:** Prepares the text data for machine learning.
3. **Feature Extraction:** TF-IDF converts news text into numerical features.
4. **Model Training:** Logistic Regression learns patterns from the labeled training data.
5. **Model Saving:** Saves the trained classifier and vectorizer for future predictions.
6. **User Input:** Accepts a news article through the Flask web interface.
7. **Prediction:** Processes the submitted text using the trained vectorizer and classifier.
8. **Result Display:** Displays the predicted category and confidence score on the webpage.

## 🧠 Machine Learning Concepts

* Natural Language Processing (NLP)
* TF-IDF Vectorization
* Logistic Regression
* Supervised Learning
* Training and Testing Data
* Text Classification
* Model Evaluation

## 📊 Model Evaluation

The model can be evaluated using the following metrics:

* **Accuracy:** Measures the overall proportion of correct predictions.
* **Precision:** Measures how many predictions for a class are correct.
* **Recall:** Measures how many actual examples of a class are correctly identified.
* **F1-Score:** Combines precision and recall into a single metric.

Actual performance depends on the dataset, preprocessing, train-test split, and model implementation.

## ⚠️ Limitations

This system predicts whether an article resembles fake or real news based on patterns learned from its training dataset.

It does not independently verify news claims against trusted sources. Therefore, its predictions should not be treated as definitive proof that a news article is true or false.

## 🚀 Future Improvements

* Integrate trusted news sources for fact-checking.
* Experiment with advanced NLP and transformer-based models.
* Improve generalization to unfamiliar news topics.
* Deploy the application to a cloud platform.
* Add more datasets to improve evaluation across different sources.

## 🎯 Project Objective

The main objective of this project is to demonstrate how machine learning and natural language processing can be applied to text classification through a simple web application.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
