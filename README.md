# Cryptocurrency Liquidity Prediction for Market Stability

## Project Overview

This project aims to build a machine learning model to predict cryptocurrency liquidity levels. In the highly volatile cryptocurrency markets, liquidity plays a critical role in maintaining stability. A lack of liquidity can lead to significant price fluctuations and market instability.

The objective of this project is to develop a robust machine learning solution that can forecast liquidity variations based on various market factors, including trading volume, transaction patterns, exchange listings, and social media activity. By detecting potential liquidity crises early, this model will enable traders and exchange platforms to manage risks more effectively and make informed decisions, contributing to overall market stability.

## Problem Statement

Cryptocurrency markets are characterized by their inherent volatility. Liquidity, defined as the ease with which assets can be bought or sold without significantly impacting their price, is a crucial determinant of market stability. Insufficient liquidity can exacerbate price swings and contribute to broader market instability.

Our goal is to leverage machine learning to predict cryptocurrency liquidity levels proactively. By analyzing diverse market factors, the model will identify early indicators of liquidity crises, providing valuable insights for risk management to traders and financial institutions. The final model will serve as a tool for forecasting liquidity variations, thereby enhancing decision-making in a dynamic market environment.

## Dataset Information

The project will utilize a dataset containing historical cryptocurrency price and trading volume data.
The dataset consists of records from **2016 and 2017**.

**Dataset Source:** <https://drive.google.com/drive/folders/10BRgPip2Zj_56is3DilJCowjfyT6E9AM>

_(Note: Please ensure you download the data from the provided Google Drive link and place it in the `data/raw/` directory as `historical_crypto_data.csv`.)_

## Project Development Steps

The project development follows a structured approach encompassing the following key phases:

1.  **Data Collection:** Gathering historical cryptocurrency price, volume, and liquidity-related data.
2.  **Data Preprocessing:** Handling missing values, ensuring data consistency, and normalizing/scaling numerical features.
3.  **Exploratory Data Analysis (EDA):** Analyzing data patterns, trends, and correlations to gain insights.
4.  **Feature Engineering:** Creating relevant liquidity-related features such as moving averages, volatility, and liquidity ratios.
5.  **Model Selection:** Choosing appropriate machine learning models (e.g., time-series forecasting, regression, deep learning) suitable for liquidity prediction.
6.  **Model Training:** Training the selected model using the thoroughly processed dataset.
7.  **Model Evaluation:** Assessing model performance using key metrics such as Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared ($R^2$) score.
8.  **Hyperparameter Tuning:** Optimizing model parameters to achieve better accuracy and generalization.
9.  **Model Testing & Validation:** Rigorous testing of the optimized model on unseen data to analyze its predictive capabilities.
10. **Local Deployment:** Deploying the trained model using a simple interface (e.g., Flask or Streamlit API) for testing predictions locally.

## Expected Deliverables

Upon completion, the project will deliver the following:

1.  **Machine Learning Model:**
    - A trained model capable of predicting cryptocurrency liquidity.
    - Detailed evaluation metrics demonstrating the model's performance.
2.  **Data Processing & Feature Engineering:**
    - A cleaned and prepared dataset.
    - A brief explanation of all new features engineered.
3.  **Exploratory Data Analysis (EDA) Report:**
    - A comprehensive summary of dataset statistics.
    - Basic visualizations illustrating trends, correlations, and distributions within the data.
4.  **Project Documentation:**
    - **High-Level Design (HLD) Document:** An overview of the system and its architecture.
    - **Low-Level Design (LLD) Document:** A detailed breakdown of how each component is implemented.
    - **Pipeline Architecture Document:** An explanation of the data flow from preprocessing to prediction.
    - **Final Report:** A concise summary of findings, model performance, and key insights.

## Guidelines & Submission Requirements

- **Code Documentation:** All scripts must be well-commented and easy to follow.
- **Report Structure:** All reports must be structured clearly, explaining the methodology followed.
- **Diagrams & Visuals:** Appropriate diagrams and plots should be used to explain data processing, model selection, and performance evaluation.
- **Deployment:** If possible, the model should be deployed using a simple interface (e.g., Streamlit or Flask API) for testing predictions.

## Submission Format

The project must be submitted as a GitHub repository or a zipped folder containing:

- Source Code (`src/`, `notebooks/`, `app/` etc.)
- EDA Report (`docs/EDA_Report.pdf`)
- HLD & LLD Documents (`docs/HLD_Document.pdf`, `docs/LLD_Document.pdf`)
- Pipeline Architecture Document (`docs/Pipeline_Architecture.pdf`)
- Final Report (`docs/Final_Report.pdf`)

## Project Structure

The project follows a standard ML project directory structure for better organization and maintainability. For details, refer to the project's file system structure.

```
cryptocurrency_liquidity_prediction/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── data_processing/
│   ├── models/
│   ├── utils/
│   └── visualization/
├── trained_models/
├── app/
│   ├── app.py
│   ├── templates/
│   └── static/
├── docs/
│   ├── HLD_Document.pdf
│   ├── LLD_Document.pdf
│   ├── Pipeline_Architecture.pdf
│   ├── Final_Report.pdf
│   └── EDA_Report.pdf
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup and Installation

To set up the project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/cryptocurrency_liquidity_prediction.git](https://github.com/your-username/cryptocurrency_liquidity_prediction.git)
    cd cryptocurrency_liquidity_prediction
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download the raw data:**
    - Access the dataset from the Google Drive link provided in the Dataset Information section.
    - Place the downloaded `historical_crypto_data.csv` file into the `data/raw/` directory.

## Usage

Once the setup is complete, you can:

- **Explore data and features:** Run the Jupyter notebooks in the `notebooks/` directory in sequential order.
- **Train the model:** Execute the `src/models/model_trainer.py` script.
- **Run the local deployment:** Navigate to the `app/` directory and run `python app.py` to start the Flask/Streamlit application for testing predictions.

## Contact

For any queries or further information, please contact:
Ram Gopal Gupta E-mail : gopalram781@gmail.com linkedIn : https://www.linkedin.com/in/ramgopagenai/
