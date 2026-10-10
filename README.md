# Email Spam Classifier

A production-ready email spam detection web application built with Flask and scikit-learn. The app downloads the SMS spam dataset automatically if it is missing, trains three models on a TF-IDF pipeline, generates evaluation charts, and serves a dark-mode dashboard for classification and inspection.

**Live demo:** https://email-spam-classifier-zil6.onrender.com

**Author:** Christina Jenifer, B.Tech Artificial Intelligence and Data Science, Jayalakshmi Institute of Technology

## Features

- Spam vs. ham detection with probability scoring
- Real-time text analysis with debounce, plus a manual Analyze button
- Three model options: Multinomial Naive Bayes, Logistic Regression and Linear SVM
- Animated probability ring, confidence score and spam word highlighting
- Model performance dashboard with confusion matrices and comparison cards
- Keyword insights for the top spam and ham terms
- Local history of recent checks stored in browser localStorage
- Render-ready deployment configuration

## Tech Stack

Python, Flask, gunicorn, scikit-learn, pandas, NumPy, NLTK, Matplotlib, Seaborn, HTML, CSS, JavaScript

## Dataset

UCI SMS Spam Collection (5,572 labelled messages)
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

The dataset is downloaded automatically by `train.py` if it is not already present in `data/`.

## Project Structure

```
├── app.py              Flask API and web app
├── train.py            Dataset download, training, evaluation, chart generation
├── preprocess.py       Text cleaning and preprocessing utilities
├── requirements.txt    Pinned dependencies
├── render.yaml         Render deployment config
├── runtime.txt         Python version
├── data/               Dataset storage
├── models/             Trained model bundle and artifacts
├── static/
│   ├── css/            UI styling
│   ├── js/             Frontend logic
│   └── charts/         Generated evaluation charts
└── templates/
    └── index.html      Single-page UI
```

## Local Setup

1. Create and activate a virtual environment (Python 3.11 recommended):

   Windows:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```

   macOS / Linux:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Train the models and generate charts:
   ```
   python train.py
   ```

4. Run the app:
   ```
   python app.py
   ```

5. Open http://localhost:5000 in your browser.

## Production Run

```
gunicorn app:app
```

## Render Deployment

1. Push this project to a GitHub repository.
2. In Render, create a new Web Service and connect the repository.
3. Set the build command:
   ```
   pip install -r requirements.txt && python train.py
   ```
4. Set the start command:
   ```
   gunicorn app:app
   ```
5. Select Python 3.11 in the environment settings.
6. Deploy.

Render also reads the included `render.yaml` and `runtime.txt`.

## API Endpoints

| Method | Route | Description |
|---|---|---|
| GET | `/` | Web UI |
| POST | `/api/predict` | Classify a message |
| GET | `/api/metrics` | Model performance metrics |
| GET | `/api/keywords` | Top spam and ham keywords |
| GET | `/health` | Health check |

Example request:

```
POST /api/predict
{"text": "You won a free prize! Click now", "model": "lr"}
```

## Notes

- Training saves a copy of the dataset into `data/` and generates evaluation images under `static/charts/`.
- On Render's free tier the app sleeps after inactivity, so the first load can take 30 to 50 seconds.

## Links

- Live demo: https://email-spam-classifier-zil6.onrender.com
- GitHub: https://github.com/Jeniferyesuraja/EMAIL-SPAM-CLASSIFIER

## License

This project is intended for educational and demonstration use.