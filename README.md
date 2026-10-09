# Email Spam Classifier

A production-ready email spam detection web application built with Flask, scikit-learn, and a pre-trained TF-IDF + classifier pipeline. The app automatically downloads the SMS spam dataset if it is missing, trains three models, generates evaluation charts, and serves a premium dark-mode dashboard for classification and inspection.

## Features

- Spam vs. ham email detection with probability scoring
- Real-time text analysis with debounce and manual analyze button
- Three model options: Multinomial Naive Bayes, Logistic Regression, and Linear SVM
- Probability ring, confidence score, and word highlighting
- Model performance dashboard with confusion matrices and comparison cards
- Keyword insights for spam and ham terms
- Local history stored in browser localStorage
- Render-ready deployment configuration

## Project structure

- app.py — Flask API and web app
- train.py — dataset download, training pipeline, evaluation, and chart generation
- preprocess.py — text cleaning and preprocessing utilities
- data/ — dataset storage and generated CSV copy
- models/ — trained model bundle and artifacts
- static/css/ — premium dark UI styling
- static/js/ — frontend logic and client-side interactions
- static/charts/ — generated evaluation charts
- templates/index.html — single-page UI

## Local setup

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Train the model and generate charts:

   ```bash
   python train.py
   ```

4. Run the app:

   ```bash
   python app.py
   ```

5. Open the UI in a browser at:

   ```text
   http://localhost:5000
   ```

## Production run

```bash
gunicorn app:app
```

## Render deployment

1. Push this project to a GitHub repository.
2. In Render, create a new Web Service and connect the repository.
3. Set the build command:

   ```bash
   pip install -r requirements.txt && python train.py
   ```

4. Set the start command:

   ```bash
   gunicorn app:app
   ```

5. Select Python 3.11 in the environment settings.
6. Deploy.

Render also reads the included `render.yaml` and `runtime.txt` configuration.

## Notes

- The project downloads the UCI SMS spam dataset automatically if it is missing using a built-in fallback workflow.
- The training step saves a copy into `data/` and generates evaluation images under `static/charts/`.
- The API includes `/health`, `/api/predict`, `/api/metrics`, and `/api/keywords` endpoints.

## License

This project is intended for educational and demonstration use.
