finance-sentiment-intelligence/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── reddit_scraper.py
│   │   └── data_loader.py
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── cleaner.py
│   │   ├── tokenizer.py
│   │   └── wrangler.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_engineer.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── tuner.py
│   │   └── evaluator.py
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── visualizer.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       └── config.py
│
├── backend/
│   ├── main.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── analyze.py
│   │   └── stats.py
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── models.py
│   │
│   └── services/
│       ├── __init__.py
│       └── predictor.py
│
├── frontend/
│   ├── index.html
│   ├── dashboard.html
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
│
├── tests/
│   ├── test_cleaner.py
│   ├── test_api.py
│   └── test_models.py
│
├── .env
├── .gitignore
├── requirements.txt
├── README.md
├── run_pipeline.py
└── Dockerfile