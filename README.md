# Project Description
This project serves as a hand-in for the final exam for 02476 Machine Learning Operations (January 2024). Our group (number 10) consists of the following memebers.
1. Laura Paz
2. Shah Bekhsh
3. Tomasz 
4. Aiax
5. Diana


## Inspiration & Data
As online news articles are being quoted and shared by billions of people everyday, the validity of these news articles and the information contained within them is continuously called into question. The goal of this project is to create a text classifier model that analyzes a given news article and identifies whether it is authentic or fake. It is trained on a dataset ([obtained from kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification/data)) that contains the following information:

1. The name of the publishing organization (for eg. Reuters, New York Times, etc.)
2. The title of the news article
3. The text of the news article

There are 72,133 unique articles and one binary classification label column. **Note: In the data available on Kaggle, 0=Real and 1=Fake.**

## Framework & Pipeline
For this project, the base framework will be PyTorch, which we will be using to create, train and validate our model (exact model algorithm to be decided). We will also be using an NLP [transformer](https://huggingface.co/docs/transformers/index) from huggingface (exact transformer to be decided) for additional functionality and robustness regarding NLP. To make this project more streamlined and available, we will be implementing version controlling for our data using [DVC](https://dvc.org/) and there will be docker images available for the training and prediction. 

# src



## Project Structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── src  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
