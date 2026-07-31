{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 4.0 Modelling\n",
    "Owners: [one per model \u2014 Linear Regression / KNN / Random Forest / Gradient Boosting]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys, pathlib\n",
    "sys.path.append(str(pathlib.Path.cwd().parent))\n",
    "import pandas as pd\n",
    "from src import config\n",
    "from sklearn.model_selection import train_test_split\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv(\"../data/apartments_cleaned.csv\")\n",
    "X = df[config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES]\n",
    "y = df[config.TARGET]\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Build pipeline, fit, evaluate with RMSE / MAE / R\u00b2 here"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}