{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 3.0 Data Preparation\n",
    "Owner: [you]\n"
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
    "from src.data_cleaning import load_and_clean\n",
    "from src import config\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = load_and_clean(\"../data/apartments_for_rent_classified_100K.csv\")\n",
    "df.to_csv(\"../data/apartments_cleaned.csv\", index=False)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3.3 Data Transformation / 3.4 Outlier Treatment\n",
    "Build on the cleaned df above. Keep encoding logic here in sync with src/config.py feature lists."
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