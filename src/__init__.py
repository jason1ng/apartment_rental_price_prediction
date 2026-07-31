{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 2.0 Data Understanding \u2014 EDA\n",
    "Owner: [teammate name]\n"
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
    "from src.data_cleaning import load_raw\n",
    "import pandas as pd, matplotlib.pyplot as plt, seaborn as sns\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load RAW (uncleaned) data for this section \u2014 2.0 describes the data before cleaning\n",
    "df_raw = load_raw(\"../data/apartments_for_rent_classified_100K.csv\")\n",
    "df_raw.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2.3\u20132.6: summary stats, missingness, EDA plots go here\n",
    "Save each figure to ../outputs/figures/ so it can be pulled into the report."
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