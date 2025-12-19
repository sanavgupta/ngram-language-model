# N-Gram Language Models 🗣️

## Overview
This project implements probabilistic Language Models (LMs) from scratch using Python. It explores the foundational mathematics behind Natural Language Processing (NLP) by building models that estimate the likelihood of token sequences based on training corpora (e.g., Shakespeare, Project Gutenberg books).

The project progresses from baseline models to a fully functional **N-Gram Model** capable of generating coherent text based on the Markov assumption.

## Key Features
* **Corpus Pipeline:** Custom web scraper and tokenizer to fetch and clean raw text data from Project Gutenberg.
* **Probabilistic Modeling:** Implementation of Uniform, Unigram, and N-Gram models using conditional probability and the chain rule.
* **Text Generation:** A sampling method that generates synthetic text by selecting tokens based on their computed conditional probabilities.
* **Optimization:** Utilizes Pandas and NumPy for efficient vectorization and probability matrix handling.

## Technical Implementation
* **Language:** Python 3.x
* **Libraries:** Pandas, NumPy, Requests, Regex
* **Concepts:** Markov Chains, Conditional Probability, Tokenization, Data Structures (Tries/Hash Maps via Pandas)

## Usage
To run the models and generate text:

1. Clone the repository.
2. Open `notebooks/language_models_demo.ipynb`.
3. Run the cells to train the model on the provided Shakespeare corpus and generate new sentences.
