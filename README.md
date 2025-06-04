# Sprint 16 – IMDB Review Classification

Notebook:
[Notebook del Proyecto](./project_3_machine_learning__for_text.ipynb)

**Film Junky Union** is building a system to filter and categorize classic movie reviews. In this project, the goal is to train a machine learning model that automatically classifies IMDB reviews as **positive** or **negative**, achieving an F1 ≥ 0.85.

---

## Project Structure

1. **Data Loading and Normalization**  
   - Read `imdb_reviews.tsv`.  
   - Clean the text (lowercase, remove digits and punctuation).  
   - Check for missing or duplicate values.

2. **Exploratory Data Analysis (EDA)**  
   - Visualize class distribution (positive vs. negative).  
   - Verify train/test split balance.  
   - Explore review length distribution and temporal trends.

3. **Preprocessing for Modeling**  
   - Split into *train* / *test* sets using the `ds_part` column.  
   - (Optional) Lemmatize with spaCy to reduce vocabulary.  
   - Vectorize text using TF-IDF.

4. **Model Training**  
   At least four different approaches were trained:  
   - **Model 0 (DummyClassifier)**: constant classifier as a baseline.  
   - **Model 1 (TF-IDF + LogisticRegression)**: simple logistic regression over TF-IDF vectors.  
   - **Model 2 (spaCy + TF-IDF + LogisticRegression)**: adds a lemmatization step.  
   - **Model 3 (spaCy + TF-IDF + LGBMClassifier)**: LightGBM over lemmatized TF-IDF vectors.  
   - **Model 4 (BERT + LogisticRegression)** *(optional)*: BERT embeddings + LR, used on a small subset due to computational cost.

5. **Evaluation**  
   - Metrics: Accuracy, Precision, Recall, F1, AUC-PR, and AUC-ROC.  
   - Plots: Precision-Recall curves, ROC curves, and confusion matrices.  
   - Compare results across models and analyze trade-offs between speed and performance.

6. **Validation with Sample Reviews**  
   - Include a small set of sample reviews (`my_reviews`) to demonstrate each model’s predictions.  
   - Normalize, lemmatize, and vectorize those reviews before passing them to the classifier.  
   - Display the probability of the positive class and a snippet of the original text.

7. **Conclusions**  
   - Summarize final metrics for each model.  
   - Recommend the optimal model based on precision vs. execution time.  
