# Urban Feedback Classifier

This project implements a **text classification system** for UrbanTech’s transit app.  
The goal is to automatically route user feedback messages into three categories:

- **Navigation Issues** — problems with routes, maps, or directions  
- **Service Updates** — issues with bus/train schedules or transit information  
- **App Experience** — problems with the app interface, crashes, or permissions  

By automating classification, UrbanTech can respond faster and more consistently to user feedback.

---

## 📂 Repository Contents
- `text_modeling_lab.ipynb` — Jupyter notebook with the full workflow (Steps 0–7)  
- `urban_feedback.csv` — dataset of user feedback messages  

---

## 🚀 Workflow

1. **Data Exploration**  
   - 197 feedback messages evenly distributed across 3 classes.  

2. **Preprocessing**  
   - Lowercasing, punctuation/number removal  
   - Tokenization, stopword removal  
   - POS-based lemmatization  

3. **Feature Extraction**  
   - Bag of Words (BoW) and TF-IDF representations  
   - Comparison of vocabulary and feature matrices  

4. **Model Training**  
   - Multinomial Naive Bayes with BoW and TF-IDF  

5. **Evaluation**  
   - Metrics: Accuracy, Precision, Recall, F1-score  
   - TF-IDF outperformed BoW (0.70 vs. 0.675 accuracy)  

6. **Model Improvement**  
   - GridSearchCV with TF-IDF parameters (`max_features=100, min_df=3, max_df=0.85, alpha=1.0`)  
   - Improved model achieved **0.85 accuracy** and balanced F1 across all classes  

7. **Prediction Function**  
   - `classify_feedback()` returns predicted department, confidence score, and all class probabilities.  
   - Example:  
     ```python
     classify_feedback("The subway times were completely wrong today.")
     # Output → {'department': 'service_updates', 'confidence': 0.68, ...}
     ```

---

## 📊 Results

- **Best Model:** TF-IDF + Multinomial Naive Bayes  
- **Test Accuracy:** 85%  
- **Macro F1 Score:** 0.85  
- Reliable classification across all three categories, ready for production use.  

---

## 🔧 Future Improvements
- Add bigrams (e.g., *bus schedule*, *route update*)  
- Increase vocabulary size  
- Periodically retrain with new data to capture evolving feedback patterns  

---

## 👩‍💻 Author
Developed as part of a Data Science lab project on **Text Modeling with Naive Bayes**.
