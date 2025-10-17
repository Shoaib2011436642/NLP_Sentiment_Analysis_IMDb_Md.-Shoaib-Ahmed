# 🎬 IMDB Movie Review Sentiment Analysis using NLP & Machine Learning

This project titled **"IMDB Movie Review Sentiment Analysis using NLP & Machine Learning"** was developed as part of my AI Engineering work. The goal was to classify IMDB movie reviews as *positive* or *negative* by applying various Natural Language Processing (NLP) techniques and machine learning models. Through this project, I explored the impact of different text representation methods — **TF-IDF**, **Word2Vec embeddings**, and **DistilBERT** — on model performance, using **Logistic Regression** and transformer-based approaches for classification.

---

##  Key Steps

- **Data Preprocessing:**  
  Cleaned and prepared IMDB review text by removing noise, punctuation, stopwords, and performing lemmatization.

- **Feature Representation:**  
  - **TF-IDF:** Captured the relative importance of words based on frequency.  
  - **Word2Vec:** Learned semantic word relationships through vector embeddings after tokenization.  
  - **DistilBERT:** Utilized a pre-trained transformer for deep contextual embeddings.

- **Model Training:**  
  Implemented and compared **Logistic Regression** (for TF-IDF and Word2Vec) and **DistilBERT fine-tuning** using the HuggingFace Transformers library.

- **Evaluation Metrics:**  
  Used **accuracy**, **precision**, **recall**, and **F1-score** to assess model performance.

- **Visualization:**  
  Displayed word clouds for positive and negative reviews, bar charts for feature importance, and accuracy plots for model comparisons.

---

##  What I Learned

- The effect of tokenization quality on Word2Vec embedding accuracy.  
- Why TF-IDF remains competitive for small and balanced datasets.  
- Practical experience fine-tuning **DistilBERT** for text classification.  
- How to interpret and evaluate model results using multiple metrics.

---

##  Technologies Used

- **Python**, **Scikit-learn**, **NLTK**, **Gensim**, **HuggingFace Transformers**, **Matplotlib**, **Seaborn**, **Pandas**, **NumPy**

---

##  Future Work

- Deploy as a web-based interactive sentiment analysis tool.  
- Experiment with **LSTM** and **RoBERTa** for improved context understanding.  

---

##  Author

**Md. Shoaib Ahmed (Afif)**    



