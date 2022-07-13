## Master Thesis: Tilburg University MSc Data Science and Society

# Thesis Title: Novelty In The Purchase Intent Prediction - The Anomaly Detection Methodology

### Author: Milan Fedorik 
#### LinkedIn: 
[linkedin.com/in/milan-fedorik](www.linkedin.com/in/milan-fedorik)

#### Abstract:
The E-commerce market has been growing over the past years. So has the demand for technologies analyzing the user’s behavior. Correct identification of potential buyers or nonbuyers can help the resellers with the aimed commercial and bonuses. However, purchase sessions are rare and have anomalous behavior. This paper takes advantage of the imbalanced dataset and analyzes the strength of anomaly detection models for the purchase intent prediction task using the clickstream data. The sequential data originating from SIGIR eCOM 2021 Data Challenge were trained using a variety of reconstruction and probabilistic models and supplemented by already studied one-class classification and proximity-based models. Gained results confirmed the advantage of this application and proved the preference for deep models in this specific task. Furthermore, the research showed the possibility for early prediction by predicting the intent with trimmed sessions.

#### Data:
- Source:   [SIGIR eCOM 2021 Data Challenge Dataset](https://github.com/coveooss/SIGIR-ecom-data-challenge)

- Type:     Consumer data - clickstream data

- Full dataset:     4 934 699 sessions consisting of 36 079 307 clicks

- Subset used for prediction: 376 558 sessions consisting of 5 798 962 clicks

- Characteristics:  imbalanced data - minority class of purchase sessions 

#### Code source:
- Package PyOD: Zhao, Y., Nasrullah, Z. and Li, Z., 2019. PyOD: [A Python Toolbox for Scalable Outlier Detection](https://www.jmlr.org/papers/volume20/19-011/19-011.pdf). Journal of machine learning research (JMLR), 20(96), pp.1-7.
- Parts of preprocessing code were part of course Analysis of Customer Data (880655-M-6) taught at Tilburg University as part of DSS program by [dr. Giovanni Cassani](https://giovannicassani.github.io/) and dr. Boris Čule.
