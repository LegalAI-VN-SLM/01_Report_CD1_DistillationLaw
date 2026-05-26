**Chapter 1 — Introduction**  
1.1 Reason for choosing the topic  
1.2 Research objectives  
- 1.2.1 General objective  
- 1.2.2 Specific objectives  
1.3 Research subjects and scope  
- 1.3.1 Research subjects  
- 1.3.2 Research scope  
1.4 Contributions of the study  
1.5 Report structure  

**Chapter 2 — Background and Literature Review**  
2.1 Recommender Systems  
- 2.1.1 Collaborative filtering  
- 2.1.2 Content-based recommendation  
- 2.1.3 Hybrid recommendation systems  

2.2 Graph-based recommendation systems and LightGCN  
- 2.2.1 User–item graph  
- 2.2.2 LightGCN propagation  
- 2.2.3 Brief comparison with other graph-based approaches  

2.3 Multimodal features in fashion recommendation  
- 2.3.1 Visual features (CLIP, MobileNetV2)  
- 2.3.2 Text features (TF-IDF, BERT)  
- 2.3.3 Multimodal fusion  

2.4 Evaluation metrics  
- 2.4.1 Precision at K  
- 2.4.2 Recall at K  
- 2.4.3 Normalized Discounted Cumulative Gain  
- 2.4.4 Mean Average Precision  
- 2.4.5 Mean Reciprocal Rank  
- 2.4.6 Hit Ratio  

**Chapter 3 — Proposed Methods and Models**  
3.1 Overview of the proposed pipeline  
- 3.1.1 Raw data to sampling  
- 3.1.2 N-core filtering  
- 3.1.3 Temporal split for train and test  

3.2 Models used in this study  
- 3.2.1 LightGCN  
- 3.2.2 CombiGCN  
- 3.2.3 BM3  
- 3.2.4 FREEDOM  

3.3 Multimodal integration design (descriptive content only, no sub-sections)  

**Chapter 4 — Experiments**  
4.1 Dataset  
- 4.1.1 Data description and statistics  
- 4.1.2 Preprocessing and train/test creation  

4.2 Experimental setup  
- 4.2.1 Configuration space  
- 4.2.2 Key hyperparameters  
- 4.2.3 Choice of K values and rationale  

4.3 Results and analysis  
- 4.3.1 Encoder comparison  
- 4.3.2 Fusion strategy comparison  
- 4.3.3 Model comparison  

**Chapter 5 — Conclusion and Future Work**  
5.1 Conclusion  
5.2 Limitations  
5.3 Future work  
