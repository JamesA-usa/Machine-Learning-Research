# Machine Learning for Cybersecurity Log Analysis

## Overview
This project applies machine learning and natural language processing (NLP) to detect anomalous behavior in large-scale cybersecurity logs. The goal is to determine whether system-generated logs should be prioritized by threat hunters compared to user-generated activity.

Using Azure Log Analytics data, this project processes ~90 days of telemetry (~5 million records) and applies a two-stage machine learning pipeline:
- **BERT (NLP embeddings)** for contextual understanding of command-line activity  
- **Isolation Forest** for anomaly detection  

The output is a ranked dataset of anomalous events to support threat hunting and security investigations.

---

## Research Question
**Should cybersecurity threat hunters prioritize monitoring system-generated logs?**

---

## Key Results
- System-generated logs accounted for **11.7% of anomalies**
- User/root-generated logs accounted for **88.3% of anomalies**
- **0% of system-generated anomalies were malicious** (based on manual review)
- Only **36.7% of anomalies were truly malicious**, indicating need for model tuning

### Conclusion
Threat hunters should prioritize **user and root activity**, as system-generated logs showed minimal malicious behavior and lower anomaly significance.

---

## Architecture

### Data Pipeline
1. **Data Collection**
   - Source: Azure Log Analytics (DeviceProcessEvents table)
   - Time Range: 90 days
   - Method: API extraction in 3-hour increments with throttling controls

2. **Data Preparation**
   - Reduced dataset from **74 → 12 columns**
   - Normalized user accounts into:
     - `root`
     - `system`
     - `user`
   - Feature engineering on command-line and process fields

3. **Modeling**
   - **BERT (Sentence Transformers)** → Text embeddings
   - **Isolation Forest** → Anomaly scoring
   - Output: Top **1,000 anomalous logs**

4. **Evaluation**
   - Manual review of anomalies
   - Statistical validation using **Welch’s t-test**
   - Visualization via Matplotlib

---

## Machine Learning Approach

### BERT (NLP)
- Converts log text into contextual embeddings
- Captures relationships between commands and arguments
- Enables semantic understanding of activity

### Isolation Forest
- Unsupervised anomaly detection
- Identifies rare and unusual behaviors
- Outputs anomaly scores:
  - Negative = more anomalous
  - Near zero = borderline
  - Positive = normal

---

## Tools & Technologies

- **Languages:** Python, SQL, KQL  
- **Libraries:** Pandas, Scikit-learn, PyTorch, SentenceTransformers, Matplotlib  
- **Cloud:** Azure Log Analytics  
- **IDE:** PyCharm  
- **Hardware:** NVIDIA GPU (for accelerated processing)

---

## Performance Considerations

| Component            | Impact |
|---------------------|--------|
| GPU (PyTorch)       | Reduced processing from ~1 week → ~1 day |
| API Throttling      | Required chunking + delays |
| Dimensionality Reduction | Enabled scalable processing |

---

## Statistical Analysis

Welch’s t-tests confirmed statistically significant differences in anomaly scores:

- Root vs User: **p < 0.001**
- Root vs System: **p < 0.001**
- User vs System: **p < 0.001**

This validates that anomaly differences across account types are meaningful.

---

## Visualizations

- Bar Chart → Distribution of anomalies by account type  
- Pie Chart → Percentage breakdown  
- Histogram → Anomaly score distribution  

---

## Key Findings

- High anomaly score ≠ malicious activity  
- User/root accounts generate most meaningful anomalies  
- System logs introduce noise in detection pipelines  

---

## Recommendations

1. **Filter system-generated logs** to improve detection precision  
2. **Upgrade embedding model** (e.g., `all-mpnet-base-v2`)  
3. **Increase training dataset size** (300K–1M samples)  
4. **Tune Isolation Forest parameters**  
5. **Incorporate human-in-the-loop validation**

---

## Real-World Applications

- Threat hunting (SIEM/SOC environments)  
- Fraud detection (pattern anomaly detection in financial/claims data)  
- Insider threat detection  
- Data exfiltration monitoring  

---

## Limitations

- Requires GPU for optimal performance  
- Dependent on Azure environment  
- Loss of granularity due to feature reduction  
- High false positive rate without tuning  

---

## Future Work

- Deep learning anomaly detection (Autoencoders, LSTMs)  
- Real-time streaming detection  
- Integration with SOAR platforms  
- Enhanced labeling for supervised learning  

---

## References

- Kent & Souppaya (2006) – NIST Log Management  
- Chandola et al. (2009) – Anomaly Detection Survey  
- Devlin et al. (2019) – BERT  
- Liu et al. (2008) – Isolation Forest  
- Zhang et al. (2021) – BERT for Cybersecurity  

---

## Repository

GitHub: https://github.com/JamesA-usa/Machine-Learning-Research

---

## Author

**Andrew J. Ahring**  
M.S. Geographic Information Systems  
B.S. Data Analytics (WGU – Expected 2026)

---
