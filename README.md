# AAECG: Anomaly detection on ECG signal using Adversarial AutoEncoder - Deep Generative Model

Can we teach a machine the normal behaviour of the heart and then have it use this knowledge to assess whether something is wrong? 

AAECG framework faces the problem of recognition of abnormal heartbeats inside the ECG signal by using modern data-driven techniques. Since the abnormal patterns are plentiful, rare and challenging to collect in a balanced dataset while the normal ones are pretty common, the framework tackles this problem in a/an semi-supervised/unsupervised manner. The model will learn the normal heartbeats variability, and it will use the knowledge acquired to infer the abnormality of new data. 

AAECG is a Deep Generative Model derived by an Adversarial Autoencoder(AAE) which captures the Sinus heartbeat distribution throughout a set of latent variables and additional patient gender information. The model is intended to monitor 24/7 patients in intensive care by alarming doctors only when abnormal heartbeats are detected. Tested on the MIT-BIH arrhythmia database, It reached 0.95 ROC-AUC and 0.92 PR-AUC outperforming the baselines and competing with the state-of-art.

Furthermore, the model shows to understand the sex differences between heartbeats, opening the possibility to study the effects of some conditions, drugs or other particular details on the normal heartbeat wave-form, opening a path towards more patient-specific diagnosis.

## The framework

![AAECG](images/AAECG.png)
