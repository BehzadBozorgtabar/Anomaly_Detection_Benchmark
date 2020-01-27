# Anomaly_Detection_Benchmark

--------------------------------------------------------------------------------
This repo contains PyTorch implementations of few methods for unsupervised anomaly detection with the focus on Chest X-Rays.


## Deep-SVDD

Implementation of DeepSVDD baseline *(Ruff, L., Vandermeulen, R., Goernitz, N., Deecke, L., Siddiqui, S.A., Binder, A., Müller, E. & Kloft, M.. (2018). Deep One-Class Classification. Proceedings of the 35th International Conference on Machine Learning, in PMLR 80:4393-4402)*

The training pipeline allows pretraining of autoencoder. After some pretraining epochs, it computes the center coordinate of the hypersphere. The final training is an end-to-end training based on reconstruction loss and clustering loss.


## GAN-GP

Implementation of GAN baseline to learn distribution of normal samples. It's based on the paper : *Y. Tang, Y. Tang, M. Han, J. Xiao and R. M. Summers, "Abnormal Chest X-Ray Identification With Generative Adversarial One-Class Classifier," 2019 IEEE 16th International Symposium on Biomedical Imaging (ISBI 2019), Venice, Italy, 2019, pp. 1358-1361.*


## MemDAE

Baseline to detect anomalies by storing normal samples in a memory bank
Trains an autoencoder which normalizes the features. It stores the corresponding features in a memory bank. The anomaly score is the mean radian between the test feature and the K nearest neighbors stored in the memory bank.


## RLAD

Another baseline to detect anomalies with the help of a memory bank. It models correlation between samples with the help of instance recognition *(Wu, Zhirong & Xiong, Yuanjun & Yu, Stella & Lin, Dahua. (2018). Unsupervised Feature Learning via Non-Parametric Instance-level Discrimination.)* and estimation of local cluster relationship in small neighborhood *(Huang, Jiabo, Qi Dong, Shaogang Gong and Xiatian Zhu. “Unsupervised Deep Learning by Neighbourhood Discovery.” ICML (2019).)*. The training of an autoencoder is added to have a good representation of the normal samples in lower dimensional space.
