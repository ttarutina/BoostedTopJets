# Inferring correlated distributions: boosted top jets

In this work we consider top-quark related observables, in particular those from hadronically decaying boosted top quark. Two relevant characterizing observables of these objects are  the number of clusters within the jet and the jet mass that we take as the input to the inference procedure. We perform the bayesian inference assuming a mixture model with correlations between these two observables. The correlation is taken into account by introducing transfer matrices for each class obtained using the expectation-maximization(EM) algorithm on the additional dataset.

After calculating the transfer matrices, we set the priors and perform the inference and assess the quality of the inferred distributions, by computing the quantifiers of the goodness of the inference. Finally we get the Maximum a Posteriori (MAP) for the posterior and  calculate Kull-back–Leibler(KL) divergence, which is a measure of the statistical difference between a model PDF and the true PDF.

The minimal system requirements are declared in the file "requirements_minimal.txt"

The data to be used in the code is in the file "data.dat.gz"

Instructions to run the code:

1)in the working directory create folders

     input
     figs
     output
     
1)uncompress the data contained in input

2)run the code using:

     python3 inference.py

The plots and the other output produced as a resut of running the code correspond to the case described in Fig.7 of arxiv:2505.11438.
