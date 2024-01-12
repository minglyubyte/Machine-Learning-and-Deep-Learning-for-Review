1. **ML基础概念类 (ML Basic Concepts)**
   - **Overfitting/Underfitting是指的什么 (What are Overfitting and Underfitting)?**
     - **A:** 
       - **Underfitting**: Occurs when a model is too simplistic, failing to capture the complex patterns in the data. Results in high training and validation errors.
       - **Overfitting**: Happens when a model is too complex and memorizes noise in the training data, leading to poor performance on new, unseen data.
   - **Bias/Variance Trade-off 是指的什么 (What is the Bias/Variance Trade-off)?**
     - **A:** It's the balance between a model's complexity (variance) and its accuracy in capturing the underlying trend (bias). A high-variance model overfits the data, while a high-bias model underfits it.
   - **过拟合一般有哪些预防手段 (What are the general ways to prevent overfitting)?**
     - **A:** Common methods include:
       - Simplifying the model (reducing complexity).
       - Increasing data samples.
       - Using regularization techniques (like L1/L2).
       - Cross-validation.
       - Early stopping during training.
       - Pruning (in decision trees).
   - **Generative和Discrimitive的区别 (Difference between Generative and Discriminative)?**
     - **A:** Generative models (e.g., Naive Bayes) learn the joint probability distribution P(X, Y) and can generate new data instances. Discriminative models (e.g., Logistic Regression) learn the conditional probability distribution P(Y|X) and focus on distinguishing between different classes.
   - **Give a set of ground truths and 2 models, how do you be confident that one model is better than another?**
     - **A:** Compare their performance using appropriate metrics (like accuracy, precision, recall, F1-score, AUC-ROC) on a validation/test set. Consistency in performance across multiple datasets and robustness to variations in data can also indicate a superior model.

1.1 **Reguarlization:**
   - **L1 vs L2, which one is which and difference**
     - **A:** L1 regularization (Lasso) adds the absolute value of coefficients as penalty term to the loss function. It can lead to sparse solutions. L2 regularization (Ridge) adds the squared value of coefficients as penalty. It tends to distribute error among all terms.
   - **Lasso/Ridge的解释 (prior分别是什么）(Explanation of Lasso/Ridge and their priors)**
     - **A:** Lasso regression uses a Laplace prior that leads to sparsity, thus performing feature selection. Ridge regression employs a Gaussian prior, which doesn't result in sparse models but minimizes the impact of less important features.
   - **Lasso/Ridge的推导 (Derivation of Lasso/Ridge)**
     - **A:** N/A
   - **为什么L1比L2稀疏 (Why is L1 sparser than L2)?**
     - **A:** L1 penalty has the geometric property of shrinking coefficients to zero, resulting in feature selection and sparser solutions. L2 penalty, however, tends to distribute penalty among all coefficients, not setting them to zero.
   - **为什么regularization works (Why does regularization work)?**
     - **A:** Regularization works by adding a penalty term to the loss function, which helps to avoid overfitting by keeping the weights small. This prevents the model from becoming too complex and fitting noise in the training data, thus improving generalization to new data.
   - **为什么regularization用L1 L2，而不是L3, L4.. (Why use L1, L2 regularization and not L3, L4, etc.)?**
     - **A:** L1 and L2 regularization are mathematically tractable and have proven effective in practice. Higher-order terms like L3 or L4 are less common as they can introduce additional computational complexity and may not provide significant benefits over L1 and L2.

1.2 **Metric:**
   - **Precision and recall, trade-off**
     - **A:** Precision measures the accuracy of positive predictions, while recall measures the proportion of actual positives identified correctly. The trade-off is that increasing precision often reduces recall and vice versa, as focusing on one can lead to missing out on the other.
   - **Label 不平衡时用什么metric (What metric to use when labels are imbalanced)?**
     - **A:** In cases of label imbalance, metrics like F1-score, precision-recall curves, or AUC-ROC are more informative than accuracy.
   - **分类问题该选用什么metric，and why (Which metric to choose for classification problems, and why)?**
     - **A:** The choice depends on the specific problem and goals. For balanced classes, accuracy might be sufficient. For imbalanced classes, precision, recall, F1-score, or AUC-ROC are better choices.
   - **Confusion matrix**
     - **A:** A confusion matrix is a table used to evaluate the performance of a classification model. It shows the true positives, false positives, true negatives, and false negatives, helping to understand the types of errors made by the model.
   - **AUC的解释 (Explanation of AUC)**
     - **A:** AUC (Area Under the Curve) measures the area under the ROC curve. It represents the likelihood that a randomly chosen positive instance is ranked higher than a randomly chosen negative one by the model.
   - **True positive rate, false positive rate, ROC**
     - **A:** True positive rate (sensitivity) measures the proportion of actual positives correctly identified. False positive rate measures the proportion of actual negatives incorrectly identified as positive. ROC (Receiver Operating Characteristic) curve plots true positive rate against false positive rate at various threshold settings.
   - **Log-loss是什么，什么时候用logloss (What is log-loss, and when is it used)?**
     - **A:** Log-loss, or logistic loss, measures the performance of a classification model where the prediction is a probability between 0 and 1. It is used in binary classification problems to quantify the accuracy of the predictions.

1.3 **Loss与优化 (Loss and Optimization):**
   - **用MSE做loss的Logistic Regression是convex problem吗 (Is Logistic Regression with MSE as loss a convex problem)?**
     - **A:** Logistic regression with MSE as the loss function is not typically a convex problem. Logistic regression usually uses a log-loss function to ensure convexity.
   - **解释并写出MSE的公式, 什么时候用到MSE? (Explain and write out the MSE formula, when is MSE used)?**
     - **A:** Mean Squared Error (MSE) is calculated as the average of the squares of the errors between predicted and actual values. Formula: MSE = (1/n) Σ (actual - predicted)². It's used in regression problems.
   - **Linear Regression最小二乘法和MLE关系 (Relationship between Least Squares in Linear Regression and Maximum Likelihood Estimation)?**
     - **A:** In the context of linear regression, the least squares method can be seen as a special case of maximum likelihood estimation (MLE) under the assumption that the errors are normally distributed.
   - **什么是relative entropy/crossentropy, 以及K-L divergence 他们intuition (What is relative entropy/crossentropy, and K-L divergence, their intuition)?**
     - **A:** Relative entropy, or Kullback-Leibler divergence, measures how one probability distribution diverges from a second, expected distribution. Cross-entropy measures the difference between two probability distributions over the same set of events. They are used to quantify the difference between distributions.
Great, I'll continue answering the remaining questions in the same format:
   - **Logistic Regression的loss是什么 (What is the loss for Logistic Regression)?**
     - **A:** The loss function for logistic regression is typically the log-loss or binary cross-entropy, which measures the difference between the predicted probabilities and the actual class labels.
   - **Logistic Regression的 Loss 推导 (Derivation of Logistic Regression's Loss)**
     - **A:** N/A
   - **SVM的loss是什么 (What is the loss for SVM)?**
     - **A:** SVM uses the hinge loss function, which penalizes misclassified points and points that are too close to the decision boundary.
   - **Multiclass Logistic Regression然后问了一个为什么用cross entropy做cost function (Why use cross entropy as cost function for Multiclass Logistic Regression)?**
     - **A:** Cross entropy is used because it measures the difference between the predicted probability distribution and the actual distribution. It's effective for multi-class classification as it penalizes wrong classifications more heavily.
   - **Decision Tree split node的时候优化目标是啥 (What is the optimization goal when splitting nodes in a Decision Tree)?**
     - **A:** The goal is to maximize the information gain or minimize the impurity (like Gini impurity or entropy) in the child nodes compared to the parent node.

2. **DL基础概念类 (DL Basic Concepts)**
   - **DNN为什么要有bias term, bias term的intuition是什么 (Why does DNN need a bias term, what is the intuition behind the bias term)?**
     - **A:** The bias term in DNN allows the activation function to be shifted, providing the model with more flexibility to fit the data. It helps in fitting the output when input features are zero or have small variations.
   - **什么是Back Propagation (What is Back Propagation)?**
     - **A:** Back Propagation is an algorithm used in training neural networks, where the error is propagated backwards through the network to update the weights, minimizing the difference between actual and predicted outputs.
   - **梯度消失和梯度爆炸是什么，怎么解决 (What are vanishing and exploding gradients, and how to solve them)?**
     - **A:** Vanishing gradients occur when the gradients are too small, slowing down the learning process. Exploding gradients happen when the gradients become too large, causing instability. Solutions include using activation functions like ReLU, gradient clipping, better weight initialization, and LSTM units in RNNs.
   - **神经网络初始化能不能把weights都initialize成0 (Can neural networks be initialized with weights all set to 0)?**
     - **A:** Initializing all weights to 0 leads to symmetry problems where all neurons learn the same features. It's better to initialize weights randomly to break symmetry.
   - **DNN和Logistic Regression的区别 (Difference between DNN and Logistic Regression)?**
     - **A:** DNNs are capable of learning complex, non-linear relationships due to their deep structure and non-linear activation functions, unlike logistic regression which is a linear model.
   - **你为什么觉得DNN的拟合能力比Logistic Regression强 (Why do you think DNN has stronger fitting capability than Logistic Regression)?**
     - **A:** DNNs, with their multiple layers and non-linear activations, can capture complex patterns and interactions in the data, which is not possible with the linear nature of logistic regression.
   - **How to do hyperparameter tuning in DL/ random search, grid search**
     - **A:** Hyperparameter tuning in DL can be done using methods like grid search, where a grid of hyperparameter values is defined and each combination is evaluated, or random search, where random combinations of hyperparameters are evaluated. Advanced methods include Bayesian optimization.
   - **Deep Learning有哪些预防overfitting的办法 (What are the methods to prevent overfitting in Deep Learning)?**
     - **A:** Methods include:
       - Regularization (L1, L2).
       - Dropout.
       - Early stopping.
       - Data augmentation.
       - Using simpler models.
       - Increasing training data.
   - **什么是Dropout，why it works，dropout的流程是什么 (What is Dropout, why it works, and what is the process of Dropout)?**
     - **A:** Dropout is a regularization technique where randomly selected neurons are ignored during training. It works by preventing co-adaptation of neurons, thus reducing overfitting. During training, neurons are dropped out randomly, while during testing, all neurons are used with adjusted weights.
   - **什么是Batch Norm, why it works, BN的流程是什么 (What is Batch Norm, why it works, and what is the process of Batch Norm)?**
     - **A:** Batch Normalization is a technique to normalize the inputs of each layer, making training faster and more stable. It works by reducing internal covariate shift. During training, it normalizes the layer inputs using the mean and variance of the current batch, while during inference, it uses the overall statistics.
   - **Common activation functions (sigmoid, tanh, relu, leaky relu) 是什么以及每个的优缺点 (What are these common activation functions and their pros and cons)?**
     - **A:** 
       - **Sigmoid**: Smooth and bounded between 0 and 1, but suffers from vanishing gradients.
       - **Tanh**: Similar to sigmoid but ranges from -1 to 1. Also suffers from vanishing gradients.
       - **ReLU**: Solves vanishing gradient problem, but neurons can die (i.e., stop activating) if large negative bias accumulates.
       - **Leaky ReLU**: Similar to ReLU but allows a small, non-zero gradient when the unit is not active, preventing dead neurons.
   - **为什么需要non-linear activation functions (Why are non-linear activation functions needed)?**
     - **A:** Non-linear activation functions allow neural networks to learn complex, non-linear mappings from inputs to outputs, which is not possible with linear activations.
   - **Different optimizers (SGD, RMSprop, Momentum, Adagrad, Adam) 的区别 (Differences between these optimizers)?**
     - **A:** 
       - **SGD**: Basic optimizer, updates weights in the opposite direction of the gradient.
       - **RMSprop**: Adapts the learning rate for each weight.
       - **Momentum**: Helps to accelerate SGD by navigating along relevant directions.
       - **Adagrad**: Adapts learning rates based on the frequency of parameters.
       - **Adam**: Combines elements of RMSprop and Momentum.
   - **Batch 和 SGD的优缺点, Batch size的影响 (Advantages and disadvantages of Batch and SGD, impact of Batch size)**
     - **A:** 
       - **Batch Gradient Descent**: Processes the entire dataset in one go. It's stable but can be slow and computationally expensive for large datasets.
       - **Stochastic Gradient Descent (SGD)**: Processes one sample at a time. Faster and can handle large datasets, but the updates are noisy and less stable.
       - **Batch Size**: A larger batch size provides a more accurate estimate of the gradient, but is computationally heavier. A smaller batch size leads to faster but noisier updates.
   - **Learning rate过大过小对于模型的影响 (Impact of too high or too low learning rates on the model)**
     - **A:** A too high learning rate can cause the model to converge too quickly to a suboptimal solution or even diverge. A too low learning rate makes the training process very slow and the model might get stuck in local minima.
   - **Problem of Plateau, saddle point**
     - **A:** Plateaus and saddle points are areas where the gradient is very low or zero, making it hard for gradient-based optimizers to make progress. Techniques like momentum or learning rate schedules can help to overcome these issues.

3. **ML模型类 (ML Model Types)**
   - **3.1 Regression:**
     - **Linear Regression的基础假设是什么 (What are the basic assumptions of Linear Regression)?**
       - **A:** Assumptions include linearity, independence of errors, homoscedasticity (constant variance of errors), and normal distribution of errors.
     - **What will happen when we have correlated variables, how to solve**
       - **A:** Correlated variables can lead to multicollinearity, affecting the model’s interpretability and stability. Solutions include removing correlated predictors, regularization, or using principal component analysis (PCA).
     - **Explain regression coefficient**
       - **A:** Regression coefficients represent the mean change in the response variable for one unit of change in the predictor variable while holding other predictors constant.
     - **What is the relationship between minimizing squared error and maximizing the likelihood**
       - **A:** Minimizing squared error in linear regression is equivalent to maximizing the likelihood under the assumption that errors are normally distributed.
     - **How could you minimize the inter-correlation between variables with Linear Regression?**
       - **A:** Use techniques like PCA for feature extraction or regularization methods that penalize large weights to reduce multicollinearity.
     - **If the relationship between y and x is non-linear, can linear regression solve that**
       - **A:** Linear regression cannot model non-linear relationships effectively. However, transforming variables or using polynomial regression might capture some non-linear aspects.
     - **Why use interaction variables**
       - **A:** Interaction variables capture the effect of two or more variables acting together on the dependent variable, which cannot be captured by these variables independently.

   - **3.2 Clustering and EM:**
     - **K-means clustering (explain the algorithm in detail; whether it will converge, whether to global or local optimums; how to stop)**
       - **A:** K-means clustering algorithm partitions the data into K clusters. It starts with initial centroids, assigns points to the nearest centroid, recalculates centroids, and repeats until convergence (no change in centroids). It generally converges to a local optimum. The process stops when centroids stabilize or after a set number of iterations.
     - **EM算法是什么 (What is the EM algorithm)?**
       - **A:** The Expectation-Maximization (EM) algorithm is used for finding maximum likelihood estimates in models with latent variables. It alternates between expectation (E) step, estimating the missing data, and maximization (M) step, optimizing the parameters.
     - **GMM是什么，和Kmeans的关系 (What is GMM, and its relation to Kmeans)?**
     - **A:** Gaussian Mixture Model (GMM) is a probabilistic model that assumes the data is generated from a mixture of several Gaussian distributions. Unlike K-means which assigns each data point to a single cluster, GMM assigns a probability of belonging to each cluster. GMM is a generalization of K-means with more flexibility due to its probabilistic nature.

   - **3.3 Decision Tree**
     - **How regression/classification Decision Trees split nodes?**
       - **A:** Decision trees split nodes by choosing the feature and threshold that result in the largest information gain or the greatest reduction in impurity (like Gini impurity or entropy).
     - **How to prevent overfitting in Decision Trees?**
       - **A:** To prevent overfitting, techniques like pruning (removing parts of the tree that provide little power), setting a maximum depth for the tree, and requiring a minimum number of samples per leaf node are used.
     - **How to do regularization in Decision Trees?**
       - **A:** Regularization in decision trees can be achieved through pruning, limiting the depth of the tree, reducing the minimum number of samples required at a leaf node, or combining trees in an ensemble method like Random Forest.

   - **3.4 Ensemble Learning**
     - **Difference between bagging and boosting**
       - **A:** Bagging (Bootstrap Aggregating) involves training multiple models (usually of the same type) in parallel on different subsets of the data and then averaging their predictions. Boosting involves training multiple models sequentially, each trying to correct the errors of the previous one.
     - **Gbdt和random forest 区别，pros and cons (Differences between GBDT and Random Forest, pros and cons)**
       - **A:** GBDT (Gradient Boosted Decision Trees) trains trees sequentially, with each tree correcting the errors of the previous one, focusing on difficult cases. Random Forest trains trees in parallel and combines their predictions. GBDT often performs better but is more prone to overfitting and is harder to tune than Random Forest.
     - **Explain GBDT/Random Forest**
       - **A:** GBDT builds trees one at a time, where each new tree helps to correct errors made by previously trained trees. Random Forest builds multiple decision trees and merges them together to get a more accurate and stable prediction.
     - **Will Random Forest help reduce bias or variance/why Random Forest can help reduce variance**
       - **A:** Random Forest helps primarily in reducing variance through its ensemble approach of averaging multiple decision trees, which individually might have high variance. By combining the predictions, the variance gets averaged out, resulting in a more robust model.

   - **3.5 Generative Model**
     - **和Discrimitive模型比起来，Generative 更容易overfitting还是underfitting (Compared to Discriminative models, are Generative models more prone to overfitting or underfitting)?**
       - **A:** Generative models are generally more prone to overfitting compared to discriminative models because they attempt to model the distribution of each class and the data itself, which can be complex.
     - **Naïve Bayes的原理，基础假设是什么 (What is the principle of Naïve Bayes, and what are its basic assumptions)?**
       - **A:** Naïve Bayes is based on Bayes' Theorem and assumes that the predictors are independent of each other given the class. It calculates the probability of each class given a set of inputs and classifies the instance into the class with the highest probability.
     - **LDA/QDA是什么，假设是什么 (What are LDA and QDA, and what are their assumptions)?**
       - **A:** LDA (Linear Discriminant Analysis) and QDA (Quadratic Discriminant Analysis) are techniques used in statistics and machine learning to find a linear or quadratic combination of features that best separate two or more classes. LDA assumes equal covariance matrices for the different classes, while QDA does not.

   - **3.6 Logistic Regression**
     - **Logistic regression和svm的差别 (Difference between logistic regression and SVM)?**
       - **A:** Logistic regression is a probabilistic model that estimates probabilities using a logistic function, ideal for estimating class probabilities. SVM (Support Vector Machine) is a margin-based technique focused on finding the hyperplane that maximally separates the classes. SVM works well for margin optimization but doesn't directly provide probability estimates.

   - **3.7 其他模型 (Other Models)**
     - **Explain SVM, 如何引入非线性 (How to introduce non-linearity in SVM)?**
       - **A:** SVM can model non-linear relationships by using kernel functions (like RBF, polynomial), which transform the input space into a higher-dimensional space where linear separation is possible.
     - **Explain PCA**
       - **A:** PCA (Principal Component Analysis) is a dimensionality reduction technique that transforms data into a new coordinate system, reducing it to its most significant components while retaining most of the variance in the data.
     - **Explain kernel methods, why to use**
       - **A:** Kernel methods allow linear models to learn non-linear boundaries by implicitly mapping input data to high-dimensional spaces. They're used because they can handle complex, non-linear relationships without explicitly transforming the data.
     - **What kernels do you know**
       - **A:** Common kernel functions include the linear kernel, polynomial kernel, radial basis function (RBF) kernel, and sigmoid kernel.
     - **怎么把SVM的output按照概率输出 (How to output probabilities from an SVM)?**
       - **A:** SVMs typically don't output probabilities directly. However, techniques like Platt scaling or logistic regression can be applied to SVM outputs to estimate class probabilities.
     - **Explain KNN**
       - **A:** KNN (k-Nearest Neighbors) is a simple, non-parametric algorithm that classifies a sample based on the majority class of its k nearest neighbors in the feature space.
     - **!所有模型的pros and cons （最高频的一个问题）(Pros and cons of all models - a frequent question)**
       - **A:** N/A

4. **数据处理类 (Data Processing)**
   - **怎么处理imbalanced data (How to handle imbalanced data)?**
     - **A:** Techniques include resampling (oversampling the minority class or undersampling the majority class), using synthetic data generation methods like SMOTE, and using appropriate evaluation metrics that are insensitive to class imbalance.
   - **High-dim classification有什么问题，以及如何处理 (What are the issues with high-dimensional classification, and how to handle them)?**
     - **A:** High-dimensional spaces can lead to overfitting and the curse of dimensionality. Handling methods include dimensionality reduction techniques like PCA or feature selection, regularization, and ensuring sufficient data to support the high dimensionality.

   - **Missing data如何处理 (How to handle missing data)?**
     - **A:** Approaches include imputing missing values using statistical methods (mean, median), prediction models, or advanced techniques like multiple imputation; or removing rows/columns with missing values.

   - **How to do feature selection**
     - **A:** Feature selection can be performed using methods like filter methods (based on statistical tests), wrapper methods (use a predictive model to evaluate feature combinations), and embedded methods (incorporate feature selection as part of the model training process, like Lasso).

   - **How to capture feature interaction**
     - **A:** Feature interactions can be captured by creating new features that represent interactions (like products of features), using models that naturally model interactions (like decision trees), or using techniques like polynomial feature expansion.

5. **Implementation、推导类 (Implementation, Derivation)**
   - **写代码实现两层fully connected网络 (Write code to implement a two-layer fully connected network)**
     - **A:** N/A
   - **手写CNN (Handwrite a CNN)**
     - **A:** N/A
   - **手写KNN (Handwrite a KNN)**
     - **A:** N/A
   - **手写K-means (Handwrite a K-means)**
     - **A:** N/A
   - **手写softmax的backpropagation (Handwrite the backpropagation for softmax)**
     - **A:** N/A
   - **给一个LSTM network的结构要你计算how many parameters (Given an LSTM network structure, calculate how many parameters)**
     - **A:** N/A
   - **Convolution layer的output size怎么算? 写出公式 (How to calculate the output size of a convolution layer? Write out the formula)**
     - **A:** The output size of a convolution layer can be calculated as: \[Output Size = ((Input Size - Filter Size + 2 × Padding) / Stride) + 1\], where Input Size is the size of the input, Filter Size is the size of the filter/kernel, Padding is the number of padding layers added, and Stride is the step size of the convolution operation.

6. **项目经验类 (Project Experience)**
   - **训练好的模型在现实中不work,问你可能的原因 (Why might a well-trained model not work well in reality)?**
     - **A:** Potential reasons include data distribution shift between training and deployment environments, overfitting to the training data, underrepresented scenarios in the training set, and poor model generalization.

   - **Loss趋于Inf或者NaN的可能的原因 (Possible reasons for Loss tending towards Inf or NaN)**
     - **A:** This can happen due to numerical instability, such as very large learning rates leading to exploding gradients, improper scaling of input data, or issues with the data itself like missing values or incorrect labels.

   - **生产和开发时候data发生了一些shift应该如何detect和补救 (How to detect and remedy data shift between production and development)?**
     - **A:** Data shift can be detected by monitoring model performance metrics over time and comparing distributions of key features between training and production data. Remedies include retraining the model with updated data, using techniques like domain adaptation, or applying robust and adaptive models.

   - **Annotation有限的情況下你要怎麼Train model (How to train a model with limited annotations)?**
     - **A:** Techniques include using semi-supervised learning, data augmentation to increase the dataset size, transfer learning from a similar task with more data, and active learning to selectively annotate the most informative samples.

   - **假设有个model要放production了但是发现online one important feature missing不能重新train model 你怎么办 (What if a model is ready for production but an important feature is missing online and the model can't be retrained)?**
     - **A:** Options include developing a fallback mechanism to handle cases without the feature, using imputation techniques to estimate the missing feature, or building a simpler model that doesn’t rely on that specific feature.

7. **NLP/RNN相关 (NLP/RNN Related)**
   - **LSTM的公式是什么 (What is the formula for LSTM)?**
     - **A:** The LSTM (Long Short-Term Memory) formulas involve several gates (forget, input, and output gates) and cell state updates, which are complex and beyond the scope of a brief answer.
   - **Why use RNN/LSTM**
     - **A:** RNNs and LSTMs are used for sequential data where the current output depends on previous computations. LSTMs, in particular, are effective in capturing long-range dependencies, avoiding the vanishing gradient problem common in traditional RNNs.
   - **LSTM比RNN好在哪 (What makes LSTM better than RNN)?**
     - **A:** LSTMs are better at capturing long-term dependencies and avoiding vanishing gradient problems due to their specialized architecture involving gates that regulate the flow of information.
   - **Limitation of RNN**
     - **A:** RNNs are limited by their inability to handle long-range dependencies (due to the vanishing gradient problem), high computational costs for long sequences, and difficulties in parallelization.

   - **How to solve gradient vanishing in RNN**
     - **A:** Solutions include using gated architectures like LSTMs or GRUs, gradient clipping, using better weight initialization methods, and shorter sequences.

   - **What is attention, why attention**
     - **A:** Attention mechanisms in neural networks allow the model to focus on specific parts of the input sequence when producing the output, improving the ability to capture long-range dependencies and relevant information, especially in tasks like machine translation.

   - **Language Model的原理，N-Gram Model**
     - **A:** Language models predict the probability of a sequence of words. The N-gram model does this by using the conditional probability of a word given the previous N-1 words. It assumes that the probability of a word depends only on the previous N-1 words.

   - **What’s CBOW and skip-gram?**
     - **A:** CBOW (Continuous Bag of Words) and skip-gram are two architectures used in the Word2Vec model for generating word embeddings. CBOW predicts a target word based on context words, while skip-gram does the opposite, predicting context words from a target word.

   - **什么是Word2Vec， loss function是什么， negative sampling是什么 (What is Word2Vec, its loss function, and what is negative sampling)?**
     - **A:** Word2Vec is a method to produce word embeddings. Its loss function depends on the architecture (CBOW or skip-gram). Negative sampling is a technique to speed up training by sampling a small number of negative examples (words not in the context) along with the positive example during training.

8. **CNN/CV相关 (CNN/CV Related)**
   - **Maxpooling, conv layer是什么, 为什么做pooling，为什么用conv lay，什么是equivariant to-translation, invariant to translation**
     - **A:** Maxpooling is a downsampling operation in CNNs that reduces the spatial dimensions. Convolutional layers apply filters to extract features. Pooling reduces computational load and achieves spatial invariance. Convolution layers are used for feature extraction and are equivariant to translation (output shifts as input shifts), whereas pooling helps to make the model invariant to translation (tolerant to small shifts and distortions in the input).
   - **1x1 filter**
     - **A:** A 1x1 convolution, often used in CNNs, is a way of channel-wise pooling and dimensionality reduction. It combines the depth-wise information and can be used to alter the number of channels in feature maps.
   - **什么是skip connection (What is a skip connection)?**
     - **A:** Skip connections in neural networks, such as those in ResNet, allow the output of one layer to “skip” some layers and be added to the output of a later layer. This helps to alleviate the vanishing gradient problem and enables training of very deep networks.
