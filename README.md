# ES-335-Assignment-3-2024-Fall

## Task 3: MNIST Classification and Analysis [3 marks]

1. Train on the MNIST dataset using an MLP. The original training dataset contains 60,000 images and the test set contains 10,000 images. If you are short on compute, use a stratified subset of a smaller number of images, but the test set remains the same 10,000 images. Compare against Random Forest (RF) and Logistic Regression models. The metrics can be: F1-score, confusion matrix. What do you observe? What all digits are commonly confused?

2. Let us assume your MLP has 30 neurons in the first layer, 20 in the second layer, and then 10 neurons in the output layer (corresponding to 10 classes). On the trained MLP, plot the t-SNE for the output from the layer containing 20 neurons for the 10 digits. Contrast this with the t-SNE for the same layer but for an untrained model. What do you conclude?

3. Now, use the trained MLP to predict on the Fashion-MNIST dataset. What do you observe? How do the embeddings (t-SNE visualization for the second layer) compare for MNIST and Fashion-MNIST images?


---



**Submission Format**: Share a GitHub repo with your training notebooks named *`question\<number\>.ipynb`*.  Include textual answers in the notebook itself. For Question 1, put the link to streamlit app at the top of the notebook.

