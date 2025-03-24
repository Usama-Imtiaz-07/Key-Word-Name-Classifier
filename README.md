# **Name Language Classifier Using RNN**

## **Project Overview**
This project is a **Recurrent Neural Network (RNN)-based language classifier** that predicts the language of origin for a given name. The model is trained on a dataset of names from **18 different languages**, including Arabic, Chinese, English, French, and more. The project involves data preprocessing, model creation, training, and evaluation, along with visualization of results and user interaction for predictions.

---

## **Features**
1. **Data Preparation**:
   - Loads and preprocesses name datasets from text files.
   - Converts Unicode names to ASCII for consistency.
   - Organizes names by language into a dictionary for easy access.

2. **Name-to-Tensor Conversion**:
   - Converts names into one-hot encoded tensors for input into the RNN.
   - Handles variable-length names by dynamically creating tensors.

3. **RNN Model**:
   - Implements a custom RNN using PyTorch for sequence classification.
   - Uses a hidden state to capture sequential dependencies in names.

4. **Training**:
   - Trains the RNN on randomly sampled names from the dataset.
   - Uses Negative Log Likelihood Loss (NLLLoss) for optimization.
   - Tracks and visualizes training loss over iterations.

5. **Evaluation**:
   - Evaluates the model using a confusion matrix to assess classification accuracy.
   - Visualizes the confusion matrix to show performance across languages.

6. **User Interaction**:
   - Allows users to input names and receive predictions for the top 3 most likely languages.

---

## **How It Works**
1. **Data Retrieval**:
   - Names are loaded from text files, with each file corresponding to a specific language.
   - Names are normalized and converted to ASCII for consistency.

2. **Model Architecture**:
   - The RNN consists of an input layer, a hidden layer, and an output layer.
   - The hidden state is updated at each step of the sequence (each letter in the name).

3. **Training Process**:
   - The model is trained using stochastic gradient descent (SGD) with a fixed learning rate.
   - Training examples are randomly sampled from the dataset to ensure diversity.

4. **Evaluation**:
   - The model's performance is evaluated using a confusion matrix, which shows how often the model correctly predicts the language of origin for a given name.

5. **User Interaction**:
   - Users can input names, and the model predicts the most likely languages of origin along with confidence scores.

---

## **Technologies Used**
- **Python Libraries**: PyTorch, NumPy, Matplotlib, Glob, OS.
- **Data Source**: Text files containing names from 18 different languages.
- **Visualization**: Matplotlib for plotting training loss and confusion matrix.

---

## **How to Run the Project**
1. **Install Dependencies**:
   ```bash
   pip install torch numpy matplotlib
   ```

2. **Run the Script**:
   - Execute the Python script containing the code.

3. **Interact with the Model**:
   - After training, input names to see the model's predictions.

---

## **Future Enhancements**
- **Improve Model Performance**:
  - Experiment with more advanced architectures like LSTM or GRU.
  - Use pre-trained embeddings for better feature representation.

- **Expand Dataset**:
  - Include names from more languages and cultures.
  - Add more examples per language to improve generalization.

- **User Interface**:
  - Develop a web or desktop interface for easier interaction with the model.

- **Deployment**:
  - Deploy the model as a REST API or a Streamlit app for broader accessibility.

---

## **Contributors**
- [Your Name]

---

This project demonstrates the power of RNNs in sequence classification tasks and provides a foundation for building more advanced language classification systems.
