import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Helper function to switch between pages
def switch_page(page_name):
    if page_name == "Introduction":
        introduction()
    elif page_name == "Theory":
        theory()
    elif page_name == "Analysis":
        analysis()
    elif page_name == "Survey":
        survey()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to", ["Introduction", "Theory", "Analysis", "Survey"]
)

# Page: Introduction
def introduction():

    st.title("Divorce Prediction App")
    st.write(
        """
        ### Welcome!

        This tool analyzes data from psychologists to predict divorce likelihood based on questionnaire responses.

        Divorce is a growing challenge worldwide. What if we could pinpoint the warning signs before it’s too late?

        Using the Divorce Predictors Scale (DPS), inspired by Gottman Couples Therapy, we can identify critical factors that lead to divorce.

        **Sections**:

        - **Theory** : Background and context for the prediction model.
        - **Analysis** : Data exploration and insights.
        - **Survey** : Fill out a questionnaire to get predictions.
        """
    )
    # Add an image
    image = Image.open("couple.jpg")  # Ensure you have an image named 'couple.jpg'
    st.image(image, caption="Predicting Marital Outcomes", use_column_width=True)

def theory():
    st.title("Theory")
    st.write(
        """
        ## Background

        Divorce rates are rising globally, affecting families and societies at large. Early prediction and intervention can save relationships and reduce the emotional and financial toll associated with divorce.

        This Divorce Prediction App leverages psychological research and data science to provide insights into marital stability. Based on the Divorce Predictors Scale (DPS), which is grounded in Gottman Couples Therapy, we analyze key behavioral indicators that may predict divorce.

        ### The Four Horsemen of the Apocalypse

        Dr. John Gottman identified four negative communication patterns that are strong predictors of divorce, known as the "Four Horsemen":

        - **Criticism**: Attacking the partner's personality or character.
        - **Contempt**: Showing disrespect or disdain towards the partner.
        - **Defensiveness**: Deflecting blame to avoid responsibility for one's own behavior.
        - **Stonewalling**: Withdrawing from interaction and refusing to engage.

        Understanding and identifying these patterns can help couples address issues before they escalate.

        ## Dataset

        We utilize a dataset consisting of responses to 54 questions related to marital interactions and personal perceptions.

        - **Features**: Responses to 54 questions (Q1 to Q54), reflecting behaviors and attitudes in the relationship.
        - **Target**: Marital status (0 = Not Divorced, 1 = Divorced).

        The dataset was collected from participants who have undergone a detailed questionnaire designed to uncover underlying issues in their relationships.

        ## Methodology

        Our predictive model is built using logistic regression, a statistical method suitable for binary classification problems.

        ### Logistic Regression

        Logistic regression estimates the probability that a given input point belongs to a certain class. The model uses the logistic function (sigmoid function) to map predicted values to probabilities.

        The logistic function is defined as:

        """
    )

    # LaTeX formula
    st.latex(
        r"P(y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1X_1 + \ldots + \beta_nX_n)}}"
    )

    st.write("where:")
    st.latex(r"""
        \begin{align*}
        P(y=1|X)  \text{ is the probability of the instance belonging to the 'Divorced' class.}\\
        \beta_0 \text{ is the intercept term.}\\
        \beta_i  \text{ are the coefficients for each feature }  X_i \\
        \end{align*}
    """)
    st.write(
        """
        The coefficients represent the influence of each feature on the probability of divorce.

        ### Data Preprocessing

        - **Scaling**: Features are scaled using StandardScaler to normalize the data. This ensures that each feature contributes equally to the model training.
        - **Train-Test Split**: The data is split into training and testing sets to evaluate the model's performance on unseen data.

        ### Model Selection and Validation

        In addition to logistic regression, we experimented with other models:

        - **Random Forest Classifier**
        - **Neural Network (MLPClassifier)**

        Models were evaluated based on accuracy, precision, recall, and F1-score. Cross-validation techniques were used to ensure the model generalizes well to new data.

        Currently, all models are giving similar accuracy scores due to the limited size and nature of the dataset.

        ## Feature Importance

        Analyzing the coefficients of the logistic regression model helps identify which features (questions) have the most significant impact on the prediction. This can provide valuable insights into specific areas of the relationship that may need attention.

        ## Future Work

        While our current model provides valuable insights, there are areas for improvement:

        - **Expanding the Dataset**: By saving the survey results from users (with consent and ensuring anonymity), we can collect more data. A larger dataset would enable us to train more robust models and potentially reveal new patterns.

        - **Assessing Model Differences**: With more data, we can better assess the performance differences between models. Currently, the models yield similar accuracy, but additional data may highlight strengths and weaknesses of each approach.

        - **Feature Engineering**: Exploring new features or combining existing ones may improve model performance. This could involve creating composite scores for the Four Horsemen or identifying interaction effects between questions.

        - **User Feedback Integration**: Incorporating feedback from users about the accuracy and usefulness of the predictions can help refine the model and the questionnaire.

        ### References

        Yöntem, M., Adem, K., İlhan, T., & Kılıçarslan, S. (2019). Divorce Prediction Using Correlation Based Feature Selection and Artificial Neural Networks. *Nevşehir Hacı Bektaş Veli University SBE Dergisi*, 9(1), 259-273.
        """
    )

# Page: Analysis
def analysis():
    st.title("Analysis")

    # Load dataset
    @st.cache_resource
    def load_data():
        return pd.read_csv("divorce_data.csv", delimiter=";")

    data = load_data()
    st.write("### Dataset Overview")
    st.dataframe(data.head())

    # Feature and target split
    X = data.drop("Divorce", axis=1)
    y = data["Divorce"]

    # Preprocess data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=1
    )

    # Train logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    st.write("### Model Performance")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    st.write("""
    Precision: Out of all predicted cases for a class, how many were correct.\\
    Recall: Out of all actual cases for a class, how many were correctly predicted.\\
    F1-Score: A balance between precision and recall.\\
    Support: The number of actual cases in the dataset for each class.\\
    Accuracy: The percentage of all predictions that were correct.\\
    Macro Average: The average performance across all classes, treating each class equally.\\
    Weighted Average: The average performance across all classes, weighted by the number of actual cases in each class.
    """)
             
    st.write("### Confusion Matrix")
    conf_matrix = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)


    # Correlation Heatmap
    st.write("### Correlation Heatmap")
    corr_matrix = data.corr()
    fig3, ax3 = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)


    # Pairplot
    st.write("### Pairplot of Selected Features (Please wait to load)")
    selected_features = X.columns[:5]  # Select first 5 features for simplicity
    pairplot = sns.pairplot(data[selected_features], diag_kind="kde")
    fig = pairplot.fig  # Extract the figure object
    st.pyplot(fig)

    # Model Comparison
    st.write("### Model Comparison")
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "Neural Network": MLPClassifier(),
    }
    results = {}
    for model_name, model_instance in models.items():
        model_instance.fit(X_train, y_train)
        y_pred = model_instance.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[model_name] = acc
    results_df = pd.DataFrame.from_dict(results, orient="index", columns=["Accuracy"])
    st.table(results_df)

def survey():
    st.title("Survey")
    st.write("Fill out the survey below to predict marital status.")
    
    # Read the reference.tsv to get the questions
    ref = pd.read_csv("reference.tsv", delimiter="|", index_col=0)
    ref = ref['description']
    
    # Store user responses
    user_responses = {}
    
    # Loop through the questions and collect responses
    for i, question in ref.items():
        user_responses[i] = st.slider(question, 0, 4, 2)
    
    # Convert responses to a DataFrame
    user_data = pd.DataFrame(user_responses, index=[0])
    user_data.columns = [f"Q{i}" for i in range(1, len(user_data.columns) + 1)]

    # Display user input
    st.write("### Your Input")
    st.dataframe(user_data)
    
    # Load pre-trained model
    @st.cache_resource
    def load_model():
        """
        Loads the pre-trained model and scaler for prediction.
        Caches the model and scaler to avoid reloading during each run.
        """
        # Load the dataset (needed to train and return the scaler and model)
        data = pd.read_csv("divorce_data.csv", delimiter=";")
        feature_columns = [f'Q{i}' for i in range(1, 55)] 
        X = data[feature_columns]
        y = data["Divorce"]

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train the Logistic Regression model
        model = LogisticRegression()
        model.fit(X_scaled, y)

        return model, scaler

    # Cache issue debugging logs
    st.write("Loading cached model...")
    model, scaler = load_model()
    st.write("Model and scaler loaded.")

    # Predict
    user_data_scaled = scaler.transform(user_data)
    prediction = model.predict(user_data_scaled)
    prediction_proba = model.predict_proba(user_data_scaled)

    # Display Prediction
    st.write("### Prediction Result")
    result = "Divorced" if prediction[0] == 1 else "Not Divorced"
    st.write(f"Predicted marital status: **{result}**")
    st.write(f"Probability of Divorce: {prediction_proba[0][1]*100:.2f}%")
    st.write(f"Probability of Not Divorce: {prediction_proba[0][0]*100:.2f}%")



    # Provide advice based on prediction
    if prediction[0] == 1:
        st.warning(
            "According to the survey, there is a high likelihood of divorce. Consider seeking professional counseling."
        )
    else:
        st.success(
            "According to the survey, the likelihood of divorce is low. Keep nurturing your relationship!"
        )

# Call the function based on the selected page
switch_page(page)
