import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import json

def app():
    # --- PAGE CONFIG ---
    st.set_page_config(page_title="Project 1", layout="centered")

    # --- HEADER ---
    st.markdown("<h1 style='text-align: center; background-color: black; color: white; padding: 10px;'>Project 1</h1>", unsafe_allow_html=True)

    # --- PROJECT DESCRIPTION ---
    st.markdown("""
    <div style='background-color: #FFD782; padding: 15px; border-radius: 5px; text-align: center;'>
        <h3>Email Spam Detection, Document Classification, Platform Price prediction</h3>
    </div>
    """, unsafe_allow_html=True)

    # --- INPUT & BUTTON ---
    col1, col2 = st.columns([3, 1])
    with col1:
        user_input = st.text_input("", placeholder="Enter dataset name or path...")
    with col2:
        if st.button("Select DS", key="select_ds"):
            # add dropdown for dataset selection
            ds = st.selectbox("Select Dataset", ["emails.csv", "flights.csv", "spam.csv"])
            # st.write(ds)
            # st.success("Dataset selected!")

    # --- ML ALGORITHMS PANEL ---
    st.markdown("### Machine Learning Algorithms")
    cols = st.columns(2)

    with cols[0]:
        st.checkbox("Unsupervised", key="unsupervised")
        st.checkbox("Linear Regression", key="linear_reg")
        st.checkbox("KNN", key="knn")
        st.checkbox("Decision Tree", key="dt")
        st.checkbox("Naive Bayes", key="nb")

    with cols[1]:
        st.radio("Type", ["Classification", "Regression"], key="type")
        st.checkbox("K-mean", key="kmean")

    # --- VALIDATION SECTION ---
    st.markdown("### Validation")
    val_cols = st.columns(2)
    with val_cols[0]:
        st.checkbox("Confusion Matrix", key="confusion_matrix")
    with val_cols[1]:
        st.checkbox("Accuracy", key="accuracy")

    # --- BAR CHART (ACCURACY) ---
    st.markdown("### Accuracy")
    categories = ['Category 1']

    ## -- Height --
    series1 = [4.3]
    series2 = [2.5]
    series3 = [2.0]

    x = np.arange(len(categories))  # label locations
    width = 0.07  # bar width

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width, series1, width, label='Series 1', color='#4B9CD3')
    bars2 = ax.bar(x, series2, width, label='Series 2', color='#FF6F00')
    bars3 = ax.bar(x + width, series3, width, label='Series 3', color='#A9A9A9')

    ax.set_ylabel('Score')
    ax.set_title('Accuracy')
    ax.set_xticks(x, categories)
    ax.legend()

    st.pyplot(fig)

    # Optional: Add some styling for fun
    st.markdown("""
    <style>
        .css-1v3fvcr {
            background-color: #FFD782;
        }
    </style>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    app()