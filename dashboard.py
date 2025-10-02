import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

st.title("📊 Machine Learning Models Performance Dashboard")
# Not using anymore
#Download zip folder

# st.download_button(
#     "Download Dataset '.zip'",
#     "https://github.com/your-username/your-repo/archive/refs/heads/main.zip",
#     "main.zip",
#     "application/zip",
# )

# قراءة ملفات الـ accuracy
with open("jsons/results_spam.json", "r") as f:
    spam_results = json.load(f)

with open("jsons/results.json", "r") as f:
    price_results = json.load(f)

with open("jsons/results_DOC.json", "r") as f:
    doc_results = json.load(f)

# قراءة ملفات الـ confusion matrix
with open("jsons/confs_spam.json", "r") as f:
    spam_conf = json.load(f)

with open("jsons/results_DOC_conf.json", "r") as f:
    doc_conf = json.load(f)

# dict للـ accuracies
all_results = {
    "Spam Detection": spam_results["Spam"],
    "Price Prediction": price_results["Price Prediction"],
    "Document Classification": doc_results["DOC_Class"],
}

# dict للـ confusion matrices
all_conf = {
    "Spam Detection": spam_conf["Spam"],
    "Document Classification": doc_conf["DOC_Class"],
}

# تحويل النتائج ل DataFrame
records = []
for proj, res in all_results.items():
    for alg, acc in res.items():
        records.append({"Algorithm": alg, "Project": proj, "Test Accuracy": acc})
df = pd.DataFrame(records)

# اختيار المشروع
project = st.selectbox("Select Project", list(all_results.keys()))

# اختيار الخوارزميات
algorithms = st.multiselect(
    "Select Algorithms",
    options=df[df["Project"] == project]["Algorithm"].unique()
)

# Radio Button للاختيار بين Accuracy أو Confusion Matrix
view_type = st.radio(
    "Choose View Type",
    ["Accuracy", "Confusion Matrix"]
)

# لو Accuracy
if view_type == "Accuracy":
    filtered_df = df[(df["Project"] == project) & (df["Algorithm"].isin(algorithms))]
    st.subheader(f"Results {project}")
    st.dataframe(filtered_df)

    if not filtered_df.empty:
        st.subheader("📊 Selected algorithms for Bar chart")
        st.bar_chart(filtered_df.set_index("Algorithm")["Test Accuracy"])
    else:
        st.info("Choose at least one algorithm and one project to display the chart.")

# لو Confusion Matrix
elif view_type == "Confusion Matrix":
    if project in all_conf:
        for alg in algorithms:
            if alg in all_conf[project]:
                cm = all_conf[project][alg]

                st.subheader(f"Confusion Matrix - {project} ({alg})")
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)
    else:
        st.warning("Confusion matrix not available for this project.")