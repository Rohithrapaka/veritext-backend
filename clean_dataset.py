import pandas as pd

df = pd.read_csv("questions.csv", encoding="latin-1", on_bad_lines="skip")

texts = []

for q1, q2 in zip(df["question1"], df["question2"]):
    if isinstance(q1, str):
        texts.append(q1.strip())

    if isinstance(q2, str):
        texts.append(q2.strip())

clean_df = pd.DataFrame({
    "id": range(len(texts)),
    "text": texts
})

# limit dataset for project
clean_df = clean_df.head(20000)

clean_df.to_csv("dataset.csv", index=False)

print("Dataset cleaned successfully")