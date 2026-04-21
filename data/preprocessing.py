import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split

# ---------- Arabic Normalization ----------

def normalize_arabic(text):

    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ة", "ه", text)

    return text


# ---------- General Cleaning ----------

def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^\w\s]', '', text)

    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# ---------- Load Dataset ----------

def load_data(file_path):

    data = []

    with open(file_path, encoding="utf-8") as file:

        for line in file:

            parts = line.strip().split("\t")

            if len(parts) == 2:

                english = clean_text(parts[0])
                arabic = normalize_arabic(
                    clean_text(parts[1])
                )
                data.append([arabic, english])

    if not data:
        raise ValueError(f"No valid bilingual lines found in {file_path}")

    return pd.DataFrame(
        data,
        columns=["arabic", "english"]
    )


# ---------- Split Dataset ----------

def split_data(df):

    n = len(df)

    if n == 0:
        raise ValueError("Dataset must contain at least one sample.")

    if n == 1:
        return df, df.iloc[:0], df.iloc[:0]

    if n == 2:
        train = df.iloc[:1]
        val = df.iloc[1:2]
        test = df.iloc[:0]
        return train, val, test

    train, temp = train_test_split(
        df,
        test_size=0.2,
        random_state=42
    )

    val, test = train_test_split(
        temp,
        test_size=0.5,
        random_state=42
    )

    return train, val, test


# ---------- Save Files ----------

def save_data(train, val, test):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(script_dir, "processed")

    os.makedirs(
        processed_dir,
        exist_ok=True
    )

    train.to_csv(
        os.path.join(processed_dir, "train.csv"),
        index=False
    )

    val.to_csv(
        os.path.join(processed_dir, "val.csv"),
        index=False
    )

    test.to_csv(
        os.path.join(processed_dir, "test.csv"),
        index=False
    )


# ---------- Main ----------

def main():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "raw", "ara_.txt")

    df = load_data(file_path)

    train, val, test = split_data(df)

    save_data(train, val, test)

    print("Dataset processed successfully!")
    print("Train size:", len(train))
    print("Validation size:", len(val))
    print("Test size:", len(test))


if __name__ == "__main__":

    main()