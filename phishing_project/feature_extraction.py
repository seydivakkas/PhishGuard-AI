import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

try:
    stop_words = set(stopwords.words("english"))
except:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))


df = pd.read_csv("data/final_dataset.csv")
print(f"Yüklendi: {len(df)} satır")


#Stilometri
def extract_stylometric(text):
    text = str(text)
    
    words     = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if s.strip()]

    num_chars       = len(text)
    num_words       = len(words)
    num_sentences   = len(sentences)
    num_upper_chars = sum(1 for c in text if c.isupper())
    num_special     = sum(1 for c in text if not c.isalnum() and not c.isspace())
    avg_word_size   = sum(len(w) for w in words) / num_words if num_words > 0 else 0
    unique_words    = len(set(words))

    words_to_chars          = num_words / num_chars if num_chars > 0 else 0
    special_to_chars        = num_special / num_chars if num_chars > 0 else 0
    unique_to_words         = unique_words / num_words if num_words > 0 else 0
    exclamation_to_chars    = text.count("!") / num_chars if num_chars > 0 else 0
    question_to_chars       = text.count("?") / num_chars if num_chars > 0 else 0

    words_per_sent = [len(s.split()) for s in sentences]
    avg_words_in_sent = sum(words_per_sent) / len(words_per_sent) if words_per_sent else 0
    max_words_in_sent = max(words_per_sent) if words_per_sent else 0

    return {
        "num_chars":                num_chars,
        "num_words":                num_words,
        "num_sentences":            num_sentences,
        "num_upper_chars":          num_upper_chars,
        "num_special_chars":        num_special,
        "avg_word_size":            avg_word_size,
        "words_to_chars":           words_to_chars,
        "special_chars_to_chars":   special_to_chars,
        "unique_words_to_word":     unique_to_words,
        "exclamationmark_to_chars": exclamation_to_chars,
        "questionmark_to_chars":    question_to_chars,
        "avg_words_in_sentence":    avg_words_in_sent,
        "max_words_in_sentence":    max_words_in_sent,
    }

print("Stilometrik özellikler çıkarılıyor...")
features = df["body"].apply(extract_stylometric)
df_features = pd.DataFrame(features.tolist())

#Metin temizleme
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)     
    text = re.sub(r'\S+@\S+', '', text)     
    text = re.sub(r'[^a-z\s]', '', text) 
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

print("Metin temizleniyor...")
df["body_clean"] = df["body"].apply(clean_text)


df_final = pd.concat([
    df[["body", "body_clean", "label", "source"]],
    df_features
], axis=1)

df_final.to_csv("data/featured_dataset.csv", index=False)
print(f"\nKaydedildi: data/featured_dataset.csv")
print(f"Kolonlar: {df_final.columns.tolist()}")
print(f"Shape: {df_final.shape}")