import pandas as pd
import hashlib

TARGET_SIZE = 18650


ceas = pd.read_csv("data/CEAS_08.csv")
enron = pd.read_csv("data/Enron.csv")
ling = pd.read_csv("data/Ling.csv")
nazario = pd.read_csv("data/Nazario.csv")
nigerian = pd.read_csv("data/Nigerian_Fraud.csv")
spam = pd.read_csv("data/SpamAssasin.csv")

#Column kontrolü
datasets = {
    "CEAS": ceas, "Enron": enron, "Ling": ling,
    "Nazario": nazario, "Nigerian": nigerian, "Spam": spam
}

print("\n--- KOLON KONTROLÜ ---")
for name, d in datasets.items():
    has_body  = "body"  in d.columns
    has_label = "label" in d.columns
    print(f"{name:10}: shape={d.shape} | body={'✅' if has_body else '❌'} | label={'✅' if has_label else '❌'}")
    if not has_body or not has_label:
        raise ValueError(f"{name} dataseti 'body' veya 'label' kolonu eksik!")

# Column temizliği ve Source
ceas["source"]     = "CEAS"
enron["source"]    = "Enron"
ling["source"]     = "Ling"
nazario["source"]  = "Nazario"
nigerian["source"] = "Nigerian"
spam["source"]     = "Spam"

print("\n--- GEREKSİZ KOLONLAR ---")
for name, d in datasets.items():
    extra = [c for c in d.columns if c not in ["body", "label", "source"]]
    if extra:
        print(f"{name}: düşürüldü → {extra}")

ceas     = ceas[["body", "label", "source"]]
enron    = enron[["body", "label", "source"]]
ling     = ling[["body", "label", "source"]]
nazario  = nazario[["body", "label", "source"]]
nigerian = nigerian[["body", "label", "source"]]
spam     = spam[["body", "label", "source"]]


df = pd.concat([ceas, enron, ling, nazario, nigerian, spam], ignore_index=True)

#Dublicate remove md5
def make_hash(text):
    return hashlib.md5(str(text).encode()).hexdigest()

df["hash"] = df["body"].astype(str).apply(make_hash)
df = df.drop_duplicates(subset="hash")



def balanced_sample(df, target_size):
    
    print("\nKaynak × Label dağılımı (ham veri):")
    print(df.groupby(["source", "label"]).size().unstack(fill_value=0))
    
    sources = df["source"].unique()
    per_source = target_size // len(sources)  # ~3108

    parts = []

    for src in sources:
        df_src = df[df["source"] == src]
        df0 = df_src[df_src["label"] == 0]
        df1 = df_src[df_src["label"] == 1]

        half = per_source // 2

        s0 = df0.sample(n=min(len(df0), half), random_state=42)
        s1 = df1.sample(n=min(len(df1), half), random_state=42)

        taken = pd.concat([s0, s1])
        parts.append(taken)
        
        # her kaynaktan ne kadar alındı
        print(f"  {src}: {len(s0)} ham + {len(s1)} phishing = {len(taken)}")

    df_bal = pd.concat(parts)

    
    current_0 = len(df_bal[df_bal["label"] == 0])
    current_1 = len(df_bal[df_bal["label"] == 1])
    target_each = target_size // 2  # 9325

    remain = df.drop(df_bal.index)

    # label=0 eksikse tamamla
    if current_0 < target_each:
        need_0 = target_each - current_0
        pool_0 = remain[remain["label"] == 0]
        extra0 = pool_0.sample(n=min(len(pool_0), need_0), random_state=42)
        df_bal = pd.concat([df_bal, extra0])
        print(f"\n  label=0 eksikti, {len(extra0)} ek örnek eklendi")

    # label=1 eksikse tamamla
    if current_1 < target_each:
        need_1 = target_each - current_1
        pool_1 = remain[remain["label"] == 1]
        extra1 = pool_1.sample(n=min(len(pool_1), need_1), random_state=42)
        df_bal = pd.concat([df_bal, extra1])
        print(f"  label=1 eksikti, {len(extra1)} ek örnek eklendi")

    df_bal_0 = df_bal[df_bal["label"] == 0].sample(n=target_size//2, random_state=42)
    df_bal_1 = df_bal[df_bal["label"] == 1].sample(n=target_size//2, random_state=42)

    df_bal = pd.concat([df_bal_0, df_bal_1]).sample(frac=1, random_state=42).reset_index(drop=True)
   
    return df_bal


df = balanced_sample(df, TARGET_SIZE)

#Kontrol
print("\nCLASS DISTRIBUTION:")
print(df["label"].value_counts())

print("\nSOURCE DISTRIBUTION:")
print(df["source"].value_counts())

print("\nFINAL SIZE:", len(df))



df.to_csv("data/final_dataset.csv", index=False)

print("\nSaved: data/final_dataset.csv") 