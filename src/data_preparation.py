import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Definovanie cesty k priečinkom s layoutmi
data_folder = "data"
layouts = ["Layout1", "Layout2", "Layout3", "Layout4"]

# Načítaj všetky CSV súbory zo všetkých layoutov
all_files = []
for layout in layouts:
    layout_path = os.path.join(data_folder, layout)
    for file in os.listdir(layout_path):
        if file.endswith(".csv"):
            all_files.append(os.path.join(layout_path, file))

# Náhodné rozdelenie na tréningovú a testovaciu množinu (80% tréning, 20% testovanie)
train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)

# Funkcia na načítanie a zlúčenie CSV súborov
def load_and_merge(files):
    df_list = [pd.read_csv(file) for file in files]
    return pd.concat(df_list, ignore_index=True)

# Načítanie a zlúčenie dát
train_df = load_and_merge(train_files)
test_df = load_and_merge(test_files)

# Uloženie do nových CSV súborov
train_df.to_csv("data/train_data.csv", index=False)
test_df.to_csv("data/test_data.csv", index=False)

print(f"Tréningových súborov: {len(train_files)}")
print(f"Testovacích súborov: {len(test_files)}")
