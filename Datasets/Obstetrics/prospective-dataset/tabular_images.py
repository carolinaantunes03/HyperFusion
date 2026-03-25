import os
import pandas as pd
import re

# Define paths
base_dir = os.path.dirname(os.path.abspath("PT_external_dataset_processed.csv"))
csv_path = os.path.join(base_dir, "PT_external_dataset_processed.csv")
images_dir = os.path.join(base_dir, "prospective_images")

# Read dataset
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

# Add new columns
df["abdomen_image"] = None
df["femur_image"] = None
df["head_image"] = None

# Helper function to find one image per plane
def find_images_for_processo(processo_folder, processo_id):
    mapping = {"abdomen": "", "femur": "", "head": ""}
    if not os.path.isdir(processo_folder):
        print(f"Folder not found for Processo {processo_id}")
        return mapping

    all_files = [f for f in os.listdir(processo_folder) if f.lower().endswith(".png")]

    for plane in ["abdomen", "femur", "head"]:
        plane_files = [f for f in all_files if plane in f.lower()]
        if not plane_files:
            print(f" No {plane} images found for Processo {processo_id}")
            continue

        # Try to find the preferred "plane1" image
        preferred = next((f for f in plane_files if re.search(rf"{plane}1", f.lower())), None)
        chosen = preferred if preferred else plane_files[0]

        if not preferred:
            print(f" Using fallback for {plane} in Processo {processo_id}: {chosen}")

        mapping[plane] = os.path.join("prospective_images", os.path.basename(processo_folder), chosen)

    return mapping

# Iterate and fill in the new columns
for idx, row in df.iterrows():
    processo_value = str(row["Processo"]).strip()
    processo_folder = os.path.join(images_dir, processo_value)

    images = find_images_for_processo(processo_folder, processo_value)
    df.at[idx, "abdomen_image"] = images["abdomen"]
    df.at[idx, "femur_image"] = images["femur"]
    df.at[idx, "head_image"] = images["head"]

# Save new CSV
output_path = os.path.join(base_dir, "all_prospective_data.csv")
df.to_csv(output_path, index=False)

print(f"\nDone! Saved to: {output_path}")
