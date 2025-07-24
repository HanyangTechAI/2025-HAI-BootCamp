
import os
import random

def generate_csv(image_dir, label_dir, output_file, num_examples):
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
    label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]

    image_names = {os.path.splitext(f)[0] for f in image_files}
    label_names = {os.path.splitext(f)[0] for f in label_files}

    common_files = list(image_names.intersection(label_names))

    if len(common_files) < num_examples:
        print(f"Warning: Not enough common files ({len(common_files)}) to generate {num_examples} examples.")
        num_examples = len(common_files)

    selected_files = random.sample(common_files, num_examples)

    with open(output_file, "w") as f:
        f.write("img,label\n")
        for name in selected_files:
            f.write(f"{name}.jpg,{name}.txt\n")

if __name__ == "__main__":
    generate_csv("images", "labels", "val_27.csv", 27)

