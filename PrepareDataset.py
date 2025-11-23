import os
import shutil

def create_yolo_testset(image_folder, label_folder, output_folder="yolo_testset"):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Valid image extensions
    img_exts = {".jpg", ".jpeg", ".png"}

    # Collect images
    images = [
        f for f in os.listdir(image_folder)
        if os.path.splitext(f)[1].lower() in img_exts
    ]

    # Collect labels
    labels = {
        os.path.splitext(f)[0]: f
        for f in os.listdir(label_folder)
        if f.endswith(".txt")
    }

    # Open pair file in output folder
    pairs_file = os.path.join(output_folder, "pairs.txt")
    with open(pairs_file, "w") as pf:

        for img in images:
            name, ext = os.path.splitext(img)
            if name in labels:
                label_file = labels[name]

                # Copy image
                shutil.copy(
                    os.path.join(image_folder, img),
                    os.path.join(output_folder, img)
                )

                # Copy label
                shutil.copy(
                    os.path.join(label_folder, label_file),
                    os.path.join(output_folder, label_file)
                )

                # Write pair entry
                pf.write(f"{img}; {label_file}\n")

    print(f"Test dataset created in: {output_folder}")
    print(f"Pairs saved to: {pairs_file}")

# Example usage
create_yolo_testset(".\images", "C:/Users/inigo.infante/Downloads/Releaseme2/LeRobotHackaton2025/labels", ".\yolo_testset")