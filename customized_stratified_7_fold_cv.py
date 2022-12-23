'''
Before running this code, make sure you have our COBA dataset in the working directory
'''
##################################################################################################################################

import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os
import shutil

print("Generating the COBA_unsplitted dataset to create the 7 fold splits...")

if not os.path.exists("COBA_unsplitted"):
    os.mkdir("COBA_unsplitted")
    os.mkdir("COBA_unsplitted/images")
    os.mkdir("COBA_unsplitted/labels")

for folder in ["train", "valid"]:
    for name in ["images", "labels"]:
        src = f"COBA/{folder}/{name}"
        all_files = os.listdir(src)
        for file in all_files:
            src_path = os.path.join(src, file)
            dest_path = f"COBA_unsplitted/{name}/{file}"
            shutil.copy(src_path, dest_path)

print("Success! COBA_unsplitted dataset created!\n")

BASE_DIR = "COBA_unsplitted"
images = os.listdir(f"{BASE_DIR}/images")
labels = os.listdir(f"{BASE_DIR}/labels")

print(f"Total number of image-label pairs: {len(labels)}")
print(f"Total background images (with no annotations): {len(images) - len(labels)}\n")

# All classes dictionary
classes = {"aircraft":0, "ambulance": 1, "watercraft":2, "barrel": 3, "briefcase": 4, "cannon": 5, "car":6, 
           "dagger":7, "dog":8, "handgun":9, "helmet":10, "horse":11, "missile_rocket":12, "motorcycle":13, "civilian": 14,
           "rifle":15, "tank":16, "truck":17, "soldier":18}

# Swapping the key-value pairs to generate the opposite class mapping
classes_opp = dict(list(zip(*(classes.values(), classes.keys()))))

# Adding the image_id column in the columns list
columns = ["image_id"] + list(classes.keys())

##################################################################################################################################
# We count the number of annotations for each object present in the image

print("Counting the number of annotations for each object present in the images...")

# Dataframe to store the count
s7f_data = pd.DataFrame(columns=columns)

# Here we loop over the labels because images also contain the background images which is not necessary during the 
# dataset split evaluation
for i, label_file in enumerate(labels):
    
    # Storing the image id in image_id column
    per_image_annot = {"image_id": label_file.split(".")[0]}
    
    # Reading the annotations for one single image from the associated label_file
    with open(f"{BASE_DIR}/labels/{label_file}", 'r') as f:
        annots = f.readlines()
    
    # Looping over all annotations in the selected label_file
    for annot in annots:
        label = int(annot.split(" ")[0])
        
        # Mapping the object index to the class name for the current annotation in the label_file
        label_name = classes_opp[label]
        
        # Counting the number of annotations for each label_name present in the label_file
        per_image_annot.setdefault(label_name, 0)
        per_image_annot[label_name] += 1
    
    # Storing the total count of objects present in the image (Note: objects not presented will return NaN)
    s7f_data = s7f_data.append(per_image_annot, ignore_index = True)

print("Success! Counting completed!\n")

##################################################################################################################################
# Filling the NaN with 0 since those objects are not present in that image_id
s7f_data = s7f_data.fillna(0)

# Here we drop the civilian since most of the images has civilian class, which may dominate other classes
s7f_data = s7f_data.drop(columns=["civilian"])

# Separating the image_id from the count values
s7f_data.set_index("image_id", inplace=True)

##################################################################################################################################
# Here we assign stratified label based on max_count of different objects
# if max == 0; then we assign civilian

stratified_labels_list = list()

for row in s7f_data.iterrows():
    max_val = max(row[1])
    if max_val == 0:
        label = "civilian"
    else:
        label = row[1].idxmax()
    stratified_labels_list.append(label)

# Now we assign the stratified labels for each image id
s7f_data.loc[:, "stratified_labels"] = stratified_labels_list

# reset index and get only required columns
s7f_data_2cols = s7f_data.reset_index()[["image_id", "stratified_labels"]]

##################################################################################################################################
# Now, we use the stratified 7 fold cross validation on our customized stratified labels

# For reproducibility, we use random_state=9
s7f = StratifiedKFold(n_splits=7, shuffle=True, random_state=9)

# Splitting the images and stratified_labels into 7 fold indices
X = s7f_data_2cols["image_id"]
y = s7f_data_2cols["stratified_labels"]

# Assigning the fold number to each of the images
for fold_number, (train_index, val_index) in enumerate(s7f.split(X=X, y=y)):
    s7f_data_2cols.loc[s7f_data_2cols.iloc[val_index].index, 'fold'] = int(fold_number)

# Creating the dataframe with only the classes column
stratified_folds_df = pd.DataFrame({"classes":classes.keys()})

# Looping over 6 folds
for i in range(7):
    
    # for each fold, we count the total number of entries of each class
    temp_dict = s7f_data_2cols[s7f_data_2cols["fold"] == i]["stratified_labels"].value_counts().to_dict()
    
    # Here, we map the count of each class for all the respective folds
    stratified_folds_df[f"fold_{i}"] = stratified_folds_df["classes"].map(temp_dict)

# Finally, we calculate the sum of total stratified entries in each fold, and append at the end of dataframe
folds_sum = {f"fold_{i}": sum(stratified_folds_df[f"fold_{i}"]) for i in range(7)}
folds_sum["classes"] = "Total"
stratified_folds_df = stratified_folds_df.append(folds_sum, ignore_index=True).set_index("classes")

##################################################################################################################################
## Here we create select one fold and make it as validation set and merge all other folds as training set.
## We repeat the process for all 7 folds to create 7 sets of training and validation data.

if not os.path.exists("COBA_7fold_split_sets"):
    os.mkdir("COBA_7fold_split_sets")
    
for i in range(7):
    
    # Creating the necessary directories
    print(f"Creating fold {i} dataset...")
    root_folder = f"COBA_7fold_split_sets/fold_{i}"
    if not os.path.exists(root_folder):
        os.mkdir(root_folder)
        os.mkdir(f"{root_folder}/valid")
        os.mkdir(f"{root_folder}/valid/images")
        os.mkdir(f"{root_folder}/valid/labels")
        os.mkdir(f"{root_folder}/train")
        os.mkdir(f"{root_folder}/train/images")
        os.mkdir(f"{root_folder}/train/labels")
    
    # Looping over each entry in the dataframe
    for row in s7f_data_2cols.iterrows():
        
        # Checking the path to ensure the copypath (either train or valid folders)
        if row[1]["fold"] == i:
            dest_path = "valid"
        else:
            dest_path = "train"
        
        # Here, we copy the image from source to the designated destination
        img_src = f"{BASE_DIR}/images/{row[1]['image_id']}.jpg"
        img_dest = f"{root_folder}/{dest_path}/images/{row[1]['image_id']}.jpg"
        shutil.copy(img_src, img_dest)
        
        # Here, we copy the labels from source to the designated destination
        label_src = f"{BASE_DIR}/labels/{row[1]['image_id']}.txt"
        label_dest = f"{root_folder}/{dest_path}/labels/{row[1]['image_id']}.txt"
        shutil.copy(label_src, label_dest)

    print(f"Fold {i} dataset created!!!\n")

print("Successfully splitted the dataset into 7 folds!\n")

##################################################################################################################################
# Delete the COBA_unsplitted folder
print("Now we do not need COBA_unsplitted folder, so we delete it!")
print("Deleting COBA_unsplitted...")
shutil.rmtree("COBA_unsplitted")
print("COBA_unsplitted deleted!")

print("\nThe 7fold of training and validation sets are stored in COBA_7fold_split_sets/fold_0")
print("\t\t\t\t\t\t\t\t\t      fold_1")
print("\t\t\t\t\t\t\t\t\t      fold_2")
print("\t\t\t\t\t\t\t\t\t      fold_3")
print("\t\t\t\t\t\t\t\t\t      fold_4")
print("\t\t\t\t\t\t\t\t\t      fold_5")
print("\t\t\t\t\t\t\t\t\t      fold_6")
##################################################################################################################################