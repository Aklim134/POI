import os
import numpy as np
import cv2
from skimage import io, color
from skimage.feature import graycomatrix, graycoprops
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import glob

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

# Konfiguracja
INPUT_IMAGE_DIR = r'D:\git_poi\POI\zad3\src'
PATCHES_OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'wycinki_mini')
FEATURES_CSV_FILE = os.path.join(SCRIPT_DIR, 'cechy_mini.csv')

PATCH_SIZE = (128, 128)
GLCM_DISTANCES = [1, 3, 5]
GLCM_ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]
GLCM_LEVELS = 64

# Wycinanie probek
def extract_patches(input_dir, output_dir, patch_h, patch_w):
    os.makedirs(output_dir, exist_ok=True)
    total_patches_saved = 0
    categories = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    if not categories:
        print(f"Brak podkatalogow (kategorii) w {input_dir}")
        return

    for category in categories:
        category_input_path = os.path.join(input_dir, category)
        category_output_path = os.path.join(output_dir, category)
        os.makedirs(category_output_path, exist_ok=True)
        
        image_files = []
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff'):
            image_files.extend(glob.glob(os.path.join(category_input_path, ext)))

        if not image_files:
            print(f"Brak obrazow w {category_input_path} dla kategorii {category}")
            continue

        category_patches_count = 0
        for img_path in image_files:
            try:
                img = cv2.imread(img_path)
                if img is None or img.shape[0] < patch_h or img.shape[1] < patch_w:
                    continue
                img_filename_base = os.path.splitext(os.path.basename(img_path))[0]
                for r in range(0, img.shape[0] - patch_h + 1, patch_h):
                    for c in range(0, img.shape[1] - patch_w + 1, patch_w):
                        patch = img[r:r + patch_h, c:c + patch_w]
                        cv2.imwrite(os.path.join(category_output_path, f"{img_filename_base}_p_{r}_{c}.png"), patch)
                        category_patches_count += 1
            except Exception as e:
                print(f"Blad (extract) przy {img_path}: {e}")
        print(f"Kategoria '{category}': zapisano {category_patches_count} probek.")
        total_patches_saved += category_patches_count
    print(f"Lacznie zapisano {total_patches_saved} probek.")
    print("Wycinanie probek zakonczone.")


# Obliczanie cech GLCM
def calculate_glcm_features(patches_dir, distances, angles, levels):
    all_features = []
    properties = ['dissimilarity', 'correlation', 'contrast', 'energy', 'homogeneity', 'ASM']
    categories = [d for d in os.listdir(patches_dir) if os.path.isdir(os.path.join(patches_dir, d))]

    if not categories:
        print(f"Brak podkatalogow (kategorii) w {patches_dir} (katalogu z wycinkami).")
        return all_features

    for category in categories:
        print(f"Cechy dla: {category}")
        patch_files_path = os.path.join(patches_dir, category, '*.png')
        patch_files = glob.glob(patch_files_path)

        if not patch_files:
            print(f"Brak plikow .png w {os.path.join(patches_dir, category)}")
            continue
            
        for patch_path in patch_files:
            try:
                patch = io.imread(patch_path)
                patch_gray = color.rgb2gray(patch) if patch.ndim == 3 else patch
                patch_gray_uint8 = (patch_gray * 255).astype(np.uint8) if patch_gray.max() <= 1.0 else patch_gray.astype(np.uint8)

                img_quantized = np.floor(patch_gray_uint8 / (256.0 / levels)).astype(np.uint8)
                img_quantized[img_quantized >= levels] = levels - 1

                glcm = graycomatrix(img_quantized, distances, angles, levels, symmetric=True, normed=True)
                feature_vector = {'category': category, 'patch_file': os.path.basename(patch_path)}
                for prop in properties:
                    prop_values = graycoprops(glcm, prop)
                    for i, dist in enumerate(distances):
                        feature_vector[f'{prop}_d{dist}'] = np.mean(prop_values[i, :])
                all_features.append(feature_vector)
            except Exception as e:
                print(f"Blad (GLCM) przy {patch_path}: {e}")
    return all_features

# Zapis i Klasyfikacja
def save_and_classify(features_list, csv_filepath, test_size=0.25, random_state=42, use_stratify=True):
    if not features_list:
        print("Brak cech do przetworzenia.")
        return

    df = pd.DataFrame(features_list)
    df.to_csv(csv_filepath, index=False)
    print(f"Zapisano {len(df)} cech do {csv_filepath}")

    X = df.drop(['category', 'patch_file'], axis=1, errors='ignore')
    if X.empty:
        print("Brak kolumn z cechami po usunieciu 'category' i 'patch_file'. Klasyfikacja niemozliwa.")
        return
    y = LabelEncoder().fit_transform(df['category'])

    n_classes = len(np.unique(y))
    
    unique_elements, counts_elements = np.unique(y, return_counts=True)

    stratify_param = None
    if use_stratify:
        if len(X) * (1 - test_size) >= n_classes and (counts_elements >= 2).all():
            stratify_param = y
        else:
            print("Ostrzezenie: Warunki dla stratyfikacji nie sa spelnione (za malo probek w klasie lub za maly zbior treningowy). Wylaczam stratyfikacje.")


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify_param)

    if len(X_train) == 0 or len(X_test) == 0 :
        print(f"Blad: Jeden ze zbiorow (treningowy: {len(X_train)}, testowy: {len(X_test)}) jest pusty. Sprawdz test_size i liczbe danych.")
        return

    # Wyswietlanie statusu stratyfikacji
    print(f"\nKlasyfikacja (test_size={test_size}, stratify={stratify_param is not None})")
    for clf_name, classifier in [('KNN', KNeighborsClassifier(n_neighbors=min(5, len(X_train)) if len(X_train) > 0 else 1)),
                                 ('SVM', SVC(kernel='linear', random_state=random_state))]:
        if len(X_train) == 0:
            print(f"Brak danych treningowych dla {clf_name}.")
            continue
        try:
            model = classifier.fit(X_train, y_train)
            if len(X_test) > 0:
                accuracy = accuracy_score(y_test, model.predict(X_test))
                print(f"  {clf_name} Accuracy: {accuracy:.4f}")
            else:
                print(f"  {clf_name}: Brak danych testowych do oceny.")
        except Exception as e:
            print(f"  Blad klasyfikacji {clf_name}: {e}")


# Glowna czesc skryptu
if __name__ == "__main__":
    print(f"Skrypt uruchomiony z: {SCRIPT_DIR}")
    print(f"Katalog na wycinki: {PATCHES_OUTPUT_DIR}")
    print(f"Plik CSV na cechy: {FEATURES_CSV_FILE}")
    print("Start przetwarzania...")
    
    extract_patches(INPUT_IMAGE_DIR, PATCHES_OUTPUT_DIR, PATCH_SIZE[0], PATCH_SIZE[1])
    features = calculate_glcm_features(PATCHES_OUTPUT_DIR, GLCM_DISTANCES, GLCM_ANGLES, GLCM_LEVELS)

    if features:
        save_and_classify(features, FEATURES_CSV_FILE, test_size=0.8, use_stratify=True)
        #save_and_classify(features, os.path.join(SCRIPT_DIR, "cechy_mini_nostratify.csv"), test_size=0.8, use_stratify=False)
    else:
        print("Nie wygenerowano zadnych cech. Klasyfikacja nie zostanie przeprowadzona.")

    print("Zakonczono.")