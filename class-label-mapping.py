# Load images and labels for JAFFE and CK+ datasets
X_jaffe_train, y_jaffe_train = load_images_from_folders(DATA_PATHS['jaffe_train'])
X_jaffe_test, y_jaffe_test = load_images_from_folders(DATA_PATHS['jaffe_test'])
X_ck_train, y_ck_train = load_images_from_folders(DATA_PATHS['ck_train'])
X_ck_test, y_ck_test = load_images_from_folders(DATA_PATHS['ck_test'])

# Combine all training and testing data for a comprehensive model
X_train_raw = np.array(X_jaffe_train + X_ck_train)
y_train_labels = np.array(y_jaffe_train + y_ck_train)

X_test_raw = np.array(X_jaffe_test + X_ck_test)
y_test_labels = np.array(y_jaffe_test + y_ck_test)

# Check if any data was loaded
if X_train_raw.size == 0 or X_test_raw.size == 0:
    print("\nFATAL ERROR: No images were loaded from the specified paths.")
    print("Please ensure the paths are correct and contain subfolders with images.")
else:
    print(f"Total Combined Training Images: {len(X_train_raw)}")
    print(f"Total Combined Testing Images: {len(X_test_raw)}")

    # Encode string labels (e.g., 'Angry', 'Happy') into integers (0, 1, 2, ...)
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train_labels)
    y_test_encoded = le.transform(y_test_labels)

    # Store the actual class names for the final report
    CLASS_LABELS = le.classes_
