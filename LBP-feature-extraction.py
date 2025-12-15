X_train_features = extract_lbp_features(X_train_raw, desc="Extracting Training LBP")
X_test_features = extract_lbp_features(X_test_raw, desc="Extracting Testing LBP")

print(f"LBP Feature vector dimension: {X_train_features.shape[1]} (N_bins)")
print(f"Total training feature vectors: {X_train_features.shape[0]}")
