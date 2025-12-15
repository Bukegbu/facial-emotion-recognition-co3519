# Initialize the KNN classifier
knn_model = KNeighborsClassifier(n_neighbors=5, metric="euclidean")

# Train the model using the LBP feature vectors
knn_model.fit(X_train_features, y_train_encoded)

print("KNN Training complete.")
