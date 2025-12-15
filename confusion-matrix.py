# Predict the labels for the test set
y_pred_encoded = knn_model.predict(X_test_features)

# Decode predictions back to original labels for clarity in reports
y_test_decoded = le.inverse_transform(y_test_encoded)
y_pred_decoded = le.inverse_transform(y_pred_encoded)

# Calculate and print accuracy
accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
print(f"Overall Accuracy: {accuracy:.4f}")

# Generate a detailed classification report
print("\nClassification Report:")
print(classification_report(y_test_decoded, y_pred_decoded, target_names=CLASS_LABELS, digits=3))

# Display Confusion Matrix (decoded labels)
cm_data = confusion_matrix(y_test_decoded, y_pred_decoded, labels=CLASS_LABELS)
cm = pd.DataFrame(cm_data, columns=CLASS_LABELS, index = CLASS_LABELS)
print("\nConfusion Matrix (Raw Counts):\n", cm)
