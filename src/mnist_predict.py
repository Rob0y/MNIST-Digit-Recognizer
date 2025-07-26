# ============================
# MNIST Digit Recognition - Prediction
# ============================

# Load the model
model = keras.models.load_model("model.h5")

# Pick some samples from test set
sample_images = x_test[:25]
sample_labels = y_test[:25]

# Predict
predictions = model.predict(sample_images)

# Show images and predictions
plt.figure(figsize=(8, 8))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(sample_images[i], cmap=plt.cm.binary)
    plt.xlabel(f"Pred: {np.argmax(predictions[i])} (True: {sample_labels[i]})")
plt.tight_layout()
plt.savefig("sample_predictions.png")
plt.show()
