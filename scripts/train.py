from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Define Callbacks for training optimization
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',   # Monitor validation loss for learning rate adjustments
    factor=0.2,           # Reduce learning rate by a factor of 0.2
    patience=3,           # Wait for 3 epochs before reducing the learning rate
    min_lr=0.0001         # Set a minimum learning rate
)

early_stopping = EarlyStopping(
    monitor='val_loss',   # Monitor validation loss to stop early if necessary
    patience=5,           # Stop if no improvement for 5 epochs
    restore_best_weights=True  # Restore the best model weights after early stopping
)

# Compile the model
model.compile(
    optimizer='adam',                          # Optimizer: Adam for adaptive learning rate
    loss='categorical_crossentropy',           # Loss function: Categorical Crossentropy for multi-class classification
    metrics=['accuracy', 'Precision', 'Recall', 'AUC', 'MSE', 'CosineSimilarity']  # Metrics to evaluate the model
)

# Train the model for 60 epochs with validation and callbacks
history = model.fit(
    train_generator,                           # Training data generator
    epochs=60,                                 # Train for 60 epochs
    validation_data=validation_generator,      # Validation data generator
    callbacks=[reduce_lr, early_stopping],     # Apply callbacks to optimize training
    verbose=1                                   # Display progress for each epoch
)

# Evaluate the model on validation data
loss, accuracy, precision, recall, auc, mse, cosine_similarity = model.evaluate(validation_generator)

# Display evaluation metrics
print("\nModel Evaluation Results:")
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Validation Precision: {precision:.4f}")
print(f"Validation Recall: {recall:.4f}")
print(f"Validation AUC: {auc:.4f}")
print(f"Validation MSE: {mse:.4f}")
print(f"Validation Cosine Similarity: {cosine_similarity:.4f}")
