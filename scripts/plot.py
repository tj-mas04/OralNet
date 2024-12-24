
import matplotlib.pyplot as plt
import pandas as pd

def plot_history(history):
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    precision = history['precision']
    val_precision = history['val_precision']
    recall = history['recall']
    val_recall = history['val_recall']
    auc = history['auc']
    val_auc = history['val_auc']
    mse = history['mse']
    val_mse = history['val_mse']


    if 'cosine_similarity' in history:
        cosine_similarity = history['cosine_similarity']
        val_cosine_similarity = history['val_cosine_similarity']
    else:
        cosine_similarity = None
        val_cosine_similarity = None

    epochs = range(1, len(acc) + 1)

    # Plotting
    plt.figure(figsize=(14, 12))

    # Plot accuracy
    plt.subplot(3, 2, 1)
    plt.plot(epochs, acc, 'r', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(3, 2, 2)
    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot precision
    plt.subplot(3, 2, 3)
    plt.plot(epochs, precision, 'r', label='Training Precision')
    plt.plot(epochs, val_precision, 'b', label='Validation Precision')
    plt.title('Training and Validation Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()

    # Plot recall
    plt.subplot(3, 2, 4)
    plt.plot(epochs, recall, 'r', label='Training Recall')
    plt.plot(epochs, val_recall, 'b', label='Validation Recall')
    plt.title('Training and Validation Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()

    # Plot AUC
    plt.subplot(3, 2, 5)
    plt.plot(epochs, auc, 'r', label='Training AUC')
    plt.plot(epochs, val_auc, 'b', label='Validation AUC')
    plt.title('Training and Validation AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()

    # Plot MSE
    plt.subplot(3, 2, 6)
    plt.plot(epochs, mse, 'r', label='Training MSE')
    plt.plot(epochs, val_mse, 'b', label='Validation MSE')
    plt.title('Training and Validation MSE')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()


    if cosine_similarity is not None and val_cosine_similarity is not None:
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 1, 1)
        plt.plot(epochs, cosine_similarity, 'r', label='Training Cosine Similarity')
        plt.plot(epochs, val_cosine_similarity, 'b', label='Validation Cosine Similarity')
        plt.title('Training and Validation Cosine Similarity')
        plt.xlabel('Epochs')
        plt.ylabel('Cosine Similarity')
        plt.legend()


    plt.tight_layout()
    plt.show()

    # Tabular representation of metrics
    metrics_df = pd.DataFrame({
        'Accuracy': acc,
        'Val Accuracy': val_acc,
        'Loss': loss,
        'Val Loss': val_loss,
        'Precision': precision,
        'Val Precision': val_precision,
        'Recall': recall,
        'Val Recall': val_recall,
        'AUC': auc,
        'Val AUC': val_auc,
        'MSE': mse,
        'Val MSE': val_mse,
        'Cosine Similarity': cosine_similarity if cosine_similarity is not None else [None]*len(acc),
        'Val Cosine Similarity': val_cosine_similarity if val_cosine_similarity is not None else [None]*len(acc),
    })

    print("\nMetrics:")
    print(metrics_df)


history = {
    'accuracy': [0.8015, 0.8059, 0.8068, 0.8104, 0.8093],
    'val_accuracy': [0.817, 0.8183, 0.8192, 0.8179, 0.8153],
    'loss': [0.5104, 0.4932, 0.485, 0.4829, 0.4793],
    'val_loss': [0.7586, 0.7513, 0.7895, 0.7638, 0.7802],
    'precision': [0.8423, 0.8397, 0.8428, 0.8435, 0.8474],
    'val_precision': [0.8518, 0.851, 0.848, 0.8475, 0.8452],
    'recall': [0.7515, 0.7615, 0.7604, 0.7645, 0.7648],
    'val_recall': [0.7483, 0.7285, 0.7453, 0.7474, 0.7504],
    'auc': [0.9721, 0.974, 0.9747, 0.9749, 0.9752],
    'val_auc': [0.9495, 0.9495, 0.948, 0.9491, 0.9482],
    'mse': [0.0463, 0.0453, 0.0448, 0.0441, 0.044],
    'val_mse': [0.0521, 0.0522, 0.0517, 0.0517, 0.051],
    'cosine_similarity': [0.8399, 0.8426, 0.8438, 0.8466, 0.8472],
    'val_cosine_similarity': [0.8292, 0.8289, 0.8315, 0.83, 0.8324]
}

plot_history(history)