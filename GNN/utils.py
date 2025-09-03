import matplotlib.pyplot as plt
import os

def plot_history(history, output_root):
    epochs = range(1, len(history['train_loss']) + 1)
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    train_acc = history['train_acc']
    val_acc = history['val_acc']
    test_acc = history['test_acc']  

    test_acc = [x if x is not None else test_acc[i-1] for i, x in enumerate(test_acc)]  

    if not os.path.exists(output_root):
        os.makedirs(output_root)    

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), dpi=300)  

    axs[0].plot(epochs, train_loss, 'b', label='Training loss')
    axs[0].plot(epochs, val_loss, 'r', label='Validation loss')
    axs[0].set_title('Training and validation loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend() 

    axs[1].plot(epochs, train_acc, 'b', label='Training accuracy')
    axs[1].plot(epochs, val_acc, 'r', label='Validation accuracy')
    axs[1].plot(epochs, test_acc, 'g', label='Test accuracy')  
    axs[1].set_title('Training, validation, and test accuracy')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend() 

    plt.tight_layout()  

    figure_path = os.path.join(output_root, 'combined_plot.png')
    plt.savefig(figure_path)
    plt.close() 