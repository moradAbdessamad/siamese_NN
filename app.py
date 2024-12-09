import os
import torch
import matplotlib.pyplot as plt

# Fix OpenMP error
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# Load the loss histories safely
models_dir = 'Models'
loss_history_path = os.path.join(models_dir, 'loss_histories.pth')
histories = torch.load(loss_history_path, map_location=torch.device('cpu'))

# Rest of your plotting code remains the same
avg_training_loss_history = histories['avg_training_loss_history']
avg_test_loss_history = histories['avg_test_loss_history']

# Create the plot
num_epochs = len(avg_training_loss_history)
epochs = range(1, num_epochs + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, avg_training_loss_history, label='Training Loss', color='blue')
plt.plot(epochs, avg_test_loss_history, label='Test Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss Over Time')
plt.legend()
plt.grid(True)
plt.show()