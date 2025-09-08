import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torchvision
import torchvision.transforms as transforms

# ===== YOUR ORIGINAL NAG IMPLEMENTATION (ADAPTED) =====

class NAGOptimizerNumPy:
    def __init__(self, input_size, hidden_size, output_size, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum

        # Initialize weights and biases
        self.w_input = np.random.randn(hidden_size, input_size) * np.sqrt(2.0/input_size)
        self.b_input = np.zeros(hidden_size)
        self.w_output = np.random.randn(output_size, hidden_size) * np.sqrt(2.0/hidden_size)
        self.b_output = np.zeros(output_size)

        # Initialize velocity
        self.v_w_input = np.zeros_like(self.w_input)
        self.v_w_output = np.zeros_like(self.w_output)
        self.v_b_input = np.zeros_like(self.b_input)
        self.v_b_output = np.zeros_like(self.b_output)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x)**2

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def forward_pass(self, x):
        z_input = np.dot(x, self.w_input.T) + self.b_input
        z_input_activation = self.tanh(z_input)
        z_output = np.dot(z_input_activation, self.w_output.T) + self.b_output
        return z_input_activation, z_output, z_input

    def mse_loss(self, y_true, y_pred):
        e = y_true - y_pred
        loss = np.mean(e**2)
        return loss, e

    def cross_entropy_loss(self, y_true, y_pred):
        # Softmax
        exp_scores = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Cross-entropy loss
        correct_logprobs = -np.log(probs[range(len(y_true)), y_true])
        loss = np.mean(correct_logprobs)

        # Gradient
        probs[range(len(y_true)), y_true] -= 1
        e = probs / len(y_true)
        return loss, e

    def backward_pass_regression(self, x_batch, y_batch, z_output, z_input, z_input_activation, batch_size):
        # Compute lookahead weights and biases
        w_input_lookahead = self.w_input - self.lr * self.momentum * self.v_w_input
        w_output_lookahead = self.w_output - self.lr * self.momentum * self.v_w_output
        b_input_lookahead = self.b_input - self.lr * self.momentum * self.v_b_input
        b_output_lookahead = self.b_output - self.lr * self.momentum * self.v_b_output

        # Forward pass with lookahead weights and biases
        z_input_lookahead = np.dot(x_batch, w_input_lookahead.T) + b_input_lookahead
        z_input_activation_lookahead = self.tanh(z_input_lookahead)
        z_output_lookahead = np.dot(z_input_activation_lookahead, w_output_lookahead.T) + b_output_lookahead

        # Compute gradients at lookahead position
        e_lookahead = y_batch - z_output_lookahead
        deltas_output_lookahead = -2 * e_lookahead / batch_size
        dw_output_lookahead = np.dot(z_input_activation.T, deltas_output_lookahead) / batch_size
        db_output_lookahead = np.sum(deltas_output_lookahead, axis=0) / batch_size

        activation_prime_lookahead = self.tanh_derivative(z_input_lookahead)
        deltas_input_lookahead = np.dot(deltas_output_lookahead, w_output_lookahead) * activation_prime_lookahead / batch_size
        dw_input_lookahead = np.dot(x_batch.T, deltas_input_lookahead) / batch_size
        db_input_lookahead = np.sum(deltas_input_lookahead, axis=0) / batch_size

        # Update velocity for weights and biases
        self.v_w_input = self.momentum * self.v_w_input + (1 - self.momentum) * dw_input_lookahead.T
        self.v_w_output = self.momentum * self.v_w_output + (1 - self.momentum) * dw_output_lookahead.T
        self.v_b_input = self.momentum * self.v_b_input + (1 - self.momentum) * db_input_lookahead
        self.v_b_output = self.momentum * self.v_b_output + (1 - self.momentum) * db_output_lookahead

        # Update weights and biases using velocity
        self.w_input -= self.lr * self.v_w_input
        self.w_output -= self.lr * self.v_w_output
        self.b_input -= self.lr * self.v_b_input
        self.b_output -= self.lr * self.v_b_output

    def backward_pass_classification(self, x_batch, y_batch, z_output, z_input, z_input_activation, batch_size):
        # Compute lookahead weights and biases
        w_input_lookahead = self.w_input - self.lr * self.momentum * self.v_w_input
        w_output_lookahead = self.w_output - self.lr * self.momentum * self.v_w_output
        b_input_lookahead = self.b_input - self.lr * self.momentum * self.v_b_input
        b_output_lookahead = self.b_output - self.lr * self.momentum * self.v_b_output

        # Forward pass with lookahead weights and biases
        z_input_lookahead = np.dot(x_batch, w_input_lookahead.T) + b_input_lookahead
        z_input_activation_lookahead = self.tanh(z_input_lookahead)
        z_output_lookahead = np.dot(z_input_activation_lookahead, w_output_lookahead.T) + b_output_lookahead

        # Cross-entropy gradients at lookahead position
        _, e_lookahead = self.cross_entropy_loss(y_batch, z_output_lookahead)

        dw_output_lookahead = np.dot(z_input_activation.T, e_lookahead)
        db_output_lookahead = np.sum(e_lookahead, axis=0)

        activation_prime_lookahead = self.tanh_derivative(z_input_lookahead)
        deltas_input_lookahead = np.dot(e_lookahead, w_output_lookahead) * activation_prime_lookahead
        dw_input_lookahead = np.dot(x_batch.T, deltas_input_lookahead)
        db_input_lookahead = np.sum(deltas_input_lookahead, axis=0)

        # Update velocity for weights and biases
        self.v_w_input = self.momentum * self.v_w_input + (1 - self.momentum) * dw_input_lookahead.T
        self.v_w_output = self.momentum * self.v_w_output + (1 - self.momentum) * dw_output_lookahead.T
        self.v_b_input = self.momentum * self.v_b_input + (1 - self.momentum) * db_input_lookahead
        self.v_b_output = self.momentum * self.v_b_output + (1 - self.momentum) * db_output_lookahead

        # Update weights and biases using velocity
        self.w_input -= self.lr * self.v_w_input
        self.w_output -= self.lr * self.v_w_output
        self.b_input -= self.lr * self.v_b_input
        self.b_output -= self.lr * self.v_b_output

    def train_regression(self, x_train, y_train, epochs=100, batch_size=32):
        losses = []
        num_samples = x_train.shape[0]

        for epoch in range(epochs):
            # Shuffle the dataset
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0
            num_batches = 0

            # Process mini-batches
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                x_batch = x_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                current_batch_size = x_batch.shape[0]

                # Forward pass
                z_input_activation, z_output, z_input = self.forward_pass(x_batch)

                # Compute loss
                loss, e = self.mse_loss(y_batch, z_output)
                epoch_loss += loss
                num_batches += 1

                # Backward pass with NAG
                self.backward_pass_regression(x_batch, y_batch, z_output, z_input, z_input_activation, current_batch_size)

            # Average loss for the epoch
            epoch_loss /= num_batches
            losses.append(epoch_loss)

        return losses

    def train_classification(self, x_train, y_train, epochs=100, batch_size=32):
        losses = []
        accuracies = []
        num_samples = x_train.shape[0]

        for epoch in range(epochs):
            # Shuffle the dataset
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0
            correct_predictions = 0
            num_batches = 0

            # Process mini-batches
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                x_batch = x_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                current_batch_size = x_batch.shape[0]

                # Forward pass
                z_input_activation, z_output, z_input = self.forward_pass(x_batch)

                # Compute loss
                loss, e = self.cross_entropy_loss(y_batch, z_output)
                epoch_loss += loss
                num_batches += 1

                # Compute accuracy
                predictions = np.argmax(z_output, axis=1)
                correct_predictions += np.sum(predictions == y_batch)

                # Backward pass with NAG
                self.backward_pass_classification(x_batch, y_batch, z_output, z_input, z_input_activation, current_batch_size)

            # Average loss and accuracy for the epoch
            epoch_loss /= num_batches
            epoch_accuracy = correct_predictions / num_samples
            losses.append(epoch_loss)
            accuracies.append(epoch_accuracy)

        return losses, accuracies

# ===== PYTORCH IMPLEMENTATION =====

class PyTorchNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PyTorchNet, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

        # Initialize weights to match NumPy implementation
        nn.init.xavier_normal_(self.hidden.weight)
        nn.init.zeros_(self.hidden.bias)
        nn.init.xavier_normal_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        x = torch.tanh(self.hidden(x))
        x = self.output(x)
        return x

def train_pytorch_regression(model, train_loader, optimizer, criterion, epochs, device):
    model.train()
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        epoch_loss /= num_batches
        losses.append(epoch_loss)

    return losses

def train_pytorch_classification(model, train_loader, optimizer, criterion, epochs, device):
    model.train()
    losses = []
    accuracies = []

    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        num_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        epoch_loss /= num_batches
        accuracy = correct / total
        losses.append(epoch_loss)
        accuracies.append(accuracy)

    return losses, accuracies

# ===== BENCHMARKING FUNCTIONS =====

def benchmark_california_housing():
    print("=== CALIFORNIA HOUSING REGRESSION BENCHMARK ===")

    # Load and prepare data
    housing = fetch_california_housing()
    X, y = housing.data, housing.target.reshape(-1, 1)

    # Standardize features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    input_size = X_train.shape[1]
    hidden_size = 64
    output_size = 1
    lr = 0.01
    momentum = 0.9
    epochs = 50
    batch_size = 32

    print(f"Dataset size: {X_train.shape[0]} training samples")
    print(f"Input size: {input_size}, Hidden size: {hidden_size}, Output size: {output_size}")
    print(f"Hyperparameters: lr={lr}, momentum={momentum}, epochs={epochs}, batch_size={batch_size}")

    # NumPy Implementation
    print("\n--- NumPy NAG Implementation ---")
    start_time = time.time()
    numpy_model = NAGOptimizerNumPy(input_size, hidden_size, output_size, lr, momentum)
    numpy_losses = numpy_model.train_regression(X_train, y_train, epochs, batch_size)
    numpy_time = time.time() - start_time
    print(f"Training time: {numpy_time:.2f} seconds")
    print(f"Final loss: {numpy_losses[-1]:.6f}")

    # PyTorch Implementation
    print("\n--- PyTorch NAG Implementation ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    start_time = time.time()
    pytorch_model = PyTorchNet(input_size, hidden_size, output_size).to(device)
    optimizer = optim.SGD(pytorch_model.parameters(), lr=lr, momentum=momentum, nesterov=True)
    criterion = nn.MSELoss()
    pytorch_losses = train_pytorch_regression(pytorch_model, train_loader, optimizer, criterion, epochs, device)
    pytorch_time = time.time() - start_time
    print(f"Training time: {pytorch_time:.2f} seconds")
    print(f"Final loss: {pytorch_losses[-1]:.6f}")

    # Speedup calculation
    speedup = numpy_time / pytorch_time
    print(f"\nSpeedup (NumPy/PyTorch): {speedup:.2f}x")

    return numpy_losses, pytorch_losses, numpy_time, pytorch_time

def benchmark_fashion_mnist():
    print("\n=== FASHION-MNIST CLASSIFICATION BENCHMARK ===")

    # Load Fashion-MNIST
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

    # Convert to numpy for our implementation
    X_train = train_dataset.data.numpy().reshape(-1, 28*28).astype(np.float32) / 255.0
    y_train = train_dataset.targets.numpy()

    # Subset for faster benchmarking
    subset_size = 10000
    indices = np.random.choice(len(X_train), subset_size, replace=False)
    X_train = X_train[indices]
    y_train = y_train[indices]

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    input_size = 784
    hidden_size = 128
    output_size = 10
    lr = 0.01
    momentum = 0.9
    epochs = 20
    batch_size = 64

    print(f"Dataset size: {X_train.shape[0]} training samples")
    print(f"Input size: {input_size}, Hidden size: {hidden_size}, Output size: {output_size}")
    print(f"Hyperparameters: lr={lr}, momentum={momentum}, epochs={epochs}, batch_size={batch_size}")

    # NumPy Implementation
    print("\n--- NumPy NAG Implementation ---")
    start_time = time.time()
    numpy_model = NAGOptimizerNumPy(input_size, hidden_size, output_size, lr, momentum)
    numpy_losses, numpy_accuracies = numpy_model.train_classification(X_train, y_train, epochs, batch_size)
    numpy_time = time.time() - start_time
    print(f"Training time: {numpy_time:.2f} seconds")
    print(f"Final loss: {numpy_losses[-1]:.6f}")
    print(f"Final accuracy: {numpy_accuracies[-1]:.4f}")

    # PyTorch Implementation
    print("\n--- PyTorch NAG Implementation ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    start_time = time.time()
    pytorch_model = PyTorchNet(input_size, hidden_size, output_size).to(device)
    optimizer = optim.SGD(pytorch_model.parameters(), lr=lr, momentum=momentum, nesterov=True)
    criterion = nn.CrossEntropyLoss()
    pytorch_losses, pytorch_accuracies = train_pytorch_classification(pytorch_model, train_loader, optimizer, criterion, epochs, device)
    pytorch_time = time.time() - start_time
    print(f"Training time: {pytorch_time:.2f} seconds")
    print(f"Final loss: {pytorch_losses[-1]:.6f}")
    print(f"Final accuracy: {pytorch_accuracies[-1]:.4f}")

    # Speedup calculation
    speedup = numpy_time / pytorch_time
    print(f"\nSpeedup (NumPy/PyTorch): {speedup:.2f}x")

    return (numpy_losses, numpy_accuracies, pytorch_losses, pytorch_accuracies, numpy_time, pytorch_time)

def plot_results(regression_results, classification_results):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Regression Loss
    numpy_reg_losses, pytorch_reg_losses, numpy_reg_time, pytorch_reg_time = regression_results
    axes[0, 0].plot(numpy_reg_losses, 'b-', label=f'NumPy NAG ({numpy_reg_time:.1f}s)', linewidth=2)
    axes[0, 0].plot(pytorch_reg_losses, 'r-', label=f'PyTorch NAG ({pytorch_reg_time:.1f}s)', linewidth=2)
    axes[0, 0].set_title('California Housing - Regression Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Classification Loss
    numpy_cls_losses, numpy_cls_acc, pytorch_cls_losses, pytorch_cls_acc, numpy_cls_time, pytorch_cls_time = classification_results
    axes[0, 1].plot(numpy_cls_losses, 'b-', label=f'NumPy NAG ({numpy_cls_time:.1f}s)', linewidth=2)
    axes[0, 1].plot(pytorch_cls_losses, 'r-', label=f'PyTorch NAG ({pytorch_cls_time:.1f}s)', linewidth=2)
    axes[0, 1].set_title('Fashion-MNIST - Classification Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Cross-Entropy Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Classification Accuracy
    axes[1, 0].plot(numpy_cls_acc, 'b-', label='NumPy NAG', linewidth=2)
    axes[1, 0].plot(pytorch_cls_acc, 'r-', label='PyTorch NAG', linewidth=2)
    axes[1, 0].set_title('Fashion-MNIST - Classification Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Performance Comparison
    implementations = ['NumPy\n(CPU)', 'PyTorch\n(GPU/CPU)']
    reg_times = [numpy_reg_time, pytorch_reg_time]
    cls_times = [numpy_cls_time, pytorch_cls_time]

    x = np.arange(len(implementations))
    width = 0.35

    axes[1, 1].bar(x - width/2, reg_times, width, label='Regression (CA Housing)', alpha=0.8)
    axes[1, 1].bar(x + width/2, cls_times, width, label='Classification (Fashion-MNIST)', alpha=0.8)
    axes[1, 1].set_title('Training Time Comparison')
    axes[1, 1].set_ylabel('Time (seconds)')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(implementations)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ===== MAIN BENCHMARK =====

if __name__ == "__main__":
    print("NAG Optimizer Benchmark: NumPy vs PyTorch")
    print("=" * 50)

    # Run benchmarks
    regression_results = benchmark_california_housing()
    classification_results = benchmark_fashion_mnist()

    # Plot results
    plot_results(regression_results, classification_results)

    # Summary
    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)

    reg_speedup = regression_results[2] / regression_results[3]
    cls_speedup = classification_results[4] / classification_results[5]

    print(f"Regression Task (California Housing):")
    print(f"  NumPy time: {regression_results[2]:.2f}s")
    print(f"  PyTorch time: {regression_results[3]:.2f}s")
    print(f"  Speedup: {reg_speedup:.2f}x")

    print(f"\nClassification Task (Fashion-MNIST):")
    print(f"  NumPy time: {classification_results[4]:.2f}s")
    print(f"  PyTorch time: {classification_results[5]:.2f}s")
    print(f"  Speedup: {cls_speedup:.2f}x")
