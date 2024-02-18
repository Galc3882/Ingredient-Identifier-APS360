import threading
from typing import Iterable, Any
import data_processing
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import timm
from tqdm import tqdm
from queue import Queue


class _ThreadedIterator(threading.Thread):
    """Prefetch the next queue_length items from iterator in a background thread."""
    class _End:
        pass

    def __init__(self, generator: Iterable, maxsize: int) -> None:
        threading.Thread.__init__(self)
        self.queue: Queue = Queue(maxsize)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self) -> None:
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(self._End)

    def __iter__(self) -> Any:
        return self

    def __next__(self) -> Any:
        next_item = self.queue.get()
        if next_item == self._End:
            raise StopIteration
        return next_item


def bg_iterator(iterable: Iterable, maxsize: int) -> Any:
    return _ThreadedIterator(iterable, maxsize=maxsize)


class DNN(nn.Module):
    def __init__(self, num_classes, model_name):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x  # loss function has sigmoid built-in


def train(model, dataloader, loss_fn, optimizer, device, precision_recall_weight):
    model.train()
    total_loss = 0
    total_accuracy = 0
    for images, labels in tqdm(bg_iterator(dataloader, 1000), total=len(dataloader), desc="Training"):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        accuracy = accuracy_fn(labels, outputs, precision_recall_weight)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_accuracy += accuracy
        print(f"Val Loss: {loss.item():.2f}, Val Accuracy: {accuracy:.2f}", end="\r")
    return total_loss / len(dataloader), total_accuracy / len(dataloader)


def evaluate(model, dataloader, loss_fn, device, precision_recall_weight):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for images, labels in tqdm(bg_iterator(dataloader, 1000), total=len(dataloader), desc="Evaluating"):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            accuracy = accuracy_fn(labels, outputs, precision_recall_weight)
            total_loss += loss.item()
            total_accuracy += accuracy
            print(f"Val Loss: {loss.item():.2f}, Val Accuracy: {accuracy:.2f}", end="\r")
    return total_loss / len(dataloader), total_accuracy / len(dataloader)


def accuracy_fn(labels, outputs, positive_weight):
    '''Calculate the accuracy of the model.'''
    with torch.no_grad():
        predicted_positives = (torch.sigmoid(outputs) > 0.5).float()
        true_positives = (predicted_positives * labels).sum()
        true_negatives = ((1 - predicted_positives) * (1 - labels)).sum()
        false_positives = (predicted_positives * (1 - labels)).sum()
        false_negatives = ((1 - predicted_positives) * labels).sum()

        total_positives = labels.sum()
        total_negatives = (1 - labels).sum()

        weighted_correct = true_positives * positive_weight + true_negatives
        weighted_total = total_positives * positive_weight + total_negatives
        weighted_accuracy = weighted_correct / weighted_total

        tp_percent = 100 * true_positives / total_positives if total_positives > 0 else torch.tensor(0.)
        tn_percent = 100 * true_negatives / total_negatives if total_negatives > 0 else torch.tensor(0.)
        fp_percent = 100 * false_positives / total_negatives if total_negatives > 0 else torch.tensor(0.)
        fn_percent = 100 * false_negatives / total_positives if total_positives > 0 else torch.tensor(0.)

        # Print confusion matrix components, percentages, and weighted accuracy
        print(f"TP: {true_positives.item() / len(labels)} ({tp_percent.item():.2f}%), FP: {false_positives.item() / len(labels)} ({fp_percent.item():.2f}%), "
              f"TN: {true_negatives.item() / len(labels)} ({tn_percent.item():.2f}%), FN: {false_negatives.item() / len(labels)} ({fn_percent.item():.2f}%)")

    return weighted_accuracy.item()


if __name__ == "__main__":
    # Set Hyperparameters
    model_name = 'vit_base_patch32_384'  # https://github.com/kentaroy47/timm_speed_benchmark
    num_epochs = 10
    lr = 1e-5
    weight_decay = 1e-5
    step_size = 7
    gamma = 0.1
    batch_size = 64
    img_size = 384
    precision_recall_weight = 3 # 10

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the dataset
    train_data, val_data, test_data = data_processing.load_dataset(img_size)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Define the model, loss function and optimizer
    print("Creating model...")
    model = DNN(model_name=model_name, num_classes=len(train_loader.dataset.classes)).to(device, non_blocking=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos_weight = (torch.ones([len(train_loader.dataset.classes)]) * precision_recall_weight).to(device, non_blocking=True)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)

    # Statistics
    train_loss_list = []
    train_accuracy_list = []
    val_loss_list = []
    val_accuracy_list = []

    # Train the model
    print("Training model...")
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        train_loss, train_accuracy = train(model, train_loader, loss_fn, optimizer, device, precision_recall_weight)
        val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device, precision_recall_weight)
        scheduler.step()

        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_accuracy)
        val_loss_list.append(val_loss)
        val_accuracy_list.append(val_accuracy)
        print(f"Epoch {epoch}: Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")

    # # Evaluate the model
    # test_loss, test_accuracy = evaluate(model, test_loader, loss_fn, device)

    # Plot the statistics in two subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(train_loss_list, label="Train Loss")
    axs[0].plot(val_loss_list, label="Val Loss")
    axs[0].set_title("Loss")
    axs[0].legend()
    axs[1].plot(train_accuracy_list, label="Train Accuracy")
    axs[1].plot(val_accuracy_list, label="Val Accuracy")
    axs[1].set_title("Accuracy")
    axs[1].legend()
    plt.show()
