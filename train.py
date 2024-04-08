import threading
from typing import Iterable, Any
import data_processing
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import timm
from tqdm import tqdm
from queue import Queue
import torch.nn.functional as F


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


class ZLPRLoss(nn.Module):
    '''https://arxiv.org/abs/2208.02955'''

    def __init__(self, weights=None):
        super(ZLPRLoss, self).__init__()
        self.weights = weights

    def forward(self, logits, targets):
        pos_mask = targets == 1
        neg_mask = targets == 0

        # Compute exponential terms
        pos_exp = torch.exp(-logits) * pos_mask
        neg_exp = torch.exp(logits) * neg_mask

        # Apply weights before summing
        weighted_pos_exp = pos_exp * self.weights if self.weights is not None else pos_exp
        weighted_neg_exp = neg_exp * self.weights if self.weights is not None else neg_exp

        # Compute log-sum-exp
        pos_loss = torch.log(1+torch.sum(weighted_pos_exp, dim=1))
        neg_loss = torch.log(1+torch.sum(weighted_neg_exp, dim=1))

        # Combine and average the loss
        loss = pos_loss + neg_loss
        return loss.mean()


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=3, padding=1)
        # Max Pooling Layer 1
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(in_channels=9, out_channels=18, kernel_size=3, padding=1)
        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(in_channels=18, out_channels=36, kernel_size=3, padding=1)
        # Fully Connected Layer 1
        self.fc1 = nn.Linear(36 * 7 * 7, 1764)
        # Fully Connected Layer 2
        self.fc2 = nn.Linear(1764, 34)

    def forward(self, x):
        # Applying conv1 -> relu -> max pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Applying conv2 -> relu -> max pooling
        x = self.pool(F.relu(self.conv2(x)))
        # Applying conv3 -> relu -> max pooling
        x = self.pool(F.relu(self.conv3(x)))
        # Flattening the tensor for the fully connected layer
        x = x.view(-1, 36 * 7 * 7)
        # Applying fully connected layer 1 -> relu
        x = F.relu(self.fc1(x))
        # Applying fully connected layer 2 -> relu
        x = F.relu(self.fc2(x))
        return x


def train(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    total_accuracy = 0
    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0

    scaler = torch.cuda.amp.GradScaler(init_scale=64)
    for images, labels in tqdm(bg_iterator(dataloader, 32), total=len(dataloader), desc="Training"):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = loss_fn(outputs, labels)
        accuracy, tp, tn, fp, fn = accuracy_fn(labels, outputs)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
        if not torch.logical_or(total_norm.isnan(), total_norm.isinf()):
            # print mean and stdv model gradients to weight ratio of all layers except head
            grads_mean = [torch.mean(torch.abs(param.grad)) for param in model.parameters() if param.name is None and param.grad is not None]
            grads_stdv = [torch.std(torch.abs(param.grad)) for param in model.parameters() if param.name is None and param.grad is not None]
            data_mean = [torch.mean(torch.abs(param.data)) for param in model.parameters() if param.name is None and param.grad is not None]
            data_stdv = [torch.std(torch.abs(param.data)) for param in model.parameters() if param.name is None and param.grad is not None]
            # # Remove head weights
            # grads_head_mean = [torch.mean(torch.abs(weight.grad)) for weight in model.model.head.parameters() if weight.grad is not None]
            # grads_head_stdv = [torch.std(torch.abs(weight.grad)) for weight in model.model.head.parameters() if weight.grad is not None]
            # data_head_mean = [torch.mean(torch.abs(weight.data)) for weight in model.model.head.parameters() if weight.grad is not None]
            # data_head_stdv = [torch.std(torch.abs(weight.data)) for weight in model.model.head.parameters() if weight.grad is not None]
            # for g in grads_head_mean:
            #     grads_mean.remove(g)
            # for g in grads_head_stdv:
            #     grads_stdv.remove(g)
            # for d in data_head_mean:
            #     data_mean.remove(d)
            # for d in data_head_stdv:
            #     data_stdv.remove(d)

            if len(grads_mean) > 0:
                print(f"Grad Mean: {torch.mean(torch.stack(grads_mean)):.6f}, Grad Stdv: {torch.std(torch.stack(grads_stdv)):.6f}, Weight Mean: {torch.mean(torch.stack(data_mean)):.6f}, Weight Stdv: {torch.std(torch.stack(data_stdv)):.6f}")

            # print mean and stdv of head gradients to weight ratio
            # print(f"FGrad Mean: {torch.mean(torch.stack(grads_head_mean)):.6f}, FGrad Stdv: {torch.std(torch.stack(grads_head_stdv)):.6f}, FWeight Mean: {torch.mean(torch.stack(data_head_mean)):.6f}, FWeight Stdv: {torch.std(torch.stack(data_head_stdv)):.6f}")

        total_loss += loss.item()
        total_accuracy += accuracy
        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn
        print(f"Train Loss: {loss.item():.2f}, Train Accuracy: {accuracy:.2f}")
    return total_loss / len(dataloader), total_accuracy / len(dataloader), (total_tp / len(dataloader), total_tn / len(dataloader), total_fp / len(dataloader), total_fn / len(dataloader))


def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0

    with torch.no_grad():
        for images, labels in tqdm(bg_iterator(dataloader, 32), total=len(dataloader), desc="Evaluating"):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            accuracy, tp, tn, fp, fn = accuracy_fn(labels, outputs)
            total_loss += loss.item()
            total_accuracy += accuracy
            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn
            print(f"Val Loss: {loss.item():.2f}, Val Accuracy: {accuracy:.2f}")
    return total_loss / len(dataloader), total_accuracy / len(dataloader), (total_tp / len(dataloader), total_tn / len(dataloader), total_fp / len(dataloader), total_fn / len(dataloader))


def accuracy_fn(labels, outputs):
    '''Calculate the accuracy of the model.'''
    with torch.no_grad():
        predicted_positives = (torch.sigmoid(outputs) > 0.5).float()
        true_positives = (predicted_positives * labels).sum()
        true_negatives = ((1 - predicted_positives) * (1 - labels)).sum()
        false_positives = (predicted_positives * (1 - labels)).sum()
        false_negatives = ((1 - predicted_positives) * labels).sum()

        total_positives = labels.sum()
        total_negatives = (1 - labels).sum()

        weighted_correct = true_positives + true_negatives
        weighted_total = total_positives + total_negatives
        weighted_accuracy = weighted_correct / weighted_total

        tp_percent = 100 * true_positives / total_positives if total_positives > 0 else torch.tensor(0.)
        tn_percent = 100 * true_negatives / total_negatives if total_negatives > 0 else torch.tensor(0.)
        fp_percent = 100 * false_positives / total_negatives if total_negatives > 0 else torch.tensor(0.)
        fn_percent = 100 * false_negatives / total_positives if total_positives > 0 else torch.tensor(0.)

        # Print confusion matrix components, percentages, and weighted accuracy
        print(f"TP: {true_positives.item() / len(labels)} ({tp_percent.item():.2f}%), FP: {false_positives.item() / len(labels)} ({fp_percent.item():.2f}%), "
              f"TN: {true_negatives.item() / len(labels)} ({tn_percent.item():.2f}%), FN: {false_negatives.item() / len(labels)} ({fn_percent.item():.2f}%)")

    return 100*weighted_accuracy.item(), tp_percent.item(), tn_percent.item(), fp_percent.item(), fn_percent.item()


if __name__ == "__main__":
    # Set Hyperparameters
    num_epochs = 150
    lr = 3e-4
    weight_decay = 1e-5
    step_size = 6
    gamma = 0.55
    batch_size = 64
    img_size = 224
    precision_recall = 1

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the dataset
    train_data, val_data, test_data, weights = data_processing.load_dataset(img_size)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Define the model, loss function and optimizer
    print("Creating model...")
    model = CNN().to(device, non_blocking=True)
    # Set model to train only the head
    for param in model.model.parameters():
        param.requires_grad = True
    # for param in model.model.head.parameters():
    #     param.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos_weight = (weights * precision_recall).to(device, non_blocking=True)
    loss_fn = ZLPRLoss(pos_weight)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=step_size, threshold=1)

    # Statistics
    train_loss_list = []
    train_accuracy_list = []
    val_loss_list = []
    val_accuracy_list = []
    train_precision_recall_list = []
    val_precision_recall_list = []

    # Train the model
    print("Training model...")
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        # Train the model
        train_loss, train_accuracy, train_precision_recall = train(model, train_loader, loss_fn, optimizer, device)

        # Save the model with identifying information (model, current time, epoch, batch size, learning rate, weight decay, step size, gamma) in model folder
        # torch.save(model.state_dict(), f"model\{model_name}_{time.strftime('%Y-%m-%d_%H-%M-%S')}_{epoch}_{batch_size}_{lr}_{weight_decay}_{step_size}_{gamma}.pt")

        # Evaluate the model
        val_loss, val_accuracy, val_precision_recall = evaluate(model, val_loader, loss_fn, device)

        # Step the scheduler
        scheduler.step(val_loss)
        print(f"Learning rate: {scheduler.get_last_lr()}")

        # if epoch == 2:
        #     # Set model to train all layers
        #     for param in model.model.parameters():
        #         param.requires_grad = True
        #     optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=weight_decay)
        #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=step_size, threshold=1)

        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_accuracy)
        val_loss_list.append(val_loss)
        val_accuracy_list.append(val_accuracy)
        train_precision_recall_list.append(train_precision_recall)
        val_precision_recall_list.append(val_precision_recall)
        print(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.6f}, Val Accuracy: {val_accuracy:.2f}%")
        print(f"Train True Positives: ({train_precision_recall[0]:.2f})%, Train True Negatives: ({train_precision_recall[1]:.2f}%), "
              f"Train False Positives: ({train_precision_recall[2]:.2f}%), Train False Negatives: ({train_precision_recall[3]:.2f}%)")
        print(f"Val True Positives: ({val_precision_recall[0]:.2f}%), Val True Negatives: ({val_precision_recall[1]:.2f}%), "
              f"Val False Positives: ({val_precision_recall[2]:.2f}%), Val False Negatives: ({val_precision_recall[3]:.2f}%)")

    # Evaluate the model
    test_loss, test_accuracy, test_precision_recall = evaluate(model, test_loader, loss_fn, device)
    print(f"Test Loss: {test_loss:.6f}, Test Accuracy: {test_accuracy:.2f}%")
    print(f"Test True Positives: ({test_precision_recall[0]:.2f}%), Test True Negatives: ({test_precision_recall[1]:.2f}%), "
          f"Test False Positives: ({test_precision_recall[2]:.2f}%), Test False Negatives: ({test_precision_recall[3]:.2f}%)")

    # Plot the statistics in three subplots
    fig, axs = plt.subplots(4, 1, figsize=(10, 10))
    axs[0].plot(train_loss_list, label="Train Loss")
    axs[0].plot(val_loss_list, label="Val Loss")
    axs[0].set_title("Loss")
    axs[0].legend()
    axs[1].plot(train_accuracy_list, label="Train Accuracy")
    axs[1].plot(val_accuracy_list, label="Val Accuracy")
    axs[1].set_title("Accuracy")
    axs[1].legend()
    axs[2].plot([train_precision_recall_list[i] for i in range(num_epochs)],
                label=["Train True Positives", "Train True Negatives", "Train False Positives", "Train False Negatives"])
    axs[2].legend()
    axs[3].plot([val_precision_recall_list[i] for i in range(num_epochs)],
                label=["Val True Positives", "Val True Negatives", "Val False Positives", "Val False Negatives"])
    axs[3].legend()
    plt.show()
