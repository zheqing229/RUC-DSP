import torch.optim as optim
import torch
import torch.nn as nn
from model import ASTModel
from data import create_dataloaders



def train_model(model, data_loader, criterion, optimizer, scheduler, num_epochs=10, device='cuda:1'):
    model.to(device)
    model.train()  # 设置为训练模式

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            _, outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计损失和准确率
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(data_loader)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        # if epoch >= 5:
        #     scheduler.step()

    print("Training complete.")


def evaluate_model(model, data_loader, criterion, device='cuda:1'):
    model.to(device)
    model.eval()  # 设置为评估模式

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            _, outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 统计损失和准确率
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    loss = running_loss / len(data_loader)
    accuracy = correct / total
    print(f"Test Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    return loss, accuracy
