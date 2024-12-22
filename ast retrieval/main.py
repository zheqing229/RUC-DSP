import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.cuda.amp import autocast
import torch.optim as optim
import os
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import pickle
import librosa
import numpy as np

from model import ASTModel
from data import create_dataloaders
from train import train_model, evaluate_model
from evalute import match_query,evaluate

def _load_audio(file_path, sr=160000):
    signal, _ = librosa.load(file_path, sr=sr, mono=True)
    return signal, sr

def _parse_file_name(file_name):
    fold, _, _, target = file_name.split("-")
    target = int(target.split(".")[0])
    return int(fold), target

def _extract_features(signal, sr):
    mel_spec = librosa.feature.melspectrogram(
        y=signal, sr=sr, n_fft=6200, hop_length=1550, n_mels=128
    )

    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)  # 转换为dB单位
    mel_spec = torch.tensor(mel_spec)[:, :512]
    return mel_spec

def model_feature(model, input, device):
    with torch.no_grad():
        inputs = torch.tensor(input).to(device)
        inputs = inputs.unsqueeze(0)
        # print(inputs.shape)
        features, outputs = model(inputs)
        return features


if __name__ == '__main__':
    # 假设使用CUDA
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # 初始化模型
    ast_mdl = ASTModel(label_dim=50, verbose=True).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(ast_mdl.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.850)

    train_loader, query_loader = create_dataloaders(file_dir='audio/', query_fold=5, batch_size=32, num_workers=2)

    train_model(ast_mdl, train_loader, criterion, optimizer, scheduler, num_epochs=20, device=device)

    evaluate_model(ast_mdl, query_loader, criterion, device)

    torch.save(ast_mdl.state_dict(), "ast_model.pth")

    file_dir = 'audio/'
    cache_dir="cache"
    file_names = os.listdir('audio/')
    database = []
    query = []
    for file_name in tqdm(file_names, desc="Loading dataset", dynamic_ncols=True, unit="file"):
        file_path = os.path.join(file_dir, file_name)
        fold, target = _parse_file_name(file_name)
        cache_path = os.path.join(cache_dir, f"{file_name}_melspec.pkl")
        
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                features = pickle.load(f)
                features = model_feature(ast_mdl, features, device)
        else:
            signal, sr = _load_audio(file_path)
            features = _extract_features(signal, sr)
            features = model_feature(ast_mdl, features, device)
        
        if fold < 5:
            database.append((features, target))
        else:
            query.append((features, target))

    
    database_features = np.array([item[0].cpu().numpy() for item in database])
    database_labels = [item[1]for item in database]
    query_features = np.array([item[0].cpu().numpy() for item in query])
    query_labels = [item[1] for item in query]

    top_labels = match_query(query_features, database_features, database_labels, k=20)
    
    top_1_acc = evaluate(query_labels, top_labels, k=1)
    top_5_acc = evaluate(query_labels, top_labels, k=5)
    top_10_acc = evaluate(query_labels, top_labels, k=10)
    top_20_acc = evaluate(query_labels, top_labels, k=20)
    print(f"Top-1 Accuracy: {top_1_acc * 100:.2f}%")
    print(f"Top-5 Accuracy: {top_5_acc * 100:.2f}%")
    print(f"Top-10 Accuracy: {top_10_acc * 100:.2f}%")
    print(f"Top-20 Accuracy: {top_20_acc * 100:.2f}%")


