import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torchvision.models import resnet101
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class RN101(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet101(pretrained=True)
        self.input_head = nn.Conv2d(1, 3, 7, padding=3)
        self.output_head = nn.Linear(1000, 50)
    
    def forward(self, x:torch.Tensor):
        x = x.unsqueeze(1).to(torch.float32)
        x = self.input_head(x)
        x = self.backbone(x)
        return self.output_head(x)


def test(model, dataloader):
    result = []
    target = []
    model.eval()
    with torch.no_grad():
        for spec, label in tqdm(dataloader):
            output = model(spec.float().to(device)).cpu()
            output = output.reshape((-1, 10, 50)).mean(dim=1)
            pred = output.argmax(dim=1)
            result.append(pred)
            target.append(label)
    result = torch.cat(result)
    target = torch.cat(target)
    acc = accuracy_score(target, result)
    model.train()
    return acc,result

class Trainer():
    def __init__(self, model:nn.Module, train_dataloader, test_dataloader, lr=1e-5):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optim = torch.optim.Adam(lr=lr, params=self.model.parameters())


    def train(self, epoches=100):
        best_acc = .0
        for _ in range(epoches):
            bar = tqdm(self.train_dataloader)
            for spec, label in bar:
                output = self.model(spec.to(device))
                loss = F.cross_entropy(output, label.to(device))

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
                bar.set_description(f"loss:{loss.item()}")
            acc,_ = test(self.model, self.test_dataloader)
            print(acc)
            if acc>best_acc:
                best_acc = acc
                torch.save(self.model.state_dict(), "RN{}.pth".format(acc))


if __name__ == "__main__":
    from dataset import build_dataset
    config_train = {
    "dataset_path":r"D:\\24-25fall\\智能信息检索导论\\dataset\\ESC-50-master\\ESC-50-master",
    "dataset_type":"train",
    "batch_size":32,
    #unnecessary options:
    "sample_rate":16000,
    "device":"cpu",
    "n_fft":512,
    "hop_length":128, #帧移
    "n_mel":128,
    "preemphasis_coef":0.97
}
    config_test = {
    "dataset_path":r"D:\\24-25fall\\智能信息检索导论\\dataset\\ESC-50-master\\ESC-50-master",
    "dataset_type":"test",
    "batch_size":8,
    #unnecessary options:
    "sample_rate":16000,
    "device":"cpu",
    "n_fft":512,
    "hop_length":128, #帧移
    "n_mel":128,
    "preemphasis_coef":0.97
}
    train_ = 0
    test_dataloader = build_dataset(config_test)
    model = RN101().to(device)
    if train_:
        train_dataloader = build_dataset(config_train)
        trainer = Trainer(
        model, 
        train_dataloader, 
        test_dataloader,
        lr=1e-4)
        trainer.train()
    else:
        model.load_state_dict(torch.load(r'D:\checkpoints\RN0.845.pth', map_location=torch.device('cpu')))
        acc,_ = test(model, test_dataloader)
        print(acc)