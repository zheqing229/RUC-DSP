from dataloader import build_dataset
from models import RN101, test
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config_test = {
        "dataset_path":r"D:\\24-25fall\\智能信息检索导论\\dataset\\ESC-50-master\\ESC-50-master",
        "dataset_type":"test",
        "batch_size":8,
        #unnecessary options:
        "sample_rate":16000,
        "device":"cpu",
        "n_fft":512, #帧长
        "hop_length":128, #帧移
        "n_mel":128, #滤波器组数量
        "preemphasis_coef":0.97
    }

test_dataloader = build_dataset(config_test)
print('Finish loading dataloader!')

for spec,label in test_dataloader:
    print(spec.shape)
    break


if __name__ == "__main__": 
    print('Input the model id(1 for clap encoder/0 for resnet encoder)')
    # model_id = int(input())
    model_id = 0
    print('test!')
    model = RN101().to(device)
    model.load_state_dict(torch.load(r'D:\checkpoints\RN0.845.pth', map_location=torch.device('cpu')))
    acc,_ = test(model, test_dataloader)
    print(acc)  #0.845（512/128）


# 不同setting下的比较
# model: load RN0.845.pth
# 帧长   帧移   test_acc
# 256    64     0.5025
# 256    128    0.5325
# 256    256    0.4325
# 512    64     0.51
# 512    128    0.845
# 512    256    0.6375
# 1024    64    0.2475
# 1024    128   0.5675
# 1024    256   0.6875
