from model import MyAlexNet
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from datasets import load_datasets

data_path = "D:/Works/Data/"
momentum = 0.5
train_epoch = 300
print_interval = 30  # 间隔print_interval的batch打印一次loss信息

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 256
    train_dataloader, test_dataloader = load_datasets("cifar100", data_path, batch_size)
    model = MyAlexNet(100)
    if device == 'cuda':
        model = model.cuda()
    sgd = SGD(model.parameters(), lr=1e-2, momentum=momentum)
    loss_fn = CrossEntropyLoss()
    if device == 'cuda':
        loss_fn = loss_fn.cuda()

    train_loss = []
    train_loss_counter = []
    for current_epoch in range(train_epoch):
        print("---start the {}th train---".format(current_epoch))
        model.train()
        for idx, (data, target) in enumerate(train_dataloader):
            if device == 'cuda':
                data = data.cuda()
                target = target.cuda()
            sgd.zero_grad()  # 每一个batch清空sgd参数
            y_hat = model(data.float())
            # y_predict = y_hat.argmax(-1)
            loss = loss_fn(y_hat, target.long())
            if idx % print_interval == 0:
                print('idx: {}, loss: {}'.format(idx, loss.sum().item()))
                train_loss.append(loss.sum().item())
                train_loss_counter.append(idx * batch_size + (current_epoch * len(train_dataloader.dataset)))
            loss.backward()
            sgd.step()

        all_correct_num = 0
        all_sample_num = 0
        model.eval()
        for idx, (data, target) in enumerate(test_dataloader):
            if device == 'cuda':
                data = data.cuda()
                target = target.cuda()
            y_hat = model(data.float()).detach()
            y_predict = y_hat.argmax(-1)

            current_correct_num = y_predict == target
            current_correct_num = current_correct_num.cpu()
            all_correct_num += np.sum(current_correct_num.numpy(), axis=-1)
            all_sample_num += current_correct_num.shape[0]
        acc = all_correct_num / all_sample_num
        #     correct_predict_num += y_predict == target
        # all_sample_num = len(target)
        # acc = correct_predict_num / all_sample_num
        print('accuracy: {:.2f}'.format(acc))
        print("---save the model---")
        torch.save(model, './model/cifar100_{:.2f}.pkl'.format(acc))
        if acc > 0.98:
            break
    plt.plot(train_loss_counter, train_loss)
    plt.show()
