from data.load_data import load_data
import torch
import torch.nn as nn
from tqdm import tqdm
from matplotlib import pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 放在cuda或者cpu上训练
TRAIN_PERCENTAGE = 0.7
TEST_PERCENTAGE = 0.3

config = {
    'hidden_dim':128,
    'layer_dim':2,
}

class MY_LSYM(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,layer_dim):
        super(MY_LSYM,self).__init__()
        self.lstm = nn.LSTM(input_dim,hidden_dim,layer_dim,batch_first=True).to(DEVICE)
        self.fc = nn.Linear(hidden_dim,output_dim).to(DEVICE)
    def forward(self,x):
        out,(hn,cn) = self.lstm(x)
        out = self.fc(out[:,-1,:])
        return out

def train():
    total_data = load_data()['xyt']
    total_label = load_data()['label']

    for i in range(len(total_label)):
        total_label[i] = int(total_label[i])
        if(total_label[i] == 1):
            total_label[i] = [0.0,1.0]
        else:
            total_label[i] = [1.0,0.0]
        
    for i in range(len(total_data)):
        for j in range(len(total_data[i])):
            total_data[i][j] = total_data[i][j].split(',')
            total_data[i][j] = [float(x) for x in total_data[i][j]]
        total_data[i] = torch.tensor(total_data[i]).to(DEVICE)
        total_data[i] = torch.unsqueeze(total_data[i],0)

    # 划分数据集
    train_data = total_data[:int(len(total_data)*TRAIN_PERCENTAGE)]
    train_label = total_label[:int(len(total_label)*TRAIN_PERCENTAGE)]
    test_data = total_data[int(len(total_data)*TEST_PERCENTAGE):]
    test_label = total_label[int(len(total_label)*TEST_PERCENTAGE):]
    model = MY_LSYM(3,128,2,1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # 预期的输入[batch_size,len,3]
    EPOCHS = 10
    loss_func = nn.CrossEntropyLoss()
    for epoch in tqdm(range(EPOCHS)):
        for i in range(len(train_data)):
            optimizer.zero_grad()
            output = model(train_data[i])
            output = output[0]
            loss = loss_func(output,torch.tensor(train_label[i]).to(DEVICE))
            loss.backward()
            optimizer.step()
            #print('epoch:{},loss:{}'.format(epoch,loss))
            #plt.plot(i,loss.item(),'r.')
        #plt.show()
        #plt.savefig('./result/train_loss/epoch_{}.png'.format(epoch))
    
    # save model
    torch.save(model.state_dict(), './result/model.pth')
    
    # 测试
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(len(test_data)):
            output = model(test_data[i])
            output = output[0]
            _, predicted = torch.max(output.data, 0)
            total += 1
            if(test_label[i][0] == 1):
                label = 0
            else:
                label = 1
            correct += (predicted == label)
            #print('predicted:{},label:{}'.format(predicted,label))
    print('correct:{},total:{},acc:{}'.format(correct,total,correct/total))
    print('Accuracy of the network on the test data: %d %%' % (100 * correct / total))

train()