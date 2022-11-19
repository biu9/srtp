import random

def load_data():
    # 原文件信息:(x,y,t),1为真人,0为机器人,最后两个数为目标坐标
    with open ('data/train.txt', 'r') as f:
        data = f.read().splitlines()
    trace = {
        'xyt':[],
        'label':[],
        'target':[]
    }

    # 打乱数据
    random.shuffle(data)

    for i in range(len(data)):
        trace['xyt'].append(data[i].split(' ')[1].split(';'))
        trace['label'].append(data[i].split(' ')[3])
        trace['target'].append(data[i].split(' ')[2])
        trace['xyt'][i].pop()
    
    return trace