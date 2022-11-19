from data.load_data import load_data
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = load_data()
    print(len(data['xyt'][0]))
    print((data['label'][0]))
    '''
    # init visualize trace
    for i in range(len(data['xyt'])):
        plt.plot([float(x.split(',')[0]) for x in data['xyt'][i]], [float(x.split(',')[1]) for x in data['xyt'][i]])
        plt.savefig('visualize/init_data/{}.png'.format(i))
        plt.close()
    '''