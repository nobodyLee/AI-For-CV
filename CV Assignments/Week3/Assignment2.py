"""Logistic Regression
"""
import random
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def inference(w, b, x):
    """Run model after training."""
    return sigmoid(w * x + b)


def eval_loss(w, b, x_list, gt_y_list):
    """evaluate loss function."""
    total_loss = sum(map(lambda x, y: -y * np.log(sigmoid(w * x + b)) - (1 - y) * np.log(1 - sigmoid(w * x + b)),
                         x_list, gt_y_list))
    return total_loss / len(x_list)


def gradient(pred_y, gt_y, x):
    diff = pred_y - gt_y
    dw = diff * x
    db = diff
    return dw, db


def cal_step_gradient(batch_x_list, batch_gt_y_list, w, b, lr):
    batch_size = len(batch_x_list)
    pred_y = list(map(inference, [w] * batch_size, [b] * batch_size, batch_x_list))
    dw_db = list(map(gradient, pred_y, batch_gt_y_list, batch_x_list))
    avg_dw = sum([i[0] for i in dw_db]) / batch_size
    avg_db = sum([i[1] for i in dw_db]) / batch_size
    w -= lr * avg_dw
    b -= lr * avg_db
    return w, b


def train(x_list, gt_y_list, batch_size, lr, max_iter):
    w, b = 0, 0
    num_samples = len(x_list)
    for i in range(max_iter):
        batch_idxs = np.random.choice(num_samples, batch_size)
        batch_x = [x_list[j] for j in batch_idxs]
        batch_y = [gt_y_list[j] for j in batch_idxs]
        w, b = cal_step_gradient(batch_x, batch_y, w, b, lr)
        print('w:{0}, b:{1} loss is {2}'.format(w, b, eval_loss(w, b, x_list, gt_y_list)))
    return w, b


def gen_sample_data(num):
    w = random.random()
    b = random.random()
    x_list = [random.random() for _ in range(num)]
    y_list = [sigmoid(w * x_list[i] + b) + 0.01*random.random() * random.randint(-1, 1) for i in range(num)]
    return x_list, y_list, w, b


def run():
    x_list, y_list, w0, b0 = gen_sample_data(100)
    batch_size, lr, max_iter = 50, 0.1, 1000
    w, b = train(x_list, y_list, batch_size, lr, max_iter)
    # Plot final result
    x = [i*0.01 for i in range(100)]
    y = [1/(1+np.exp(-w*i-b)) for i in x]
    plt.plot(x_list, y_list, '.')
    plt.plot(x, y)
    plt.title('{}*x+{}'.format(w, b))
    plt.show()


if __name__ == '__main__':
    run()
