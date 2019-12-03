# 实现二分类的感知器算法

import numpy as np


# 感知器算法
def perceptron(w1_data, w2_data, C, w_default):
    # 首先将w1和w2的训练数据取增广形式
    w1 = np.array([])
    w2 = np.array([])

    for i in range(0, w1_data.shape[0]):
        w1 = np.append(w1, np.append(w1_data[i], 1))
        w2 = np.append(w2, np.append(w2_data[i], 1))

    w1 = w1.reshape(w1_data.shape[0], w1_data.shape[1]+1)
    w2 = w2.reshape(w2_data.shape[0], w2_data.shape[1]+1)

    # 然后将w2的全部分量乘-1
    w2 = w2 * -1

    # 将w1和w2整合成一个训练集X
    X = w1[:]
    X = np.append(X, w2)
    X = X.reshape(w1.shape[0]+w2.shape[0], w1.shape[1])

    # 迭代过程
    counter = 1  # 迭代计数器
    w = w_default  # 待学习到的权重
    while(True):
        print('第', counter, '轮迭代：')
        error_flag = False  # 在迭代过程中是否产生了分类错误
        for i in range(0, X.shape[0]):
            print('w的取值：',w)
            if np.dot(w, X[i]) <= 0:
                w = w + C * X[i]
                error_flag = True
        if(error_flag == False):
            return w
        counter+=1


# main函数
def main():
    # 自定义w1和w2的训练数据，维数和数据个数不限，但w1和w2的数据个数需要保持一致
    w1_data = [[0, 0, 0],
               [1, 0, 0],
               [1, 0, 1],
               [1, 1, 0]]
    w2_data = [[0, 0, 1],
               [0, 1, 1],
               [0, 1, 0],
               [1, 1, 1]]
    # w1_data = [[0, 0],
    #            [0, 1]]
    # w2_data = [[1, 0],
    #            [1, 1]]
    w1_data = np.array(w1_data)
    w2_data = np.array(w2_data)

    C = 1  # 设置矫正增量C
    w_default = [0, 0, 0, 0]  # 设置初始的权重
    w_default = np.array(w_default)

    w = perceptron(w1_data, w2_data, C, w_default)

    # 格式化输出判别函数
    result_formula = 'd(x)='
    for i in range(0, w.size):
        if i==0:
            if w[i] != 0:
                result_formula += str(w[i]) + 'x' + str(i+1)
        elif i ==w.size-1:
            if w[i]>0:
                result_formula += '+' + str(w[i])
            elif w[i] < 0:
                result_formula += str(w[i])
            else:
                result_formula == result_formula
        else:
            if w[i]>0:
                result_formula += '+' + str(w[i]) + 'x' + str(i+1)
            elif w[i] < 0:
                result_formula += str(w[i]) + 'x' + str(i+1)
            else:
                result_formula == result_formula
    print(result_formula)


if __name__=='__main__':
    main()