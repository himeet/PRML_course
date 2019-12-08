# 实现两类模式的高斯贝叶斯分类器，即两类的训练数据均服从高斯分布
# 同时应该满足两类的协方差矩阵相等
import numpy as np
import math
import matplotlib.pyplot as plt


# Bayes分类器核心
def gauss_bayes_classifier(w1_data, w2_data, p_w1, p_w2):
    # 计算二者的均值向量
    m1 = np.zeros(w1_data.shape[1])  # w1类别的均值向量
    m2 = np.zeros(w2_data.shape[1])
    for i in range(0, w1_data.shape[0]):
        m1 += w1_data[i]
    m1 = m1 / w1_data.shape[0]
    for i in range(0, w2_data.shape[0]):
        m2 += w2_data[i]
    m2 = m2 / w2_data.shape[0]

    # 计算二者的协方差矩阵
    C1 = np.zeros((w1_data.shape[1], w1_data.shape[1]))
    C2 = np.zeros((w2_data.shape[1], w2_data.shape[1]))
    for i in range(0, w1_data.shape[0]):
        C1 += matrix_multi_with_vector(w1_data[i]-m1, w1_data[i]-m1)
    C1 = C1 / w1_data.shape[0]
    for i in range(0, w2_data.shape[0]):
        C2 += matrix_multi_with_vector(w2_data[i]-m2, w2_data[i]-m2)
    C2 = C2 / w2_data.shape[0]
    if((C1==C2).all()):
        C = C1[:]
        C_inv = np.linalg.inv(C)  # C的逆矩阵
        result_formula = ''

        # 根据公式计算出二者的判别界面
        # 计算常数项
        constant0 =  math.log(p_w1, math.e) -  math.log(p_w2, math.e)
        constant1 = -0.5 * np.dot(np.dot(m1, C_inv), m1)
        constant2 = 0.5 * np.dot(np.dot(m2, C_inv), m2)
        constant_sum = constant0 + constant1 + constant2

        # 计算x项
        coff_vec = np.dot(m1-m2, C_inv)  # x的系数向量
        for i in range(0, coff_vec.size):
            if coff_vec[i] < 0:
                result_formula += str(coff_vec[i]) + 'x' + str(i+1)
            elif coff_vec[i] == 0:
                result_formula = result_formula
            else:
                if i==0:
                    result_formula += str(coff_vec[i]) + 'x' + str(i+1)
                else:
                    result_formula += '+' + str(coff_vec[i]) + 'x' + str(i + 1)
        if(constant_sum < 0):
            result_formula += str(constant_sum)
        elif(constant_sum==0):
            result_formula = result_formula
        else:
            result_formula += '+' + str(constant_sum)
        result_formula += '=0'
        coff_vec = np.append(coff_vec, constant_sum)
        return result_formula, coff_vec
    else:  # 不满足C1=C2的条件
        return None


# 两个一维向量实现矩阵的乘法
def matrix_multi_with_vector(vec1, vec2):
    matrix = np.zeros((vec1.size, vec2.size))
    for i in range(0, vec1.size):
        for j in range(0, vec2.size):
            matrix[i][j] = vec1[i] * vec2[j]
    return matrix


# 可视化判别界面
def plot(w1_x, w1_y, w2_x, w2_y, coff_vec):
    fig = plt.figure()
    ax1 = fig.add_subplot()
    # 绘制训练样本点
    plt.scatter(w1_x, w1_y, c='b')
    plt.scatter(w2_x, w2_y, c='r')
    # 绘制判别界面
    x = w1_x[:]
    x = np.append(x, w2_x[:])
    y = -(coff_vec[0]*x + coff_vec[2]) / coff_vec[1]
    plt.plot(x, y, c='g')
    plt.show()
# main函数
def main():
    w1_data = [[0, 0],
               [2, 0],
               [2, 2],
               [0, 2]]
    w2_data = [[4, 4],
               [6, 4],
               [6, 6],
               [4, 6]]
    w1_data = np.array(w1_data)
    w2_data = np.array(w2_data)
    p_w1 = 0.5  # w1类别的先验概率
    p_w2 = 0.5  # w2类别的先验概率

    result_formula, coff_vec = gauss_bayes_classifier(w1_data, w2_data, p_w1, p_w2)
    if result_formula is None:
        print('该题目不满足C1=C2的条件，即两类的协方差矩阵不相同')
    else:
        print('判别界面为：\n', result_formula)
        w1_x = w1_data[:,0]
        w1_y = w1_data[:,1]
        w2_x = w2_data[:,0]
        w2_y = w2_data[:,1]
        plot(w1_x, w1_y, w2_x, w2_y, coff_vec)


if __name__=='__main__':
    main()