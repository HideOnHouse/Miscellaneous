from matplotlib import pyplot as plt

if __name__ == '__main__':
    """
    모든 원소가 1인 A벡터와 -1인 B벡터의 차원에 따른 코사인유사도, 유클리드유사도, 맨하탄유사도의 비교
    """
    cosine_list = []
    euclid_list = []
    manhattan_list = []
    for i in range(2, 102):
        A = list(1 for i in range(i))
        B = list(-1 for i in range(i))
        cosine = sum((A[i] * B[i]) for i in range(len(A))) / (sum(A[i] ** 2 for i in range(len(A))))
        euclid = sum((A[i] - B[i]) ** 2 for i in range(len(A))) ** (1 / 2)
        manhattan = sum((A[i] - B[i]) for i in range(len(A)))
        cosine_list.append(cosine)
        euclid_list.append(euclid)
        manhattan_list.append(manhattan)

    x_list = list(i for i in range(1, len(cosine_list) + 1))
    plt.plot(x_list, manhattan_list, label='manhattan')
    plt.plot(x_list, euclid_list, label='euclid')
    plt.plot(x_list, cosine_list, label='cosine')
    plt.legend()
    plt.ylim(-2, 25)
    plt.xlabel("Vector Dimension")
    plt.ylabel("Dissimilarity")
    plt.show()
