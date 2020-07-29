import hashlib


def get_index(value):
    temp = str.encode(value)
    temp = hashlib.md5(temp)
    temp = int(temp.hexdigest(), 16)
    index = temp % table_size
    return index


def get_sign(value):
    """

    :param value:
    :return: minus if 0, else 1
    """
    temp = str.encode(value)
    temp = hashlib.sha256(temp)
    temp = int(temp.hexdigest(), 16)
    if temp % 2 == 0:
        sign = 1
    else:
        sign = -1

    return sign


def get_hashed_feature(value):
    hash_table = [0 for i in range(table_size)]
    index = get_index(value)
    sign = get_sign(value)
    hash_table[index] += sign
    return hash_table


def main():
    values = ['dog', 'cat', 'horse', 'cow', 'sheep', 'pig', 'chicken', 'duck']
    for value in values:
        print(get_hashed_feature(value))


if __name__ == '__main__':
    table_size = 20
    main()
