import hashlib
import numpy as np
from PIL import Image


def convert(array):
    img = Image.fromarray(np.uint8(array)).convert('RGB')
    img.show()


def generator(string):
    temp = ""
    for i in string:
        temp += str(hashlib.sha256(str.encode(i)).hexdigest())
    for i in temp:
        temp += str(hashlib.sha512(str.encode(i)).hexdigest())
    for i in temp:
        temp += str(hashlib.md5(str.encode(i)).hexdigest())
    temp = list(str.encode(temp))
    temp = np.uint8(temp[:480000])
    temp = temp.reshape(400, 400, 3)
    return temp


def main():
    name = 'sha256'
    temp = generator(name)
    convert(temp)


if __name__ == '__main__':
    main()
