import os


def ae_chunk(file_path, window_size):
    answer = list()
    with open(file_path, 'rb') as f:
        stream = f.read()
        temp = -1
        max_point = stream[0]
        chunked = [0]
        for idx in range(len(stream) - 1):
            temp += 1
            if stream[idx] > max_point:
                max_point = stream[idx]
                temp = 0
            elif temp == window_size:
                chunked.append(idx)
                answer.append(chunked)
                temp = -1
                chunked = [idx + 1]
                max_point = stream[idx + 1]
        chunked.append(idx + 1)
        answer.append(chunked)
    return answer


if __name__ == '__main__':
    fp = os.path.join(r'C:\Windows', 'notepad.exe')
    chunked_fp = ae_chunk(fp, 3)
    for each in chunked_fp:
        print(each)
