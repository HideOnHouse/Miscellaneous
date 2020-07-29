from hashlib import md5

from AE_Chunking import ae_chunk


def fuzzz(file_path_1, file_path_2, window_size):
    file_1_chunked = ae_chunk(file_path_1, window_size)
    file_2_chunked = ae_chunk(file_path_2, window_size)
    stream1 = open(file_path_1, 'rb').read()
    stream2 = open(file_path_2, 'rb').read()

    matched = 0

    for chunk_index1, chunk_index2 in zip(file_1_chunked, file_2_chunked):
        hash1 = md5(stream1[chunk_index1[0]:chunk_index2[1]]).hexdigest()
        hash2 = md5(stream2[chunk_index2[0]:chunk_index2[1]]).hexdigest()
        if hash1 == hash2:
            matched += 1

    return matched / len(file_1_chunked)


if __name__ == '__main__':
    file_1 = 'foo1.txt'
    file_2 = 'foo2.txt'
    similarity = fuzzz(file_1, file_2, 2)
    print("similarity : %s%%" % (similarity * 100))
