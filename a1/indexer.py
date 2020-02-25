import math
import os
import regex as re
import pickle
from scipy import spatial



def get_files(dir, suffix):
    files = []
    for file in os.listdir(dir):
        if file.endswith(suffix):
            files.append(file)
    return files


def words(text): return re.finditer(r'\w+', text.lower())


def create_dictionary(wordlist):
    word_index = {}

    for word in wordlist:
        try:
            word_index[word.group()].append(word.start())
        except:
            word_index[word.group()] = [word.start()]
    return word_index


def save_pickle(dictionary, filename):
    save_path = "./pickle/" + filename
    print(save_path)
    pickle.dump(dictionary, open(save_path, "wb"))
    # read = pickle.load(open(save_path, "rb"))
    # print(read)


def create_pickles(files):
    for file in files:
        text = open("./Selma/" + file, 'r').read()
        wordlist = words(text)

        word_dict = create_dictionary(wordlist)
        save_pickle(word_dict, file.strip(".txt") + '.p')


def create_master_indexer(files):
    master_index = {}
    for file in files:
        save_path = "./pickle/" + file.strip(".txt") + '.p'
        file_index = pickle.load(open(save_path, "rb"))

        for key in file_index:
            if master_index.get(key) is None:
                index = {}
            else:
                index = master_index.get(key)

            index[file] = file_index[key]
            master_index[key] = index
            #if key == 'ände':
               # print(key)
               # print(master_index[key])

    save_pickle(master_index, "master_index.p")
    return master_index


def create_tf_idf(files, master_index):
    master_tf_idf = {}
    for file in files:
        save_path = "./pickle/" + file.strip(".txt") + '.p'
        file_index = pickle.load(open(save_path, "rb"))
        total_words = len(file_index)
        total_doc = len(files)
        print(file)


        file_tfidf = {}
        list_tfidf = []
        for key in master_index:
            # file_tfidf = {}
            if key in file_index:
                # tf = n / N
                freq = len(file_index[key]) / total_words

                # idf = log_e( total nbr of documents / total nbt of documents with t)
                doc_with_t = len(master_index[key])
                idf = math.log(total_doc/doc_with_t)
                # out = key + ' ' + str(freq*idf)
                out = freq*idf
            else:
                # out = key + ' ' + str(0.0)
                out = 0.0

            # file_tfidf[key] = out
            list_tfidf.append(out)
            # if key == 'nils':
            # print('\t' + str(out))

        # print('SIZE ' + str(len(list_tfidf)))
        master_tf_idf[file] = list_tfidf

    return master_tf_idf



def create_matrix(master_tf_df):
    print("")
    print('\t'*0 + '\t'.join(master_tf_df.keys()))

    for key_i in master_tf_df:

        # print(key_i.strip('.txt') + '\t'*3, end="")

        for key_j in master_tf_df:
            di = master_tf_df[key_i]
            dj = master_tf_df[key_j]
            res = 1 - spatial.distance.cosine(di, dj)
            print("%.3f" % res, end="\t")

        print('')

    print('hej då')


# START
files = get_files("Selma", "txt")
create_pickles(files)
m_index = create_master_indexer(files)
m_tf_idf = create_tf_idf(files, m_index)
create_matrix(m_tf_idf)




