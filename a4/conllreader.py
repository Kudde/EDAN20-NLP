"""
CoNLL-X and CoNLL-U file readers and writers
"""
__author__ = "Pierre Nugues"

import os
import operator
import regex as re


def get_files(dir, suffix):
    """
    Returns all the files in a folder ending with suffix
    Recursive version
    :param dir:
    :param suffix:
    :return: the list of file names
    """
    files = []
    for file in os.listdir(dir):
        path = dir + '/' + file
        if os.path.isdir(path):
            files += get_files(path, suffix)
        elif os.path.isfile(path) and file.endswith(suffix):
            files.append(path)
    return files


def read_sentences(file):
    """
    Creates a list of sentences from the corpus
    Each sentence is a string
    :param file:
    :return:
    """
    f = open(file).read().strip()
    sentences = f.split('\n\n')
    return sentences


def split_rows(sentences, column_names):
    """
    Creates a list of sentence where each sentence is a list of lines
    Each line is a dictionary of columns
    :param sentences:
    :param column_names:
    :return:
    """
    new_sentences = []
    root_values = ['0', 'ROOT', 'ROOT', 'ROOT', 'ROOT', 'ROOT', '0', 'ROOT', '0', 'ROOT']
    start = [dict(zip(column_names, root_values))]
    for sentence in sentences:
        rows = sentence.split('\n')
        sentence = [dict(zip(column_names, row.split())) for row in rows if row[0] != '#']
        sentence = start + sentence
        new_sentences.append(sentence)
    return new_sentences


def save(file, formatted_corpus, column_names):
    f_out = open(file, 'w')
    for sentence in formatted_corpus:
        for row in sentence[1:]:
            # print(row, flush=True)
            for col in column_names[:-1]:
                if col in row:
                    f_out.write(row[col] + '\t')
                else:
                    f_out.write('_\t')
            col = column_names[-1]
            if col in row:
                f_out.write(row[col] + '\n')
            else:
                f_out.write('_\n')
        f_out.write('\n')
    f_out.close()


def getSV(formatted_corpus):
    freq = {}
    for sen in formatted_corpus:

        for word in sen:
            # print(word)

            if word["deprel"] == "SS":
                subWord = word['form'].lower()
                vIndex = int(word["head"])
                verb = sen[vIndex]
                # print('Verb' + str(verb))
                verbWord = verb["form"].lower()

                if (subWord, verbWord) in freq:
                    freq[(subWord, verbWord)] += 1
                else:
                    freq[(subWord, verbWord)] = 1

    sort_freq = sorted(freq.items(), key=operator.itemgetter(1), reverse=True)
    return sort_freq


def getSVO(formatted_corpus):
    freq = {}
    for sen in formatted_corpus:

        for word in sen:
            # print(word)

            if word["deprel"] == "SS":
                subWord = word['form'].lower()
                vIndex = int(word["head"])
                verb = sen[vIndex]
                # print('Verb' + str(verb))

                for w in sen:
                    # print(obj)

                    if w["deprel"] == "OO" and w["head"] == verb['id']:
                        verbWord = verb['form'].lower()
                        objWord = w["form"].lower()

                        if (subWord, verbWord, objWord) in freq:
                            freq[(subWord, verbWord, objWord)] += 1
                        else:
                            freq[(subWord, verbWord, objWord)] = 1

    sorted_freq = sorted(freq.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_freq


def getMulti(formatted_corpus):
    freq = {}
    for sen in formatted_corpus:

        for word in sen:
            # print(word)

            if word["deprel"] == "nsubj":
                subWord = word['form'].lower()
                vIndex = int(word["head"])
                verb = sen[vIndex]

                for x in sen:

                    if x["deprel"] == "obj" and x["head"] == verb['id']:
                        # print(x)
                        objWord = x["form"].lower()
                        verbWord = verb['form'].lower()

                        if (subWord, verbWord, objWord) in freq:
                            freq[(subWord, verbWord, objWord)] += 1
                        else:
                            freq[(subWord, verbWord, objWord)] = 1

    sorted_freq = sorted(freq.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_freq

def get_files(dir, suffix):
    f = []

    for subdir, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(suffix):
                f.append(subdir + "/" + file)

    return f


if __name__ == '__main__':
    column_names_2006 = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']
    train_file = './swedish_talbanken05_train.conll'
    # train_file = 'test_x'
    test_file = './swedish_talbanken05_test.conll'

    sentences = read_sentences(train_file)
    formatted_corpus = split_rows(sentences, column_names_2006)
    print(train_file, len(formatted_corpus))
    print(formatted_corpus[0])

    sv = getSV(formatted_corpus)
    print("\n-------- Subject - Verb ---------")
    for s in sv[:5]:
        print(s)

    svo = getSVO(formatted_corpus)
    print("\n-------- Subject - Verb - Object ---------")
    for s in svo[:5]:
        print(s)

    print("\n-------- Multilingual Corpora ---------")
    column_names_u = ['id', 'form', 'lemma', 'upostag', 'xpostag', 'feats', 'head', 'deprel', 'deps', 'misc']

    files = get_files("ud-treebanks-v2.2", "train.conllu")
    # for f in files:
        # print(f)

    for train_file in files:
        print(re.search(r'(?<=UD_)(.*)(?=\/)', train_file).group(1))
        sentences = read_sentences(train_file)
        formatted_corpus = split_rows(sentences, column_names_u)
        result = getMulti(formatted_corpus)
        for r in result[:5]:
            print(r)

    print("K. Done . Bye")
