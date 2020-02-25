__author__ = "Ludde och Kevin"

import regex as re
import math
from collections import Counter


def words(text): return re.findall(r'\w+|<s>|<\/s>+', text.lower())


WORDS = Counter(words(open('d_selma.txt').read()))


def tokenize4(text):
    """uses the punctuation and symbols to break the text into words
    returns a list of words"""
    spaced_tokens = re.sub('([\p{S}\p{P}])', r' \1 ', text)
    one_token_per_line = re.sub('\s+', '\n', spaced_tokens)
    tokens = one_token_per_line.split()
    return tokens


# ----------------------------------------- Normilize and adds tags
def sentenize(text, file_out):
    pattern = re.compile(r'([A-ZÅÄÖ][^!?\.]*[!?\.])', re.M)
    sen_list = pattern.findall(text)
    new_text = ''
    new_sen_list = []

    for sen in sen_list:
        sen = '<s> ' + sen + ' </s>\n'
        sen = sen.lower()
        sen = re.sub('[!?\.(),\"\*-]', '', sen)
        new_text += sen
        new_sen_list.append(sen)

    new_text = new_text.lower()
    new_text = re.sub('[!?\.()\"\*-]', ' ', new_text)
    text_file = open(file_out, "w")
    text_file.write(new_text)
    text_file.close()

    return new_sen_list


# ---------------------------------------- UNIGRAM
def Pword(word, N=sum(WORDS.values())):
    return WORDS[word] / N


def Psentence(sen):
    product = 1
    for w in sen.split():
        p = Pword(w)
        product *= p
        print(w + '  ' + str(WORDS[w]) + '\t' + str(sum(WORDS.values())) + '\t' + str(p))

    return product


# ========================================== BIGRAM

def count_ngrams(words, n):
    ngrams = [tuple(words[inx:inx + n])
              for inx in range(len(words) - n + 1)]
    # "\t".join(words[inx:inx + n])
    frequencies = {}
    for ngram in ngrams:
        if ngram in frequencies:
            frequencies[ngram] += 1
        else:
            frequencies[ngram] = 1
    return frequencies


def P2sentence(sen, text_words):
    uni_freq = count_ngrams(text_words, 1)
    bi_freq = count_ngrams(text_words, 2)
    product = 1

    N = len(sen)
    for i in range(N):
        ci = uni_freq[(sen[i],)]

        if not i == N - 1:
            next_word = sen[i + 1]

            try:
                cici = bi_freq[(sen[i], sen[i + 1])]
                prob = cici / ci
            except:
                prob = ci / len(sen)

            product *= prob
            print(sen[i] + " " + str(next_word) + " " + str(cici) + " " + str(ci) + " " + str(prob))

    return product


# ------------------------------ MAIN
if __name__ == '__main__':

    # --- - Norm and tags
    text = open("selma.txt", 'r').read()
    token = tokenize4(text)
    sen_list = sentenize(text, "d_selma.txt")

    print("========= Unigram model ==============")
    print('wi\tC(wi)\t#words\tP(wi)')
    print("======================================")

    # for sen in sen_list:
    sen =  "det var en gång en katt som hette nils </s>"
    #print(sen)

    prob = Psentence(sen)

    print("======================================")
    print('Prob. unigrams: ' + str(prob))
    l = len(sen.split())
    geo = prob**(1/float(l))
    print('Geometric mean prob.: ' + str(geo))
    entropy = math.log2(prob) * (-1 / l)
    print('Entropy rate: ' + str(entropy))
    perplexity = math.pow(2, entropy)
    print('Perplexity: ' + str(perplexity))
    print()




    print("========= Bigram model ==============")
    print('wi\twi+wi+1\tCi,i+1\tC(i)\tP(wi+1|wi)')
    print("======================================")
    sen2 = "<s> det var en gång en katt som hette nils </s>"

    t2 = open("d_selma.txt", 'r').read()
    word_list = words(t2)

    prob2 = P2sentence(sen2.split(), word_list)

    print("======================================")
    geo = prob**(1/float(l))

    print("Prob. bigrams: " + str(prob2))
    geo = prob2**(1/float(l))

    print('Geometric mean prob.: ' + str(geo))
    entropy = math.log2(prob2) * (-1 / l)
    print("Entropy rate: " + str(entropy))
    perplexity = math.pow(2, entropy)
    print("Perplexity " + str(perplexity))


    # sort token.txt | uniq -c | sort -n
