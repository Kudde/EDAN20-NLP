"""
Gold standard parser
"""
__author__ = "Pierre Nugues"

import transition
import conll
import pickle


from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
from sklearn import linear_model
from sklearn import metrics
from sklearn import tree
from sklearn import preprocessing


def reference(stack, queue, graph):
    """
    Gold standard parsing
    Produces a sequence of transitions from a manually-annotated corpus:
    sh, re, ra.deprel, la.deprel
    :param stack: The stack
    :param queue: The input list
    :param graph: The set of relations already parsed
    :return: the transition and the grammatical function (deprel) in the
    form of transition.deprel
    """
    # Right arc
    if stack and stack[0]['id'] == queue[0]['head']:
        # print('ra', queue[0]['deprel'], stack[0]['cpostag'], queue[0]['cpostag'])
        deprel = '.' + queue[0]['deprel']
        stack, queue, graph = transition.right_arc(stack, queue, graph)
        return stack, queue, graph, 'ra' + deprel
    # Left arc
    if stack and queue[0]['id'] == stack[0]['head']:
        # print('la', stack[0]['deprel'], stack[0]['cpostag'], queue[0]['cpostag'])
        deprel = '.' + stack[0]['deprel']
        stack, queue, graph = transition.left_arc(stack, queue, graph)
        return stack, queue, graph, 'la' + deprel
    # Reduce
    if stack and transition.can_reduce(stack, graph):
        for word in stack:
            if (word['id'] == queue[0]['head'] or
                        word['head'] == queue[0]['id']):
                # print('re', stack[0]['cpostag'], queue[0]['cpostag'])
                stack, queue, graph = transition.reduce(stack, queue, graph)
                return stack, queue, graph, 're'
    # Shift
    # print('sh', [], queue[0]['cpostag'])
    stack, queue, graph = transition.shift(stack, queue, graph)
    return stack, queue, graph, 'sh'


def extract(stack, queue, graph, feature_names, sentence):
    features = {}

    # INIT
    features["next_word_POS"] = "nil"
    features["next_word_word"] = "nil"
    features["stack0_POS"] = "nil"
    features["stack0_word"] = "nil"
    features["stack1_POS"] = "nil"
    features["stack1_word"] = "nil"
    features["queue0_POS"] = "nil"
    features["queue0_word"] = "nil"
    features["queue1_POS"] = "nil"
    features["queue1_word"] = "nil"

    # STACK
    if len(stack) > 0:
        features["stack0_POS"] = stack[0]["postag"]
        features["stack0_word"] = stack[0]["form"]
        features["next_word_POS"] = sentence[int(stack[0]['id']) + 1]['postag']
        features["next_word_word"] = sentence[int(stack[0]['id']) + 1]['form']

    if len(stack) > 1:
        features["stack1_POS"] = stack[1]["postag"]
        features["stack1_word"] = stack[1]["form"]

    # QUEUE
    if len(queue) > 0:
        features["queue0_POS"] = queue[0]["postag"]
        features["queue0_word"] = queue[0]["form"]

    if len(queue) > 1:
        features["queue1_POS"] = queue[1]["postag"]
        features["queue1_word"] = queue[1]["form"]

    if len(queue) > 2:
        features["queue2_POS"] = queue[2]["postag"]
        features["queue2_word"] = queue[2]["form"]

    features["can-re"] = str(transition.can_reduce(stack, graph))
    features["can-la"] = str(transition.can_leftarc(stack, graph))

    r = {}
    for f in feature_names:
        r[f] = features[f]

    return r


def extract_f(file):
    column_names_2006 = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']
    column_names_2006_test = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats']
    feature_names = ["stack0_POS", "stack0_word", "queue0_POS", "queue0_word", "can-re", "can-la"]
    feature_names2 = ["stack0_POS", "stack0_word", "stack1_POS", "stack1_word",
                      "queue0_POS", "queue0_word", "queue1_POS", "queue1_word",
                      "can-re", "can-la"]
    feature_names3 = ["stack0_POS", "stack0_word", "stack1_POS", "stack1_word",
                      "queue0_POS", "queue0_word", "queue1_POS", "queue1_word", "queue1_POS", "queue1_word",
                      "can-re", "can-la", "next_word_POS", "next_word_word"]

    sentences = conll.read_sentences(file)
    formatted_corpus = conll.split_rows(sentences, column_names_2006)
    sent_cnt = 0
    x = []
    y = []

    for sentence in formatted_corpus:
        sent_cnt += 1
        stack = []
        queue = list(sentence)
        graph = {}
        graph['heads'] = {}
        graph['heads']['0'] = '0'
        graph['deprels'] = {}
        graph['deprels']['0'] = 'ROOT'
        transitions = []

        while queue:
            featureRow = extract(stack, queue, graph, feature_names3, sentence)
            stack, queue, state, trans = reference(stack, queue, graph)
            transitions.append(trans)

            x.append(featureRow)
            y.append(trans)

        stack, graph = transition.empty_stack(stack, graph)

        # Poorman's projectivization to have well-formed graphs.
        for word in sentence:
            word['head'] = graph['heads'][word['id']]

    return x, y


def parse_ml(stack, queue, graph, trans):
    if transition.can_rightarc(stack) and trans[:2] == 'ra':
        stack, queue, graph = transition.right_arc(stack, queue, graph, trans[3:])
        return stack, queue, graph, 'ra'
    elif transition.can_leftarc(stack, graph) and trans[:2] == 'la':
        stack, queue, graph = transition.left_arc(stack, queue, graph, trans[3:])
        return stack, queue, graph, 'la'
    elif transition.can_reduce(stack, graph) and trans[:2] == 're':
        stack, queue, graph = transition.reduce(stack, queue, graph)
        return stack, queue, graph, 're'
    else:
        stack, queue, graph = transition.shift(stack, queue, graph)
        return stack, queue, graph, 'sh'


def parse(file, model, vec):
    column_names_2006 = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']
    column_names_2006_test = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats']
    feature_names = ["stack0_POS", "stack0_word", "queue0_POS", "queue0_word", "can-re", "can-la"]
    feature_names2 = ["stack0_POS", "stack0_word", "stack1_POS", "stack1_word",
                      "queue0_POS", "queue0_word", "queue1_POS", "queue1_word",
                      "can-re", "can-la"]
    feature_names3 = ["stack0_POS", "stack0_word", "stack1_POS", "stack1_word",
                      "queue0_POS", "queue0_word", "queue1_POS", "queue1_word", "queue1_POS", "queue1_word",
                      "can-re", "can-la", "next_word_POS", "next_word_word"]

    sentences = conll.read_sentences(file)
    formatted_corpus = conll.split_rows(sentences, column_names_2006)
    x = []
    y = []

    for sentence in formatted_corpus:
        stack = []
        queue = list(sentence)
        graph = {}
        graph['heads'] = {}
        graph['heads']['0'] = '0'
        graph['deprels'] = {}
        graph['deprels']['0'] = 'ROOT'

        while queue:
            features = extract(stack, queue, graph, feature_names3, sentence)
            trans_nr = model.predict(vec.transform(features))
            stack, queue, graph, trans = parse_ml(stack, queue, graph, trans_nr[0])

            x.append(features)
            y.append(trans)

        stack, graph = transition.empty_stack(stack, graph)

        for word in sentence:
            word['head'] = graph['heads'][word['id']]
            word['deprel'] = graph['deprels'][word['id']]

    conll.save("out", sentences, column_names_2006)


if __name__ == '__main__':
    train_file = './train.conll'
    test_file = './test.conll'

    x_train, y_train = extract_f(train_file)
    x_test, y_test = extract_f(test_file)

    for e in x_train[:4]:
        for k in e:
            print(e[k])
        print("----")

    for e in y_train[:4]:
        print(e)

    # ---------------------------------

    print("Vectorizing...")
    vec = DictVectorizer(sparse=True)
    classifier = linear_model.LogisticRegression(penalty='l2', dual=True, solver='liblinear')
    pickle.dump(classifier, open("class3.pkl", 'wb'))
    pickle.dump(vec, open("vec3.pkl", 'wb'))
    X = vec.fit_transform(x_train)

    print("Training the model...")
    model = classifier.fit(X, y_train)
    pickle.dump(model, open("model3.pkl", 'wb'))
    print(model)

    print("Predicting..")
    X_test = vec.transform(x_test)
    y_test_predicted = classifier.predict(X_test)

    print("Classification report for classifier %s:\n%s\n" % (
    classifier, metrics.classification_report(y_test, y_test_predicted)))

    """ 
    vec = pickle.load(open("vec3.pkl", 'rb'))
    classifier = pickle.load(open("class3.pkl", 'rb'))
    model = pickle.load(open("model3.pkl", "rb"))
    X = pickle.load(open("x3.pkl", "rb"))
    """

    print("VEC ----------")
    print(vec)
    print("CLASS ----------")
    print(classifier)
    print("MODEL ----------")
    print(model)
    print("X ----------")
    print(X)

    parse(test_file, model, vec)


