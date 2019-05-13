from gensim.models import KeyedVectors
import pandas as pd
from sklearn import svm
import pickle

model = KeyedVectors.load_word2vec_format('cc.sv.300.vec')
# print (model.most_similar('desk'))

sentences = [['this', 'is', 'the', 'good', 'machine', 'learning', 'book'],
             ['this', 'is', 'another', 'machine', 'learning', 'book'],
             ['one', 'more', 'new', 'book'],
             ['this', 'is', 'about', 'machine', 'learning', 'post'],
             ['orange', 'juice', 'is', 'the', 'liquid', 'extract', 'of', 'fruit'],
             ['orange', 'juice', 'comes', 'in', 'several', 'different', 'varieties'],
             ['this', 'is', 'the', 'last', 'machine', 'learning', 'book'],
             ['orange', 'juice', 'comes', 'in', 'several', 'different', 'packages'],
             ['orange', 'juice', 'is', 'liquid', 'extract', 'from', 'fruit', 'on', 'orange', 'tree'],  #8
             # 2:
             ['har', 'du', 'en', 'kille'],
             ['har', 'du', 'en', 'tjej'],
             ['har', 'du', 'en', 'pojkvän'],
             ['har', 'du', 'en', 'flickvän'],
             ['är', 'du', 'singel'],
             ['är', 'du', 'ledig'],
             ['har', 'du', 'pojkvän'],
             ['har', 'du', 'flickvän'],
             ['har', 'du', 'någon', 'pojkvän'],
             ['har', 'du', 'någon', 'flickvän'],
             ['har', 'du', 'en', 'pojkvän', 'eller', 'flickvän'],
             ['har', 'du', 'en', 'flickvän', 'eller', 'pojkvän'],
             ['har', 'du', 'någon', 'pojkvän', 'eller', 'flickvän'],
             ['har', 'du', 'någon', 'flickvän', 'eller', 'pojkvän'],
             ['har', 'du', 'en', 'pojkvän'],
             ['har', 'du', 'nån', 'pojkvän'],
             ['har', 'du', 'nån', 'flickvän'],
             ['har', 'du', 'nån', 'flickvän', 'eller', 'pojkvän'], #26
             # 3:
             ['hur', 'mycket', 'tjänar'],
             ['vad', 'tjänar', 'du'],
             ['vad', 'har', 'du', 'i', 'lön'],
             ['vad', 'har', 'du', 'för', 'månadslön'],
             ['vad', 'har', 'du', 'i', 'månadslön'], #31

             # 4:
             ['vilken', 'är', 'din', 'favoriträtt'],
             ['vilken', 'är', 'ditt', 'favoritrecept'],
             ['vad', 'är', 'din', 'favoriträtt'],
             ['vad', 'är', 'din', 'favoritmat'],
             ['vad', 'är', 'ditt', 'favoritrecept'],
             ['vilken', 'vad', 'är', 'din', 'ditt', 'favoriträtt'],
             ['vilken', 'vad', 'är', 'din', 'ditt', 'favoritmat'],
             ['vilken', 'vad', 'är', 'din', 'ditt', 'favoritrecept'], #39
             # 5:
             ['vad', 'äter', 'du'],
             ['är', 'du', 'vegetarian', 'vegeterian'],
             ['är', 'du', 'vegetarian', 'vegan'],  #42
             # 6:
             ['vad', 'jobbar', 'du', 'med'],
             ['var', 'vad', 'jobbar', 'du'],
             ['vad', 'gör', 'du', 'om', 'dagarna'],
             ['har', 'du', 'ett', 'något', 'jobb'],
             ['vad', 'är', 'ditt', 'jobb', 'levebröd'], #47
             # 7:
             ['är', 'du', 'en', 'man', 'eller', 'kvinna'],
             ['är', 'du', 'en', 'kille', 'eller', 'tjej'],

             ['är', 'du', 'en', 'man'],
             ['är', 'du', 'en', 'kille'],
             ['är', 'du', 'en', 'pojke'],
             ['är', 'du', 'en', 'kvinna'],
             ['är', 'du', 'en', 'flicka'], #54
             # 8:myname
             ['vad', 'heter', 'du'],
             ['vad', 'är', 'ditt', 'namn'],
             ['vad', 'ska', 'jag', 'kalla', 'dig'],
             ['vem', 'är', 'du'],
             ['namn'],
             ['varför', 'heter', 'du'],
             ['vad', 'ska', 'vi', 'kalla', 'dig'],
                 #61
             # 9:
             ['är', 'du', 'en', 'bot'],
             ['är', 'du', 'en', 'robot'], #63
             # 10:
             ['vilken', 'är', 'din', 'favoritfärg'],
             ['vad', 'är', 'din', 'favoritfärg'],
             ['har', 'du', 'en', 'favoritfärg'], #66
             # 12:
             ['how', 'are', 'you'],
             ['hur', 'mår', 'du'],
             ['hej', 'hur', 'mår', 'du'],
             ['hur', 'är', 'läget'],
             ['läget'],  #71
             ['tjarå'],  # 104
             # 14: greetsv
             ['hej'],
             ['hejhej'],
             ['hi'],
             ['hello'],
             ['hålla'],
             ['hey'],
             ['jo'],
             ['hej', 'du', 'igen'],
             ['hej', 'på', 'dig'],
             ['hej', 'hej'],
             ['halloj'],
             ['tja'],
             ['tjo'],
             ['hejjjj'],

             # ['hallå', 'där'],
             # ['hallåigen'],
             # ['tjabba'],
             # ['tjena'],
             # ['är', 'du', 'där'],
             # ['är', 'du', 'kvar'],  # 124

             # testing:
             ['har', 'du', 'en', 'flickvän'],
             ['har', 'du', 'nån', 'pojkvän', 'eller', 'flickvän'],
             ['vad', 'är', 'ditt', 'favoritrecept'],
             ['är', 'du', 'en', 'pojke'],
             ['vad', 'ska', 'jag', 'kalla', 'dig'],
             ['vad', 'e', 'ditt', 'namn'],
             ['vad', 'händer'],
             ['är', 'du', 'en', 'tjej'],  #80
             ]


# sentences = [
#     # 2:intent:CatchAreYouSingle
#     ['har', 'du', 'en', 'kille'],
#     ['har', 'du', 'en', 'tjej'],
#     ['har', 'du', 'en', 'pojkvän'],
#     ['har', 'du', 'en', 'flickvän'],
#     ['är', 'du', 'singel'],
#     ['är', 'du', 'ledig'],
#     ['har', 'du', 'pojkvän'],
#     ['har', 'du', 'flickvän'],
#     ['har', 'du', 'någon', 'pojkvän'],
#     ['har', 'du', 'någon', 'flickvän'],
#     ['har', 'du', 'en', 'pojkvän', 'eller', 'flickvän'],
#     ['har', 'du', 'en', 'flickvän', 'eller', 'pojkvän'],
#     ['har', 'du', 'någon', 'pojkvän', 'eller', 'flickvän'],
#     ['har', 'du', 'någon', 'flickvän', 'eller', 'pojkvän'],
#     ['har', 'du', 'en', 'pojkvän'],
#     ['har', 'du', 'nån', 'pojkvän'],
#     ['har', 'du', 'nån', 'flickvän'],
#     ['har', 'du', 'nån', 'flickvän', 'eller', 'pojkvän'],  # 26
#     # 3:
#     ['hur', 'mycket', 'tjänar'],
#     ['vad', 'tjänar', 'du'],
#     ['vad', 'har', 'du', 'i', 'lön'],
#     ['vad', 'har', 'du', 'för', 'månadslön'],
#     ['vad', 'har', 'du', 'i', 'månadslön'],
#     ['hur', 'mycket', 'para', 'får', 'du'],
#     ['vad', 'är', 'din', 'inkomst'],
#     ['hur', 'mycket', 'är', 'din', 'inkomst', 'på'],
#     ['hur', 'mycket', 'får', 'du', 'för', 'det', 'här', 'jobbet'],
#     ['bra', 'betalt', 'här'],
#     ['hur', 'mycket', 'pengar', 'får', 'du', 'i', 'månaden'],
#     ['hur', 'mycket', 'pengar', 'får', 'du'],
#     ['vill', 'du', 'ha', 'mer', 'pengar'],
#     ['vill', 'du', 'har', 'bättre', 'lön'],
#     ['betalar', 'din', 'chef', 'dig'],
#     ['betalar', 'din', 'chef', 'dig', 'bra'],
#     # 4:
#     ['vilken', 'är', 'din', 'favoriträtt'],
#     ['vilken', 'är', 'ditt', 'favoritrecept'],
#     ['vad', 'är', 'din', 'favoriträtt'],
#     ['vad', 'är', 'din', 'favoritmat'],
#     ['vad', 'är', 'ditt', 'favoritrecept'],
#     ['vilken', 'vad', 'är', 'din', 'ditt', 'favoriträtt'],
#     ['vilken', 'vad', 'är', 'din', 'ditt', 'favoritmat'],
#     ['vilken', 'vad', 'är', 'din', 'ditt', 'favoritrecept'],  # 50
#     # 5:
#     ['vad', 'äter', 'du'],
#     ['är', 'du', 'vegetarian'],
#     ['är', 'du', 'vegan'],
#     ['äter', 'du', 'bara', 'grönsaker'],
#     ['äter' 'du' 'kött'],  # 55
#     # 6:
#     ['vad', 'jobbar', 'du', 'med'],
#     ['var', 'jobbar', 'du'],
#     ['vad', 'gör', 'du', 'om', 'dagarna'],
#     ['har', 'du', 'ett', 'jobb'],
#     ['vad', 'är', 'ditt', 'jobb', ],
#     ['vad', 'jobbar', 'du'],
#     ['vad', 'är', 'ditt', 'levebröd'],
#     ['har', 'du', 'något', 'jobb'],
#     ['jobbar', 'du', 'här'],
#     ['din', 'sysselsättning'],
#     ['sysselsättning'],
#     ['jobb'],
#     ['arbetar', 'du', 'här'],
#         # testing:
#     ['har', 'du', 'en', 'flickvän'],
#     ['har', 'du', 'nån', 'pojkvän', 'eller', 'flickvän'],
#     ['vad', 'är', 'ditt', 'favoritrecept'],
#     ['är', 'du', 'en', 'pojke'],
#     ['vad', 'ska', 'jag', 'kalla', 'dig'],
#     ['vad', 'e', 'ditt', 'namn'],
#     ['vad', 'händer'],  # false
#     ['är', 'du', 'en', 'tjej']  # 76
# ]  # 68
#              # 7:
#              ['är', 'du', 'en', 'man', 'eller', 'kvinna'],
#              ['är', 'du', 'en', 'kille', 'eller', 'tjej'],
#              ['är', 'du', 'en', 'man'],
#              ['är', 'du', 'en', 'kille'],
#              ['är', 'du', 'en', 'pojke'],
#              ['är', 'du', 'en', 'kvinna'],
#              ['är', 'du', 'en', 'flicka'],
#              ['är', 'du', 'en', 'flicka', 'eller', 'pojke'],
#              ['är', 'du', 'en', 'tjej', 'eller', 'kille'],
#              ['vad', 'har', 'du', 'för', 'kön'],
#              # 8:
#              ['vad', 'heter', 'du'],
#              ['vad', 'är', 'ditt', 'namn'],
#              ['vad', 'ska', 'jag', 'kalla', 'dig'],  # 81
#              # 9:
#              ['är', 'du', 'en', 'bot'],
#              ['är', 'du', 'en', 'robot'],  # 84
#              # 10:
#              ['vilken', 'är', 'din', 'favoritfärg'],
#              ['vad', 'är', 'din', 'favoritfärg'],
#              ['har', 'du', 'en', 'favoritfärg'],  # 76
#              # 12:
#              ['how', 'are', 'you'],
#              ['hur', 'mår', 'du'],
#              ['hej', 'hur', 'mår', 'du'],
#              ['hur', 'är', 'läget'],
#              ['läget'],
#              ['sup'],
#              ['hru'],
#              ['händer'],  # 92
#              # 13:Goodbye
#              ['bye'],
#              ['hej', 'då'],
#              ['ha', 'det'],
#              ['vi', 'ses'],
#              ['vi', 'hörs'],
#              ['hejdå'],
#              ['adjö'],
#              ['byebye'],
#              ['farväl'],
#              ['tjarå'],  # 104
#              # 14: greetsv
#              ['hej'],
#              ['hejhej'],
#              ['hi'],
#              ['hello'],
#              ['hålla'],
#              ['hey'],
#              ['jo'],
#              ['hej', 'du', 'igen'],
#              ['hej', 'på', 'dig'],
#              ['hej', 'hej'],
#              ['halloj'],
#              ['tja'],
#              ['tjo'],
#              ['hejjjj'],
#              ['hallå', 'där'],
#              ['hallåigen'],
#              ['tjabba'],
#              ['tjena'],
#              ['är', 'du', 'där'],
#              ['är', 'du', 'kvar'],  # 124
#              # testing:
#              ['har', 'du', 'en', 'flickvän'],
#              ['har', 'du', 'nån', 'pojkvän', 'eller', 'flickvän'],
#              ['vad', 'är', 'ditt', 'favoritrecept'],
#              ['är', 'du', 'en', 'pojke'],
#              ['vad', 'ska', 'jag', 'kalla', 'dig'],
#              ['vad', 'e', 'ditt', 'namn'],
#              ['vad', 'händer'],  # false
#              ['är', 'du', 'en', 'tjej']  # 76
#              ]

import numpy as np


def sent_vectorizer(sent, model):
    sent_vec = []
    numw = 0
    for w in sent:
        try:
            if numw == 0:
                sent_vec = model[w]
            else:
                sent_vec = np.add(sent_vec, model[w])
            numw += 1
        except:
            pass
    return np.asarray(sent_vec) / numw


V = []
for sentence in sentences:
    V.append(sent_vectorizer(sentence, model))

# X_train = V[0:125]
# X_test = V[125:132]
X_train = V[0:86]
X_test = V[87:95]
print("Aloha")
print(X_test)
print("caos")
Y_train = [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, #31
           4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 9, 9, 10, 10, 10, 12, 12, #65
           12, 12, 12, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14
           ]

# 2- CatchAreYouSingle 3- CatchHowMuchDoYouMake, 10- CatchWhatsYourFavoriteColor, 4- CatchWhatsYourFavoriteFood, 5- CatchWhatsYourDiet
# 6-CatchWhereDoYouWork  7- CatchWhatsYourGender 8- CatchWhatsYourName 9- CatchAreYouABot 11- catchwhoareyou 12- CatchHowAreYou

# Y_train = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3,
#            3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7,
#            7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 9, 9, 10, 10,
#            10, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14,
#            14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
#            14, 14, 14, 14, 14, 14
#            ]
# Y_test = [0, 1, 1]


from sklearn.neural_network import MLPClassifier

classifier = svm.SVC(gamma='scale', probability=True, decision_function_shape='ovo');
# MLPClassifier(alpha=0.7, max_iter=10000)
classifier.fit(X_train, Y_train)
# filename = 'finalized_model.sav'
# pickle.dump(model, open(filename, 'wb'))

df_results = pd.DataFrame(data=np.zeros(shape=(1, 3)), columns=['classifier', 'train_score', 'test_score'])
train_score = classifier.score(X_train, Y_train)
# test_score = classifier.score(X_test, Y_test)

print("ol na fer")
print(classifier.predict_proba(X_test))  # [:,1] >= 0.5).astype(bool)
print("tu sam")
print(classifier.predict(X_test))

df_results.loc[5, 'classifier'] = "SVM"
df_results.loc[5, 'train_score'] = train_score
# df_results.loc[5, 'test_score'] = test_score
print(df_results)
