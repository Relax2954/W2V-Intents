from gensim.models import KeyedVectors
import pandas as pd

model = KeyedVectors.load_word2vec_format('cc.sv.300.vec')
# print (model.most_similar('desk'))

words = []
for word in model.vocab:
    words.append(word)

# print("Vector components of a word: {}".format(
#     model[words[0]]
# ))

sentences = [['this', 'is', 'the', 'good', 'machine', 'learning', 'book'],
             ['this', 'is', 'another', 'machine', 'learning', 'book'],
             ['one', 'more', 'new', 'book'],
             ['this', 'is', 'about', 'machine', 'learning', 'post'],
             ['orange', 'juice', 'is', 'the', 'liquid', 'extract', 'of', 'fruit'],
             ['orange', 'juice', 'comes', 'in', 'several', 'different', 'varieties'],
             ['this', 'is', 'the', 'last', 'machine', 'learning', 'book'],
             ['orange', 'juice', 'comes', 'in', 'several', 'different', 'packages'],
             ['orange', 'juice', 'is', 'liquid', 'extract', 'from', 'fruit', 'on', 'orange', 'tree'],
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
             ['har', 'du', 'nån', 'flickvän', 'eller', 'pojkvän'],
             # 3:
             ['hur', 'mycket', 'tjänar'],
             ['vad', 'tjänar', 'du'],
             ['vad', 'har', 'du', 'i', 'lön'],
             ['vad', 'har', 'du', 'för', 'månadslön'],
             ['vad', 'har', 'du', 'i', 'månadslön'],

             # 4:
             ['vilken', 'är', 'din', 'favoriträtt'],
             ['vilken', 'är', 'ditt', 'favoritrecept'],
             ['vad', 'är', 'din', 'favoriträtt'],
             ['vad', 'är', 'din', 'favoritmat'],
             ['vad', 'är', 'ditt', 'favoritrecept'],
             ['vilken', 'vad', 'är', 'din', 'ditt', 'favoriträtt'],
             ['vilken', 'vad', 'är', 'din', 'ditt', 'favoritmat'],
             ['vilken', 'vad', 'är', 'din', 'ditt', 'favoritrecept'],
             # 5:
             ['vad', 'äter', 'du'],
             ['är', 'du', 'vegetarian', 'vegeterian'],
             ['är', 'du', 'vegetarian', 'vegan'],
             # 6:
             ['vad', 'jobbar', 'du', 'med'],
             ['var', 'vad', 'jobbar', 'du'],
             ['vad', 'gör', 'du', 'om', 'dagarna'],
             ['har', 'du', 'ett', 'något', 'jobb'],
             ['vad', 'är', 'ditt', 'jobb', 'levebröd'],
             # 7:
             ['är', 'du', 'en', 'man', 'eller', 'kvinna'],
             ['är', 'du', 'en', 'kille', 'eller', 'tjej'],
             ['är', 'du', 'en', 'man'],
             ['är', 'du', 'en', 'kille'],
             ['är', 'du', 'en', 'pojke'],
             ['är', 'du', 'en', 'kvinna'],
             ['är', 'du', 'en', 'flicka'],
             # 8:
             ['vad', 'heter', 'du'],
             ['vad', 'är', 'ditt', 'namn'],
             ['vad', 'ska', 'jag', 'kalla', 'dig'],
             # 9:
             ['är', 'du', 'en', 'bot'],
             ['är', 'du', 'en', 'robot'],
             # 10:
             ['vilken', 'är', 'din', 'favoritfärg'],
             ['vad', 'är', 'din', 'favoritfärg'],
             ['har', 'du', 'en', 'favoritfärg'],
             # 11:
             ['vem', 'är', 'du'],
             # 12:
             ['how', 'are', 'you'],
             ['hur', 'mår', 'du'],
             ['hej', 'hur', 'mår', 'du'],
             ['hur', 'är', 'läget'],
             ['läget'],
             # testing:
             ['har', 'du', 'en', 'flickvän'],
             ['har', 'du', 'nån', 'pojkvän', 'eller', 'flickvän'],
             ['vad', 'är', 'ditt', 'favoritrecept'],
             ['är', 'du', 'en', 'pojke'],
             ['vad', 'ska', 'jag', 'kalla', 'dig'],
             ['vad', 'e', 'ditt', 'namn'],
             ['vad', 'händer'],
             ['är', 'du', 'en', 'tjej'],
             ]

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

X_train = V[0:63]
X_test = V[63:71]

# 2- CatchAreYouSingle 3- CatchHowMuchDoYouMake, 10- CatchWhatsYourFavoriteColor, 4- CatchWhatsYourFavoriteFood, 5- CatchWhatsYourDiet
# 6-CatchWhereDoYouWork  7- CatchWhatsYourGender 8- CatchWhatsYourName 9- CatchAreYouABot 11- catchwhoareyou 12- CatchHowAreYou

Y_train = [0, 0, 0, 0, 1, 1, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3,
           4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 9, 9, 10, 10, 10, 11, 12, 12,
           12
           ]
# Y_test = [0, 1, 1]


from sklearn.neural_network import MLPClassifier

classifier = MLPClassifier(alpha=0.7, max_iter=400)
classifier.fit(X_train, Y_train)

df_results = pd.DataFrame(data=np.zeros(shape=(1, 3)), columns=['classifier', 'train_score', 'test_score'])
train_score = classifier.score(X_train, Y_train)
# test_score = classifier.score(X_test, Y_test)

print("ol na fer")
print(classifier.predict_proba(X_test))
print("tu sam")
print(classifier.predict(X_test))

df_results.loc[5, 'classifier'] = "MLP"
df_results.loc[5, 'train_score'] = train_score
# df_results.loc[5, 'test_score'] = test_score
print(df_results)
