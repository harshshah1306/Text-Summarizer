import json
import sys
def assignClasses():
    global allArticlesWithFeatures
    file = open('NLPData/allArticlesWithFeatures.json', 'r', encoding = 'utf-8')
    allArticlesWithFeatures = json.load(file)
    for i in range(200):
        file = open("NLPData/summary/"+str(i)+"_sum.txt", "r", encoding = "utf-8")
        sum_stringLines = file.read()
        file.close()
        sum_stringLines = sum_stringLines.replace("?", ". ")
        sum_stringLines = sum_stringLines.replace("!", ". ")
        sum_stringLines = sum_stringLines.replace("\"", "")
        sum_stringLines = sum_stringLines.replace("।", ". ")
        sum_stringLines = sum_stringLines.replace("?", ". ")
        sum_stringLines = sum_stringLines.replace("\'", "")
        sum_stringLines = sum_stringLines.replace("\n", "")
        sum_stringLines = sum_stringLines.replace("\ufeff", "")
        sum_stringLines = sum_stringLines.replace(",", "")
        sum_stringLines = sum_stringLines.replace("’", "")
        sum_Lines = [sentenceList.strip() for sentenceList in sum_stringLines.split('.')]
        del sum_Lines[-1]
        print("\nFile Name - " + str(i))

        print("Length of Summary File - "+str(len(sum_Lines)))
        print("Sentences in Summary File\n")
        for keys in sum_Lines:
            print(keys)
        print("\n\n")
        count=0

        print("Sentences in Text File")
        for keys in allArticlesWithFeatures[i].keys():
            print (keys)

        print ("\n\n")
        for sent in sum_Lines:
            if sent in allArticlesWithFeatures[i]:
                allArticlesWithFeatures[i][sent]["class"] = 1
                count+=1
            else:
                print("Line Not matched - ")
                print("\t"+ sent)
                sys.exit()
        print("Number of lines in Summary file that matched - "+ str(count))
        #sys.exit()

def saveFinalDataset():
    global allArticlesWithFeatures
    file = open('NLPData/finalDataset.csv', 'w+', encoding = 'utf-8')
    file.write("Article Number, Topic Feature, ProperNoun Feature, Unknown Word Feature, Cue Word Feature,  Bigram Feature, Tf-Idf, Sentence Position Feature, Class\n")
    articleCount = -1
    for articles in allArticlesWithFeatures:
        articleCount += 1
        for sentenceKey in articles:
            file.write(str(articleCount)+", "+ str(articles[sentenceKey]["topicFeature"])+", "+ str(articles[sentenceKey]["properWordFeature"])+", "+ str(articles[sentenceKey]["unknownWordFeature"])+", "+ str(articles[sentenceKey]["cueWordFeature"])+", "+ str(articles[sentenceKey]["bigrams"])+", "+ str(articles[sentenceKey]["tfIdf"])+", "+ str(articles[sentenceKey]["sentencePositionFeature"]) +", "+ str(articles[sentenceKey]["class"]) + "\n")
    file.close()


allArticlesWithFeatures = list()
assignClasses()
saveFinalDataset()