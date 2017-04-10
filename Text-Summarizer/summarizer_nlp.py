import json
import sys
from collections import OrderedDict

import math
import nltk
from nltk import sent_tokenize, word_tokenize, bigrams
from nltk.corpus import indian
from nltk.tag import tnt
from collections import Counter
import copy


def readData(fileName):
    global stringLines
    dataFileOpen = open(fileName, 'r', encoding = "utf-8")
    stringLines = dataFileOpen.read()
    dataFileOpen.close()

def formDataDict():
    global stringLines, titleData, wordList, sentenceList, originalSentenceList, bigramCountList
    stringLines = stringLines.replace("?", ". ")
    stringLines = stringLines.replace("!", ". ")
    stringLines = stringLines.replace("\"", "")
    stringLines = stringLines.replace("।", ". ")
    stringLines = stringLines.replace("?", ". ")
    stringLines = stringLines.replace("\'", "")
    stringLines = stringLines.replace("\n", "")
    stringLines = stringLines.replace(",", "")
    stringLines = stringLines.replace("’", "")

    originalFile = copy.deepcopy(stringLines)
    originalSentenceList = [sentenceList.strip() for sentenceList in originalFile.split('.')]
    #del originalSentenceList[0]
    del originalSentenceList[-1]

    removeStopWord()
    generateBigrams()
    sentenceList = [sentenceList.strip() for sentenceList in stringLines.split('.')]
    titleData = sentenceList[0]
    del sentenceList[0]
    del sentenceList[-1]

    for sentence in (range(len(sentenceList))):
        sentenceList[sentence] = word_tokenize(sentenceList[sentence])
        bigramCountList.append(countBigrams(sentenceList[sentence]))
    wordList = list(set(word_tokenize(stringLines)))

def generateBigrams():
    global bigrams, bigramsDict, bigramsWordsList

    wordTokenizedList = word_tokenize(stringLines)
    bigrams = list(nltk.bigrams(wordTokenizedList))
    cfd = nltk.ConditionalFreqDist(bigrams)

    for inneritems in cfd.items():
        for items in inneritems[1].items():
            if (items[1] > 2 and (inneritems[0] != '.' and items[0] != '.')):
                if items[1] not in bigramsDict:
                    bigramsDict[items[1]] = []
                    bigramsDict[items[1]].append([inneritems[0], items[0]])
                else:
                    bigramsDict[items[1]].append([inneritems[0], items[0]])
                bigramsWordsList.append([inneritems[0],items[0]])

def removeStopWord():
    global stringLines

    stopWordsFile = open("NLPData/stopwords.txt", 'r', encoding = "utf-8")
    stopWordsFileRead = stopWordsFile.readlines()
    for words in stopWordsFileRead:
        words = words.strip()
        stringLines = stringLines.replace(words, "")


def getSuffixes():
    global suffixes
    file = open('NLPData/stemmer.json', 'r', encoding="utf-8")  # name of the file containing the stemming data
    suffixes = json.load(file)  # to load the data to stem
    file.close()

def generateStemWords(word):
    global suffixes
    for key in suffixes:
        if len(word) > int(key) + 1:
            for suf in suffixes[key]:
                if word.endswith(suf):
                    return word[:-int(key)]
        return word

def partsOfSpeechTagger(localWordList, checkString):
    global partsOfSpeechTaggedData, partsOfSpeechTaggedTitleData
    train_data = indian.tagged_sents('hindi.pos')  # code to have the reference for tagging
    tnt_pos_tagger = tnt.TnT()  # parts of speech tagger
    tnt_pos_tagger.train(train_data)  # Training the tnt Part of speech tagger with hindi data
    if checkString == "textData":
        partsOfSpeechTaggedData = (tnt_pos_tagger.tag(localWordList))
    elif checkString == "titleData":
        partsOfSpeechTaggedTitleData = (tnt_pos_tagger.tag(localWordList))

def loadWordNet():
    global word_dict
    file = open('NLPData/word_dict.json', 'r', encoding="utf-8")  # name of the file containing the stemming data
    word_dict = json.load(file)  # to load the data to stem
    file.close()

def removeStopWords(localpartsOfSpeechTaggedData, flag):
    global removedTaggedData, removedTaggedTitleData
    if flag:
        for words in localpartsOfSpeechTaggedData:
            if not (words[1] == 'VAUX' or words[1] == 'SYM' or words[1] == 'VFM' or words[1] == 'CC' or words[1] == 'PRP' or
                            words[1] == 'PUNC' or words[1] == 'QF' or words[1] == 'RB' or words[1] == 'QW' or words[
                1] == 'RP' or words[1] == 'PREP'):
                removedTaggedData.append(words[0])
    else:
        for words in localpartsOfSpeechTaggedData:
            if (words[1] == 'NN' or words[1] == 'NNP' or words[1] == 'Unk'):
                removedTaggedTitleData.append(words[0])

def stemmingForData(sentenceList):
    for sentence in range(len(sentenceList)):
        stringTemp = []

        for words in sentenceList[sentence]:
            if words in removedTaggedData:
                if words in word_dict:
                    temp_word = word_dict[words]
                else:
                    temp_word = generateStemWords(words)
                    if temp_word in word_dict:
                        temp_word = word_dict[temp_word]
                stringTemp.append(temp_word)
        sentenceList[sentence] = stringTemp

def stemmingForTitle():
    global titleList
    titleList = titleData.split(" ")
    stringTemp = []
    for words in removedTaggedTitleData:
        if words in removedTaggedData:
            if words in word_dict:
                temp_word = word_dict[words]
            else:
                temp_word = generateStemWords(words)
                if temp_word in word_dict:
                    temp_word = word_dict[temp_word]
            stringTemp.append(temp_word)
        titleList = stringTemp

def properNounFeature(localPartsOfSpeechTaggedData):
    global properNounList, unknownWordList
    for items in localPartsOfSpeechTaggedData:
        if (items[1] == "NNP"):
            properNounList.append(items[0])
        if (items[1] == "Unk"):
            unknownWordList.append(items[0])

def generateCueWordList(titleList):
    global cueWordList
    for items in titleList:
        if items in word_dict:
            cueWordList.append(word_dict[items])

def calculateIdf():
    global idf
    allWords = []
    for sentence in range(len(sentenceList)):
        allWords.extend(list(set(sentenceList[sentence])))
    idf = Counter(allWords)
    for items in idf:
        idf[items] = math.log(len(sentenceList) / idf[items])

def countBigrams(sentences):
    sentenceBigrams = list(nltk.bigrams(sentences))
    count=0
    for items in sentenceBigrams:
        if list(items) in bigramsWordsList:
            count += 1
    return count

def calculateFeatures():
    global featureProbablity
    for i in range(1,len(originalSentenceList)):
        featureProbablity[originalSentenceList[i]] = {}
    i = 1
    j = len(originalSentenceList)-1
    tfIdf = [0] * len(sentenceList)
    for sentences in sentenceList:
        countTopicFeature = 0
        countCueFeature = 0
        countProperWordFeature = 0
        countUnknownWordFeature = 0
        for words in sentences:
            if words in properNounList:
                countProperWordFeature += 1
            elif words in unknownWordList:
                countUnknownWordFeature += 1
            if words in titleList:
                countTopicFeature += 1
                countCueFeature += 1
            if words in cueWordList:
                countCueFeature += 1
        if (len(sentences) == 0):
            sentences.append(" ")
        featureProbablity[originalSentenceList[i]]["topicFeature"] = countTopicFeature / len(sentences)
        featureProbablity[originalSentenceList[i]]["properWordFeature"] = countProperWordFeature / len(sentences)
        featureProbablity[originalSentenceList[i]]["unknownWordFeature"] = countUnknownWordFeature / len(sentences)
        featureProbablity[originalSentenceList[i]]["cueWordFeature"] = countCueFeature / len(sentences)
        featureProbablity[originalSentenceList[i]]["bigrams"] = bigramCountList[i-1] / len(sentences)

        tfNumerator = {}
        tfNumerator = Counter(sentences)
        for words in tfNumerator:
            tfNumerator[words] = (tfNumerator[words] / len(sentences) * idf[words])
            tfIdf[i - 1] += tfNumerator[words]
        featureProbablity[originalSentenceList[i]]["tfIdf"] = tfIdf[i - 1] / len(sentences)

        if (i <= j):
            featureProbablity[originalSentenceList[i]]["sentencePositionFeature"] = (j - (len(sentenceList) / 2)) / ((len(sentenceList) / 2))
            featureProbablity[originalSentenceList[j]]["sentencePositionFeature"] = (j - (len(sentenceList) / 2)) / ((len(sentenceList) / 2))
        featureProbablity[originalSentenceList[i]]["class"] = 0
        i += 1
        j -= 1

def writeFeatures():
    global allArticlesWithFeatures
    file = open('NLPData/allArticlesWithFeatures.json', 'w+', encoding = 'utf-8')
    json.dump(allArticlesWithFeatures, file)
    file.close()

numberOfFiles = 200
word_dict={}
suffixes={}
loadWordNet() #Call outside for loop, no need to initialize
getSuffixes() #Call outside the loop, no need to initialize
allArticlesWithFeatures = list()

for i in range(numberOfFiles):
    stringLines = ""
    titleData = ""
    wordList = list()
    sentenceList = list()
    bigramsDict = dict()
    bigramsWordsList = list()
    bigram = list()
    partsOfSpeechTaggedData = list()
    partsOfSpeechTaggedTitleData = list()
    properNounList = list()
    unknownWordList = list()
    removedTaggedData = list()
    removedTaggedTitleData = list()
    titleList = list()
    cueWordList = list()
    featureProbablity = OrderedDict()
    bigramCountList = list()
    readData("NLPData/updated_articles/"+str(i)+".txt")

    formDataDict() #titleData = "", wordList = [], sentenceList = [], bigramDict = {}, bigramWordList  = [], bigram = []
    partsOfSpeechTagger(wordList, "textData") #partsOfSpeechTaggedData = [], partsOfSpeechTaggedTitleData = []
    removeStopWords(partsOfSpeechTaggedData, True) #stringLines = ""
    partsOfSpeechTagger(titleData.split(" "), "titleData")
    properNounFeature(partsOfSpeechTaggedData) #properNounList = [], unknownWordlist = []
    removeStopWords(partsOfSpeechTaggedTitleData, False) #removedTaggedData = [], removedTaggedTitleData = []
    stemmingForData(sentenceList)
    stemmingForTitle() #titleList = []
    generateCueWordList(titleList) #cueWordList = []
    calculateIdf()
    calculateFeatures() #featureProbablity = {}
    allArticlesWithFeatures.append(featureProbablity)

writeFeatures()

print(allArticlesWithFeatures)