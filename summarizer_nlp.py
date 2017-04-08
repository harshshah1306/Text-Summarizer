import json
import sys
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import indian
from nltk.tag import tnt

def readData():
    global dataFileOpen, stringLines
    dataFileOpen = open("1160_utf.txt", 'r', encoding = "utf-8")
    stringLines = dataFileOpen.read()

def formDataDict():
    global dataDict, stringLines, titleData, wordList, sentenceList
    stringLines = stringLines.replace("?", ".")
    stringLines = stringLines.replace("!", ".")
    stringLines = stringLines.replace("\"", "")
    stringLines = stringLines.replace("।", ".")
    stringLines = stringLines.replace("\'", "")
    stringLines = stringLines.replace("\n", "")
    sentenceList = [sentenceList.strip() for sentenceList in stringLines.split('.')]
    titleData = sentenceList[0]
    del sentenceList[0]
    del sentenceList[-1]
    for sentence in (range(len(sentenceList))):
        sentenceList[sentence] = word_tokenize(sentenceList[sentence])
    wordList = set(word_tokenize(stringLines))

def getSuffixes():
    global suffixes
    file = open('stemmer.json', 'r', encoding="utf-8")  # name of the file containing the stemming data
    suffixes = json.load(file)  # to load the data to stem

def generateStemWords(word):
    for key in suffixes:
        if len(word) > int(key) + 1:
            for suf in suffixes[key]:
                if word.endswith(suf):
                    return word[:-int(key)]
        return word

def partsOfSpeechTagger(localWordList, flag):
    global partsOfSpeechTaggedData, partsOfSpeechTaggedTitleData
    train_data = indian.tagged_sents('hindi.pos')  # code to have the reference for tagging
    tnt_pos_tagger = tnt.TnT()  # parts of speech tagger
    tnt_pos_tagger.train(train_data)  # Training the tnt Part of speech tagger with hindi data
    if flag:
        partsOfSpeechTaggedData = (tnt_pos_tagger.tag(localWordList))
    else:
        partsOfSpeechTaggedTitleData = (tnt_pos_tagger.tag(localWordList))

def loadWordNet():
    global word_dict
    file = open('word_dict.json', 'r', encoding="utf-8")  # name of the file containing the stemming data
    word_dict = json.load(file)  # to load the data to stem
    file.close()

def removeStopWords(localpartsOfSpeechTaggedData, flag):
    global removedTaggedData, removedTaggedTitleData
    if flag:
        removedTaggedData = []
        for words in localpartsOfSpeechTaggedData:
            if not (words[1] == 'VAUX' or words[1] == 'SYM' or words[1] == 'VFM' or words[1] == 'CC' or words[1] == 'PRP' or
                            words[1] == 'PUNC' or words[1] == 'QF' or words[1] == 'RB' or words[1] == 'QW' or words[
                1] == 'RP' or words[1] == 'PREP'):
                removedTaggedData.append(words[0])
    else:
        removedTaggedTitleData = []
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
            else:
                continue
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
        else:
            continue
        titleList = stringTemp



def generateCueWordList(titleList):
    global cueWordList
    cueWordList = []
    for items in titleList:
        if items in word_dict:
            cueWordList.append(word_dict[items])

def calculateFeatures():
    global featureProbablity
    featureProbablity = {}
    for i in range(len(sentenceList)):
        featureProbablity[i + 1] = {}
    i = 1
    j = len(sentenceList)
    for sentences in sentenceList:
        countTopicFeature = 0
        countCueFeature = 0
        print ("here")
        for words in sentences:
            if words in titleList:
                countTopicFeature += 1
            if words in cueWordList:
                countCueFeature += 1
        featureProbablity[i]["topicFeature"] = countTopicFeature
        featureProbablity[i]["sentenceLengthFeature"] = len(sentences)
        featureProbablity[i]["cueWordFeature"] = countCueFeature
        if (i <= j):
            featureProbablity[i]["sentencePositionFeature"] = i
            featureProbablity[j]["sentencePositionFeature"] = i
        i += 1
        j -= 1


loadWordNet()
getSuffixes()
readData()
formDataDict()
partsOfSpeechTagger(wordList, True)
removeStopWords(partsOfSpeechTaggedData, True)
partsOfSpeechTagger(titleData.split(" "), False)
removeStopWords(partsOfSpeechTaggedTitleData, False)
stemmingForData(sentenceList)
stemmingForTitle()
generateCueWordList(titleList)
calculateFeatures()






