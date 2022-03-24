import xml.dom.minidom as xml
import pandas as pd 
from datetime import datetime, timedelta
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import numpy as np 
import enchant
from typing import List
from collections import deque
import json
import os
from abc import ABC
from pathlib import Path
### Classes to treat and compute OCR for each newspaper 



class Newspaper(ABC):
    name = ""
    language = ""
    def pathFromDate(self,date,folder):
        pass 
    def getBlocks(self,path):
        pass
    def checkYear(self,path): return  Path(path).exists()
    def compute_ratio(self,tokenized_text: List[List[str]], dictionnary: enchant.Dict):
        nb_words = len(tokenized_text)
        nb_OoV = len([word for word in tokenized_text if not dictionnary.check(word)]) # Number of Out-of-Vocabulary elements
        ratio = 1- nb_OoV/nb_words if nb_words > 0 else 0.0
        
        return ratio, nb_words
     
    def treat_file(self,path):
        ret = []
        for block in self.getBlocks(path):
            ratio, length = self.compute_ratio(block,self.dictionnary)
            ret.append((" ".join(block), ratio, length))
        return ret

    def saveFile(self,results,folder,date):
        df = pd.DataFrame(results, columns = ["Text","Ratio", "Length"])
        year =date.strftime("%Y")
        month = date.strftime("%m")
        day = date.strftime("%d")
        path =  folder + f"/{self.name}/{year}/{month}/{day}.csv"
        Path(path).parent.mkdir(parents=True, exist_ok=True) 
        df.to_csv( path, index = False)

    def treatDate(self,date,dataFolder, resultFolder, show = True ):
        path = self.pathFromDate(date,dataFolder)
        if self.checkYear(path):
            results = self.treat_file(path)
            if len(results) > 0: 
                if show : print(f"Average ratio for {date} : ",sum([b for (a,b,c) in results])/len(results))
                self.saveFile(results,resultFolder,date)
            else:
                print("No data available for date :",date)
        else: print("No data available for date :",date)

    

class Figaro(Newspaper):

    name = "Figaro"
    language = 'fr'
    dictionnary = enchant.Dict(language)

    def isTitle(self,text:str, threshold = 40):
        return len(text) <= threshold

    def pathFromDate(self,date,folder):
        datepath = date.strftime("%Y%m%d")
        return folder + "//" + datepath + "//" + datepath  + ".metadata.fulltext.json"

    def getBlocks(self,path):
        with open(path, encoding= "utf-8") as f:
            raw_object = json.load(f)
        blocks = "\n".join(raw_object["contentAsText"])
        tokenized_blocks = [[s for s in word_tokenize(b, language ="french") if s.isalnum()] for b in blocks.split("\n") ]
        ret = []
        current_article = deque()
        for b in tokenized_blocks:
            if self.isTitle(b):
                if len(current_article) != 0 :
                    ret.append(list(current_article))
                    current_article.clear()
            else:
                current_article += b 
        return ret

class NYT(Newspaper):

    name = "NYT"
    language = 'en'
    dictionnary = enchant.Dict(language)
    def xmlParser(self,path): 
        doc = xml.parse(path)
        blocks = []
        for block in doc.getElementsByTagName("TextBlock"):
            bl = []
            for line in block.getElementsByTagName('TextLine'):
                l = []
                for content in line.getElementsByTagName("String"):
                    l.append(content.getAttribute("CONTENT"))
                bl.append(" ".join(l))
            blocks.append(" ".join(bl))
        tokenized_blocks = [np.array([s for s in word_tokenize(block) if s.isalnum()]) for block in blocks]
        return tokenized_blocks

    def pathFromDate(self,date,folder):
        return folder + "//" + date.strftime("%Y") + "//"+   date.strftime("%m")  + "//" +date.strftime("%d")
    
    def getBlocks(self,path) : 
        files = []
        blocks = []
        for subdir, dirs, f in os.walk(path):
            if len(f)>1:
                path = subdir+"\ocr.xml"
                # print(path)
                blocks += self.xmlParser(path)
        return blocks

class Imparcial(Newspaper):

    name = "Imparcial"
    language = 'es'
    dictionnary = enchant.Dict(language)
    def pathFromDate(self,date,folder):
        return folder  + "//" + date.strftime("%Y%m%d") + "_00000.txt"

    def getBlocks(self,path):
        ret = []
        with open(path, encoding= "utf-8") as f:
            data = f.readlines()
        for line in data:
            if len(line) > 550:
                ret.append(
                    [s for s in word_tokenize(line, language ="spanish") if s.isalnum()]
                )
        return ret


def yearGenerator(year):
    start_date = datetime(year,1,1)
    end_date = datetime(year+1,1,1)
    delta = timedelta(1)
    ret = []
    while start_date < end_date:
        ret.append(start_date) 
        start_date += delta
    return ret
def treatYear(newspaper: Newspaper, year: int, dataFolder: str, resultFolder: str):
    for day in tqdm(yearGenerator(year)):
        newspaper.treatDate(day,dataFolder,resultFolder, False)

### FUNCTIONS FOR NER 

def get_nb_dates(doc,show = True, spacy = True):
    """
    Arguments : 
        - doc : the processed spaCy document from which to get the number of dates appearing in it
        - show : Boolean to decide whether to show an example of the recognized date if there is any in the doc 
    Return :
        - # of dates appearing in the doc
    """
    # For spacy :
    nb_dates = []
    if spacy:
        nb_dates = [ e for e in doc.ents if e.label_ == "DATE"] # Spacy pipeline
    else:
        nb_dates = [ e for e in doc if e["entity_group"] == "DATE"] # Hugging face pipeline
    if bool(nb_dates) and show:
        print("Example of date for this article: ",nb_dates[0])

        
    return len(nb_dates)

def get_dates_from_file(filepath, pipeline,spacy, rt = 0.75,lt = 15 ): 
    """
    Arguments : 
        - filepath : path of the file 
            The file path has to be of the form ./things/YYYY_MM__DD.csv, for example "./results/1920_10_01.csv"    
        - pipeline : spaCy pipeline ttah performs the NER
        - rt : Word ratio threshold under which we do not consider the text block 
            range : [0,1]
        - lt : Length threshold under which we do not consider the text block 
            range : (0,inf)
    Return : 
        - date: date at which the newspaper was issued
        - nb_dates : number of dates appearing in the file
        
    """
    date  = datetime.strptime(filepath[-14:-4], "%Y\\%m\\%d")
    df = pd.read_csv(filepath)
    filtered_df = df[(df.Length > lt) & (df.Ratio > rt)]
    processed_docs = [pipeline(block) for block in list(filtered_df["Text"])]
    nb_dates = sum([get_nb_dates(doc, show = False, spacy = spacy) for doc in processed_docs])
    return date,nb_dates
