from doctest import DocFileSuite
import xml.dom.minidom as xml
import pandas as pd 
from datetime import datetime, timedelta
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import numpy as np 
from glob import glob
import enchant
from typing import List
import scipy.optimize as opt
from sklearn.metrics import r2_score
from collections import deque
import json
import os
from os import path
from abc import ABC
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.graph_objects as go 
from calendar import monthrange

### Functions to access the result files

def get_df_from_path(path):
    all_files= glob(path + "/*/*.csv")
    li = []
    for file in all_files:
        date = file.split("/")[-1].split(".")[0]
        temp_df = pd.read_csv(file)
        temp_df["Date"] = date
        li.append(temp_df)
    df = pd.concat(li, axis=0, ignore_index=True)
    df = df[(df["Length"] > 0) & (df["Ratio"] > 0 )& (df["Ratio"]<1)]
    return df




### Classes to treat and compute OCR for each newspaper 



class Newspaper(ABC):
    name = ""
    language = ""
    def pathFromDate(self,date,folder):
        pass 
    def getBlocks(self,path):
        pass
    
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

    def checkPath(self,path):
        # print(path)
        files = glob(path + "*")
        return len(files) > 0
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
        if self.checkPath(path):
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

    def pathFromDate(self,date: datetime,folder):
        datepath = date.strftime("%Y%m%d")
        return folder + "//" + str(date.year) +  "//" + datepath + "//" + datepath  + ".metadata.fulltext.json"

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

class NYH(Newspaper):

    name = "NYH"
    language = 'en'
    dictionnary = enchant.Dict(language)
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

class Berlin(Newspaper):
    
    name = "Berlin"
    language = 'de'
    dictionnary = enchant.Dict(language)

    def pathFromDate(self,date,folder):
        year = date.strftime("%Y")
        yearfile = date.strftime("%Y%m%d")
        return folder + "//" + year + '//' + "F_SBB_00001_" + yearfile
    def getBlocks(self,path):
        folder = path[:-22]
        blocks = []
        for _,_,files in os.walk(folder):
            for file in files:
                if file.startswith(path[27:]):

                    blocks += self.xmlParser(folder + "//" + file)
        return blocks
    def checkPath(self,path):
        files = glob(path + "*")
        return len(files) > 0


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


def count_missing_dates(result_folder:str,year_range: list):
    result = []
    for year in year_range:
        for month in range(1,13):
            _ , nb_days = monthrange(year,month)
            prop_missing = 0
            nb_missing = 0
            month_path = result_folder + f"/{year}/{month:02d}"
            if path.isdir(month_path):
                for day in range(1,nb_days+1):
                    day_path = month_path + f"/{day:02d}.csv"
                    if not path.isfile(day_path):
                        nb_missing +=1                        
                prop_missing = nb_missing/nb_days
            else:
                prop_missing = 1
            result.append((year,month,prop_missing))
    res = pd.DataFrame(result, columns = ["Year", "Month", "missing_prop"])
    return res
            

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
    filtered_df = filtered_df.sample(min(15, len(filtered_df)))
    processed_docs = [pipeline(block) for block in list(filtered_df["Text"])]
    nb_dates = sum([get_nb_dates(doc, show = False, spacy = spacy) for doc in processed_docs])
    return date,nb_dates



def get_dates_from_year_range(year_range, source_folder, result_folder, model, useSpacy):
    beginning_year , ending_year = year_range
    for year in range(beginning_year, ending_year):
        results = []
        folder = source_folder  + f"{year}"
        for file in tqdm(glob(folder + "/*/*.csv")):        # Compute the nb of dates
            date, nb_dates = get_dates_from_file(file,model,useSpacy,rt = 0.75, lt = 15)
            results.append((date,nb_dates,pd.read_csv(file)["Length"].sum()))
        df = pd.DataFrame(results, columns = ["Date", "Number of dates", "length"])
        df.to_csv(result_folder + f"{year}.csv", index= False)
### Plotting functions

def plot_dfs_hist(dfs):
    plt.figure(figsize=(8, 6))

    for (df,name) in dfs:
        arr = df["Ratio"]
        weights = np.ones_like(arr)/float(len(arr))
        plt.hist(x = arr, bins = 20, alpha = 0.4, label = name, weights = weights)
    plt.title("Distribution of word ratio per newspaper")
    plt.xlabel("Word ratio")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig('histo.png')
    plt.show()


def plot_dfs_wr(dfs):

    def f(x,a,b): 
        if isinstance(x,pd.Series) or isinstance(x,np.ndarray):
            y = []
            for elem in x:
               y.append( a*np.log(elem) + b if elem != 0 else b)
            return y
        else:
            return a*np.log(x) + b if x != 0 else b
    
    for (df, name )in dfs:
        x = df["Length"]
        y = df["Ratio"]
        a, _ = opt.curve_fit(f,x,y)
        y_pred = f(x,a[0], a[1])
        r2 = r2_score(y,y_pred)
        plt.figure(figsize=(8, 6))

        plt.scatter(x = x, y = y,label = name, alpha = 0.3)
        x = np.linspace(0, max(x), num=5000)
        plt.plot(x, f(x,a[0], a[1]), c = "red")
        plt.title(f"Word ratio w.r.t. document length for the newspaper {name}")
        plt.xlabel("Length")
        plt.ylabel("Word ratio")
        plt.legend(["Text Block", "regression: $y = log({:0.3f}*x) + {:0.3}".format(a[0], a[1])])
        plt.savefig(f'dwr{name}.png')
        print(f"R2 score for the newspaper {name} : {r2}")
        plt.show()

def plot_missing_dates(df):
    df["date"] = pd.to_datetime(df['Year'].astype(str)  +  df['Month'].astype(str), format='%Y%m')
    beg_year = df["date"].min().year
    end_year = df["date"].max().year

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x = df["date"],
            y = df["missing_prop"].rolling(5).mean(),
            # line_shape = "spline"
        )
    )
    fig.update_layout(template = "none", title = f"Missing Le Figaro issues proportion per month between {beg_year} and {end_year}",font_size = 15)
    fig.update_xaxes(title = "Date")
    fig.update_yaxes(title = "Missing proportion per month")
    fig.show()

def plot_ner_results(result_folder):
    dfs = []
    for file in glob(result_folder  + "*.csv"):
        dfs.append(pd.read_csv(file))
    final_df = pd.concat(dfs)
    final_df = final_df.drop_duplicates()
    final_df["ratio"]= (final_df["Number of dates"]/final_df["length"]).rolling(window = 60).mean()
    # final_df["ratio"].plot()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = final_df["Date"],
        y = final_df["ratio"],
        mode = "lines"
        
    ))
    fig.update_layout(
        template = "none", 
        title = "Date ratio from 1875 to 1920 in the French newspaper Le Figaro"
    )
    fig.update_xaxes(title = "Dates")
    fig.update_yaxes(title = "Date ratio" )
    fig.show()