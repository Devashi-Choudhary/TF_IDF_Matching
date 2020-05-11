#!/usr/bin/env python
# coding: utf-8

# # Preprocessing

# In[1]:


from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from nltk.stem.porter import PorterStemmer
porter_stemmer  = PorterStemmer()
import re
import inflect
def Pre_Processing(file):
    token_files=[]
    after_lower=[]
    after_lemmatizer=[]
    after_stemming=[]
    tokenizer = RegexpTokenizer(r'\w+')
    tokens=(tokenizer.tokenize(file))
    p = inflect.engine()
    token_files=[]
    for i in range(len(tokens)):
        if tokens[i].isnumeric() and len(tokens[i])<36:
            tem=p.number_to_words((tokens[i]))
            tokenizer = RegexpTokenizer(r'\w+')
            temp=(tokenizer.tokenize(tem))
            for x in temp:
                token_files.append(x)
        elif tokens[i].isnumeric() and len(tokens[i])>36:
            for j in range(len(tokens[i])):
                token_files.append(p.number_to_words((tokens[i][j])))
        else:
            token_files.append(tokens[i])
    for i in range(len(token_files)):
        after_lower.append(token_files[i].lower())
    for i in range(len(after_lower)):
        after_stemming.append(porter_stemmer.stem(after_lower[i]))
    return after_stemming


# # TF_IDF Calculation

# In[2]:


import os
import codecs
doc=0
TF_Dictionary={}
parent="C:/Users/Devashi Jain/Desktop/Information Retrieval/Assignment2/stories/stories"
for file_name in os.listdir(os.path.join(parent)): 
        preprossed_file=[]
        if os.path.isdir(parent+'/'+file_name):
            for file in os.listdir(os.path.join(parent,file_name)):
                if file=='index.html':
                    print("no")
                else:
                    fd=codecs.open(parent+'/'+file_name+'/'+file,'r',errors='ignore',encoding='utf-8')
                    preprossed_file=Pre_Processing(fd.read())
                    for i in range(len(preprossed_file)):
                        if preprossed_file[i] in TF_Dictionary:
                            if file in TF_Dictionary[preprossed_file[i]]:
                                TF_Dictionary[preprossed_file[i]][file]+=1
                            else:
                                 TF_Dictionary[preprossed_file[i]][file]=1
                        else:
                            TF_Dictionary[preprossed_file[i]] = {file:1}
                    for term in TF_Dictionary:
                        for j in TF_Dictionary[term]:
                            if(j==file):
                                TF_Dictionary[term][j]/=len(preprossed_file)

        else:
            if file_name=='index.html':
                print("no")
            else:
                fd=codecs.open(parent+'/'+file_name,'r',errors='ignore',encoding='utf-8')
                preprossed_file=Pre_Processing(fd.read())
                for i in range(len(preprossed_file)):
                        if preprossed_file[i] in TF_Dictionary:
                            if file_name in TF_Dictionary[preprossed_file[i]]:
                                TF_Dictionary[preprossed_file[i]][file_name]+=1
                            else:
                                 TF_Dictionary[preprossed_file[i]][file_name]=1
                        else:
                            TF_Dictionary[preprossed_file[i]] = {file_name:1}
                for term in TF_Dictionary:
                    for j in TF_Dictionary[term]:
                        if(file_name==j):
                            TF_Dictionary[term][j]/=len(preprossed_file)


# In[3]:


import math
for term in TF_Dictionary:
    IDF=1+math.log(467/len(TF_Dictionary[term]))
    for file in TF_Dictionary[term]:
        TF_Dictionary[term][file]*=IDF


# In[4]:


import pandas as pd
File_title=pd.DataFrame()
File_Title_SRE=pd.DataFrame()
File_title=pd.read_html('C:/Users/Devashi Jain/Desktop/Information Retrieval/Assignment2/stories/stories/index.html', flavor='bs4')
File_Title_SRE=pd.read_html('C:/Users/Devashi Jain/Desktop/Information Retrieval/Assignment2/stories/stories/SRE/index.html',flavor='bs4')


# In[5]:


Title=pd.concat([File_title[0],File_Title_SRE[0]])
Title=Title.drop([1],axis=1)
Title=Title.reset_index(drop=True)
Title=Title.drop([0,1,2,3,452])
Title=Title.reset_index(drop=True)
Title


# In[6]:


Title_Dictionary={}
for i in range(len(Title)):
    Title_Dictionary.update({Title[0][i]:Title[2][i]})


# In[7]:


TF_Title_Dictionary={}
for term in Title_Dictionary:
    preprossed_file=Pre_Processing(Title_Dictionary[term])
    for i in range(len(preprossed_file)):
        if preprossed_file[i] in TF_Title_Dictionary:
            if term in TF_Title_Dictionary[preprossed_file[i]]:
                TF_Title_Dictionary[preprossed_file[i]][term]+=1
            else:
                TF_Title_Dictionary[preprossed_file[i]][term]=1
        else:
            TF_Title_Dictionary[preprossed_file[i]] = {term:1}
    for word in TF_Title_Dictionary:
        for file in TF_Title_Dictionary[word]:
            if(file==term):
                TF_Title_Dictionary[word][file]/=len(preprossed_file)


# In[8]:


import math
for term in TF_Title_Dictionary:
    p = len(TF_Title_Dictionary[term])
    IDF=1+math.log(467/p)
    for file in TF_Title_Dictionary[term]:
        TF_Title_Dictionary[term][file]*=IDF


# # QueryProcessing and Top k documents

# In[11]:


print("Enter the Query")
query=input()
Document={}
preprossed_query=Pre_Processing(query)
for term in preprossed_query:
    if term in TF_Title_Dictionary:
        for key in TF_Title_Dictionary[term].keys():
            if key in Document:
                Document[key]+=TF_Title_Dictionary[term][key]
            else:
                Document[key]=TF_Title_Dictionary[term][key] 
    if term in TF_Dictionary:
        for key in TF_Dictionary[term].keys():
            if key in Document:
                 Document[key]+=TF_Dictionary[term][key]
            else:
                Document[key]= TF_Dictionary[term][key]
topk=(sorted(Document.items(), key = lambda kv:(kv[1], kv[0] ),reverse=True)) 
print(len(topk))
print("enter the top doc")
t=int(input())
if t<len(Document):
    for i in range(t):
        print(topk[i])
elif len(Document)==0:
    print("No document found")
else :
    print("The top documents are :")
    for i in range(len(Document)):
        print(topk[i])


# In[ ]:




