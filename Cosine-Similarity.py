#!/usr/bin/env python
# coding: utf-8

# # PreProcessing

# In[54]:


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
        #after_lemmatizer.append(wordnet_lemmatizer.lemmatize(after_lower[i]))
        after_stemming.append(porter_stemmer.stem(after_lower[i]))
    return after_stemming


# # TF-IDF Calculation

# In[55]:


import os
import math
import codecs
TF_Dictionary={}
parent="C:/Users/Devashi Jain/Desktop/Information Retrieval/Assignment2/stories/stories"
for file_name in os.listdir(os.path.join(parent)): 
        preprossed_file=[]
        Normalized_TF={}
        if os.path.isdir(parent+'/'+file_name):
            for file in os.listdir(os.path.join(parent,file_name)):
                if file=='index.html':
                    print("no")
                else:
                    fd=codecs.open(parent+'/'+file_name+'/'+file,'r',errors='ignore',encoding='utf-8')
                    preprossed_file=Pre_Processing(fd.read())
                    print(len(preprossed_file))
                    for i in range(len(preprossed_file)):
                        
                        if preprossed_file[i] in Normalized_TF:
                            Normalized_TF[preprossed_file[i]]+=1
                        else:
                            Normalized_TF[preprossed_file[i]]=1
                    s=0
                    for term in Normalized_TF:
                        s=s+(Normalized_TF[term])**2
                    p=math.sqrt(s)
                    for term in Normalized_TF:
                        Normalized_TF[term]/=p
                    for i in range(len(preprossed_file)):
                    
                        if preprossed_file[i] in TF_Dictionary:
                            TF_Dictionary[preprossed_file[i]][file]=Normalized_TF[preprossed_file[i]]/len(preprossed_file)
                        else:
                            TF_Dictionary[preprossed_file[i]] = {file:Normalized_TF[preprossed_file[i]]/len(preprossed_file)}


        else:
            if file_name=='index.html':
                print("no")
            else:
                fd=codecs.open(parent+'/'+file_name,'r',errors='ignore',encoding='utf-8')
                preprossed_file=Pre_Processing(fd.read())
                for i in range(len(preprossed_file)):
                        if preprossed_file[i] in Normalized_TF:
                            Normalized_TF[preprossed_file[i]]+=1
                        else:
                            Normalized_TF[preprossed_file[i]]=1
                s=0
                for term in Normalized_TF:
                        s=s+(Normalized_TF[term])**2
                p=math.sqrt(s)
                for term in Normalized_TF:
                    Normalized_TF[term]/=p
                for i in range(len(preprossed_file)):
                    if preprossed_file[i] in TF_Dictionary:
                        TF_Dictionary[preprossed_file[i]][file_name]=Normalized_TF[preprossed_file[i]]/len(preprossed_file)
                    else:
                        TF_Dictionary[preprossed_file[i]] = {file_name:Normalized_TF[preprossed_file[i]]/len(preprossed_file)}


# In[56]:


import pandas as pd
File_title=pd.DataFrame()
File_Title_SRE=pd.DataFrame()
File_title=pd.read_html('index.html', flavor='bs4')
File_Title_SRE=pd.read_html('C:/Users/Devashi Jain/Desktop/Information Retrieval/Assignment2/stories/stories/SRE/index.html',flavor='bs4')
Title=pd.concat([File_title[0],File_Title_SRE[0]])
Title=Title.drop([1],axis=1)
Title=Title.reset_index(drop=True)
Title=Title.drop([0,1,2,3,452])
Title=Title.reset_index(drop=True)
Title


# In[57]:


Title_Dictionary={}
for i in range(len(Title)):
    Title_Dictionary.update({Title[0][i]:Title[2][i]})


# In[58]:


TF_Title_Dictionary={}
for term in Title_Dictionary.keys():
    Normalized_Title_TF={}
    preprossed_file=Pre_Processing(Title_Dictionary[term])
    #print(len(preprossed_file))
    for i in range(len(preprossed_file)):
                if preprossed_file[i] in Normalized_Title_TF:
                    Normalized_Title_TF[preprossed_file[i]]+=1
                else:
                    Normalized_Title_TF[preprossed_file[i]]=1
    s=0
    for term1 in Normalized_Title_TF:
            s=s+(Normalized_Title_TF[term1])**2
    p=math.sqrt(s)
    for term2 in Normalized_Title_TF:
        Normalized_Title_TF[term2]/=p
    for i in range(len(preprossed_file)):
        if preprossed_file[i] in TF_Title_Dictionary:
            TF_Title_Dictionary[preprossed_file[i]][term]=Normalized_Title_TF[preprossed_file[i]]/len(preprossed_file)
        else:
            TF_Title_Dictionary[preprossed_file[i]] ={term:Normalized_Title_TF[preprossed_file[i]]/len(preprossed_file)}


# # QueryProcessing

# In[59]:


import math
query_dic={}
print("Enter the query")
query=input()
Cosine_sim={}
Cosine_sim_title={}
preprocessed_query=Pre_Processing(query)
for i in range(len(preprocessed_query)):
    if preprocessed_query[i] in query_dic:
        query_dic[preprocessed_query[i]]+=1
    else:
        query_dic[preprocessed_query[i]]=1
#for calculating dictionary for normal documents
for term in query_dic:
    if term in TF_Dictionary:
        IDF=1+math.log(467/len(TF_Dictionary[term]))
        query_dic[term]*=IDF

    elif term in TF_Title_Dictionary:
        p=len(TF_Title_Dictionary[term])
        IDF=1+math.log(467/len(TF_Dictionary[term]))
        query_dic[term]*=IDF
    
    else :
        query_dic[term]=0

for term in query_dic:
    if term in TF_Dictionary:
        for file in TF_Dictionary[term]:
            if file in Cosine_sim:
                Cosine_sim[file]+=(query_dic[term]*TF_Dictionary[term][file])
            else:
                Cosine_sim[file]=query_dic[term]*TF_Dictionary[term][file]


for term in query_dic:
    if term in TF_Title_Dictionary:
        for file in TF_Title_Dictionary[term]:
            if file in Cosine_sim_title:
                Cosine_sim_title[file]+=(query_dic[term]*TF_Title_Dictionary[term][file])
            else:
                Cosine_sim_title[file]=query_dic[term]*TF_Title_Dictionary[term][file]


# In[60]:


for term in TF_Dictionary:
    for file in TF_Dictionary[term]:
            TF_Dictionary[term][file]*0.4
for term in TF_Title_Dictionary:
    for file in TF_Title_Dictionary[term]:
            TF_Title_Dictionary[term][file]*=0.6


# In[61]:


for term in Cosine_sim:
            Cosine_sim[term]*=0.4
for term in Cosine_sim_title:
            Cosine_sim_title[term]*=0.6


# In[62]:


Top_Dic={}
for term in Cosine_sim:
    if term in Cosine_sim_title:
        Top_Dic[term]=Cosine_sim[term]+Cosine_sim_title[term]
    else :
        Top_Dic[term]=Cosine_sim[term]
for term1 in Cosine_sim_title:
    if term1 not in Top_Dic:
        Top_Dic[term1]=Cosine_sim_title[term1]


# In[ ]:





# In[63]:


topk=(sorted(Top_Dic.items(), key = lambda kv:(kv[1], kv[0] ),reverse=True)) 


# # Top K Documents

# In[64]:


topk=(sorted(Top_Dic.items(), key = lambda kv:(kv[1], kv[0] ),reverse=True)) 
print(len(topk))
print("enter the top doc")
t=int(input())
if t<len(Top_Dic):
    for i in range(t):
        print(topk[i])
elif len(Top_Dic)==0:
    print("No document found")
else :
    print("The top documents are :")
    for i in range(len(Top_Dic)):
        print(topk[i])
        


# In[65]:


Top_Dic


# In[66]:


query_dic


# In[ ]:





# In[ ]:




