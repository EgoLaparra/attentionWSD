# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 16:31:09 2016

@author: egoitz
"""
import os
import re
import tempfile

def edit_output (tmpoutput):
    sense_dict = {}
    wn_file = open('/home/egoitz/Data/Resources/WordNet/WordNet-3.0/dict/index.sense','r')
    for line in wn_file:
        fields = line.rstrip().split()
        sensekey = fields[0]
        pos = re.sub('.+\%','',sensekey.split(':')[0])
        lemma = re.sub('\%.+','',sensekey.split(':')[0])
        if pos == '1':
            pos = 'n'
        elif pos == '2':
            pos = 'v'
        elif pos == '3' or pos == '5':
            pos = 'a'
        elif pos == '4':
            pos = 'r'
        synset = fields[1]
        sense_dict[lemma + "-" + pos + " " + synset + "-" + pos] = sensekey
    wn_file.close()
    
    forevalids = []
    forevalwords = {}
    input_file = open(tmpoutput.name, 'r')
    for line in input_file:
        if not re.match('^!!',line):
            fields = line.rstrip().split()
            doc = re.sub('\.s[0-9]+.+','',fields[0])
            tokid = re.sub('\.a[0-9]+','',fields[0])
            lemma = fields[-1]
            i = 1
            while fields[i] != "!!":
                synset = fields[i]            
                sensekey = sense_dict[lemma + " " + synset]
                if not forevalwords.has_key(doc + " " + tokid):
                    forevalwords[doc + " " + tokid] = []
                    forevalids.append(doc + " " + tokid)
                forevalwords[doc + " " + tokid].append(sensekey)
                i += 1
    input_file.close()
    
    tmpedited = tempfile.NamedTemporaryFile(delete=False)
    for tid in forevalids:
        tmpedited.write (tid + " " + " ".join(forevalwords[tid]) + "\n")
    tmpedited.close()
    
    return tmpedited

    
def run_scorer (tmpedited):
    scorer = '/home/egoitz/Data/Datasets/WSD/Senseval-3/scorer2'
    gs = '/home/egoitz/Data/Datasets/WSD/Senseval-3/EnglishAW/test/EnglishAW.test.key'
    os.system(scorer + ' ' + tmpedited.name + ' ' + gs)