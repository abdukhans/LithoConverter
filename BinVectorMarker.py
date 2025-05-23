import pandas as pd
import numpy as np
from DefnWordGraph import DefnWordGraph
import os
from tqdm import tqdm

import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from nltk import word_tokenize, pos_tag
import nltk
nltk.download('words')
from nltk.corpus import words
from tqdm import tqdm
from functools import lru_cache 
import ast

import re
# Enable tqdm for pandas
tqdm.pandas()



TAGSET = 'universal'
word_list = set(words.words())

def is_valid_english_word(word):
    return word.lower() in word_list

NOUN_TYPE = 'NOUN'
ADJ_TYPE  = 'ADJ'



LEMMATIZER = WordNetLemmatizer()
def get_key_words(sample_dec:str) -> list[str]:

    sample_dec = ' '.join(re.findall(r'\w+',sample_dec))

    sample_dec    = sample_dec.lower()
    words    =  word_tokenize(sample_dec)

    pos_tags = pos_tag(words,tagset=TAGSET)


    key_words:list[str] = []


    for (word,word_type) in pos_tags:

        if word_type in [NOUN_TYPE, ADJ_TYPE]:
            key_words.append(word)


    return key_words

class BinVectorMarker:

    def __init__(self,similar_word_csv_path:str,word_def_csv_path:str,filter_num_feat=0,cache_dir='cache_dir') -> None:


        self.SIMILAR_WORD_CSV_FP:str = similar_word_csv_path
        self.WORD_DEF_CSV_FP:str     = word_def_csv_path
        

        # This has cols: [Attribute,Frequency,Similar_Words,NUM_TOKS]
        """
            Attribute: a string 

            Frequency: a string that contains the number of times 
                       the Attribute word appeared 

            Similar_Words: A list of tuples where each tuple has
                        
                           (a word similar to the word in Attribute, Freq of that word )

            NUM_TOKS: an integer that shows how manys token are in the corresponding prompt 

        """
        self.similar_words_df:pd.DataFrame = pd.read_csv(similar_word_csv_path)

        # This has cols: [Req_id,Word,Key_words_defn]
        self.word_def_df:pd.DataFrame      = pd.read_csv(word_def_csv_path)


        self.DEFN_GRAPH:DefnWordGraph = DefnWordGraph()

        self.DEFN_GRAPH = DefnWordGraph()


        print(f"Generating Defn word graph for BinVectorMarker")
        for row in tqdm(self.word_def_df.itertuples(),total=len(self.word_def_df)):

            word     = row.Word
            word_def = row.Key_words_defn

            word_def =[ def_word.lstrip().rstrip().lower()  for def_word in word_def.split(',')]

            self.DEFN_GRAPH.add_word(word)
            self.DEFN_GRAPH.add_def_words(word,word_def)
        self.DEFN_GRAPH.get_all_def_words()


        print(f"Generating 'similar_key_word_map' and 'def_word_idx_map'")
        self.similar_key_word_map = {}
        self.def_word_idx_map    = {}

        idx_counter =0
        for row in tqdm(self.similar_words_df.itertuples(),total=len(self.similar_words_df)):

            key_word = row.Attribute
            

            similar_words_freq = ast.literal_eval(row.Similar_Words)

            for similar_word , _  in similar_words_freq:
                if similar_word not in self.similar_key_word_map:
                    self.similar_key_word_map[similar_word] = key_word


            def_words = self.DEFN_GRAPH.word_node_map[key_word].def_words

            for word in def_words:

                if word not in self.def_word_idx_map:
                    self.def_word_idx_map[word] = idx_counter

                    idx_counter +=1 


        self.num_feats = len(self.def_word_idx_map)

        self.filtered_num_feats = filter_num_feat if filter_num_feat > 0  else self.num_feats


        # print(f"Generating Definition Root Word tofreq Map")

        # self.defn_root_word_idx_map = {}
        # self.similar_words_df.set_index("Attribute")
        # for word in self.similar_words_df:





    def gen_bin_vec(self, sample_descr:str) -> np.ndarray:


        

        
        result    = np.zeros(self.num_feats)

        if sample_descr in ['','na','nan','None']:
            return result

        key_words = get_key_words(sample_descr)


        for key_word in key_words:

            if key_word in self.similar_key_word_map:
                root_key_word = self.similar_key_word_map[key_word]

                def_words = self.DEFN_GRAPH.word_node_map[root_key_word].def_words

                for def_word in def_words:

                    idx = self.def_word_idx_map[def_word]
                    result[idx] = 1 


        return result[:self.filtered_num_feats]

    
    def gen_bin_vecs(self,sample_descs:pd.Series) -> pd.Series:

        return sample_descs.progress_apply( lambda x : self.gen_bin_vec(str(x)))



