import pandas as pd
import os 
from  pathlib import Path
import chardet
import pickle
# from pattern3.en import lemma
from difflib import SequenceMatcher
from DictTree import LetterNode
from DictTree import ALPHABETS
import numpy as np
import csv
import re
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
# from pattern3.en import lemma
import nltk
nltk.download('words')
from nltk.corpus import words
from nltk.stem.snowball import SnowballStemmer
from difflib import SequenceMatcher
from tqdm import tqdm 
import tiktoken
import json
from openai import OpenAI
import jsonlines
import time
DICT_TREE= None
id_counter = 0
def sequence_matcher_similarity(str1, str2):
    return SequenceMatcher(None, str1, str2).ratio() > 0.8

def open_csv_df(path_to_csv:str )-> pd.DataFrame:
    """
       
    """
    cached_dir = "CACHED_DF"
    CWD  = os.getcwd()
    file_name = path_to_csv.split(os.sep)[-1].split('.')[0] + '.pkl'
    pickle_file_name = os.path.join(CWD,cached_dir,file_name)
    os.makedirs(cached_dir,exist_ok=True)
    if not(os.path.exists(pickle_file_name)):
        with open(path_to_csv,"rb") as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']

            df = pd.read_csv(path_to_csv,encoding=encoding,low_memory=False)
            df.to_pickle(pickle_file_name)
    else:

        
        df = pd.read_pickle(pickle_file_name)

    return df

stemmer = SnowballStemmer("english")
english_words = set(words.words())

def is_english_word(word):
    return word.lower() in english_words


# Download necessary data (run once)
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('omw-1.4')

LEMMATIZER = WordNetLemmatizer()
unique_descr = {}
node_map:dict[str,LetterNode] = {}

NOUN_TYPE = 'NN'
ADJ_TYPE = 'JJ'
TAG_SET = None
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def custom_root(word):
    # Direct mappings for exceptions or domain-specific words
    
    # General suffix rules
    suffix_rules = [
        (r'ified$', ''),     # e.g., 'modified' -> 'modify'
        (r'ation$', ''),     # e.g., 'alteration' -> 'alter'
        (r'eous$', ''),      # Handle cases like 'siliceous' -> 'silicify'
        (r'ed$', ''),           # Basic past tense removal
    ]
    
    for pattern, replacement in suffix_rules:
        if re.search(pattern, word):
            return re.sub(pattern, replacement, word)
    
    return word  # Return original if no rule applies

def get_english_sub_word(word):

    min_idx = len(word) - 1

    for i in range(min_idx,1,-1):

        if is_english_word(word[0:i]):
            min_idx = i
    

    return word[0:min_idx]





def get_root_word(word):



    eng_word = is_english_word(word)


    return word,eng_word
   


def ShouldCompress(main_word:str,lst_words:list[tuple[str,int]]):

    if len(main_word)< 3:
        return False
    a = np.array([sequence_matcher_similarity(main_word,sub_word) for sub_word, _ in lst_words])

    return any(a)

def CompressTree(node:LetterNode,root_node:LetterNode,node_map:dict[str,LetterNode],word_path:str):
    for child_char in node.childern:

        child = node.childern[child_char]

        if child:
            similar_words_freq = [  (sub_word,node_map[sub_word].word_freq) for sub_word in root_node.get_all_key_words_with_start(word_path)]


            if ShouldCompress(word_path,similar_words_freq):
                node.Compress_Node()
            else:
                CompressTree(child,root_node,node_map, word_path+child_char)
                pass

def Tree_To_Dict(node_map:dict[str,LetterNode],unique_descr,file_name:str):


    # total_words = root_node.get_total_words()

    word_freq = [ (sub_word,
                    node_map[sub_word].word_freq,
                    [ (k ,node_map[k].word_freq)  for k in node_map[sub_word].similar_words])
                                                  for sub_word  in unique_descr
                                        
                                    ]

    word_freq = sorted(word_freq,key= lambda x: x[1],reverse=True)



    with open(file_name,mode='w',newline='') as file:
        writer = csv.writer(file)

        writer.writerow(["Attribute","Frequency","Similar_Words"])
        writer.writerows(word_freq)



def save_to_pickle(obj_name:str , data):

    out_dir = 'CACHED_PY_OBJS'

    os.makedirs(out_dir,exist_ok=True)


    fp = os.path.join(os.getcwd(),out_dir,f"{obj_name}.pkl")
    with open(fp, "wb") as f:
        pickle.dump(data, f)


def get_pkl_obj(obj_name ) :
    out_dir = 'CACHED_PY_OBJS'

    fp = os.path.join(os.getcwd(),out_dir,f"{obj_name}.pkl")
    with open(fp, "rb") as f:
        data = pickle.load(f)

    return data

    
def gen_similar_word_freq_csv(df:pd.Series,out_file_csv:str,use_cache_if_exists:bool):
    
    print(f"Generating '{out_file_csv}'")


    if use_cache_if_exists and "DICT_TREE.pkl" in os.listdir("CACHED_PY_OBJS"):

        DICT_TREE = get_pkl_obj("DICT_TREE")
        node_map  = get_pkl_obj("node_map")
        unique_descr = get_pkl_obj("unique_descr")


    else:

        DICT_TREE = LetterNode('',False)                
        unique_descr:dict[str,int] = {}
        node_map:dict[str,LetterNode] = {}
        for i in tqdm(df, total=len(df)):
            descr = str(i)
            descr = ' '.join(re.findall(r'\w+',descr))

            descr    = descr.lower()
            words    =  word_tokenize(descr)

            pos_tags = pos_tag(words,tagset=TAG_SET)


            for (word,word_type) in pos_tags:


                root_word,eng_word = get_root_word(word)


                if root_word in unique_descr:
                    unique_descr[root_word] += 1
                    node_map[root_word].word_freq += 1

                elif word_type == NOUN_TYPE or word_type == ADJ_TYPE :
                    if root_word not in ['na','nan']:

                        if root_word != word:
                        

                            print(f"{root_word=} {word=} {eng_word=}")


                        is_alpah_num = list(map(lambda x: x in ALPHABETS,root_word))
                        if all(is_alpah_num):
                            DICT_TREE.addWord(root_word)
                            unique_descr[root_word] = 1
                            node_map[root_word] = DICT_TREE.get_node(root_word)
                            node_map[root_word].word_freq += 1
        save_to_pickle('DICT_TREE',DICT_TREE)
        save_to_pickle('node_map',node_map)
        save_to_pickle('unique_descr',unique_descr)

    CompressTree(DICT_TREE,DICT_TREE,node_map,'')  

    Tree_To_Dict(node_map,unique_descr,out_file_csv)




def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding('cl100k_base')
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_prompt(word:str) -> str:


    prompt =  f"""From the definition of the word {word},
give me the key words that are most important
in describing that word in a geological context.
Give the keywords as only a list seperated with commas, do not
say anything else, and make sure each keyword is only one word."""

    
    return re.sub(r"\n"," ",prompt)

def gen_eng_compressed_csv(in_fp:str,out_fp:str,FREQ_THRESHOLD=80) -> pd.DataFrame:
    

    ATTRIBUTE:str = "Attribute"
    FREQUENCY:str = "Frequency"

    freq_df:pd.DataFrame             = pd.read_csv(in_fp)
    attribs             = freq_df[ATTRIBUTE]
    freq_df['NUM_TOKS'] = attribs.apply(lambda x: num_tokens_from_string(get_prompt(str(x))) )
    english_mask        = freq_df[ATTRIBUTE].apply(lambda x: is_english_word(str(x)))

    freq_df = freq_df[english_mask]
    freq_mask = freq_df[FREQUENCY].apply(lambda x: int(x) >= FREQ_THRESHOLD)
    freq_df   = freq_df[freq_mask]

    freq_df.to_csv(out_fp)




    # TOT_IN_TOK = freq_df['NUM_TOKS'].sum()
    # AVG_TOK    = freq_df['NUM_TOKS'].mean() 

    # EST_OUT_TOK = int(2*TOT_IN_TOK)
    # ESTIMATED_PRICE  = (TOT_IN_TOK/1_000_000) * 3.00 + (EST_OUT_TOK/1_000_000) * 6.00


    # print(f"{TOT_IN_TOK  = }")

    # print(f"{EST_OUT_TOK = }")


    # print(f"{ESTIMATED_PRICE = }")
    # print(f"{AVG_TOK     = }")



    # print(freq_df.shape)
    return  freq_df



def get_json_gpt_req(word:str,model= "gpt-3.5-turbo",get_prompt=get_prompt):
    MODEL = model
    global id_counter
    id_counter += 1
    return {
        "custom_id" : f"request-{id_counter}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": 
            {
                "model": MODEL,
                "messages": 
                    [
                        {"role": "system", 
                        "content": "You are a helpful assistant."
                        },
                        {"role": "user",
                        "content": get_prompt(word)
                        }
                    ],
                "max_tokens": 1000
            }


        }


def create_batch_json(eng_df:pd.DataFrame,req_word_map:dict[str,str],out_fp:str,get_prompt = get_prompt):
    



    english_attribs = eng_df['Attribute']



    batch_json = []

    for word in english_attribs:
        req_obj = get_json_gpt_req(word,get_prompt=get_prompt)
        req_word_map[req_obj['custom_id']] = word
        batch_json.append(req_obj)
    
    
    with open(out_fp, 'w') as file:
        for obj in batch_json:
            file.write(json.dumps(obj) + "\n")

    return batch_json

def load_jsonl(file_path) -> list:
    with jsonlines.open(file_path, "r") as reader:
        return list(reader)


def gen_llm_word_def_csv(in_fp:str , req_fp:str , res_fp:str,out_fp:str, get_prompt = get_prompt) -> None:
    """
        Descr:
            This function will generate a .csv file with file path of 'out_fp' that will contain the 
            key word definition of each word. The argument 'in_fp' is a file path to a csv file that
            ATLEAST has the following columns

                Attribute,NUM_TOKS

            The 'Attribute' column contains the key words you want the key word definitions of.
            The 'NUM_TOKS' column contains the number of tokens and is used to esitmate the price
            of these request to the LLM. The way this function works is by sending each of the words
            outlined in 'Attribute' and asking an LLM (in this case an OPENAI LLM model) what are the
            list of key words in describing that word, it then saves this info to a csv file, that has
            the following columns:

                Req_id, Word, Key_words_defn

            Where 'Req_id' is just an identifier for a row, 'Word' is the word we want the definition of
            and   'Key_words_defn' contains the list of words, an example row could be:

                    Req_id   |Word   | Key_words_defn
                    _________+_______+_________________________________________________________________
                    request-1|igneous|"Magma, Crystallization, Melting, Volcanic, Rock, Solidification."
    
        Args:

            'in_fp': A string that is the file path of a .csv file you want the key word definitions of.
                    IT MUST HAVE THE FOLLOWING COLUMNS:  
        
                        Attribute, NUM_TOKS

                    The 'Attribute' colums contatins the words that you want the keyword definition of.
                    And the 'NUM_TOKS' contains the number of tokens which is used to estimate the price
                    of the LLM batch request. This function will ask your permission to continute to 
                    execute this function after it has calculated and displayed the price

            'out_fp': A string that is the file path of a .csv file that contains the definitons of each word.
                    This function will generate a .csv file and save to the file path: 'out_fp'. The gerneated
                    .csv file will have the following structure:


                        Req_id   |Word   | Key_words_defn
                        _________+_______+_________________________________________________________________
                        request-1|igneous|"Magma, Crystallization, Melting, Volcanic, Rock, Solidification."

            'req_fp': A string that is the file path to a .jsonl file that will be generated by this function.
                    This .jsonl file will inlcude the OPENAI batch request.

            'res_fp': A string that is the file path to a .jsonl file that will be generated by this funciton.
                    This .jsonl file will include the OPENAI batch response data, from the LLM.

            'get_prompt': A function that takes in a word as string and returns  a prompt that will be sent to the LLM. 
        
        Ret:
            No Return. This function will generate one .csv file that contatins the definition of each word, and two .jsonl files
            that which are the requests and responses to the OPENAI API. 
    """

    global id_counter

    # A variable used as a unique identifier for each request to the LLM
    id_counter = 0

    # Load the dataframe from 'in_fp'
    eng_df:pd.DataFrame = pd.read_csv(in_fp)

    # Calculte info needed for price estimate
    TOT_IN_TOK:int   = eng_df['NUM_TOKS'].sum()
    AVG_TOK:int      = eng_df['NUM_TOKS'].mean() 
    EST_OUT_TOK:int  = int(2*TOT_IN_TOK)

    # Calculate an UPPER BOUND estimate of the price. (this is a formula I saw online and on the website)
    ESTIMATED_PRICE:int  = (TOT_IN_TOK/1_000_000) * 3.00 + (EST_OUT_TOK/1_000_000) * 6.00



    print(f"{TOT_IN_TOK  = }")
    print(f"{EST_OUT_TOK = }")
    print(f"{ESTIMATED_PRICE = }")
    print(f"{AVG_TOK     = }")
    print(eng_df.shape)



    # This is a dictionary that will map each request-id to a word.
    # so for example  req_word_map['request-17'] = 'igneous' this
    # means that the request with id : 'request-17' is asking for 
    # key word definitions of 'igneous'.
    req_word_map:dict[str,str] = {}


    # This function populates 'req_word_map' and generates
    # a .jsonl file that contains the list of requests that will be made to
    # the LLM, and will save it to the file path: "req_fp"
    create_batch_json(eng_df,req_word_map,req_fp)


    # This sends batch file generated 'create_batch_json' 
    # to the OpenAI api. Its important to note that 
    # you must have the OPEN AI API key set up in order to 
    # run this function. If you don't this 
    # function will crash the program. Note it just send the batch
    # request file to the open AI API, and will not actually
    # process the batch request
    client:OpenAI = OpenAI()
    
    # This returns a file object class, that can be used
    # to query info about the batch request to the LLM
    batch_input_file = client.files.create(
        file=open(req_fp, "rb"),
        purpose="batch"
    )

    




    ans = input("""
[WARNING] Continuing this function will cost real money. The above ESTIMATED_PRICE 
          is a rough estimate of the price you will pay.
          Are you sure you want to continue? [Y/N]
""")
    



    if ans.lower() not in ['y','yes']:
        print(f"Aborted '{out_fp}'  generation")
        return



    # This is needed to get the id of the batch request.
    # This info is needed to kickstart the processsing of the
    # batch file 
    batch_input_file_id:str = batch_input_file.id

    # This processess the batch request [this is the part of the code that costs MONEY]
    meta_data = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
        "description": "Get key word definitions of attribute words"
        }
    )



    # We continually query the progress of the batch request and stop once it is finished
    while meta_data.status != 'completed' and meta_data.errors == None:
        
        # Display the progress of the batch
        print(f"\r[OPENAI_API] Completed {meta_data.request_counts.completed}/{meta_data.request_counts.total}",end='',flush=True)

        # Make a query to the API to get the progress info
        meta_data = client.batches.retrieve(meta_data.id)
        
        # The OPENAI API has a rate limiter.
        # So we have make sure the line of code above does not
        # execute too quickly in the while loop.
        # So we wait 0.1s to give avoid triggering  
        # OPENAI API rate limiter
        time.sleep(0.1)

    if meta_data.errors:
        print(f"[OPENAI_API] Error: {meta_data.errors}")
        return
    

    # This gets the results of each request  
    file_response = client.files.content(meta_data.output_file_id)

    # We write to the file path: 'res_fp' the resuls of each 
    # request from the batch request 
    with open(res_fp, 'wb') as f:
        f.write(file_response.content)

    # We then load the results from the LLM.
    results:list = load_jsonl(res_fp)
    


    # Each request had a unique request id parameter that had to be specified.
    # Each response has also has a unique response id that corressponeds to the
    # request id. So the response with id '17' was the response to the
    # the request with id '17'. The dictionary below maps each response/req id
    # to the corresponding output from the LLM. 
    req_llm_map:dict[str,str] = { i['custom_id']: i['response']['body']['choices'][0]['message']['content'] for i in results}
    
    # This maps each word from the 'Attribute' column in 'in_fp'
    # to the corresponding output from the LLM 
    data:list[tuple[str,str,str]] = [ (req_id,req_word_map[req_id],req_llm_map[req_id]) for req_id in req_word_map]



    # we save this data to 'out_fp'
    with open(out_fp, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        
        # Writing the header
        writer.writerow(["Req_id", "Word", "Key_words_defn"])

    
        writer.writerows(data)



