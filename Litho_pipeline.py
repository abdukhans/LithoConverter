import pandas as pd 
import numpy as np 
import os
import csv
import pickle
from .GenSimKWFreq import gen_similar_word_freq_csv,open_csv_df,gen_eng_compressed_csv,gen_llm_word_def_csv
from .BinVectorMarker import BinVectorMarker

def save_to_pickle(obj_name:str , data) -> None:
    """
        Descr:
            This function will just save a 

        Args:


        Ret:
    """
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

#Function which reduces the memory allocation of outputted .csv files
def compress_csv_memory(input_csv_path, output_csv_path):
    with open(input_csv_path, mode='r', newline='', encoding='utf-8') as infile, \
         open(output_csv_path, mode='w', newline='', encoding='utf-8') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        header = next(reader)
        writer.writerow(header)  # Write the header unchanged

        for row in reader:
            # Compress binary vector from "1.0,0.0,1.0,..." to "101..."
            compressed_bin_vec = ''.join(bit.strip()[0] for bit in row[2].split(','))
            writer.writerow([row[0], row[1], compressed_bin_vec])

    print(f"Compressed CSV written to: {output_csv_path}")
 
def gen_csvs(sample_descs:pd.Series,freq_threshold, out_dir='LITHO_CSVS', use_cache_if_exists=True) :
    

    CWD:str            = os.getcwd()
    OUT_CSV_FOLDER:str = os.path.join(CWD,out_dir)


    DEFAULT_COL:str  = '0'
    RED_COL:str      = '31'


    SIMILAR_KEY_WORD_FREQ:str           = "similar_keywords_freq.csv"
    FILITERED_SIMILAR_KEY_WORD_FREQ:str = "similar_keywords_compressed_freq.csv"

    SIMILAR_KEY_WORD_FREQ_FP:str    = os.path.join(OUT_CSV_FOLDER,SIMILAR_KEY_WORD_FREQ) 
    FILITERED_SIMILAR_KEY_WORD_FREQ_FP:str = os.path.join(OUT_CSV_FOLDER,FILITERED_SIMILAR_KEY_WORD_FREQ)

    CACHED_PY_OBJS = "CACHED_PY_OBJS"

    # Create the csv folders
    os.makedirs(OUT_CSV_FOLDER,exist_ok=True)

    # Create the cached py obj folder
    os.makedirs(CACHED_PY_OBJS,exist_ok=True)


    # Create the similar_keywords_freq.csv file
    gen_similar_word_freq_csv(sample_descs,SIMILAR_KEY_WORD_FREQ_FP,use_cache_if_exists)


    # Create the filtered "similar_keywords_freq.csv" csv file which is called "similar_keywords_compressed_freq.csv"
    gen_eng_compressed_csv(SIMILAR_KEY_WORD_FREQ_FP,FILITERED_SIMILAR_KEY_WORD_FREQ_FP,freq_threshold)
    


    # Query the open AI API to get 'word_def.csv'
    req_fp = os.path.join(OUT_CSV_FOLDER,'gpt_batch_req.jsonl')
    res_fp = os.path.join(OUT_CSV_FOLDER,'gpt_batch_res.jsonl')
    out_fp = os.path.join(OUT_CSV_FOLDER,'word_def.csv')

    gen_llm_word_def_csv(FILITERED_SIMILAR_KEY_WORD_FREQ_FP,req_fp,res_fp,out_fp)

    # Compress the filtered keyword frequency CSV
    compressed_fp = os.path.join(OUT_CSV_FOLDER, 'similar_keywords_compressed_freq.csv')
    compress_csv_memory(FILITERED_SIMILAR_KEY_WORD_FREQ_FP, compressed_fp)
