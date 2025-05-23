import pandas as pd 
import numpy as np 
import os
import pickle
from GenSimKWFreq import gen_similar_word_freq_csv,open_csv_df,gen_eng_compressed_csv,gen_llm_word_def_csv
from BinVectorMarker import BinVectorMarker

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

 
def gen_csvs(sample_descs:pd.Series,out_dir='LITHO_CSVS',use_cache_if_exists=True) :
    

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
    gen_eng_compressed_csv(SIMILAR_KEY_WORD_FREQ_FP,FILITERED_SIMILAR_KEY_WORD_FREQ_FP)
    


    # Query the open AI API to get 'word_def.csv'
    req_fp = os.path.join(OUT_CSV_FOLDER,'gpt_batch_req.jsonl')
    res_fp = os.path.join(OUT_CSV_FOLDER,'gpt_batch_res.jsonl')
    out_fp = os.path.join(OUT_CSV_FOLDER,'word_def.csv')

    gen_llm_word_def_csv(FILITERED_SIMILAR_KEY_WORD_FREQ_FP,req_fp,res_fp,out_fp)


if __name__ == "__main__":
    CWD             = os.getcwd()
    BRIT_DF_PATH    = os.path.join(CWD,"British Columbia","Lithogeochemical","lithogeochem_data.csv")
    USGS_DF_PATH    = os.path.join(CWD,"USGS","Concentration_lithology_data.csv")
    SARGI_DF_PATH   = os.path.join(CWD,"SARIG","rock_lith_mod_counts_sarig.csv")
    ONTARIO_DF_PATH = os.path.join(CWD,"Ontario", "Ontario_rock.csv")


    
    brit_df       = open_csv_df(BRIT_DF_PATH)
    usgs_df       = open_csv_df(USGS_DF_PATH)
    sarig_df      = open_csv_df(SARGI_DF_PATH)
    ontario_df    = open_csv_df(ONTARIO_DF_PATH)

    df:pd.Series = pd.concat([
                    brit_df['Sample_Desc'], 
                    # usgs_df['ADDL_ATTR'],
                    # usgs_df['SPEC_NAME'],
                    # usgs_df["XNDRYCLASS"],
                    # sarig_df['feature'],
                    # ontario_df['ROCK'],
                    # ontario_df['ROCK.1'],
                    # ontario_df['ROCK.2']
                    ],axis=0)


    print(df.shape)
    csv_out_folder = 'LITHO_CSVS'


    if not(os.path.exists(csv_out_folder)):
        gen_csvs(df,out_dir=csv_out_folder)
    


    
    sim_key_word_csv_path = os.path.join(CWD,csv_out_folder,"similar_keywords_compressed_freq.csv")
    word_def_csv_path    = os.path.join(CWD,csv_out_folder,"word_def.csv")
    BIN_VEC_MARKER = BinVectorMarker(sim_key_word_csv_path,word_def_csv_path)


    bin_vecs= BIN_VEC_MARKER.gen_bin_vecs(df).apply(lambda x: ','.join(map(str, x)))


    bin_vec_df = pd.concat([df,bin_vecs],keys=['Sample_Descr','Bin_Vec'],axis=1)

    bin_vec_df.to_csv("bin_vec.csv")

