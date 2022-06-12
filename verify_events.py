import re
import pandas as pd
import sent2vec

from nltk import word_tokenize

from string import punctuation
from gensim.utils import simple_preprocess
from scipy.spatial import distance
from nltk.stem import WordNetLemmatizer

import pandas as pd
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.exporter.xes import exporter as xes_exporter


def time_analysis(df):
    p1 = df
    p1['Date_x'] = pd.to_datetime(p1['Date_x'])
    p1['Date_y'] = pd.to_datetime(p1['Date_y'])
    p1['Date Difference'] = (p1['Date_x'] - p1['Date_y']).abs()

    pat = re.compile(r"([AP])")

    li = []
    for index, row in p1.iterrows():
        v = pat.sub(" \\1", str(row.Time_x))
        li.append(v)

    p1['Time_x'] = li
    p1['Time_x'] = p1['Time_x'].replace("nan", "")
    p1['Time_x'] = p1['Time_x'].replace("AM", "")
    return p1

def compare_events(extracted_events, eventlog_events, matched_events, time_matched_events):
    d1=pd.read_csv(extracted_events)
    d2=pd.read_csv(eventlog_events)

    d1['Timestamp'] = d1['Timestamp'].str.lstrip()

    d1[['Date', 'Time']] = d1["Timestamp"].str.split(" ", expand=True, n=1)
    d2[['Date', 'Time']] = d2["Timestamp"].str.split(" ", expand=True, n=1)

    d1=d1.rename(columns ={'Activity':'extracted_Activity'})
    d2 =d2.rename(columns={'Activity': 'eventlog_Activity'})
    d3o2 = d1.merge(d2, on=['HADM_ID'])
    out=time_analysis(d3o2)
    out.to_csv(matched_events)
    d3o1= d3o2[d3o2['Date Difference']=='0 days']
    d3o1.to_csv(time_matched_events)

def calcute_semantic_similarity(pre_trained_model_path, input_csv, semantic_similariy_csv):

    lemmatizer = WordNetLemmatizer()
    
    model_path =pre_trained_model_path
    model = sent2vec.Sent2vecModel()
    try:
        model.load_model(model_path)
    except Exception as e:
        print(e)
    print('model successfully loaded')

    stop = open("smart_stopword.txt", "r")
    k = stop.readlines()
    k = ([i.strip('\n') for i in k])
    stop_words = [p.lower() for p in k]

    def preprocess_sentence(text):
        text = text.replace('/', ' / ')
        text = text.replace('.-', ' .- ')
        text = text.replace('.', ' . ')
        text = text.replace('\'', ' \' ')
        text = text.lower()
        tokens = [token for token in simple_preprocess(text, min_len=0, max_len=float("inf")) if
                  token not in stop_words]
        # tokens = [token for token in word_tokenize(text) if token not in punctuation and token not in stop_words]

        return ' '.join(tokens)

    d0 = pd.read_csv(input_csv)
    d0.dropna()
    d0['extracted_Activity'] = d0['extracted_Activity'].str.strip()

    li = []
    for index, row in d0.iterrows():
        w_l1 = preprocess_sentence(str(row.extracted_Activity))
        w_l2 = preprocess_sentence(str(row.eventlog_Activity))

        vec_1 = model.embed_sentence(w_l1)
        vec_2 = model.embed_sentence(w_l2)
        cosine_sim = 1 - distance.cosine(vec_1, vec_2)
        li.append(cosine_sim)

    d0['Similarity_Tweet'] = li
    #d0['Similarity_Wiki'] = li
    #d0['Similarity_Bio'] = li

    d0.to_csv(semantic_similariy_csv)
    
def get_matched_events(semantic_similariy_csv, matched_events_csv, timestamp_updated_csv):
    d1= pd.read_csv(semantic_similariy_csv)
    matched_events = d1.loc[d1.groupby(["HADM_ID","eventlog_Activity","Timestamp_y"])['Sim_Bio'].idxmax()].reset_index(drop=True)

    # Bio threshld avg+ 2*std  = 0.295439
    matched_events= matched_events[matched_events['Sim_Bio'] > 0.295439]
    matched_events = matched_events.loc[matched_events.groupby(["HADM_ID","extracted_Activity","Timestamp_x"])['Sim_Bio'].idxmax()].reset_index(drop=True)
    matched_events.to_csv(matched_events_csv)
    
    #Update timestamps
    dis_charge= matched_events[(matched_events['Source_x'] == 'Discharge summary')]
    event_specific = ['Radiology', 'Echo']
    events = matched_events[matched_events['Source_x'].isin(event_specific)]
    events['Timestamp_y'] = events['Timestamp_x']
    
    df=pd.DataFrame()
    df=pd.concat([dis_charge, events])
    df.to_csv(timestamp_updated_csv)

    
def add_new_events(matched_events_csv, all_events_csv, event_log_csv, enhanced_log_csv):
    d1= pd.read_csv(matched_events_csv)
    d2= pd.read_csv(all_events_csv)
    d3= pd.read_csv(event_log_csv)
    d4= pd.read_csv(enhanced_log_csv)
    
    d2["Activity"]=d2["Activity"].str.lower()
    d1["Activity"]=d1["Activity"].str.lower()
    d2['Activity'] = d2['Activity'].str.strip()

    d1_index = d1.set_index(['HADM_ID', 'Activity', 'Timestamp']).index
    d2_index = d2.set_index(['HADM_ID', 'Activity','Timestamp']).index
    mask = ~d2_index.isin(d1_index)
    result = d2.loc[mask]
    enh_event_log = pd.concat([d3, result])

    enh_event_log.to_csv(enhanced_log_csv)
    
def convert_csv_xes(eventlog_csv, eventlog_xes):
    log_csv = pd.read_csv(eventlog_csv)
    log_csv.rename(columns = {'HADM_ID':'case:concept:name', 'Activity':'concept:name', 'Timestamp':'time:timestamp'}, inplace = True)

    log_csv = dataframe_utils.convert_timestamp_columns_in_df(log_csv)
    log_csv = log_csv.sort_values('time:timestamp')
    event_log = log_converter.apply(log_csv)

    xes_exporter.apply(event_log, eventlog_xes)
    
    
if __name__ == '__main__':
    compare_events("all_events.csv", "event_log_evaluation.csv", "matched_events.csv", "time_matched_events.csv")
    #calcute_semantic_similarity(pre_trained_model_path, "time_matched_events.csv", "semantic_similariy.csv")
    # pre_trained_model_path = "/BioSentVec_PubMed_MIMICIII-bigram_d700.bin"
    # pre_trained_model_path = "/wiki_bigrams.bin"
    # pre_trained_model_path = "/twitter_bigrams.bin"
    get_matched_events("semantic_similariy.csv", "matched_events.csv", "timestamp_updated_csv")
    add_new_events("matched_events.csv", "all_events.csv", "event_log.csv", "enhanced_log.csv")
    convert_csv_xes('eventlog.csv', 'eventlog_xes')


