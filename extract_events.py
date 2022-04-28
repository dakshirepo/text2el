import re
import pandas as pd
import glob
import ntpath
import os
from os import listdir
from os.path import isfile, join


#Extract events based on temporal expressions from MIMIC-Discharge summaries
def apply_regex_rules(collection_filepath, output_file1):
    time = []
    activity = []
    caseID = []
    folders = [f for f in listdir(collection_filepath)]
    df = pd.DataFrame()

    for dir in folders:
        for f in glob.glob(collection_filepath + '\\' + dir + "\\*Discharge summary.txt"):
            with open(f, "r+") as inputfile:
                con = inputfile.read()
                x = con.split("\n")

                pattern = re.compile(r"(.*[[][*]{2}[0-9]{4}[-][0-9]*[-][0-9]*[*]{2}\])")
                frontkeyonlypattern = re.compile(
                    r"(.*)[[][*]{2}[0-9]{4}[-][0-9]*[-][0-9]{1,2}[*]{2}\]([ ][0-9]*[:][0-9]{2}[ ]*(AM|A\.M\.|am|a\.m\.|PM|P\.M\.|pm|p\.m\.))?")
                backkeyonlypattern = re.compile(
                    r"[[][*]{2}[0-9]{4}[-][0-9]*[-][0-9]{1,2}[*]{2}\]([ ][0-9]*[:][0-9]{2}[ ]*(AM|A\.M\.|am|a\.m\.|PM|P\.M\.|pm|p\.m\.))?(.*)")
                datetimepattern = re.compile(
                    r"([[][*]{2}[0-9]{4}[-][0-9]*[-][0-9]{1,2}[*]{2}\]([ ][0-9]*[:][0-9]{2}[ ]*(AM|A\.M\.|am|a\.m\.|PM|P\.M\.|pm|p\.m\.))?)")
                pattern1 = re.compile(r"([[][*]{2}[0-9]{4}[-][0-9]*[-][0-9]{1,2}[*]{2}\].*)")

                for k in x:
                    m = pattern.findall(k)
                    m1 = pattern1.findall(k)
                    for i in m:
                        caseID.append(dir)
                        if not datetimepattern.match(i):
                            rgx = frontkeyonlypattern.search(i)
                            if rgx.group(1) is not None:
                                activity.append(rgx.group(1))
                            rgx1 = datetimepattern.search(i)
                            if rgx1.group(1) is not None:
                                time.append(rgx1.group(1))

                        else:
                            for i1 in m1:
                                rgx = backkeyonlypattern.search(i1)
                                if rgx.group(3) is not None:
                                    activity.append(rgx.group(3))
                                rgx1 = datetimepattern.search(i1)
                                if rgx1.group(1) is not None:
                                    time.append(rgx1.group(1))


    df["HADM_ID"] = caseID
    df['Activity'] = activity
    df['Timestamp'] = time
    df['Source']="Discharge summary"


    df.to_csv(output_file1)

def extract_specific_notes(collection_filepath, output_file2):
    folders = [f for f in listdir(collection_filepath)]
    df = pd.DataFrame()

    table1 = {"HADM_ID": [], "Activity": [], "Timestamp": [], "Source": []}
    for dir in folders:

        for f in glob.glob(collection_filepath + '\\' + dir + "\\*Echo.txt"):
            with open(f, "r+") as inputfile:

                con = inputfile.read()

                pattern1 = re.compile(r"Date/Time:(.*)")
                pattern2 = re.compile(r"Test:(.*)")

                rem_characters= ["['","']", "at"]
                m1 = pattern1.findall(con)
                m2 = pattern2.findall(con)

                for i in rem_characters:
                    m1 = str(m1).replace(i, "")
                    m2 = str(m2).replace(i, "")

                table1["HADM_ID"].append(dir)
                table1["Activity"].append(m2)
                table1["Timestamp"].append(m1)
                table1["Source"].append(os.path.basename(f))

        for f in glob.glob(collection_filepath + '\\' + dir + "\\*Radiology.txt"):
            with open(f, "r+") as inputfile:

                con2 = inputfile.readlines()
                pattern = re.compile(r"([[][*]{2}[0-9]{4}[-][0-9]*[-][0-9]{1,2}[*]{2}\].*)")

                n1 = con2[:1]
                for k in n1:
                    m = pattern.findall(k)

                n2 = con2[1]

                pattern3 = re.compile(r"(.*)Clip \# \[\*\*Clip Number")
                pattern4 = re.compile(r"(.*);")
                mm = pattern3.findall(n2)
                mm1 = pattern4.findall(n2)

                rem_characters = ["['", "']", ]
                for i in rem_characters:
                    m = str(m).replace(i, "")
                    mm = str(mm).replace(i, "")
                    mm1 = str(mm1).replace(i, "")

                mm = str(mm).replace(") ", ")")
                mm = re.sub('\s+', " ", mm)
                mm = re.sub("[\[].*?[\]]", "", mm)
                mm1 = str(mm1).replace(") ", ")")
                mm1 = re.sub('\s+', " ", mm1)
                mm1 = re.sub("[\[].*?[\]]", "", mm1)


                table1["HADM_ID"].append(dir)
                table1["Activity"].append(mm + mm1)
                table1["Timestamp"].append(m)
                table1["Source"].append(os.path.basename(f))

        df = df.append(pd.DataFrame(table1), ignore_index=True)

    df = df[["HADM_ID", "Activity", "Timestamp", "Source"]]
    df = df.dropna()
    df['Source'] = df['Source'].str.replace('^[^a-zA-Z]*', '')
    df['Source'] = df['Source'].str.replace('.txt', '')

    df = df.drop_duplicates(keep='first')
    df = df.dropna()
    df = df[df.Activity != '']
    df.to_csv(output_file2)

def refine_events(input_csv, exclude_list, stopword_list, output_csv):
    d = pd.read_csv(input_csv)
    f1 = open(exclude_list , "r")
    w_list = f1.read().split('\n')

    suffix_list = ['in', 'on', 'of', 'On']

    d['Activity'] = d['Activity'].str.strip()

    d['Activity'] = d['Activity'].str.rstrip(',')
    d['Activity'] = d['Activity'].str.rstrip(',')
    d['Activity'] = d['Activity'].str.lstrip(',')

    for j in suffix_list:
        d['Activity'] = d['Activity'].str.removesuffix(j)


    d['Activity'] = d['Activity'].str.replace('^[^a-zA-Z]*', '')
    d['Activity'] = d['Activity'].str.replace('^and*', '')

    d = d[d['Activity'].str.len() > 0]
    d = d.dropna()

    out_df = d[~d['Activity'].str.lower().isin([x.lower() for x in w_list])]
    f2 = open(stopword_list, "r")
    among_list = f2.readlines()
    rem=str.maketrans('', '', '\n')
    b=[s.translate(rem) for s in among_list]
    print(b)
    n_l = []
    for i in b:
        out_df['Activity'] = out_df['Activity'].str.replace(re.escape(i),'')

    out_df.dropna()
    out_df['Activity'] = out_df['Activity'].str.replace('^[^a-zA-Z]*', '')
    out_df = out_df[out_df['Activity'].str.len() > 0]

    out_df.to_csv(output_csv)

def get_all_events(general_events_file, specific_event_file, all_events):
    d1=pd.read_csv(general_events_file)
    d2=pd.read_csv(specific_event_file)
    d1=d1[["HADM_ID",	"Activity",	"Timestamp",	"Source"]]
    d2 = d2[["HADM_ID", "Activity", "Timestamp", "Source"]]

    d3 = pd.concat([d1, d2])

    d3['Timestamp'] = d3['Timestamp'].str.replace(re.escape("[**"), '')
    d3['Timestamp'] = d3['Timestamp'].str.replace(re.escape("**]"), '')
    d3['Timestamp'] = d3['Timestamp'].str.replace(re.escape("-"), '/')
    d3['Timestamp'] = d3['Timestamp'].str.replace(re.escape("at"), ' ')
    d3['Timestamp'] = d3['Timestamp'].str.replace(re.escape("**"), '')
    d3=d3.drop_duplicates()
    d3=d3.sort_values(['HADM_ID', 'Timestamp'], ascending=[True, True])
    d3.to_csv(all_events)

if __name__ == '__main__':
    apply_regex_rules("C:\Research\\Note_Evaluation", "C:\Research\MIMIC_Exp\Test\\extracted_events.csv")
    extract_specific_notes("C:\Research\\Note_Evaluation", "C:\Research\MIMIC_Exp\Test\\specific_events.csv")
    refine_events("C:\Research\MIMIC_Exp\Test\\extracted_events.csv", "C:\\Research\MIMIC_Exp\\exclude_list.txt", "C:\\Research\MIMIC_Exp\\stopwords.txt", "C:\Research\MIMIC_Exp\Test\\refined_events.csv")
    get_all_events("C:\Research\MIMIC_Exp\Test\\refined_events.csv", "C:\Research\MIMIC_Exp\Test\\specific_events.csv", "C:\Research\MIMIC_Exp\Test\\all_events.csv")


