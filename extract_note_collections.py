import pandas as pd
import os

#Extract the notes related to patients admitted during 01-01-2100 – 01-01-2101  
def extract_notes(admission_csv, noteevents_csv, path_to_collection):
    d1=pd.read_csv(admission_csv)
    d2=pd.read_csv(noteevents_csv)
    output_folder=path_to_collection


    d1[['Adm_Date','Adm_Time']] = d1["ADMITTIME"].str.split(" ", expand=True, n=1)
    d1['Adm_Date'] = pd.to_datetime(d1['Adm_Date'])
    d1['Adm_Date'] = d1['Adm_Date'].replace('/', '-')

    #Extract 2100 - 2101
    adm_extracted=d1[(d1['Adm_Date'] > '01-01-2100') & (d1['Adm_Date'] < '01-01-2101')]


    d2=d2.dropna(subset=[ 'HADM_ID'])

    d2b=d2[['ROW_ID' ,'HADM_ID' ,'CATEGORY', 'TEXT']]
    
    #Filter required note categories
    d2b=d2b[d2b['CATEGORY'].isin(['Discharge summary', 'Echo','Radiology'])]
    merged=adm_extracted.merge(d2b, on=["HADM_ID"])

    merged=merged[['ROW_ID_y' ,'HADM_ID' ,'CATEGORY', 'TEXT']]
    merged=merged.rename(columns ={'ROW_ID_y':'ROW_ID'})

    #write notes to collection
    g= merged.groupby(['ROW_ID', 'HADM_ID','CATEGORY', 'TEXT'])

    for (r, a, c, t), group in g:
        if not os.path.exists(output_folder+str(a)):
            os.makedirs(output_folder+str(a))
        with open(output_folder+str(a)+"\\"+str(r) +"_"+ str(c)+ ".txt", "w") as fout:
            fout.write(t.replace('Discharge Date:', '\nDischarge Date:'))


if __name__ == '__main__':
    extract_notes("mimic-iii-clinical-database-1.4\\ADMISSIONS.csv", "mimic-iii-clinical-database-1.4\\noteevents.csv", "Note_collection\\")
