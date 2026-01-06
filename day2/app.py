import streamlit as st
import pandas as pd 
import numpy as np
import os 
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from datetime import datetime
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

def log(message):
    timestamp =datetime.now().strftime("%Y-%m-%d:%s")
    print(f"[{timestamp}]{message}")

if "cleaned_saved" not in st.session_state:
    st.session_state.cleaned_saved="False"

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
RAW_DIR=os.path.join(BASE_DIR,"Data","raw")
CLEAN_DIR=os.path.join(BASE_DIR,"Data","cleaned")

os.makedirs(RAW_DIR,exist_ok=True)
os.makedirs(CLEAN_DIR,exist_ok=True)

log('Applicatoion Started')
log(f"RAW_DIR={RAW_DIR}")
log(f"CLEAN_DIR={CLEAN_DIR}")

st.set_page_config("End-to-End SVM",layout="wide")
st.title("End to End SVM Platfrom")
                       

st.sidebar.header("SVM Settings")
kernel=st.sidebar.selectbox("kernel",["linear","rbf","poly","sigmoid"])
C=st.sidebar.slider("C (Regularization)",0.01,10.0,1.0)
gamma=st.sidebar.selectbox("Gamma",["scale","auto"])
log(f"SVM Settings --> Kernel = {kernel},c={C},Gamma={gamma}")

st.header("Step1:Data Ingestion")
log("Step1 started:Data Ingestion")
option = st.radio("Choose Data Source",["Download Dataset","Upload CSV"])
df=None
raw_path=None
if option=="Download Dataset":
    if st.button("Download Iris Dataset"):
        log("Downloading Iris Dataset")
        url="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
        response=requests.get(url)
        raw_path=os.path.join(RAW_DIR,"iris.csv")
        with open (raw_path,"wb") as f:
            f.write(response.content)
        
        df=pd.read_csv(raw_path)
        st.success("Dataset Downloaded Succesfully")
        log(f"Iris datset saved at {raw_path}")

if option== "Upload CSV":
    uploaded_file =st.file_uploader("upload CSV File",type=["csv"])
    if uploaded_file:
        raw_path=os.path.join(RAW_DIR,uploaded_file.name)
        with open (raw_path,"wb") as f:
            f.write(uploaded_file.getbuffer())
        df=pd.read_csv(raw_path)
        st.success("File uploaded Successfully")
        log(f"uploaded data saved at {raw_path}")

if df is not None :
    st.header("Step 2 : Exploaratory Data Analysis")
    log("Step 2 Started: EDA")
    st.dataframe(df.head())
    st.write("Shape",df.shape)
    st.write("Missing Values :",df.isnull().sum())
    fig,ax=plt.subplots()
    sns.heatmap(df.corr(numeric_only=True),annot=True,cmap="coolwarm",ax=ax)
    st.pyplot(fig)
    log("EDA Completed")


if df is not None:
        st.header("Step3: Data Cleaning")
        strategy=st.selectbox("Missing VAlues Strategy",["Mean","Median","Drop Rows"])
        df_clean=df.copy()
        if strategy=="Drop Rows":
            df_clean=df_clean.dropna()
        else:
            for col in df_clean.select_dtypes(include=np.number):
                if strategy=="Mean":
                    df_clean[col]=df_clean[col].fillna(df_clean[col].mean())
                else:
                    df_clean[col]=df_clean[col].fillna(df_clean[col].median())
        st.session_state.df_clean=df_clean
        st.success("Data Cleaning Completed")

else:
    st.info("Please complete Step 1(Data Ingestion) first...")
    

if st.button("Saved Cleaned Dataset"):
    if st.session_state.df_clean is None:
        st.error("No Cleaned Data Found . Please Complete step 3 first")
    else:
        timestamp=datetime.now().strftime("Y%m%d_%H%M%S")
        clean_filename=f"cleaned_dataset{timestamp}.csv"
        clean_path=os.path.join(CLEAN_DIR,clean_filename)
        st.session_state.df_clean.to_csv(clean_path,index=False)
        st.success("Cleaned Dataset Successfully")
        st.info(f"Saved at :{clean_path}")
        log(f"Cleaned dataset saved at {clean_path}")

st.header("Step 5 : Load Cleaned Dataset")
clean_files=os.listdir(CLEAN_DIR)
if not clean_files:
    st.warning("No Cleaned Dataset found ,Please Save  atleaset one dataset in step4...")
    log("No cleaned datasets avilable")

else:
    selected=st.selectbox("Select Cleaned Dataset",clean_files)
    df_model=pd.read_csv(os.path.join(CLEAN_DIR,selected))
    st.success(f"Loaded dataset :{selected}")
    log(f"Loaded cleaned datset :{selected}")

    st.dataframe(df_model.head())


    st.header("Step 6:Train SVM")
    log("Step 6 Started : SVM Training ")

    target=st.selectbox("Select Target Column",df_model.columns)
    y=df_model[target]
    if y.dtype =="object":
        y=LabelEncoder().fit_transform(y)
        log("Target Column Encoded")

x= df_model.drop(columns=[target])
x=x.select_dtypes(include=np.number)
if x.empty:
    st.error("No numeric features available for Training..")
    st.stop()

#Scale features
scaler=StandardScaler()
x=scaler.fit_transform(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
 
model=SVC(kernel =kernel,C=C,gamma=gamma)
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
acc=accuracy_score(y_test,y_pred)

st.success(f"Accuracy : {acc:.2f}")
log(f"SVM trained successfully  | accuracy ={acc:.2f}")

cm=confusion_matrix(y_test,y_pred)
fig,ax=plt.subplots()
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",ax=ax)
st.pyplot(fig)

