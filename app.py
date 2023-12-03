from flask import Flask,render_template, request,render_template,send_file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import joblib
from sklearn.preprocessing import StandardScaler
import os
import json

app=Flask(__name__)
model=joblib.load('customer_segmentation.pkl')

def load_and_clean_data(file_path):

        # Possible encodings to try
    encodings_to_try = ['utf-8', 'ISO-8859-1', 'latin1']  # Add more encodings as needed
    
    # Try different encodings to read the file
    for encoding in encodings_to_try:
        try:
            # Read the file using the current encoding
            df = pd.read_csv(file_path, encoding=encoding)
            
            # Clean the DataFrame (add your cleaning logic here)
            df = df.dropna()
            df['CustomerID']=df['CustomerID'].astype(int)

            df['CustomerID'] = df['CustomerID'].astype(str)
            df = df[df['Quantity'] > 0]
            
            df['Total_amount_spent']=df['Quantity']*df['UnitPrice']

            cust_df=pd.DataFrame(df.groupby('CustomerID')['Total_amount_spent'].sum()).reset_index(inplace=True)

            cust_df['No_of_items_purchased']=df.groupby('CustomerID')['InvoiceNo'].count().values

            cust_df['No_of_times_ordered']=df.groupby('CustomerID')['InvoiceNo'].nunique().values

            cust_df['Amount_spent_per_order']=np.round((df.groupby('CustomerID')['Total_amount_spent'].sum()/
                                   df.groupby('CustomerID')['InvoiceNo'].nunique()).values,2)

            df['InvoiceDate']=pd.to_datetime(df['InvoiceDate'],format='%m/%d/%Y %H:%M')

            df['Date_diff']=max(df['InvoiceDate'])-df['InvoiceDate']
            df['Date_diff']=df['Date_diff'].dt.days

            cust_df['Recency']=df.groupby('CustomerID')['Date_diff'].min().values

            cust_df['History_of_customer']=df.groupby('CustomerID')['Date_diff'].max().values

            return cust_df


        except UnicodeDecodeError:
            continue
    
    raise ValueError("Unable to read the file with the specified encodings.")
    
    
def outlier_treat(cust_df):

    for col in cust_df.iloc[:,1:].columns:
        q1=cust_df[col].quantile(0.25)
        q3=cust_df[col].quantile(0.75)
        IQR=q3-q1
        lower_bound=q1-(IQR*1.5)
        upper_bound=q3+(IQR*1.5)
        cust_df1=cust_df[(cust_df[col]>=lower_bound) & (cust_df[col]<=upper_bound)]
        return cust_df1

def preprocess_data(file_path):
    cust_df=load_and_clean_data(file_path)

    cust_df1 = outlier_treat(cust_df)


    scaler=StandardScaler()
    cust_scaled=scaler.fit_transform(cust_df1.drop('CustomerID',axis=1))
    cust_scaled=pd.DataFrame(cust_scaled)
    cust_scaled.columns=['Amount_spent','No_of_items_purchased','No_of_times_ordered','Amount_spent_per_order',
                         'Recency','History_of_customer']

    return cust_df1,cust_scaled

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:

        file = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        print("File saved successfully:", file_path)  # Add this print statement

        # Preprocess data
        cust_df1, df_scaled = preprocess_data(file_path)
        result_df = model.predict(df_scaled)
        
        df_scaled['Cluster_ID'] = result_df
        
        cust_df1['Label'] = result_df
        # Plot and save images
        sns.stripplot(x='Cluster_ID', y='Total_amount_spent', data=df_scaled, hue='Cluster_ID')
        Tot_amt_img_path = 'static/ClusterID_Total_amt.jpg'
        plt.savefig(Tot_amt_img_path)
        plt.clf()
        
        sns.stripplot(x='Cluster_ID', y='No_of_items_purchased', data=df_scaled, hue='Cluster_ID')
        items_img_path = 'static/ClusterID_No_of_items_purchased.jpg'
        plt.savefig(items_img_path)
        plt.clf()
        
        sns.stripplot(x='Cluster_ID', y='No_of_times_ordered', data=df_scaled, hue='Cluster_ID')
        freq_img_path = 'static/ClusterID_No_of_times_ordered.jpg'
        plt.savefig(freq_img_path)
        plt.clf()
        
        sns.stripplot(x='Cluster_ID', y='Amount_spent_per_order', data=df_scaled, hue='Cluster_ID')
        amt_per_order_img_path = 'static/ClusterID_Amt_per_order.jpg'
        plt.savefig(amt_per_order_img_path)
        plt.clf()

        sns.stripplot(x='Cluster_ID', y='Recency', data=df_scaled, hue='Cluster_ID')
        recency_img_path = 'static/ClusterID_Recency.jpg'
        plt.savefig(recency_img_path)
        plt.clf()

        sns.stripplot(x='Cluster_ID', y='History_of_customer', data=df_scaled, hue='Cluster_ID')
        history_img_path = 'static/ClusterID_History_of_cust.jpg'
        plt.savefig(history_img_path)
        plt.clf()

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.csv')
        cust_df1.to_csv(file_path)

        
        output_saved_message = f"Output file 'output.csv' is saved in the same directory'"


        response = {
            'tot_amt_img': '/get-image/ClusterID_Total_amt.jpg',
            'items_img': '/get-image/ClusterID_No_of_items_purchased.jpg',
            'freq_img': '/get-image/ClusterID_No_of_times_ordered.jpg',
            'amt_per_order_img': '/get-image/ClusterID_Amt_per_order.jpg',
            'recency_img': '/get-image/ClusterID_Recency.jpg',
            'history_img': '/get-image/ClusterID_History_of_cust.jpg',
            'output_saved_message': output_saved_message 
        }

        print("Sending response:", response)  

        return json.dumps(response)


    except Exception as e:
        print("Error occurred:", str(e))  

        return str(e)
    
@app.route('/get-image/<path:filename>')
def get_image(filename):
    return send_file(os.path.join(app.root_path, 'static', filename))


if __name__=="__main__":
    app.run(debug=True)

