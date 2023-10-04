import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import boto3 
import datetime
import botocore
import io
import os
import base64
import json
import openai
from dotenv import load_dotenv
from utils import read_file_from_s3, download_file_from_s3
from time import sleep
st.set_option('deprecation.showPyplotGlobalUse', False)

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


def ask_gpt(question, context):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You're a marketing specialist that build marketing campaigns."},
            {"role": "user", "content": context},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message['content'].strip()


def page_1():

    # Load your data into a DataFrame (replace 'data.csv' with your data file)
    df1 = pd.read_csv('data/ml_project1_data.csv')
    old_columns = ['ID', 'Year_Birth', 'Education', 'Marital_Status', 'Income', 'Kidhome',
                'Teenhome', 'Dt_Customer', 'Recency', 'MntWines', 'MntFruits',
                'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
                'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
                'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
                'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1',
                'AcceptedCmp2', 'Complain', 'Z_CostContact', 'Z_Revenue', 'Response']

    lower_case = lambda x: x.lower()

    new_columns = list(map(lower_case, old_columns))

    df1.columns = new_columns
    df1['dt_customer'] = pd.to_datetime(df1['dt_customer'])

    # Creating the loyalty variable - days
    df1['loyalty_days'] = df1['dt_customer'].max() - df1['dt_customer']
    df1['loyalty_days'] = df1['loyalty_days'] //  np.timedelta64(1, 'D')

    # Creating the loyalty variable - convert to month
    df1['loyalty_months'] = df1['loyalty_days'] / 30

    # Creating the total amount spent by each customer - MONETARY
    df1['amount_spent'] = df1['mntmeatproducts'] + df1['mntwines'] + df1['mntfruits'] + df1['mntsweetproducts'] + df1['mntfishproducts']
    # Creating the amount spent per month 
    df1['amount_spent_month'] = df1.apply(lambda x:  x['amount_spent'] / x['loyalty_months'] if x['loyalty_months'] > 1 else x['amount_spent'], axis = 1)

    # Creating the total number of purchases - FREQUENCY
    df1['total_purchases'] = df1['numdealspurchases'] + df1['numwebpurchases'] + df1['numcatalogpurchases'] + df1['numstorepurchases']

    df1['accepted_campaigns'] = df1['acceptedcmp1'] + df1['acceptedcmp2'] + df1['acceptedcmp3'] + df1['acceptedcmp4'] + df1['acceptedcmp5'] + df1['response']

    # Create the age of the customer
    df1['age'] = df1['dt_customer'].dt.year.max() - df1['year_birth']

    # Creating the total number of purchasese by each month
    df1['purchases_month'] = df1['total_purchases'] / (df1['loyalty_months'] + 1)

    # Creating percentage purchases on different channel types
    df1['percentage_type_deals'] = df1['numdealspurchases'] / df1['total_purchases']
    df1['percentage_type_web'] = df1['numwebpurchases'] / df1['total_purchases']
    df1['percentage_type_catalog'] = df1['numcatalogpurchases'] / df1['total_purchases']
    df1['percentage_type_store'] = df1['numstorepurchases'] / df1['total_purchases']

    # Creating percentage spent on different products
    df1['percentage_spent_wines'] = df1['mntwines'] / df1['amount_spent']
    df1['percentage_spent_fruits'] = df1['mntfruits'] / df1['amount_spent']
    df1['percentage_spent_meat'] = df1['mntmeatproducts'] / df1['amount_spent']
    df1['percentage_spent_fish'] = df1['mntfishproducts'] / df1['amount_spent']
    df1['percentage_spent_sweet'] = df1['mntsweetproducts'] / df1['amount_spent']
    df1['percentage_spent_gold'] = df1['mntgoldprods'] / df1['amount_spent']


    # Sidebar filters for data analysis
    st.sidebar.subheader('Data Filters')

    # Filter by education level
    selected_education = st.sidebar.selectbox('Select Education Level', ['All'] + df1['education'].unique().tolist())

    # Filter by marital status
    selected_marital_status = st.sidebar.selectbox('Select Marital Status', ['All'] + df1['marital_status'].unique().tolist())

    # Apply filters to the data
    if selected_education != 'All':
        filtered_data = df1[df1['education'] == selected_education]
    else:
        filtered_data = df1.copy()  # No filter for education

    if selected_marital_status != 'All':
        filtered_data = filtered_data[filtered_data['marital_status'] == selected_marital_status]

    # Calculate KPI 1: Customer Acquisition Rate
    total_customers = filtered_data['id'].nunique()
    new_customers = filtered_data[filtered_data['recency'] <= 30]['id'].nunique()
    acquisition_rate = (new_customers / total_customers) * 100

    # Calculate KPI 2: Average Amount Spent per Customer
    total_amount_spent = filtered_data['amount_spent'].sum()
    average_amount_spent_per_customer = total_amount_spent / total_customers

    # Calculate KPI 3: Customer Response Rate
    total_campaigns = 6
    total_customers_contacted = filtered_data['accepted_campaigns'].mean()
    response_rate = (total_customers_contacted / total_campaigns) * 100

    # Calculate KPI 4: Customer Response Rate of Last Campaign
    total_customers_contacted = filtered_data['response'].sum()
    response_rate_last_cmp = (total_customers_contacted / total_customers) * 100

    # Display KPIs
    #st.subheader('Key Performance Indicators (KPIs)')
    #st.write(f'KPI 1 - Customer Acquisition Rate: {acquisition_rate:.2f}%')
    #st.write(f'KPI 2 - Average Amount Spent per Customer: ${average_amount_spent_per_customer:.2f}')
    #st.write(f'KPI 3 - Customer Response Rate: {response_rate:.2f}%')
    #st.write(f'KPI 4 - Customer Response Rate of Last Campaign: {response_rate_last_cmp:.2f}%')

    # Create a Streamlit app
    st.subheader('Key Performance Indicators (KPIs)')

    # Display KPIs side by side using st.metric()
    col1, col2, col3, col4 = st.columns(4)  # Create 4 columns for KPIs
    with col1:
        st.metric("Customer Acquisition Rate", f"{acquisition_rate:.2f}%")

    with col2:
        st.metric("Average Amount Spent per Customer", f"${average_amount_spent_per_customer:.2f}")

    with col3:
        st.metric("Customer Response Rate", f"{response_rate:.2f}%")

    with col4:
        st.metric("Customer Response Rate of Last Campaign", f"{response_rate_last_cmp:.2f}%")

    # Visualization 1: Customer Age Distribution
    st.subheader('Customer Age Distribution')
    plt.hist(filtered_data['age'], bins=20, edgecolor='k')
    plt.xlabel('Age')
    plt.ylabel('Number of Customers')
    plt.title('Customer Age Distribution')
    st.pyplot()
    sleep(0.2)

    # Visualization 2: Percentage of Spending on Product Categories
    st.subheader('Percentage of Spending on Product Categories')
    product_categories = ['mntwines', 'mntfruits', 'mntmeatproducts', 'mntfishproducts', 'mntsweetproducts', 'mntgoldprods']
    total_spending = filtered_data[product_categories].sum().sum()
    percentage_spending_prod = filtered_data[product_categories].sum() / total_spending * 100
    st.bar_chart(percentage_spending_prod)   

    # Visualization 3: Percentage of Spending on Channels
    st.subheader('Percentage of Spending on Channels')
    channel_categories = ['percentage_type_deals','percentage_type_web','percentage_type_catalog','percentage_type_store']
    total_spending = filtered_data[channel_categories].sum().sum()
    percentage_spending_channel = filtered_data[channel_categories].sum() / total_spending * 100
    st.bar_chart(percentage_spending_channel)   


def page_2():

  
    bucket_name = 'jooaobrum-projects'
    output_folder = 'crm-project/output'
    s3_client = boto3.client('s3')

    # List objects in the bucket
    objects = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=output_folder)

    # Initialize variables to track the most recent 'propensity' and 'clustering' objects and their timestamps
    most_recent_propensity = {'object': None, 'timestamp': None}
    most_recent_clustering = {'object': None, 'timestamp': None}

    # Iterate through the objects and find the most recent ones for each category
    for obj in objects.get('Contents', []):
        object_key = obj['Key']
        obj_timestamp = obj['LastModified']
        
        # Check if the object key contains 'propensity'
        if 'propensity' in object_key:
            if most_recent_propensity['timestamp'] is None or obj_timestamp > most_recent_propensity['timestamp']:
                most_recent_propensity['object'] = obj
                most_recent_propensity['timestamp'] = obj_timestamp
        
        # Check if the object key contains 'clustering'
        if 'cluster' in object_key:
            if most_recent_clustering['timestamp'] is None or obj_timestamp > most_recent_clustering['timestamp']:
                most_recent_clustering['object'] = obj
                most_recent_clustering['timestamp'] = obj_timestamp

    
    
    df_cluster = read_file_from_s3(s3_client, bucket_name, most_recent_clustering['object']['Key'])
    df_cluster = df_cluster.iloc[:, 1:]
    df_cluster.columns = ['Date', 'ID', 'Main Cluster',	'Alternative Cluster 1', 'Alternative Cluster 2']

    df_prop = read_file_from_s3(s3_client, bucket_name, most_recent_propensity['object']['Key'])
    df_prop = df_prop[['ID', 'Date', 'Prob']]
    df_prop.columns = ['ID', 'Date', 'Prob. to Accept']

    df = pd.merge(df_cluster, df_prop, on = ['ID', 'Date'], how = 'left')
    df = df.sort_values(["Date", "Prob. to Accept"], ascending=[False, False])

    # Create a Streamlit app
    st.subheader('Customer Clusters and Target')
    st.write("Legend: Products_Channel_Behavior")
    selected_main_cluster = st.sidebar.selectbox('Select Main Cluster', ['All'] + df['Main Cluster'].unique().tolist())

    selected_sec_cluster = st.sidebar.selectbox('Select Alternative Cluster 1', ['All'] + df['Alternative Cluster 1'].unique().tolist())

    selected_third_cluster = st.sidebar.selectbox('Select Alternative Cluster 2', ['All'] + df['Alternative Cluster 2'].unique().tolist())

    # Filter by Prob threshold
    threshold = st.sidebar.slider('Select Probability to Accept Threshold', min_value=0.0, max_value=1.0, value=0.0, step=0.05)

    # Apply filters to the data
    if selected_main_cluster != 'All':
        filtered_data = df[df['Main Cluster'] == selected_main_cluster]
    else:
        filtered_data = df.copy() 

    if selected_sec_cluster != 'All':
        filtered_data = filtered_data[filtered_data['Alternative Cluster 1'] == selected_sec_cluster]

    if selected_third_cluster != 'All':
        filtered_data = filtered_data[filtered_data['Alternative Cluster 2'] == selected_third_cluster]

    # Apply Prob threshold filter
    filtered_data = filtered_data[filtered_data['Prob. to Accept'] > threshold]

    st.write(filtered_data)

    # Create a button to download the filtered DataFrame as a CSV file
    csv_data = filtered_data.to_csv(index=False)
    b64 = base64.b64encode(csv_data.encode()).decode()  # Encode to base64
    csv_filename = "filtered_data.csv"


    # Generate and display the download link
    st.markdown(f'<a href="data:file/csv;base64,{b64}" download="{csv_filename}">Export CSV file</a>', unsafe_allow_html=True)


    if not os.path.exists('app/artifacts'):
        # Create the folder if it doesn't exist
        os.makedirs('app/artifacts')
    download_file_from_s3(s3_client, bucket_name, 'crm-project/models/cluster_id_channel.png', 'app/artifacts')
    download_file_from_s3(s3_client, bucket_name, 'crm-project/models/cluster_id_prod.png', 'app/artifacts')
    download_file_from_s3(s3_client, bucket_name, 'crm-project/models/cluster_id_rfm.png', 'app/artifacts')

    st.subheader('Cluster Interpretation:')
    st.write("Given the plot below, create personas as you want! The values are normalized wihtin 0 and 1.")
    st.write("Cluster Channel:")
    st.image('app/artifacts/cluster_id_channel.png', caption='Cluster Channel', use_column_width=True)
    st.write("Cluster Products:")
    st.image('app/artifacts/cluster_id_prod.png', caption='Cluster Products', use_column_width=True)
    st.write("Cluster Behavior:")
    st.image('app/artifacts/cluster_id_rfm.png', caption='Cluster Behavior', use_column_width=True)



def page_3():

    st.title("Organize your campaign!")
    st.write("Powered by Open AI")
    st.write("Campaign Objective:")
    
    obj1 = st.text_input("What are the specific goals and objectives you aim to achieve with this marketing campaign?")
    kpi = st.text_input("Who do you envision measuring the success of this campaign (KPIs)?")
    
    st.write("Target Audience:")
    tar1 = st.text_input("Who is your primary target audience for this campaign? Please describe their demographics, interests, and behaviors.")
    
    st.write("Key Messages:")
    mess = st.text_input("What are the key messages or value propositions you want to convey to your audience through this campaign?")
    
    st.write("Budget Allocation:")
    budget = st.text_input("What is the budget for this campaign?")

    st.write("Additional Information (Optional):")
    add_info = st.text_input("Write down the additional information.")
    
    
    all_characteristics = [obj1, kpi, tar1, mess, budget]
    question = "\n".join(all_characteristics)
    
    st.write(question)

    context = "Act as a marketing specialist. Build a marketing campaign based on the following user input and also create a content plan. "
    if st.button('Ask to the Advisor'):
            answer = ask_gpt(question, context)
            st.write(f"Answer: {answer}")

   

st.sidebar.image("app/images/logo.png", use_column_width=True)
# Create a Streamlit app
st.sidebar.title('Marketing Campaigns Analysis')
# Create a Streamlit app
st.sidebar.title('Navigation')

# Create a sidebar menu for navigation
selected_page = st.sidebar.radio("Select a page", ("KPIs Analysis", "Marketing Targets", "Campaign Advisor"))



# Depending on the selected page, display the corresponding content
if selected_page == "KPIs Analysis":
    page_1()
elif selected_page == "Marketing Targets":
    page_2()
elif selected_page == "Campaign Advisor":
    page_3()


st.sidebar.info(
        """
        My name is João Paulo Sales Brum . I am data scientist and passionate by data pipelines.
        If you want to know more about my projects, you can acess my social medias.

        Github: [github.com/jooaobrum](https://github.com/jooaobrum)
        Linkedin: [linkedin.com/jooaobrum](https://www.linkedin.com/in/jooaobrum/)

        This report was generated using Streamlit.
	"""
    )