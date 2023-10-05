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
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
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

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Let's build your clusters based on preferences!")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter preferences on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df


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

    # Check if the folder exists
    if not os.path.exists('app/artifacts'):
        # If it doesn't exist, create it
        os.makedirs('app/artifacts')
    df1.to_csv("app/artifacts/df_customers.csv")

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

    df1 = pd.read_csv('app/artifacts/df_customers.csv')
    
    df_cluster = read_file_from_s3(s3_client, bucket_name, most_recent_clustering['object']['Key'])
    df_cluster = df_cluster.iloc[:, 1:]
    df_cluster.columns = ['Date', 'ID', 'Main Cluster',	'Alternative Cluster 1', 'Alternative Cluster 2']

    df_prop = read_file_from_s3(s3_client, bucket_name, most_recent_propensity['object']['Key'])
    df_prop = df_prop[['ID', 'Date', 'Prob']]
    df_prop['Prob'] = df_prop['Prob'].mul(100)
    df_prop.columns = ['ID', 'Date', 'Prob. to Accept (%)']

    df = pd.merge(df_cluster, df_prop, on = ['ID', 'Date'], how = 'left')
    df = df.sort_values(["Date", "Prob. to Accept (%)"], ascending=[False, False])

    cluster_summary = read_file_from_s3(s3_client, bucket_name, 'crm-project/models/summary_clusters.csv')
    cluster_summary = cluster_summary.iloc[:, 1:]


    # Create a Streamlit app
    st.subheader('Customer Clusters and Target')
    filtered_data = filter_dataframe(cluster_summary)
    #st.dataframe(filtered_data)
    selected_clusters = filtered_data['Cluster'].unique().tolist()

    df_filt = df[df['Main Cluster'].isin(selected_clusters)]

    #st.write(df_filt)
    st.write("Number of Target Customers:", len(df_filt))


    st.write(df_filt)

    st.subheader("Estimation of Campaign Gains:")
    budget = st.number_input("Total Budget ($):", min_value=0.01)
    cac = st.number_input("Cost per Client ($):", min_value=0.01)
    time_discount = st.number_input("Period of the campaign (months):", min_value=1)
    coupon_discount = st.number_input("Discount (%):", min_value=0.01)
    
    
    campaign_size = int(budget // cac)


    recall_metric_experiments = 0.7

    # Apply Prob threshold filter
    df_filt = df_filt.iloc[:campaign_size]
    selected_customers = df_filt['ID'].values.tolist()

    total_cost = len(selected_customers) * cac
    total_gain = (df1[df1['id'].isin(selected_customers)]['amount_spent_month'].sum() * recall_metric_experiments * (1-coupon_discount/100)) * time_discount
    roi = 100*(total_gain-total_cost) / total_cost

    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]

    st.write(df_filt)

    if budget == 0.01:
        with col1:
            st.metric("Estimated Total Cost", f"${0:.2f}")

        with col2:
            st.metric("Estimated Total Gain", f"${0:.2f}")

        with col3:
            st.metric("Estimated ROI", f"{0:.2f}%")
    else:
        with col1:
            st.metric("Estimated Total Cost", f"${total_cost:.2f}")

        with col2:
            st.metric("Estimated Total Gain", f"${total_gain:.2f}")

        with col3:
            st.metric("Estimated ROI", f"{roi:.2f}%")



    #st.write(df_filt)
    st.write("Number of Target Customers after Budget Restriction:", len(selected_customers))



    # Create a button to download the filtered DataFrame as a CSV file
    csv_data = df_filt.to_csv(index=False)
    b64 = base64.b64encode(csv_data.encode()).decode()  # Encode to base64
    csv_filename = "exported_data.csv"


    # Generate and display the download link
    st.markdown(f'<a href="data:file/csv;base64,{b64}" download="{csv_filename}">Export CSV file</a>', unsafe_allow_html=True)


    # Streamlit App
    st.subheader("Cluster Characteristics:")

    # Create a dropdown to select a cluster
    selected_cluster = st.selectbox("Select Cluster:", cluster_summary['Cluster'].unique())

    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]
    with col1:
        # List of columns you want to display
        columns_to_display = ['Preference Wines','Preference Fruits','Preference Meat','Preference Fish','Preference Sweets','Preference Gold',
]

        for group, row in cluster_summary[cluster_summary['Cluster'] == selected_cluster].iterrows():
            st.info(f'''
                **Products**
                - {columns_to_display[0]}: **{row[columns_to_display[0]]}**
                - {columns_to_display[1]}: **{row[columns_to_display[1]]}**
                - {columns_to_display[2]}: **{row[columns_to_display[2]]}**
            ''')

    with col2:
        # List of columns you want to display
        columns_to_display = ['Preference Deals','Preference Web','Preference Catalog','Preference Store']

        for group, row in cluster_summary[cluster_summary['Cluster'] == selected_cluster].iterrows():
            st.info(f'''
                **Channel**
                - {columns_to_display[0]}: **{row[columns_to_display[0]]}**
                - {columns_to_display[1]}: **{row[columns_to_display[1]]}**
                - {columns_to_display[2]}: **{row[columns_to_display[2]]}**
            ''')

    with col3:
        # List of columns you want to display
        columns_to_display = ['Recency','Frequency','Amount Spent']

        for group, row in cluster_summary[cluster_summary['Cluster'] == selected_cluster].iterrows():
            st.info(f'''
                **Behavior (RFM)**
                - {columns_to_display[0]}: **{row[columns_to_display[0]]}**
                - {columns_to_display[1]}: **{row[columns_to_display[1]]}**
                - {columns_to_display[2]}: **{row[columns_to_display[2]]}**
            ''')
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