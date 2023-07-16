import pandas as pd
import numpy as np


def rename_columns(df):


    old_columns = ['ID', 'Year_Birth', 'Education', 'Marital_Status', 'Income', 'Kidhome',
                'Teenhome', 'Dt_Customer', 'Recency', 'MntWines', 'MntFruits',
                'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
                'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
                'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
                'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1',
                'AcceptedCmp2', 'Complain', 'Z_CostContact', 'Z_Revenue', 'Response']

    lower_case = lambda x: x.lower()

    new_columns = list(map(lower_case, old_columns))

    df.columns = new_columns

    return df

def cast_columns(df):
    
    df['dt_customer'] = pd.to_datetime(df['dt_customer'])

    return df


def feature_engineering(df):

    # Creating the loyalty variable - days
    df['loyalty_days'] = df['dt_customer'].max() - df['dt_customer']
    df['loyalty_days'] = df['loyalty_days'] //  np.timedelta64(1, 'D')
    # Creating the loyalty variable - convert to month
    df['loyalty_months'] = df['loyalty_days'] / 30

    # Creating the total amount spent by each customer - MONETARY
    df['amount_spent'] = df['mntmeatproducts'] + df['mntwines'] + df['mntfruits'] + df['mntsweetproducts'] + df['mntfishproducts']
    # Creating the amount spent per month 
    df['amount_spent_month'] = df.apply(lambda x:  x['amount_spent'] / x['loyalty_months'] if x['loyalty_months'] > 1 else x['amount_spent'], axis = 1)

    # Creating the total number of purchases - FREQUENCY
    df['total_purchases'] = df['numdealspurchases'] + df['numwebpurchases'] + df['numcatalogpurchases'] + df['numstorepurchases']

    df['accepted_campaigns'] = df['acceptedcmp1'] + df['acceptedcmp2'] + df['acceptedcmp3'] + df['acceptedcmp4'] + df['acceptedcmp5'] + df['response']

    # Create the age of the customer
    df['age'] = df['dt_customer'].dt.year.max() - df['year_birth']

    # Creating the total number of purchasese by each month
    df['purchases_month'] = df['total_purchases'] / (df['loyalty_months'] + 1)

    # Creating percentage spent on different products
    df['percentage_spent_wines'] = df['mntwines'] / df['amount_spent']
    df['percentage_spent_fruits'] = df['mntfruits'] / df['amount_spent']
    df['percentage_spent_meat'] = df['mntmeatproducts'] / df['amount_spent']
    df['percentage_spent_fish'] = df['mntfishproducts'] / df['amount_spent']
    df['percentage_spent_sweet'] = df['mntsweetproducts'] / df['amount_spent']
    df['percentage_spent_gold'] = df['mntgoldprods'] / df['amount_spent']

    # Creating percentage purchases on different channel types
    df['percentage_type_deals'] = df['numdealspurchases'] / df['total_purchases']
    df['percentage_type_web'] = df['numwebpurchases'] / df['total_purchases']
    df['percentage_type_catalog'] = df['numcatalogpurchases'] / df['total_purchases']
    df['percentage_type_store'] = df['numstorepurchases'] / df['total_purchases']

    return df



def filter_columns(df):

    # Only values lower than 100% 
    for col in ['percentage_type_deals', 'percentage_type_web', 'percentage_type_catalog', 'percentage_type_store', 'percentage_spent_wines', 'percentage_spent_fruits', 'percentage_spent_meat', 'percentage_spent_fish', 'percentage_spent_sweet','percentage_spent_gold']:
        df = df[df[col] < 1] 

    return df