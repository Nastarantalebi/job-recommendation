import pandas as pd
import mysql.connector
from mysql.connector import Error
import os

class DatabaseConnectionClass:
    
    def __init__(self, database, query1, query2):
        self.database = database  # Store the database name
        self.query1 = query1 # Store the query
        self.query2 = query2
        self.db_config = {
            'host': os.getenv('DB_HOST', "database"),
            'user': os.getenv('DB_USERNAME', "root"),
            'password': os.getenv('DB_PASSWORD', "123456"),
            'database': self.database
        }
        self.connection = self._create_connection()
       
    def _create_connection(self):
        try:
            connection = mysql.connector.connect(**self.db_config)
            return connection
        except Error as e:
            print(f"Error connecting to the database: {e}")
            return None
        
    def get_dataframes(self):
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(self.query1)  # Use the provided query
            category_id_listoftuple = cursor.fetchall()
            category_id_df = pd.DataFrame(category_id_listoftuple, columns=['sub_category_id', 'main_category_id'])
        except Error as e:
            print(f"Error reading data: {e}")
            return pd.DataFrame(), pd.DataFrame()  # Return an empty DataFrame on error
        try:
            cursor = self.connection.cursor()
            cursor.execute(self.query2)
            joblist_listoftuple = cursor.fetchall()
            joblist_df = pd.DataFrame(joblist_listoftuple, columns=['id', 'title', 'sub_category_id'])
            joblist_df.dropna(subset=['sub_category_id'], inplace=True)
            joblist_df.drop_duplicates(subset=['title', 'sub_category_id'], keep='first', inplace=True)
            #joblist_df = joblist_df.sample(n=20000)
        except Error as e:
            print(f"Error reading data: {e}")
            return pd.DataFrame(), pd.DataFrame()  # Return an empty DataFrame on error
        
        return category_id_df, joblist_df
