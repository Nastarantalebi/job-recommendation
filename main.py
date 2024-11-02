import os
import pandas as pd
import requests
# import tensorflow as tf
import json
import preprocessor
from DatabaseConnection import DatabaseConnectionClass
from LabseLanguageModel import EmbeddingModelClass
# from ClassifierModel import ModelTrainClass, ModelInferenceClass
from SVMClassifierModel import ModelTrainClass
from RecommenderModel import JobSeekerClass, RecommenderClass

def GetDatasetFromDB():
    # write database name here
    database_name = "e_estekhdam_db"
    # write quries (the tables and columns you need)
    query1 = " SELECT id, main_category_id FROM sub_category"
    query2 = "SELECT id, title, sub_category_id FROM joblist"
    # create a connection object
    db_connection = DatabaseConnectionClass(database_name, query1, query2)
    category_id_df, joblist_df = db_connection.get_dataframes()
    print("joblist_df: \n",joblist_df.tail(10).to_string())
    print("category_id_df: \n",category_id_df.tail(10).to_string())
    # merger two dataframes on on 'sub_category_id' columns
    df = pd.merge(joblist_df, category_id_df, on='sub_category_id', how='left')
    print("merged df: \n", df.tail(10).to_string())
    return df
 

def PreprocessDataframe(df):
    provinces_directory = os.path.join(os.getcwd(), "datasets/provinces.csv")
    cities_directory = os.path.join(os.getcwd(), "datasets/cities.csv")
    DFCleaner = preprocessor.DataFrameCleaner(df, provinces_directory, cities_directory)
    preprocessed_df = DFCleaner.clean_dataframe()
    return preprocessed_df

def PreprocessList(TitlesList):
    provinces_directory = os.path.join(os.getcwd(), "datasets/provinces.csv")
    cities_directory = os.path.join(os.getcwd(), "datasets/cities.csv")
    ListCleaner = preprocessor.ListCleaner(
        TitlesList, provinces_directory, cities_directory)
    preprocessed_list = ListCleaner.process_experiences()
    return preprocessed_list

def GetEmbeddings(titles_list):
    EmbeddingObject = EmbeddingModelClass()
    df_embeddings = EmbeddingObject.get_embeddings(titles_list)
    return df_embeddings

def build_classifier_model(df_embeddings, ClassifierModel_path):
    modelObject = ModelTrainClass(df_embeddings)
    modelObject.build_model(ClassifierModel_path)

# read user experiences from input.json
def read_ux_input():
    # Define the path to the input JSON file
    input_path = os.path.join(os.getcwd(), 'input.json')  # Assuming input.json is in the same directory

    # Read experiences from input.json file
    try:
        with open(input_path, 'r', encoding='utf-8') as file:
            user_data = json.load(file)
            # Combine the experiences and previous applications
            JobSeeker_experiences = user_data['experinces'] + user_data['perviousApplies']
            # Get the recommend number (p)
            recommend_number = user_data['numerOfRecommendedJobs']
            JobSeeker_city = user_data['cityId']
            JobSeeker_gender = user_data['genderId']
    except FileNotFoundError:
        print(f"Error: The file {input_path} was not found.")
        JobSeeker_experiences = []  # Return an empty list if the file is not found
        recommend_number = 0  # Set to 0 if the file is not found
        JobSeeker_city = -1
        JobSeeker_gender = 4
    except json.JSONDecodeError:
        print("Error: The JSON file is not valid.")
        JobSeeker_experiences = []  # Return an empty list if JSON is invalid
        recommend_number = 0  # Set to 0 if JSON is invalid
        JobSeeker_city = -1
        JobSeeker_gender = 4

    return JobSeeker_experiences, recommend_number, JobSeeker_city, JobSeeker_gender


def get_active_job_opportunities(top_experience_categories, cityId, genderId):
    opportunities = pd.DataFrame()
    experiences = []
    cities = []
    genders = []
    experiences = experiences.append(top_experience_categories.keys())
    cities = cities.append(cityId)
    genders = genders.append(genderId)
    url = 'https://api.soha-ats.com/api/v2/jobEmployers/filter'
    data = {
        "categories": experiences,
        "subcategories":[],
        "cities":cities,
        "genders":genders
    }
    
    # Make the POST request to the API
    response = requests.post(url, json=data)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        
        # Convert the JSON data to a DataFrame
        df = pd.DataFrame(data)
        
        # Display the DataFrame
        print("The DataFrame came from api:\n",df.head(20).to_string())
    else:
        print(f"Failed to retrieve data: {response.status_code}")
    
    return df


def main():
    # enter classifier model path here
    ClassifierModel_path = "models/SVMClassifierModel.sav"
    # check classifier model existance in local directory 
    if os.path.exists(ClassifierModel_path):
        print("Classifier model already exists.")
        # #load the model
        # classifier_model = tf.keras.models.load_model(ClassifierModel_path)
    else:
        print("Classifier model does not exist. let's create that.")
        print("Loading the dataset from database...")
        # Load the dataset to train the classifier model
        dataset = GetDatasetFromDB()
        print("dataset Loaded.")
        print("preprocessing the dataset...")
        # preprocess the dataset
        preprocessed_dataset = PreprocessDataframe(dataset)
        print("dataset preprocessed> here is the result: \n", preprocessed_dataset.head(10))
        print(preprocessed_dataset.info())
        titles_list = preprocessed_dataset['title']
        # get dataset embeddings(converting titles to numerical vectors)
        dataset_embeddings = GetEmbeddings(titles_list)
        # List of columns to copy
        columns_to_copy = ['title', 'main_category_id']
        # Copy columns of df in df_embeddings, handling potential index mismatches
        for col in columns_to_copy:
            dataset_embeddings[col] = preprocessed_dataset[col].reset_index(drop=True)
        # dataset_embeddings['title'] = preprocessed_dataset['title']
        # dataset_embeddings['main_catergory_id'] = preprocessed_dataset['main_catergory_id']
        print("building the model...")
        build_classifier_model(dataset_embeddings, ClassifierModel_path)
        #load the model
        # classifier_model = tf.keras.models.load_model(ClassifierModel_path)


    # read job seeker informations from input.json
    experinces, recommendation_number, cityId, genderId = read_ux_input()
    # preprocess job seeker experiences
    preprocessed_experinces = PreprocessList(experinces)
    # get embedding of experiesnces
    experinces_embeddings = GetEmbeddings(preprocessed_experinces)
    # get the most repeated categories from recommender
    job_seeker_object = JobSeekerClass()
    #get the predictions numpy array
    predictions = job_seeker_object.classify_experiences(ClassifierModel_path, experinces_embeddings)
    # Add experiences titles and their predicted class in predictions to experinces_embeddings
    experinces_embeddings['predicted_class'] = predictions
    print('the prediction is: ', predictions)
    # get the most repeated categories from recommender
    top_experience_categories = job_seeker_object.find_top_experience_categories(predictions)
    # get active job opportunities in the returned categories with api
    opportunities = get_active_job_opportunities(
        top_experience_categories,
        cityId,
        genderId)
    # preprocess active job opportunities
    preprocessed_opportunities = PreprocessList(opportunities)
    # get embedding of opportunities
    opportunities_embeddings = GetEmbeddings(preprocessed_opportunities)
    
    # List of columns to copy
    columns_to_copy = ['title', 'jobCategoryId', 'id', 'companyId']
    # Copy columns of df in opportunities, handling potential index mismatches
    for col in columns_to_copy:
        opportunities_embeddings[col] = opportunities[col].reset_index(drop=True)

    Recommender = RecommenderClass()

    # give embeddings and classifier model to recommender model
    recommending_jobs = Recommender.find_most_similars(
        experinces_embeddings,
        opportunities_embeddings,
        top_experience_categories,
        recommendation_number)
    print("The final recomendation is:", recommending_jobs)
    # Save the self.similar_jobs_dict to a JSON file
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(recommending_jobs.to_dict(orient="records"), f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
