import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from SVMClassifierModel import ModelInferenceClass
from collections import Counter
import heapq
from math import ceil

class JobSeekerClass:
    def classify_experiences(self, classifier_model_path, experiences_embeddings):
        InferenceModelObject = ModelInferenceClass(classifier_model_path)
        #get the predictions numpy array
        predictions = InferenceModelObject.inference(experiences_embeddings)
        return predictions

    def find_top_experience_categories(self, predictions):

        # Count the occurrences of each class using Counter
        class_counts = Counter(predictions)

        # Find the top 3 most repeated classes
        top_classes = heapq.nlargest(3, class_counts.items(), key=lambda item: item[1])

        # Convert the list of tuples to a dictionary
        top_classes_dict = {class_name: count for class_name, count in top_classes}

        return top_classes_dict

class RecommenderClass:
    def __init__(self):
        self.x_category_recommendings = pd.DataFrame(
            columns=["title", "companyId", "id", "repeatation_number"]
        )
        self.final_recommendations = pd.DataFrame(
            columns=["title", "companyId", "id"]
        )

    def find_most_similars(self, experiences_df, opportunities_embedding_df, top_experience_categories, recommend_number):
        total_count = sum(top_experience_categories.values())
        print("total_count: ", total_count)
        print("top_experience_categories dict:\n", top_experience_categories)
        top_experience_categories_array = np.array([(class_name, count) for class_name, count in top_experience_categories.items()])
        top_classes_with_recommend_number_dict = {
            class_name: ceil(recommend_number * (count / total_count))
            for class_name, count in top_experience_categories_array
        }
        print("top_classes_with_recommend_number dict:\n", top_classes_with_recommend_number_dict)
        for category in top_experience_categories.keys():
            related_experiences = experiences_df[experiences_df["predicted_class"] == category]
            related_opportunities = opportunities_embedding_df[opportunities_embedding_df["jobCategoryId"] == category]
            class_recommend_num = top_classes_with_recommend_number_dict[category]
            if related_opportunities.empty:
                print("There is no related job opportunities")
                continue
            for _, row in related_experiences.iterrows():
                ex_vector = row.iloc[:768].values
                similarities = cosine_similarity([ex_vector], related_opportunities.iloc[:, :768].values)
                top_indices = similarities[0].argsort()[-class_recommend_num:][::-1]
                similar_jobs = related_opportunities.iloc[top_indices][["title", "companyId", "id"]].to_dict('records')
                for job in similar_jobs:
                    if job["id"] in self.x_category_recommendings["id"].values:
                        self.x_category_recommendings.loc[
                            self.x_category_recommendings["id"] == job["id"],
                            "repeatation_number",
                        ] += 1
                    else:
                        self.x_category_recommendings = self.x_category_recommendings.append(
                            {
                                "title": job["title"],
                                "companyId": job["companyId"],
                                "id": job["id"],
                                "repeatation_number": 1,
                            },
                            ignore_index=True,
                        )
            self.x_category_recommendings = self.x_category_recommendings.sort_values(by="repeatation_number", ascending=False)
            self.x_category_recommendings = self.x_category_recommendings.drop(columns=["repeatation_number"])
            head_df = self.x_category_recommendings.head(class_recommend_num)
            self.final_recommendations = pd.concat([self.final_recommendations, head_df], ignore_index=True)
            self.x_category_recommendings = self.x_category_recommendings.iloc[0:0]  # Clear the DataFrame
            head_df = head_df.iloc[0:0]
        if len(self.final_recommendations) > recommend_number:
            self.final_recommendations = self.final_recommendations.head(recommend_number)
        return self.final_recommendations