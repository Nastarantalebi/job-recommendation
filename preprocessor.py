import pandas as pd
import re

class DataFrameCleaner:
    def __init__(self, df, provinces_directory, cities_directory):
        self.df = pd.DataFrame(df)
        try:
            self.provinces_df = pd.read_csv(provinces_directory, encoding='utf-8')
        except OSError:
            print(f"provinces_df does not exist in {provinces_directory}")
        try:
            self.cities_df = pd.read_csv(cities_directory, encoding='utf-8')
        except OSError:
            print(f"cities does not exist in {cities_directory}")
        self.provinces_to_remove = self.provinces_df['name'].tolist()
        self.cities_to_remove = self.cities_df['name'].tolist()
        self.words_to_remove = [
            'خانم', 'آقا', 'اقا', 'کارشناس', 'و', 'سرپرست',
            'مدیر', 'مسئول', 'ارشد', 'دستیار', 'به', 'با', 'ها', '&',
            'ای', 'شهر', 'زن', 'مرد', 'کشور', 'and',
            'دورکاری', 'در', 'with', 'تا', 'مسلط', 'استخدام'
        ]
    
    def remove_invalid_rows(self):
        # حذف سطر هایی که دارای کلمه ی عوان شغلی است
        self.df = self.df[~self.df['title'].str.contains("عنوان شغلی", case=False, na=False)]
        return self.df

    def clean_text(self, text):
        if isinstance(text, str):
            text = re.sub(r'[()\[\]]', ' ', text)
            text = text.replace('-', ' ')
            text = text.replace('_', ' ')
            text = text.replace('\\', ' ')
            text = text.replace('|', ' ')
            text = text.replace('/', ' ')
            text = text.replace(',', ' ')
            text = text.replace(':', ' ')
            text = text.replace('.', ' ')
            text = text.replace('،', ' ')
            return text
        return text

    def remove_provinces(self, text):
        pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, self.provinces_to_remove)) + r')\b')
        return pattern.sub('', text)

    def remove_cities(self, text):
        pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, self.cities_to_remove)) + r')\b')
        return pattern.sub('', text)

    def remove_words(self, text):
        pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, self.words_to_remove)) + r')\b')
        return pattern.sub('', text)

    def clean_dataframe(self):
        # Step 0: Remove rows that contain (عنوان شغلی)
        self.remove_invalid_rows()
        # Step 1: Clean text in the specified column
        self.df['title'] = self.df['title'].astype(str).apply(self.clean_text)

        # Step 2: Remove provinces
        self.df['title'] = self.df['title'].apply(self.remove_provinces)

        # Step 3: Remove cities
        self.df['title'] = self.df['title'].apply(self.remove_cities)

        # Step 4: Remove specific words
        self.df['title'] = self.df['title'].apply(self.remove_words)

        # Step 5: Drop rows with all NaN values
        self.df.dropna(inplace=True, how='all')

        # Step 6: Drop duplicate rows
        self.df.drop_duplicates(subset=['title', 'main_category_id'], inplace=True)

        return self.df

#---------------------------------------------------------------------------------
        
class ListCleaner:
    def __init__(self, titles_list, provinces_directory, cities_directory):
        self.titles_list = titles_list
        try:
            self.provinces_df = pd.read_csv(provinces_directory, encoding='utf-8')
        except OSError:
            print(f"provinces_df does not exist in {provinces_directory}")
        try:
            self.cities_df = pd.read_csv(cities_directory, encoding='utf-8')
        except OSError:
            print(f"cities does not exist in {cities_directory}")
        self.cleaned_experiences = []
        self.provinces_to_remove = self.provinces_df['name'].tolist()
        self.cities_to_remove = self.cities_df['name'].tolist()

    def clean_text(self, text):
        if isinstance(text, str):
            text = re.sub(r'[()\[\]]', ' ', text)
            text = text.replace('-', ' ')
            text = text.replace('(', ' ')
            text = text.replace(')', ' ')
            text = text.replace('_', ' ')
            text = text.replace('\\', ' ')
            text = text.replace('|', ' ')
            text = text.replace('/', ' ')
            text = text.replace(',', ' ')
            text = text.replace(':', ' ')
            text = text.replace('.', ' ')
            text = text.replace('،', ' ')
            return text
        return text

    def remove_provinces(self, text):
        pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, self.provinces_to_remove)) + r')\b')
        return pattern.sub('', text)

    def remove_cities(self, text):
        pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, self.cities_to_remove)) + r')\b')
        return pattern.sub('', text)

    def remove_words(self, text):
        words_to_remove = ['خانم', 'آقا', 'اقا', 'کارشناس', 'و', 'سرپرست',
                           'مدیر', 'مسئول', 'ارشد', 'دستیار', 'به', 'با', 'ها', '&',
                           'ای', 'شهر', 'زن', 'مرد', 'کشور', 'and',
                           'دورکاری', 'در', 'with', 'تا', 'مسلط', 'استخدام' , 'عنوان شغلی']
        pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, words_to_remove)) + r')\b')
        return pattern.sub('', text)

    def process_experiences(self):
        # Step 1: Strip whitespace
        self.cleaned_experiences = [title.strip() for title in self.titles_list]

        # Step 2: Clean text
        self.cleaned_experiences = [self.clean_text(title) for title in self.cleaned_experiences]

        # Step 3: Remove provinces
        self.cleaned_experiences = [self.remove_provinces(title) for title in self.cleaned_experiences]

        # Step 4: Remove cities
        self.cleaned_experiences = [self.remove_cities(title) for title in self.cleaned_experiences]

        # Step 5: Remove specific words
        self.cleaned_experiences = [self.remove_words(title) for title in self.cleaned_experiences]

        return self.cleaned_experiences
    
# ----------------------------------------------------------------------------
