## Name
Soha Job Opportunity Recommender | 
سیستم پیشنهاد دهنده فرصت شغلی سها

## Description
In this project, suitable job opportunities are offered to job seekers according to their work records and application history. For this purpose, first, using the real data of the job search sites, a classification neural network model is created and stored to categorize the job opportunities. In this process, the LaBSE (Language-agnostic BERT Sentence Embedding) model is used, which is a Transformer model of multilingual sentences and is developed based on the BERT language model. This model is used to understand the texts of job advertisements, job records and previous requests of job seekers. After the model is prepared, in order to make an offer to each job seeker, his career records are first classified by the classification model and the field of work and interest of the person is recognized. Then active job advertisements in the field related to the job seeker are received from the backend using an API, and then the nearest opportunities are found using Cosine Similarity and offered to him. <br>
<br>
در این پروژه ارائه پیشنهاد فرصت های شغلی مناسب به کارجویان با توجه به سوابق کاری و سابقه درخواست های کارجویان انجام میشود. برای این کار ابتدا با استفاده از داده‌های واقعی سایت‌های کاریابی، یک مدل شبکه عصبی کلاسبندی برای دسته‌بندی فرصت‌های شغلی ساخته و ذخیره میشود. در این فرآیند، از مدل LaBSE (Language-agnostic BERT Sentence Embedding) استفاده میشود که یک مدل Transformer جملات چندزبانه است و بر پایه مدل زبانی BERT توسعه داده شده است. این مدل برای درک متون آگهی‌های شغلی، سوابق شغلی و درخواست‌های قبلی کارجو و تبدیل آن‌ها به بردارهای عددی بکار میرود. پس از آماده شدن مدل، برای ارائه پیشنهاد به هر کارجو ابتدا سوابق شغلی او توسط مدل کلاس بندی دسته بندی داده میشود و حوزه کاری و مورد علاقه فرد تشخیص داده میشود. سپس آگهی های شغلی فعال در حوزه مربوط به کارجو با استفاده از یک API از بک اند دریافت شده و سپس نزدیک ترین فرصت ها به او پیشنهاد میشود. 

## Usage
Here we see a example of using this system and the input and the expected output of this project: <br>
**Input:** <br>
    The input is a JSON file containing the job seeker’s information, including work records and experiences, application histories, city, and gender. Additionally, the number of recommended jobs is provided at the end. <br>
    Here you see an example input.json file:<br>
`{

    "cityId": 12,

    "genderId": 1,
    
    "experinces": ["کارشناس فروش", "مدیر بازاریابی"],
    
    "perviousApplies": ["کارشناس فروش", "مدیر فروش"],
    
    "numerOfRecommendedJobs": 5
    
}`  

**Output:** <br>
    The output of the project is a JSON file containing information about the recommended job opportunities, including the title, link, and ID.
## Support
if you had problem in using this project you can contact me via armankhalilieng@gmail.com

## Roadmap
We can enhance the analysis of job opportunities and job seekers’ resumes by using advanced text mining techniques on job descriptions and the texts users have written about themselves. However, this approach has been set aside for now, and recommendations are currently made solely based on job titles and job seekers’ titles.

## Authors and acknowledgment
Arman Khalili, Nastaran Talebi, Amir Jafari