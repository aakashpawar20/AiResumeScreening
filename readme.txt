How to Use the Project
Clone the Repository
First, clone the repository to your local machine using Git:

bash
Copy code
git clone https://github.com/m-manu619/AI-Powered-Resume-Screening-and-Job-Matching-Tool.git
Install the Required Packages
Navigate to the project directory:

bash
Copy code
cd AI-Powered-Resume-Screening-and-Job-Matching-Tool
Then, install the necessary dependencies listed in the requirements.txt file:

Copy code
pip install -r requirements.txt
Run the Application
This project is based on Streamlit, a framework for building web applications in Python. To run the application, use:

arduino
Copy code
streamlit run app.py
This will launch the application in your default browser.

Interact with the Application

Login: Enter the predefined credentials to log in.

Username: aakash8767
Password: aakash@8767
Upload a Resume: You can upload a resume in PDF, DOCX, or TXT format.

Upload a Job Description: You can either upload a custom job description or choose from predefined ones.

View Match Scores: The app will display a match score that indicates how well the resume matches the job description based on extracted skills and qualifications.

Export Results: After processing the resume, you can export the match results (including skill overlaps) as an Excel file.

How the Project Works
This project uses Natural Language Processing (NLP) and machine learning techniques to match resumes with job descriptions. Here's a breakdown of the working components:

Text Extraction

The app extracts text from the uploaded resume files (PDF, DOCX, or TXT formats) using libraries like PyPDF2 (for PDFs) and python-docx (for DOCX files).
Skills Extraction

The app processes the extracted text to identify key skills and experience listed in the resume and job description. This may be done through regular expressions, keyword matching, or more advanced NLP methods.
Semantic Matching Using BERT

The app uses BERT (Bidirectional Encoder Representations from Transformers), a state-of-the-art NLP model, to calculate the semantic similarity between the resume and job description. This helps determine how well the candidate's qualifications align with the job requirements.
Match Scoring

The app generates a match score that reflects how well the resume fits the job description based on a comparison of the skills and qualifications. This score is based on NLP analysis using the BERT embeddings and other algorithms.
Eligibility Scoring

Based on the match score, the app determines whether the candidate is a good fit for the job role.
Visualization

The app also provides visual charts and graphs that compare the skill overlaps and match scores between the resume and job description.
Customization
Job Roles and Descriptions: You can modify the job roles and job descriptions in the job_roles dictionary to add new roles or change existing ones.
Skills Extraction: The extract_skills() function can be customized to include additional skills or improve the accuracy of skill extraction.
Login Credentials: If needed, you can change the login credentials for user authentication.