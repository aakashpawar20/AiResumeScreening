import streamlit as st
import spacy
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
from collections import Counter

# Setting a modern color palette
match_score_color = "#6C5CE7"
skill_color = "#00CEC9"
eligible_color = "#2ECC71"
not_eligible_color = "#E74C3C"

# Predefined login credentials
USERNAME = "mmanu619"
PASSWORD = "mmanu@619"

# Predefined job roles and descriptions
job_roles = {
    "Data Scientist": """
    Responsibilities include data collection, processing, and analysis to extract insights and create predictive models. 
    Proficient in Python, machine learning, deep learning frameworks, and data visualization tools like Tableau.
    """,
    "Machine Learning Engineer": """
    Responsible for developing and deploying machine learning models and algorithms. 
    Requires skills in Python, TensorFlow, PyTorch, data preprocessing, and model deployment.
    """,
    "Backend Developer": """
    Focuses on server-side application logic and integration with frontend elements. 
    Skills include proficiency in server frameworks like Node.js, Flask, or Django, as well as databases and APIs.
    """,
    "Full Stack Developer": """
    Responsible for both client-side and server-side software, including frontend and backend. 
    Proficient in HTML, CSS, JavaScript, React, Node.js, databases, and RESTful API development.
    """,
    "Cloud Engineer": """
    Manages and supports cloud computing resources, including infrastructure, platforms, and software solutions. 
    Requires knowledge of cloud platforms like AWS, Azure, or Google Cloud, as well as DevOps practices.
    """,
    "DevOps Engineer": """
    Ensures smooth development, testing, and deployment cycles through automation and collaboration. 
    Skills include CI/CD pipelines, containerization (Docker, Kubernetes), and infrastructure as code (Terraform).
    """,
    "Cybersecurity Analyst": """
    Protects an organization’s digital assets from cyber threats. 
    Proficiency in network security, intrusion detection systems, vulnerability assessment, and incident response.
    """,
    "Data Engineer": """
    Designs, builds, and maintains data pipelines and architectures for analytics and machine learning. 
    Skills include ETL processes, SQL, data warehousing, and big data technologies like Spark and Hadoop.
    """,
    "Frontend Developer": """
    Specializes in user-facing web and mobile application development. 
    Skills include HTML, CSS, JavaScript, and frontend frameworks like React, Angular, or Vue.js.
    """,
    "Database Administrator": """
    Responsible for the design, implementation, and maintenance of database systems. 
    Requires expertise in SQL, database tuning, and administration for platforms like MySQL, Oracle, or PostgreSQL.
    """,
    "AI Research Scientist": """
    Conducts research to advance AI and ML fields, often working on developing algorithms and models. 
    Skills include deep knowledge of mathematics, neural networks, machine learning frameworks, and scientific research.
    """,
    "Software Engineer": """
    Engages in the design, development, and maintenance of software systems. 
    Core skills include programming languages like Java, C++, Python, and experience with software development methodologies.
    """,
    "IT Support Specialist": """
    Provides technical support for hardware and software systems, addressing user issues and ensuring smooth IT operations. 
    Skills include troubleshooting, network management, and a solid understanding of operating systems.
    """,
    "QA Engineer": """
    Responsible for software testing to ensure quality and functionality across development cycles. 
    Skills include test automation, understanding of SDLC, and proficiency with testing tools like Selenium or JIRA.
    """
}

# Function to add custom CSS for background, fonts, and other styles
def set_custom_style():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #1f1c2c, #928dab);
            color: #EAEAEA;
            font-family: 'Roboto', sans-serif;
        }
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
        
        .main-title {
            color: #ffffff;
            font-size: 3em;
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
            text-shadow: 2px 2px 5px #000000;
        }
        .stMarkdown h2 {
            color: #e0e0e0;
            font-size: 1.75em;
            font-weight: bold;
            margin-top: 20px;
            border-bottom: 1px solid #444;
            padding-bottom: 5px;
        }
        .css-10trblm {
            background-color: #333;
            color: #ffffff;
            border-radius: 10px;
            padding: 10px;
            font-size: 1em;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            font-size: 1.1em;
            border-radius: 8px;
            cursor: pointer;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        /* Styling for the results box */
        .results-box {
            background-color: #292b2c;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4);
            margin-bottom: 20px;
        }
        /* Highlighted headings for result summary */
        .highlight-heading {
            color: #F1C40F;
            font-weight: bold;
            font-size: 1.1em;
        }
        .results-content {
            color: #e0e0e0;
            font-size: 1em;
            line-height: 1.5;
        }
        .highlight {
            color: #4CAF50;
            font-weight: bold;
        }
        .highlight-red {
            color: #e74c3c;
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function to apply custom styling
set_custom_style()

# Login function
def login():
    st.markdown("<div class='main-title'>AI-Powered Resume Screening</div>", unsafe_allow_html=True)
    with st.form("login_form", clear_on_submit=True):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            if username == USERNAME and password == PASSWORD:
                st.success("Login successful!")
                st.session_state["logged_in"] = True
                st.experimental_rerun()
            else:
                st.error("Invalid username or password.")

# Load spaCy NLP model and DistilBERT model for embeddings
@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_sm")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    return nlp, tokenizer, model

nlp, tokenizer, model = load_models()

# Function to extract text from files
def extract_text_from_file(file):
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        text = "".join(page.extract_text() for page in reader.pages)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
    elif file.type == "text/plain":
        text = file.read().decode("utf-8")
    else:
        st.warning("Unsupported file format")
        text = ""
    return text

# Function to extract skills
def extract_skills(text):
    skills = ["algorithms", "data structures", "programming", "software development", "web development", "mobile app development", "cloud computing", "cybersecurity", "artificial intelligence", "machine learning", "deep learning", "natural language processing", "computer vision", "data analysis", "data mining", "database management", "software testing", "debugging", "problem solving", "critical thinking", "teamwork", "communication", "project management", "agile methodologies", "version control", "git", "linux", "command-line interface", "networking", "network security", "computer architecture", "operating systems", "embedded systems", "internet of things", "iot", "robotics", "automation", "devops", "cloud security", "ethical hacking", "penetration testing", "vulnerability assessment", "risk management", "software engineering principles", "design patterns", "object-oriented programming", "functional programming", "scripting languages", "front-end development", "html", "css", "javascript", "back-end development", "api development", "ui/ux design", "user research", "data visualization", "statistical analysis", "mathematics", "logic", "linear algebra", "calculus", "discrete mathematics", "probability", "statistics", "big data", "hadoop", "spark", "nosql databases", "sql", "distributed systems", "parallel computing", "high-performance computing", "bioinformatics", "computational biology", "game development", "graphics programming", "virtual reality", "vr", "augmented reality", "ar", "3d modeling", "animation", "simulation", "cryptography", "blockchain", "quantum computing", "full-stack development", "technical writing", "research", "analysis", "testing", "deployment", "maintenance", "optimization", "security auditing", "code review", "refactoring", "continuous integration", "continuous delivery", "containerization", "docker", "kubernetes", "microservices", "serverless computing", "edge computing", "quantum information theory", "network protocols", "tcp/ip", "http", "wireless networking", "mobile networks", "digital signal processing", "image processing", "control systems", "mechatronics", "c", "c++", "c#", "java", "python", "javascript", "typescript", "go", "ruby", "php", "swift", "kotlin", "objective-c", "rust", "scala", "perl", "r", "julia", "ada", "assembly language", "visual basic", "delphi", "fortran", "cobol", "lisp", "prolog", "haskell", "erlang", "elixir", "clojure", "smalltalk", "lua", "matlab", "groovy",  "kotlin", "scala", "typescript", "sql", "bash", "shell scripting", "powershell", "awk", "sed", "regular expressions", "python libraries", "numpy", "pandas", "scikit-learn", "javascript frameworks", "react", "angular", "vue.js", "node.js", "express.js", "django", "flask", "ruby on rails", "spring boot", "asp.net core", "android development", "ios development", "cross-platform development", "react native", "flutter",  "game engines", "unity", "unreal engine", "cloud platforms", "aws", "azure", "gcp", "databases", "mysql", "postgresql", "mongodb", "data warehousing", "data modeling", "etl", "extract", "transform", "load", "business intelligence", "data science", "machine learning algorithms", "regression", "classification", "clustering", "deep learning frameworks", "tensorflow", "pytorch", "computer graphics", "opengl", "vulkan", "websockets", "restful apis", "graphql", "microservices architecture", "service-oriented architecture", "soa", "version control systems", "svn", "mercurial", "agile development methodologies", "scrum", "kanban", "software design principles", "solid", "testing frameworks", "junit", "pytest", "code quality tools", "sonarqube", "eslint", "ci/cd pipelines", "jenkins", "gitlab ci", "security best practices", "owasp top 10", "encryption techniques", "authentication and authorization", "access control", "network security tools", "wireshark", "nmap", "firewall management", "intrusion detection systems", "security information and event management", "siem", "data loss prevention", "dlp", "compliance and regulations", "gdpr", "hipaa", "cloud security best practices", "devsecops", "ethical hacking techniques", "penetration testing methodologies", "vulnerability scanning", "security audits", "incident response", "disaster recovery", "business continuity planning",  "technical documentation", "communication skills", "written and verbal", "presentation skills", "interpersonal skills", "time management", "project planning", "risk assessment", "decision making", "leadership", "mentoring", "problem-solving", "analytical thinking", "creativity", "innovation", "adaptability", "continuous learning", "research skills", "academic writing", "scientific computing", "numerical analysis", "optimization techniques", "algorithm design", "complexity analysis", "software architecture", "system design", "distributed systems design", "concurrency", "parallelism", "performance tuning", "code optimization", "memory management", "garbage collection", "debugging techniques", "troubleshooting", "root cause analysis", "software maintenance", "software evolution", "legacy systems", "software refactoring", "code reviews", "pair programming", "technical debt management", "software documentation", "api documentation", "user manuals", "training materials", "software licensing", "open source software", "intellectual property", "software patents", "copyright law", "data privacy", "data security", "ethics in computing", "social impact of technology", "sustainable computing", "green it", "accessibility", "human-computer interaction", "usability engineering", "user experience", "ux", "research", "user interface", "ui", "design", "information architecture", "interaction design", "visual design", "prototyping", "wireframing", "mockups", "user testing", "a/b testing", "design thinking", "lean ux", "agile ux", "accessibility standards", "wcag", "responsive design", "mobile-first design", "cross-browser compatibility", "search engine optimization", "seo", "web analytics", "conversion rate optimization", "cro", "e-commerce", "digital marketing", "social media marketing", "content marketing", "email marketing", "affiliate marketing", "pay-per-click", "ppc", "advertising", "search engine marketing", "sem", "marketing automation", "customer relationship management", "crm", "salesforce", "data analytics platforms", "google analytics", "project management tools", "jira", "trello", "communication tools", "slack", "microsoft teams", "collaboration tools", "python", "machine learning", "deep learning", "sql", "javascript", "react", "node.js", "excel", "tableau", "tensorflow", "java", "c++", "c#", "html", "css", "angular", "docker", "kubernetes", "flask", "django", "aws", "azure", "gcp", "pytorch", "scikit-learn", "nlp", "hadoop", "spark", "big data", "keras", "pandas", "numpy", "git", "bash", "linux", "r", "matlab", "data visualization", "business analysis", "mysql", "postgresql", "mongodb", "oracle", "hive", "power bi", "airflow", "selenium", "ci/cd", "jenkins", "terraform", "cloud computing", "data engineering", "google workspace", "microsoft 365", "cloud storage", "aws s3", "azure blob storage", "content management systems", "wordpress", "drupal", "e-learning platforms", "moodle", "coursera", "video conferencing tools", "zoom", "google meet", "online collaboration tools", "miro", "figma", "variables", "data types", "operators", "control flow", "functions", "classes", "objects", "inheritance", "polymorphism", "encapsulation", "abstraction", "modules", "packages", "libraries", "frameworks", "debugging", "testing", "code style", "documentation", "version control", "software development life cycle (sdlc)", "agile development", "scrum", "kanban", "waterfall", "requirements gathering", "design", "implementation", "testing", "deployment", "maintenance", "software architecture", "microservices", "monolithic architecture", "cloud-native development", "serverless computing", "containers", "databases", "relational databases", "nosql databases", "sql", "data modeling", "normalization", "acid properties", "transactions", "data warehousing", "data mining", "big data", "hadoop", "spark", "data analytics", "machine learning", "deep learning", "supervised learning", "unsupervised learning", "reinforcement learning", "computer networks", "network topologies", "network protocols", "tcp/ip", "http", "dns", "routing", "switching", "firewalls", "cybersecurity", "network security", "application security", "data security"]
    matcher = spacy.matcher.PhraseMatcher(nlp.vocab)
    patterns = [nlp(skill) for skill in skills]
    matcher.add("SKILLS", None, *patterns)
    
    doc = nlp(text.lower())
    matches = matcher(doc)
    return list({doc[start:end].text for _, start, end in matches})

# Function to calculate BERT embeddings
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze()

# Function to compute similarity score
def compute_similarity(resume_text, job_text):
    resume_emb = get_bert_embedding(resume_text)
    job_emb = get_bert_embedding(job_text)
    return cosine_similarity(resume_emb.view(1, -1), job_emb.view(1, -1))[0][0]

# Function to save results to Excel
def save_to_excel(results):
    output = BytesIO()
    df = pd.DataFrame(results)
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Results')
    return output.getvalue()

# Function to display results summary in a styled box
def display_results_summary(result):
    st.markdown(
        f"""
        <div class="results-box">
            <div class="results-title">Resume: {result['Resume Name']}</div>
            <div class="results-content">
                <span class="highlight-heading">Similarity Score:</span> <span class="highlight">{result["Similarity Score"]:.2f}</span><br>
                <span class="highlight-heading">Total Match Score:</span> <span class="highlight">{result["Total Match Score"]:.2f}</span><br>
                <span class="highlight-heading">Eligibility Status:</span> <span class="{'highlight' if result['Eligibility Status'] == 'Eligible' else 'highlight-red'}">{result["Eligibility Status"]}</span><br>
                <span class="highlight-heading">Additional Skills in Resume:</span> <span class="results-content">{result["Additional Skills"]}</span><br>
                <span class="highlight-heading">Required Skills Missing from Resume:</span> <span class="results-content">{result["Required Skills"]}</span>
            </div>
        </div>
        """, unsafe_allow_html=True
    )

# Main app
def main_app():
    st.markdown("<div class='main-title'>AI-Powered Resume Screening</div>", unsafe_allow_html=True)

    # Optional dropdown for predefined job roles
    st.subheader("Select a Job Role (Optional)")
    selected_role = st.selectbox("Choose a job role", ["Select a role"] + list(job_roles.keys()))
    job_description_text = job_roles[selected_role] if selected_role != "Select a role" else ""

    # File upload section
    st.subheader("Upload Files")
    resume_files = st.file_uploader("Upload Resumes (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    job_file = st.file_uploader("Upload Job Description (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

    # Use uploaded job description file if available, otherwise use the selected predefined job description
    if job_file:
        job_description_text = extract_text_from_file(job_file)
    elif job_description_text == "":
        st.warning("Please upload a job description or select a job role from the dropdown.")

    if resume_files and job_description_text:
        # Process resumes with the selected or uploaded job description text
        job_skills = extract_skills(job_description_text)
        eligible_count = 0  # Count for eligibility breakdown
        all_skills = []     # List for top skills chart

        results = []
        for resume_file in resume_files:
            resume_text = extract_text_from_file(resume_file)
            resume_skills = extract_skills(resume_text)
            additional_skills = list(set(resume_skills) - set(job_skills))
            required_skills = list(set(job_skills) - set(resume_skills))
            similarity_score = compute_similarity(resume_text, job_description_text)
            skill_overlap = len(set(resume_skills).intersection(set(job_skills)))
            total_match_score = 0.7 * similarity_score + 0.3 * (skill_overlap / len(job_skills)) if job_skills else similarity_score

            eligibility_status = "Eligible" if total_match_score >= 0.75 else "Not Eligible"
            if eligibility_status == "Eligible":
                eligible_count += 1

            all_skills.extend(resume_skills)

            result = {
                "Resume Name": resume_file.name,
                "Similarity Score": similarity_score,
                "Total Match Score": total_match_score,
                "Eligibility Status": eligibility_status,
                "Additional Skills": ", ".join(additional_skills) if additional_skills else "None",
                "Required Skills": ", ".join(required_skills) if required_skills else "None"
            }
            results.append(result)
            display_results_summary(result)

        # Excel download
        excel_data = save_to_excel(results)
        st.download_button(
            label="Download Results as Excel",
            data=excel_data,
            file_name="resume_screening_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # Visualization: Total Match Score Comparison
        st.subheader("Total Match Score Comparison")
        fig, ax = plt.subplots()
        resume_names = [result["Resume Name"] for result in results]
        match_scores = [result["Total Match Score"] for result in results]
        ax.barh(resume_names, match_scores, color=match_score_color)
        ax.set_xlabel("Total Match Score")
        ax.set_ylabel("Resume")
        ax.set_title("Comparison of Total Match Scores")
        for i, v in enumerate(match_scores):
            ax.text(v + 0.01, i, f"{v:.2f}", color="black", va="center")
        st.pyplot(fig)

        # Visualization: Top Skills Overlap
        st.subheader("Top Skills Overlap")
        skill_counts = Counter(all_skills)
        common_skills = skill_counts.most_common(10)
        skills, counts = zip(*common_skills)
        fig, ax = plt.subplots()
        ax.barh(skills, counts, color=skill_color)
        ax.set_xlabel("Frequency")
        ax.set_title("Most Common Skills Across All Resumes")
        for i, v in enumerate(counts):
            ax.text(v + 0.5, i, str(v), color="black", va="center")
        st.pyplot(fig)

        # Visualization: Eligibility Breakdown
        st.subheader("Eligibility Breakdown")
        fig, ax = plt.subplots()
        labels = ["Eligible", "Not Eligible"]
        sizes = [eligible_count, len(results) - eligible_count]
        colors = [eligible_color, not_eligible_color]
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140, colors=colors)
        for text in autotexts:
            text.set_color("white")
            text.set_fontweight("bold")
        ax.set_title("Percentage of Eligible vs. Not Eligible Resumes")
        st.pyplot(fig)

# Check if user is logged in
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
else:
    main_app()
