"""
data/generate_dataset.py
────────────────────────────────────────────────────────────────────────────────
Synthetic dataset generator that mirrors the Kaggle "UpdatedResumeDataSet" and
"Resume Dataset" structure.

In a real deployment you would replace this with:
    kaggle datasets download -d gauravduttakiit/resume-dataset
    kaggle datasets download -d snehaanbhawal/resume-dataset

This script generates:
  - resumes.csv         : 500 synthetic resumes with category labels
  - job_descriptions.csv: 20 JD templates for common roles
  - ats_scores.csv      : Ground-truth ATS scores for evaluation
────────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
import random
import os
import json

random.seed(42)
np.random.seed(42)

# ── Skill pools per category ──────────────────────────────────────────────────
CATEGORIES = {
    "Data Science": {
        "skills": ["Python", "R", "Machine Learning", "Deep Learning", "TensorFlow",
                   "PyTorch", "Pandas", "NumPy", "Scikit-learn", "SQL", "Tableau",
                   "Power BI", "Statistics", "NLP", "Computer Vision", "Keras",
                   "XGBoost", "Feature Engineering", "Data Wrangling", "Spark"],
        "roles": ["Data Scientist", "ML Engineer", "AI Researcher", "Data Analyst",
                  "Research Scientist"],
        "companies": ["Google", "Amazon", "Microsoft", "Meta", "Netflix"]
    },
    "Web Development": {
        "skills": ["JavaScript", "React", "Node.js", "HTML", "CSS", "TypeScript",
                   "Vue.js", "Angular", "MongoDB", "PostgreSQL", "REST API",
                   "GraphQL", "Docker", "AWS", "Git", "Redux", "Express.js",
                   "Tailwind CSS", "Next.js", "Webpack"],
        "roles": ["Frontend Developer", "Backend Developer", "Full Stack Developer",
                  "Web Developer", "Software Engineer"],
        "companies": ["Infosys", "TCS", "Wipro", "Accenture", "Cognizant"]
    },
    "DevOps": {
        "skills": ["Docker", "Kubernetes", "Jenkins", "AWS", "Azure", "GCP",
                   "Terraform", "Ansible", "Linux", "Bash", "CI/CD", "Git",
                   "Prometheus", "Grafana", "Helm", "ArgoCD", "Python",
                   "Networking", "Security", "ELK Stack"],
        "roles": ["DevOps Engineer", "SRE", "Cloud Engineer", "Infrastructure Engineer",
                  "Platform Engineer"],
        "companies": ["Amazon", "Microsoft", "Google", "IBM", "Red Hat"]
    },
    "Android Developer": {
        "skills": ["Java", "Kotlin", "Android SDK", "XML", "Firebase", "Retrofit",
                   "Room Database", "MVVM", "Jetpack Compose", "Material Design",
                   "REST API", "SQLite", "Git", "Gradle", "Unit Testing",
                   "RecyclerView", "LiveData", "Coroutines", "Dagger", "Hilt"],
        "roles": ["Android Developer", "Mobile Developer", "Software Engineer",
                  "App Developer", "Junior Android Developer"],
        "companies": ["Flipkart", "Swiggy", "Zomato", "Paytm", "BYJU'S"]
    },
    "HR": {
        "skills": ["Recruitment", "Talent Acquisition", "HRMS", "Onboarding",
                   "Performance Management", "Employee Relations", "Payroll",
                   "SAP HR", "Labor Law", "Training & Development", "MS Excel",
                   "Communication", "Compensation & Benefits", "ATS", "LinkedIn",
                   "Interviewing", "Workforce Planning", "HRBP", "Compliance"],
        "roles": ["HR Manager", "HR Executive", "Talent Acquisition Specialist",
                  "HRBP", "Recruiter"],
        "companies": ["Deloitte", "Accenture", "Infosys", "TCS", "Capgemini"]
    },
    "Java Developer": {
        "skills": ["Java", "Spring Boot", "Hibernate", "Maven", "Microservices",
                   "REST API", "SQL", "Git", "JUnit", "Docker", "Kafka",
                   "AWS", "Design Patterns", "OOP", "MySQL", "PostgreSQL",
                   "Redis", "JWT", "Swagger", "CI/CD"],
        "roles": ["Java Developer", "Backend Developer", "Software Engineer",
                  "Senior Java Developer", "Full Stack Java Developer"],
        "companies": ["Oracle", "SAP", "IBM", "Infosys", "Wipro"]
    },
    "Testing": {
        "skills": ["Selenium", "TestNG", "JUnit", "JIRA", "Postman", "API Testing",
                   "Manual Testing", "Automation Testing", "Python", "Java",
                   "SQL", "Git", "Agile", "Regression Testing", "UAT",
                   "Performance Testing", "JMeter", "BDD", "Cucumber", "Cypress"],
        "roles": ["QA Engineer", "Test Engineer", "SDET", "QA Analyst",
                  "Automation Engineer"],
        "companies": ["Infosys", "Capgemini", "Accenture", "TCS", "LTIMindtree"]
    },
    "Business Analyst": {
        "skills": ["Business Analysis", "Requirements Gathering", "SQL", "Excel",
                   "Power BI", "Tableau", "Agile", "Scrum", "JIRA", "Visio",
                   "Use Cases", "Stakeholder Management", "UML", "Process Mapping",
                   "Data Analysis", "Communication", "Presentation", "Wireframing"],
        "roles": ["Business Analyst", "Product Analyst", "Systems Analyst",
                  "BA Consultant", "Functional Analyst"],
        "companies": ["Deloitte", "KPMG", "EY", "PWC", "Accenture"]
    }
}

UNIVERSITIES = [
    "IIT Bombay", "IIT Delhi", "NIT Surathkal", "VTU", "Anna University",
    "BITS Pilani", "Manipal University", "SRM University", "Pune University",
    "Bangalore University", "Delhi University", "Jadavpur University"
]

DEGREES = ["B.E.", "B.Tech", "M.Tech", "MCA", "BCA", "M.Sc", "B.Sc"]

CERTIFICATIONS = [
    "AWS Certified Solutions Architect", "Google Cloud Professional",
    "Microsoft Azure Fundamentals", "Certified Scrum Master",
    "PMP Certification", "TensorFlow Developer Certificate",
    "HackerRank Python", "Coursera Machine Learning", "Udemy Full Stack",
    "Oracle Java SE Certified"
]

ACTION_VERBS = [
    "Developed", "Designed", "Implemented", "Built", "Optimized",
    "Led", "Managed", "Collaborated", "Delivered", "Achieved",
    "Reduced", "Improved", "Increased", "Automated", "Architected"
]


def generate_resume_text(category: str, experience_years: int, quality: str) -> str:
    """
    Generate a synthetic resume text.
    quality: 'high' | 'medium' | 'low'
    """
    cat_data = CATEGORIES[category]
    n_skills = {"high": 12, "medium": 8, "low": 4}[quality]
    skills = random.sample(cat_data["skills"], min(n_skills, len(cat_data["skills"])))
    role = random.choice(cat_data["roles"])
    company = random.choice(cat_data["companies"])
    university = random.choice(UNIVERSITIES)
    degree = random.choice(DEGREES)
    n_certs = {"high": 2, "medium": 1, "low": 0}[quality]
    certs = random.sample(CERTIFICATIONS, n_certs) if n_certs > 0 else []

    # Summary section
    summary = f"Motivated {role} with {experience_years} year(s) of experience in {category}."
    if quality == "high":
        summary += (f" Proven track record of delivering scalable solutions at {company}. "
                    f"Strong expertise in {', '.join(skills[:4])}.")

    # Experience section
    exp_lines = []
    for i in range(min(experience_years, 3)):
        verb = random.choice(ACTION_VERBS)
        skill_used = random.choice(skills)
        if quality == "high":
            exp_lines.append(
                f"• {verb} {skill_used}-based solution that reduced processing time by "
                f"{random.randint(20,60)}% at {company}."
            )
        elif quality == "medium":
            exp_lines.append(f"• {verb} {skill_used} features for internal tools.")
        else:
            exp_lines.append(f"• Worked on {skill_used}.")

    # Education
    grad_year = 2024 - experience_years
    education = f"{degree} in Computer Science, {university}, {grad_year}"

    # Build resume text
    sections = [
        f"SUMMARY\n{summary}",
        f"SKILLS\n{', '.join(skills)}",
        f"EXPERIENCE\n{chr(10).join(exp_lines) if exp_lines else 'Fresher / Intern experience'}",
        f"EDUCATION\n{education}",
    ]
    if certs:
        sections.append(f"CERTIFICATIONS\n{chr(10).join(certs)}")
    if quality == "high" and random.random() > 0.4:
        project = random.choice(skills)
        sections.append(
            f"PROJECTS\n• {project} Dashboard: Built end-to-end pipeline using {project} "
            f"and deployed on AWS. Achieved 95% accuracy."
        )

    return "\n\n".join(sections)


def generate_resumes(n: int = 500) -> pd.DataFrame:
    records = []
    categories = list(CATEGORIES.keys())
    for i in range(n):
        category = random.choice(categories)
        experience = random.randint(0, 10)
        quality = random.choices(["high", "medium", "low"], weights=[0.4, 0.4, 0.2])[0]
        text = generate_resume_text(category, experience, quality)

        # Heuristic ATS score based on quality + experience
        base = {"high": 75, "medium": 55, "low": 35}[quality]
        ats_score = min(100, base + experience * 2 + random.randint(-10, 10))

        records.append({
            "resume_id": f"R{i+1:04d}",
            "category": category,
            "experience_years": experience,
            "quality": quality,
            "resume_text": text,
            "ats_score": ats_score,
            "keyword_score": min(100, ats_score + random.randint(-5, 5)),
            "formatting_score": min(100, base + random.randint(-8, 8)),
            "completeness_score": min(100, base + random.randint(-5, 15)),
            "readability_score": min(100, base + random.randint(-5, 10)),
            "action_verb_score": min(100, base + random.randint(-10, 10)),
        })
    return pd.DataFrame(records)


def generate_job_descriptions() -> pd.DataFrame:
    jds = []
    for category, data in CATEGORIES.items():
        for role in data["roles"][:2]:
            required_skills = random.sample(data["skills"], 10)
            preferred_skills = random.sample(data["skills"], 5)
            jd_text = (
                f"We are hiring a {role} to join our team.\n\n"
                f"Required Skills: {', '.join(required_skills)}\n"
                f"Preferred Skills: {', '.join(preferred_skills)}\n"
                f"Experience: {random.randint(1,5)}+ years in {category}\n"
                f"Education: B.E./B.Tech or equivalent degree\n"
                f"Responsibilities:\n"
                f"• Design and implement {required_skills[0]} solutions\n"
                f"• Collaborate with cross-functional teams\n"
                f"• Optimize performance of {required_skills[1]} pipelines\n"
                f"• Write unit tests and documentation"
            )
            jds.append({
                "jd_id": f"JD{len(jds)+1:03d}",
                "role": role,
                "category": category,
                "required_skills": json.dumps(required_skills),
                "preferred_skills": json.dumps(preferred_skills),
                "jd_text": jd_text
            })
    return pd.DataFrame(jds)


def generate_company_profiles() -> list:
    profiles = []
    company_data = {
        "Google": {
            "focus": ["Algorithms", "System Design", "Python", "C++", "Machine Learning"],
            "preferred": ["Open Source contributions", "Research publications", "LeetCode"],
            "tip": "Google values problem-solving. Include competitive programming and research."
        },
        "Amazon": {
            "focus": ["Leadership Principles", "Java", "AWS", "Distributed Systems", "SQL"],
            "preferred": ["Microservices", "Scalability", "Customer obsession examples"],
            "tip": "Quantify your impact. Amazon loves metrics and measurable outcomes."
        },
        "Microsoft": {
            "focus": ["Azure", "C#", ".NET", "TypeScript", "Cloud Architecture"],
            "preferred": ["Growth mindset examples", "Collaboration", "Agile"],
            "tip": "Microsoft values collaboration and growth mindset. Show teamwork."
        },
        "TCS": {
            "focus": ["Java", "SQL", "Communication", "Agile", "SDLC"],
            "preferred": ["Certifications", "Training programs", "Domain knowledge"],
            "tip": "TCS values certifications and domain expertise. Add any TCS-relevant certs."
        },
        "Infosys": {
            "focus": ["Java", "Python", "Testing", "Communication", "Problem Solving"],
            "preferred": ["Client-facing experience", "Infosys certifications", "Agile"],
            "tip": "Infosys looks for communication skills and adaptability."
        },
        "Wipro": {
            "focus": ["Software Development", "Testing", "SQL", "Communication", "Teamwork"],
            "preferred": ["Cloud skills", "Automation", "Certifications"],
            "tip": "Wipro values automation and cloud skills in recent hires."
        },
        "Meta": {
            "focus": ["React", "Python", "C++", "Distributed Systems", "Machine Learning"],
            "preferred": ["Open Source", "Scale experience", "Product mindset"],
            "tip": "Meta values scale and impact. Show projects that handled large data or users."
        },
        "Accenture": {
            "focus": ["Consulting", "Agile", "Cloud", "Communication", "Digital Transformation"],
            "preferred": ["Client management", "Certifications", "Presentation skills"],
            "tip": "Accenture values consulting experience and client-facing skills."
        },
        "Flipkart": {
            "focus": ["Java", "Python", "Distributed Systems", "MySQL", "Kafka"],
            "preferred": ["E-commerce domain", "High traffic systems", "Microservices"],
            "tip": "Flipkart values high-scale system design and e-commerce domain knowledge."
        },
        "Swiggy": {
            "focus": ["Python", "Node.js", "React", "Microservices", "AWS"],
            "preferred": ["Startup mindset", "Fast delivery", "Mobile development"],
            "tip": "Swiggy values fast iteration and mobile-first thinking."
        }
    }
    for name, data in company_data.items():
        profiles.append({
            "company": name,
            "focus_skills": data["focus"],
            "preferred_extras": data["preferred"],
            "tip": data["tip"]
        })
    return profiles


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__))
    os.makedirs(out_dir, exist_ok=True)

    print("Generating resumes dataset (500 records)...")
    df_resumes = generate_resumes(500)
    df_resumes.to_csv(os.path.join(out_dir, "resumes.csv"), index=False)
    print(f"  ✓ resumes.csv — {len(df_resumes)} rows")

    print("Generating job descriptions dataset...")
    df_jds = generate_job_descriptions()
    df_jds.to_csv(os.path.join(out_dir, "job_descriptions.csv"), index=False)
    print(f"  ✓ job_descriptions.csv — {len(df_jds)} rows")

    print("Generating company profiles...")
    company_profiles = generate_company_profiles()
    with open(os.path.join(out_dir, "company_profiles.json"), "w") as f:
        json.dump(company_profiles, f, indent=2)
    print(f"  ✓ company_profiles.json — {len(company_profiles)} companies")

    print("\nDataset generation complete!")
    print(f"  Category distribution:\n{df_resumes['category'].value_counts().to_string()}")
    print(f"\n  ATS Score stats:\n{df_resumes['ats_score'].describe().to_string()}")
