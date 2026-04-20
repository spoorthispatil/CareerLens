"""
models/company_recommender.py
────────────────────────────────────────────────────────────────────────────────
Company-Specific Resume Recommendation Engine
────────────────────────────────────────────────────────────────────────────────
Generates tailored improvement tips for 10+ companies based on:
  - Company's known preferred skills
  - Skills present/absent in the resume
  - Company-specific hiring culture tips
────────────────────────────────────────────────────────────────────────────────
"""

import json
import os
from dataclasses import dataclass, field


@dataclass
class CompanyTip:
    company: str
    tip_type: str       # "skill_gap" | "culture" | "format" | "certification"
    message: str
    priority: str       # "high" | "medium" | "low"

    def to_dict(self):
        return {
            "company": self.company,
            "tip_type": self.tip_type,
            "message": self.message,
            "priority": self.priority,
        }


COMPANY_PROFILES = {
    "Google": {
        "focus_skills": ["algorithms", "system design", "python", "c++", "distributed systems",
                         "machine learning", "open source"],
        "certifications": [],
        "culture_tips": [
            "Google values deep problem-solving ability. Include competitive programming (LeetCode, Codeforces) profiles.",
            "Highlight any research publications, papers, or conference presentations.",
            "Show large-scale system design experience — mention data volumes and throughput metrics.",
            "Open source contributions are highly valued. Add GitHub profile with notable contributions.",
        ],
        "format_tips": [
            "Keep resume to 1 page if < 5 years experience.",
            "Lead every bullet with a quantified impact: '...reduced latency by 30% at 1M req/sec'.",
        ]
    },
    "Amazon": {
        "focus_skills": ["java", "aws", "distributed systems", "microservices", "sql",
                         "python", "leadership", "scalability"],
        "certifications": ["AWS Certified Solutions Architect", "AWS Developer"],
        "culture_tips": [
            "Amazon's 16 Leadership Principles are core. Frame experience bullets around them (ownership, bias for action, customer obsession).",
            "Quantify customer impact wherever possible — Amazon is obsessively customer-centric.",
            "Show experience with high-availability and fault-tolerant systems.",
            "Add AWS certifications — they are strongly weighted at Amazon.",
        ],
        "format_tips": [
            "Use STAR format (Situation, Task, Action, Result) for experience bullets.",
            "Include metrics: team size managed, cost savings, latency improvements, uptime achieved.",
        ]
    },
    "Microsoft": {
        "focus_skills": ["azure", "c#", ".net", "typescript", "cloud architecture",
                         "python", "agile", "collaboration"],
        "certifications": ["Microsoft Azure Fundamentals", "Microsoft Certified"],
        "culture_tips": [
            "Microsoft values growth mindset and collaboration. Include cross-team or cross-functional projects.",
            "Azure experience is highly preferred. Mention specific Azure services used.",
            "Show experience with enterprise-scale products or B2B software.",
        ],
        "format_tips": [
            "Highlight any open source contributions to Microsoft projects (VS Code, TypeScript, etc.).",
            "Include certifications from Microsoft Learn — AZ-900, AZ-204 etc.",
        ]
    },
    "Meta": {
        "focus_skills": ["react", "python", "c++", "distributed systems", "machine learning",
                         "large scale systems", "php", "data pipelines"],
        "certifications": [],
        "culture_tips": [
            "Meta values impact at scale. Show projects or systems that handle millions of users or events.",
            "Highlight product thinking — frame engineering work in terms of user and business impact.",
            "Open source experience is valued, especially in React, PyTorch, or related ecosystems.",
        ],
        "format_tips": [
            "Move fast and show it — mention iteration speed and rapid deployment examples.",
            "Include A/B testing or experimentation experience if applicable.",
        ]
    },
    "TCS": {
        "focus_skills": ["java", "sql", "communication", "agile", "sdlc",
                         "testing", "python", "client management"],
        "certifications": ["TCS iON", "AWS", "Oracle", "Agile Scrum Master"],
        "culture_tips": [
            "TCS heavily weights certifications. Add any relevant industry certifications.",
            "Highlight domain knowledge (BFSI, healthcare, retail) as TCS is domain-focused.",
            "Show client-facing communication and delivery experience.",
            "Mention training programs, workshops, or internal TCS tools if applicable.",
        ],
        "format_tips": [
            "Include CGPA / percentage if above 7.0/70% — TCS has academic cutoffs.",
            "List relevant internships, projects, and hackathon participation.",
        ]
    },
    "Infosys": {
        "focus_skills": ["java", "python", "testing", "communication", "problem solving",
                         "agile", "rest api", "microservices"],
        "certifications": ["Infosys Springboard", "AWS", "Salesforce", "Microsoft"],
        "culture_tips": [
            "Infosys values adaptability and continuous learning. Show upskilling via online courses.",
            "Highlight client-interaction and offshore delivery model experience.",
            "Communication skills are explicitly evaluated — mention presentations or client demos.",
        ],
        "format_tips": [
            "Ensure CGPA is above 6.5 and list it clearly in the education section.",
            "Add any Infosys-related certifications from Infosys Springboard platform.",
        ]
    },
    "Wipro": {
        "focus_skills": ["software development", "testing", "sql", "communication",
                         "cloud", "automation", "python", "java"],
        "certifications": ["AWS", "Azure", "Google Cloud", "Selenium"],
        "culture_tips": [
            "Wipro increasingly values cloud and automation skills in new hires.",
            "Mention any experience with RPA tools (UiPath, Automation Anywhere).",
            "Show diversity of project experience across multiple domains.",
        ],
        "format_tips": [
            "Include extracurricular activities and leadership roles — Wipro evaluates holistic profiles.",
            "Add CGPA/percentage (cutoff: 6.0).",
        ]
    },
    "Flipkart": {
        "focus_skills": ["java", "python", "distributed systems", "mysql", "kafka",
                         "microservices", "high traffic", "data structures"],
        "certifications": [],
        "culture_tips": [
            "Flipkart values e-commerce and high-scale system experience.",
            "Show experience with high-traffic architectures — millions of requests per day.",
            "Highlight data structures and algorithms proficiency for technical interviews.",
        ],
        "format_tips": [
            "Quantify system scale: '...handled 2M daily active users' or '...processed 50K orders/sec'.",
            "Include any experience with recommendation systems, search, or pricing algorithms.",
        ]
    },
    "Swiggy": {
        "focus_skills": ["python", "node.js", "react", "microservices", "aws",
                         "mobile development", "postgresql", "redis"],
        "certifications": [],
        "culture_tips": [
            "Swiggy values startup agility. Show fast iteration and shipping velocity.",
            "Highlight mobile-first development experience (React Native, Flutter, or Android/iOS).",
            "Logistics or real-time systems experience is a differentiator.",
        ],
        "format_tips": [
            "Show side projects and personal apps — Swiggy values self-motivated builders.",
            "Mention any experience with location-based services or real-time tracking.",
        ]
    },
    "Accenture": {
        "focus_skills": ["consulting", "agile", "cloud", "communication",
                         "digital transformation", "power bi", "salesforce"],
        "certifications": ["PMP", "Scrum Master", "Salesforce", "AWS", "SAP"],
        "culture_tips": [
            "Accenture is a consulting firm — emphasize client-facing work, presentations, and stakeholder management.",
            "Mention digital transformation projects: cloud migration, ERP, CRM implementations.",
            "Industry domain knowledge (BFSI, healthcare, retail) is highly valued.",
        ],
        "format_tips": [
            "Use business-outcome language: 'saved $X', 'increased efficiency by Y%', 'reduced costs by Z%'.",
            "Add consulting-relevant certifications: PMP, Scrum Master, Salesforce.",
        ]
    },
    "Deloitte": {
        "focus_skills": ["data analytics", "power bi", "tableau", "python",
                         "consulting", "communication", "excel", "sql"],
        "certifications": ["CPA", "CFA", "AWS", "Tableau", "Power BI"],
        "culture_tips": [
            "Deloitte values analytical thinking and data storytelling. Show dashboards and reports.",
            "Consulting experience with measurable client ROI is strongly preferred.",
            "Academic excellence matters — include CGPA if above 8.0.",
        ],
        "format_tips": [
            "Include relevant coursework in finance, accounting, or risk management.",
            "Show experience with enterprise BI tools: SAP, Oracle, Tableau, Power BI.",
        ]
    },
}


class CompanyRecommender:
    """
    Generates company-specific resume improvement tips.
    """

    def __init__(self, profiles_path: str = None):
        self.profiles = COMPANY_PROFILES
        if profiles_path and os.path.exists(profiles_path):
            with open(profiles_path) as f:
                external = json.load(f)
                for p in external:
                    name = p.get("company", "")
                    if name:
                        self.profiles[name] = p

    def list_companies(self) -> list:
        return sorted(self.profiles.keys())

    def get_tips(self, company: str, resume_text: str) -> list:
        """Generate company-specific tips for a given resume."""
        company_data = self.profiles.get(company)
        if not company_data:
            return [CompanyTip(
                company=company,
                tip_type="culture",
                message=f"No specific profile available for {company}. Ensure your resume matches the JD keywords.",
                priority="medium"
            ).to_dict()]

        tips = []
        resume_lower = resume_text.lower()

        # ── Skill gap tips ────────────────────────────────────────────────────
        focus_skills = company_data.get("focus_skills", [])
        missing_skills = [s for s in focus_skills if s.lower() not in resume_lower]

        if missing_skills:
            top_missing = missing_skills[:4]
            tips.append(CompanyTip(
                company=company,
                tip_type="skill_gap",
                message=(
                    f"{company} specifically looks for: {', '.join(top_missing)}. "
                    f"Add these to your Skills section if applicable."
                ),
                priority="high" if len(missing_skills) > 3 else "medium"
            ).to_dict())

        # ── Certification tips ────────────────────────────────────────────────
        certifications = company_data.get("certifications", [])
        missing_certs = [c for c in certifications if c.lower() not in resume_lower]
        if missing_certs:
            tips.append(CompanyTip(
                company=company,
                tip_type="certification",
                message=(
                    f"{company} values these certifications: {', '.join(missing_certs[:2])}. "
                    f"Consider earning them to boost your profile."
                ),
                priority="medium"
            ).to_dict())

        # ── Culture tips ──────────────────────────────────────────────────────
        for tip_text in company_data.get("culture_tips", [])[:2]:
            tips.append(CompanyTip(
                company=company,
                tip_type="culture",
                message=tip_text,
                priority="medium"
            ).to_dict())

        # ── Format tips ───────────────────────────────────────────────────────
        for tip_text in company_data.get("format_tips", [])[:2]:
            tips.append(CompanyTip(
                company=company,
                tip_type="format",
                message=tip_text,
                priority="low"
            ).to_dict())

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        tips.sort(key=lambda x: priority_order.get(x["priority"], 3))

        return tips[:6]

    def compare_companies(self, resume_text: str, companies: list) -> dict:
        """
        Score resume fit for multiple companies simultaneously.
        Returns a ranked list of company-fit scores.
        """
        resume_lower = resume_text.lower()
        results = {}

        for company in companies:
            data = self.profiles.get(company, {})
            focus_skills = data.get("focus_skills", [])
            if not focus_skills:
                continue

            matched = sum(1 for s in focus_skills if s.lower() in resume_lower)
            fit_score = round(matched / len(focus_skills) * 100, 1)
            results[company] = {
                "fit_score": fit_score,
                "matched_skills": [s for s in focus_skills if s.lower() in resume_lower],
                "missing_skills": [s for s in focus_skills if s.lower() not in resume_lower][:4],
            }

        return dict(sorted(results.items(), key=lambda x: -x[1]["fit_score"]))


if __name__ == "__main__":
    recommender = CompanyRecommender()

    resume = """
    SKILLS: Python, Machine Learning, TensorFlow, SQL, Docker, AWS, REST API
    EXPERIENCE: Developed ML models. Built REST APIs. Deployed on AWS.
    EDUCATION: B.Tech Computer Science, 2022
    """

    print("Available companies:", recommender.list_companies())
    print()

    for company in ["Google", "Amazon", "TCS"]:
        tips = recommender.get_tips(company, resume)
        print(f"── {company} Tips ──────────────────────────────")
        for tip in tips:
            print(f"  [{tip['priority'].upper()}] [{tip['tip_type']}] {tip['message'][:100]}...")
        print()

    print("── Company Fit Comparison ──────────────────────────────")
    fit = recommender.compare_companies(resume, ["Google", "Amazon", "TCS", "Flipkart"])
    for company, data in fit.items():
        print(f"  {company}: {data['fit_score']}% fit | Missing: {data['missing_skills'][:3]}")
