import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, callback_context, State,dash_table, ALL
import dash_bootstrap_components as dbc
from datetime import datetime
from collections import Counter
import re
import unicodedata
import os
import json
import signal
import psutil
from IPython.display import display, IFrame
import ast
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import logging

from pathlib import Path
base_dir = Path.cwd()
json_path = base_dir / "final_grouped_skills_v6.json"
# Functions for map
# # Function to normalize text
# def normalize(text):
#     if isinstance(text, str):  # Only normalize strings
#         return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
#     return text  # Return non-string values as is
def normalize(s):
    if isinstance(s, str):
        s = ''.join(c for c in unicodedata.normalize('NFD', s)
                    if unicodedata.category(c) != 'Mn')
        return s.replace(" ", "").upper()
    return s

def normalize(s):
    if isinstance(s, str):
        s = ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )
        return s.replace(" ", "").upper()
    return s

def filter_geojson_features(geojson, match_values, key):
    normalized = set(map(normalize, match_values))
    geojson["features"] = [f for f in geojson["features"] if normalize(f["properties"].get(key, '')) in normalized]
    return geojson

def compute_center_from_geojson(features):
    lats, lons = [], []

    for feature in features:
        try:
            geometry = feature.get("geometry", {})
            geom_type = geometry.get("type")
            coordinates = geometry.get("coordinates", [])

            def extract_coords(coords):
                for coord in coords:
                    if isinstance(coord[0], list):
                        yield from extract_coords(coord)
                    else:
                        yield coord

            coords = list(extract_coords(coordinates))

            for lon, lat in coords:
                lats.append(lat)
                lons.append(lon)
        except Exception as e:
            print("⚠️ Error parsing feature geometry:", e)

    if lats and lons:
        center = {"lat": sum(lats) / len(lats), "lon": sum(lons) / len(lons)}
        print(f"✅ Center computed from features: {center}")
        return center
    else:
        print("❌ No coordinates found, using fallback center")
        return {"lat": 20, "lon": 0}

def compute_bounding_box(features):
    lats, lons = [], []
    for feature in features:
        geometry = feature.get("geometry", {})
        coords = geometry.get("coordinates", [])

        def extract_coords(coords):
            for coord in coords:
                if isinstance(coord[0], list):
                    yield from extract_coords(coord)
                else:
                    yield coord

        for lon, lat in extract_coords(coords):
            lats.append(lat)
            lons.append(lon)

    if lats and lons:
        return {"min_lat": min(lats), "max_lat": max(lats), "min_lon": min(lons), "max_lon": max(lons)}
    return None

def compute_zoom(bbox):
    if not bbox:
        return 3.5

    lat_diff = bbox['max_lat'] - bbox['min_lat']
    lon_diff = bbox['max_lon'] - bbox['min_lon']
    max_diff = max(lat_diff, lon_diff)

    # Adjusted heuristic: more zoomed-out
    if max_diff > 60:
        return 2
    elif max_diff > 30:
        return 2.7
    elif max_diff > 10:
        return 3.2
    elif max_diff > 5:
        return 3.7
    elif max_diff > 2:
        return 4.2
    else:
        return 5.2  
        
zoom_levels = {
    "Canada": 2.2,
    "USA": 2.7,
    "Australia": 3.5,
    "United Kingdom": 4.0,
    "Israel": 10.0,
    "Mexico": 3.5,
    "Morocco": 4.0,
    "Vietnam": 5.0,
    "Indonesia": 4.5,
    "Singapore": 10,
    "China": 5.0
}




# Function to extract locations from a string
def extract_locations(location_string):
    """
    Extract individual locations from a string like "['location1', 'location2', ...]"
    """
    if not isinstance(location_string, str):
        return []
    
    # Use regex to find all strings enclosed in single quotes
    matches = re.findall(r"'([^']*)'", location_string)
    return matches

# Function to clean and standardize degree text
def clean_degree(degree_text):
    """Clean and standardize degree text"""
    if pd.isna(degree_text) or degree_text == 'None' or not degree_text:
        return '1_No Requirement'
    
    degree_lower = str(degree_text).lower()
    
    # Generic student without specific degree mention - group with No Requirement
    if 'student' in degree_lower and not any(term in degree_lower for term in ['bachelor', 'master', 'phd', 'doctor', 'high school']):
        return '2_Student'
    
    # High School level
    if any(term in degree_lower for term in ['high school', 'diploma']):
        return '3_High School'
    
    # Enrolled in Bachelor's or Associate's
    if any(term in degree_lower for term in ['enrolled', 'pursuing', 'student', 'studying', 'working toward']) and \
       any(term in degree_lower for term in ['bachelor', 'undergraduate', 'college', 'associate']):
        return '4_Pursuing Bachelor\'s'
    
    # Bachelor's
    if any(term in degree_lower for term in ['bachelor', 'undergraduate', 'bs ', 'b.s', 'ba ', 'b.a']):
        return '5_Bachelor\'s'
    
    # Enrolled in Master's
    if any(term in degree_lower for term in ['enrolled', 'pursuing', 'student', 'studying', 'working toward']) and \
       any(term in degree_lower for term in ['master', 'mba', 'graduate']):
        return '6_Pursuing Master\'s'
    
    # Master's
    if any(term in degree_lower for term in ['master', 'mba', 'ms ', 'm.s', 'graduate degree']):
        if 'bachelor' in degree_lower:
            return '6_Bachelor\'s/Master\'s'
        return '7_Master\'s'
    
    # Enrolled in PhD
    if any(term in degree_lower for term in ['enrolled', 'pursuing', 'student', 'studying', 'working toward']) and \
       any(term in degree_lower for term in ['phd', 'doctor']):
        return '8_Pursuing PhD'
    
    # PhD level
    if any(term in degree_lower for term in ['phd', 'doctor', 'doctorate', 'pharmd']):
        return '9_PhD'
    
    # Specific field requirements
    if any(term in degree_lower for term in ['engineer', 'comput', 'program', 'scienc', 'technical']):
        return '10_Field Specific'
    
    # Default category
    return '11_Other'

# Function to extract skills from text
def extract_skills(skills_text):
    """Extract individual skills from technical skills text"""
    if pd.isna(skills_text) or skills_text == 'None':
        return []
    
    # Split skills (usually comma-separated)
    skills = [s.strip() for s in skills_text.split(',')]
    
    # Further split (sometimes semicolon or other separators)
    all_skills = []
    for skill in skills:
        all_skills.extend([s.strip() for s in skill.split(';')])
    
    return all_skills

# Function to categorize technical skills
def categorize_tech_skills(skills_text):
    """Categorize technical skills"""
    if pd.isna(skills_text) or skills_text == 'None':
        return 'None'
    
    programming_keywords = ['Python', 'Java', 'SQL', 'C++', 'C#', 'R', 'JavaScript', 'Scala', 'Bash', 'Prolog']
    office_keywords = ['Excel', 'Word', 'PowerPoint', 'Office', 'Outlook', 'Sharepoint']
    design_keywords = ['AutoCAD', 'Revit', 'editing', 'Newsroom', 'Swift']
    
    skills_lower = skills_text.lower()
    
    if any(keyword.lower() in skills_lower for keyword in programming_keywords):
        return 'Programming'
    elif any(keyword.lower() in skills_lower for keyword in office_keywords):
        return 'Office'
    elif any(keyword.lower() in skills_lower for keyword in design_keywords):
        return 'Design'
    else:
        return 'Other'

# Function to extract soft skills keywords
softskill_dict = {
    "Communication": ["communication", "verbal", "written", "interpersonal", "listening", "presentation"],
    "Teamwork":      ["teamwork", "team", "collaborate", "support", "cooperate", "cross-functional"],
    "Problem Solving":["problem-solving", "analytical", "critical thinking", "root cause", "troubleshoot"],
    "Adaptability":   ["adaptability", "flexible", "learning", "open-minded", "resilience"],
    "Leadership":     ["leadership", "lead", "manage", "delegate", "coach", "mentor"],
    "Time Management":["time management", "prioritize", "deadline", "organize", "plan"],
    "Initiative":     ["initiative", "self-motivated", "proactive", "ownership", "drive"],
    "Emotional Intelligence":["emotional intelligence", "empathy", "self-awareness", "relationship"],
    "Creativity":     ["creativity", "innovation", "ideation", "design thinking", "brainstorm"],
    "Professionalism":["work ethic", "reliable", "professional", "integrity", "accountability"]
}

def extract_soft_skills1(text):
    """Extract soft skill *themes* from description"""
    if pd.isna(text) or text.strip().lower() == 'none':
        return []
    
    found = []
    t = text.lower()
    for theme, kws in softskill_dict.items():
        if any(kw in t for kw in kws):
            found.append(theme)
    return found

# Function to extract experience years from text
def extract_experience_years(exp_text):
    """Extract minimum and maximum years from experience requirement text"""
    if pd.isna(exp_text) or exp_text == 'None':
        return (0, 0)
    
    # Look for patterns like "0 - 2 years"
    matches = re.findall(r'(\d+)\s*-\s*(\d+)', str(exp_text))
    if matches:
        return (int(matches[0][0]), int(matches[0][1]))
    
    # Look for single year, like "1 year"
    single_match = re.findall(r'(\d+)\s*year', str(exp_text))
    if single_match:
        return (int(single_match[0]), int(single_match[0]))
    
    return (0, 0)

# Function to format degree name for display
def display_degree_name(coded_degree):
    """Extract display name from coded degree"""
    return coded_degree.split('_', 1)[1] if '_' in coded_degree else coded_degree

# Function to load qualification data
def load_qualification_data():
    """Load and clean qualification requirements data"""
    # Check if file exists
    file_path = 'dash_data/Quali_Finresults.csv'
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} does not exist!")
        return None
    
    try:
        # Load data with cp1252 encoding
        quali_data = pd.read_csv(file_path, encoding='cp1252')
        
        # Clean degree data
        quali_data['Degree_Clean'] = quali_data['Degree'].apply(clean_degree)
        
        # Process technical skills
        quali_data['Technical_Skills_List'] = quali_data['Technical Skills'].apply(extract_skills)
        
        # Extract main skill categories
        quali_data['Tech_Skill_Category'] = quali_data['Technical Skills'].apply(categorize_tech_skills)
        
        # Process soft skills
        quali_data['Soft_Skills_Keywords'] = quali_data['Soft Skills'].apply(extract_soft_skills1)
        
        # Process experience years
        quali_data['Experience_Years_Min'], quali_data['Experience_Years_Max'] = zip(*quali_data['Experience Years Requirement'].apply(extract_experience_years))
        
        # Calculate if job has experience requirements
        quali_data['Has_Experience_Req'] = quali_data['Experience_Years_Max'] > 0
        
        print(f"Successfully loaded qualification data with {len(quali_data)} rows")

        all_tech_skills = []
        for skills_list in quali_data['Technical_Skills_List']:
            all_tech_skills.extend(skills_list)
        tech_skill_counter = Counter(all_tech_skills)
        quali_data.attrs['top_tech_skills'] = [skill for skill, count in tech_skill_counter.most_common(20) if skill]

        all_soft_keywords = []
        for keywords_list in quali_data['Soft_Skills_Keywords']:
            all_soft_keywords.extend(keywords_list)
        soft_skill_counter = Counter(all_soft_keywords)
        quali_data.attrs['top_soft_skills'] = [skill for skill, count in soft_skill_counter.most_common(15) if skill]

        language_values = quali_data['Language'].dropna()
        language_values = language_values[language_values != 'None']
        if len(language_values) > 0:
            language_counter = Counter(language_values)
            quali_data.attrs['common_languages'] = [lang for lang, count in language_counter.items() if count > 5]
        else:
            quali_data.attrs['common_languages'] = []
            
        return quali_data
    except Exception as e:
        print(f"Error loading qualification data: {e}")
        return None

# Function to load job market data
def load_job_market_data():
    """Load all the necessary data files for the job market dashboard."""
    
    # Load the classified tags data
    classified_tags = None
    if os.path.exists('dash_data/classified_tags.csv'):
        classified_tags = pd.read_csv('dash_data/classified_tags.csv')
        print(f"Successfully loaded classified tags data with {len(classified_tags)} rows")
    else:
        print("Warning: classified_tags.csv not found. Skills tab may not function properly.")
    
    # Load location data
    location_data = {}
    location_files = {
        'all_locations': 'dash_data/all_locations.csv',
        'location_counts': 'dash_data/location_counts.csv',
        'top_locations': 'dash_data/top_30_locations.csv'
    }
    
    for key, filepath in location_files.items():
        try:
            location_data[key] = pd.read_csv(filepath)
            print(f"Successfully loaded {key} data with {len(location_data[key])} rows")
        except FileNotFoundError:
            print(f"Warning: {filepath} not found. Some location features may be unavailable.")
            location_data[key] = None
    
    # Try to load job data file for job count information and time series analysis
    job_data = None
    try:
        job_data = pd.read_csv('dash_data/job_data.csv')
        # Convert date column to datetime if it exists
        if 'postedDate' in job_data.columns:
            job_data['postedDate'] = pd.to_datetime(job_data['postedDate'], errors='coerce')
        print(f"Successfully loaded job data with {len(job_data)} rows")
    except FileNotFoundError:
        print("Warning: job_data.csv not found. Using location data for job counts.")
    
    return classified_tags, location_data, job_data

# NLP model
# === Load model & skill dictionary (只需执行一次即可) ===
notebook_dir = Path(os.getcwd())
json_path = base_dir / "final_grouped_skills_v6.json"

with open(json_path, "r", encoding="utf-8") as f:
    skill_dict = json.load(f)["technical_skills"]

model_name = "jjzha/jobbert_skill_extraction"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
skill_pipeline = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# === Utility: clean text ===
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s\.\+\-#]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# === Degree Pattern Matching ===
degree_pattern = r"\b(Bachelor(?:'s)?|Master(?:'s)?|Ph\.?D\.?|M\.Sc\.|B\.Sc\.|MBA|JD|MD)[^.,;\n]{0,50}"

def extract_degree(text):
    if isinstance(text, list):
        text = " ".join(text)
    if not isinstance(text, str):
        return []
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8", "ignore")
    matches = re.findall(degree_pattern, text, flags=re.IGNORECASE)
    results = []
    for match in matches:
        pattern = rf"{match}[^.,;\n]{{0,50}}"
        found = re.findall(pattern, text, flags=re.IGNORECASE)
        results.extend(found)
    return [r.strip() for r in results]

# === Technical Skills Extraction ===
def extract_skills_technical(text, skill_list):
    text_cleaned = clean_text(text)
    extracted = []
    for skill in skill_list:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_cleaned):
            extracted.append(skill)
    return extracted

# === Soft Skills Extraction via Hugging Face model ===
def extract_soft_skills(text):
    try:
        if isinstance(text, list):
            text = " ".join(text)
        elif not isinstance(text, str) or not text.strip():
            return []

        #results = skill_pipeline(text, clean_up_tokenization_spaces=True)
        skills = []
        current_skill = ""
        for entity in results:
            if entity["entity_group"] == "B":
                if current_skill:
                    skills.append(current_skill.strip())
                current_skill = entity["word"]
            elif entity["entity_group"] == "I":
                current_skill += " " + entity["word"]
            else:
                if current_skill:
                    skills.append(current_skill.strip())
                    current_skill = ""
        if current_skill:
            skills.append(current_skill.strip())

        return list(set(skills))  # Remove duplicates
    except Exception as e:
        print(f"Error: {e}")
        return []

# === Unified NLP Extraction Function ===
def extract_all_qualifications(text):
    """
    Given a text or list of text items, extract:
    - degree (学历)
    - technical skills (技术技能)
    - soft skills (软技能 from model)
    
    Returns a dict:
    {
        "degree": [...],
        "technical_skills": [...],
        "soft_skills": [...]
    }
    """
    if isinstance(text, list):
        text = " ".join(text)
    if not isinstance(text, str) or not text.strip():
        return {"degree": [], "technical_skills": [], "soft_skills": []}
    
    return {
        "degree": extract_degree(text),
        "technical_skills": extract_skills_technical(text, skill_dict),
        "soft_skills": extract_soft_skills(text)
    }

# ========================= JOB MATCHER HELPER FUNCTIONS =========================
def normalize_skill_name(skill):
    """Normalize skill name for matching"""
    if not skill:
        return ""
    # Remove extra spaces
    skill = ' '.join(skill.strip().split())
    # Handle common variations
    skill_mapping = {
        'MS Excel': 'Excel',
        'Microsoft Excel': 'Excel',
        'MS Word': 'Word',
        'Microsoft Word': 'Word',
        'MS PowerPoint': 'PowerPoint',
        'Microsoft PowerPoint': 'PowerPoint',
        'MS Office': 'Microsoft Office',
        'JS': 'JavaScript',
        'PowerBI': 'Power BI',
        'Node': 'Node.js',
        'ReactJS': 'React',
        'React.js': 'React'
    }
    return skill_mapping.get(skill, skill)

def create_job_card(match, job_data, badge_color):
    """Create a job card for display"""
    # Get job details
    if job_data is not None and 'roleTitle' in job_data.columns:
        job_info = job_data.iloc[match['index']]
        title = str(job_info.get('roleTitle', 'N/A'))[:60]
        company = str(job_info.get('companyName', 'N/A'))[:40]
        location = str(job_info.get('locations', 'N/A'))
        apply_link = job_info.get('applyLink', None)
        
        # Process location
        if location.startswith('[') and location.endswith(']'):
            try:
                locations = eval(location)
                location = locations[0] if locations else 'N/A'
            except:
                location = location[:40]
        else:
            location = location[:40]
    else:
        title = f"Position #{match['index'] + 1}"
        company = "Company Info"
        location = "Location"
        apply_link = None
    
    # Match percentage
    match_percentage = match['match_ratio'] * 100
    
    # Skills breakdown
    matched_count = len(match['matched_skills'])
    unmatched_count = len(match['unmatched_skills'])
    job_skills = match['job_skills']
    missing_skills = [skill for skill in job_skills if skill not in match['matched_skills']]
    
    soft_skills = match.get('soft_skills', [])
    
    # Apply button
    if apply_link and isinstance(apply_link, str) and apply_link.strip() and apply_link != 'nan':
        view_button = dbc.Button("Apply", href=apply_link, target="_blank", size="sm", 
                                color="primary", external_link=True)
    else:
        view_button = dbc.Button("No Link", size="sm", color="secondary", disabled=True)
    
    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H5(title, style={'marginBottom': '5px', 'fontWeight': 'bold'}),
                    html.P(f"{company} • 📍 {location}", className="text-muted", style={'fontSize': '15px'})
                ], md=9),
                dbc.Col([
                    dbc.Badge(
                        f"{match_percentage:.0f}%", 
                        color=badge_color, 
                        style={'fontSize': '20px', 'padding': '10px'}
                    )
                ], md=2, className="text-center"),
                dbc.Col([
                    view_button
                ], md=1, className="text-end")
            ]),
            html.Hr(style={'margin': '10px 0'}),
            html.P([
                html.Span("📚 Degree: ", style={'fontWeight': 'bold', 'marginRight': '5px'}),
                html.Span(f"{display_degree_name(match['degree'])}", style={'marginRight': '20px'}),
                
                html.Span("💼 Experience: ", style={'fontWeight': 'bold', 'marginRight': '5px'}),
                html.Span(f"{match['experience_min']}-{match['experience_max']} years" if match['experience_max'] > 0 else "No experience required", style={'marginRight': '20px'}),
                
                html.Span("🌐 Language: ", style={'fontWeight': 'bold', 'marginRight': '5px'}),
                html.Span(f"{match['language']}" if match.get('language', 'None') != 'None' else "Not specified")
            ], className="text-muted", style={'fontSize': '14px', 'lineHeight': '1.6'}),
            html.P([
                html.Span("Skills: ", style={'fontWeight': 'bold'}),
                html.Span(f"{matched_count} matched", style={'color': '#27ae60', 'marginRight': '10px'}),
                html.Span("•", style={'margin': '0 10px'}),
                html.Span(f"{len(missing_skills)} missing", style={'color': '#f39c12'})
            ], style={'fontSize': '15px', 'marginBottom': '8px'}),
            
            html.Div([
                html.Span("💡 Soft Skills: ", style={'fontWeight': 'bold', 'color': '#9b59b6'}),
                html.Span(', '.join(soft_skills[:5]) if soft_skills else "None specified", 
                         style={'fontSize': '14px'}),
                html.Span(f" (+{len(soft_skills)-5} more)" if len(soft_skills) > 5 else "", 
                         style={'fontSize': '14px', 'fontStyle': 'italic'})
            ], style={'marginBottom': '8px'}) if soft_skills else html.Div([
                html.Span("💡 Soft Skills: ", style={'fontWeight': 'bold', 'color': '#9b59b6'}),
                html.Span("None specified", style={'fontSize': '14px', 'color': '#95a5a6'})
            ], style={'marginBottom': '8px'}),
            
            html.Div([
                html.Span("✅ Matched: ", style={'fontWeight': 'bold', 'color': '#27ae60'}),
                html.Span(', '.join(match['matched_skills'][:5]), style={'fontSize': '14px'}),
                html.Span(f" (+{len(match['matched_skills'])-5} more)" if len(match['matched_skills']) > 5 else "", 
                         style={'fontSize': '14px', 'fontStyle': 'italic'})
            ]) if match['matched_skills'] else None,
            
            html.Div([
                html.Span("❌ Not needed: ", style={'fontWeight': 'bold', 'color': '#e74c3c'}),
                html.Span(', '.join(match['unmatched_skills'][:5]), style={'fontSize': '14px'}),
                html.Span(f" (+{len(match['unmatched_skills'])-5} more)" if len(match['unmatched_skills']) > 5 else "", 
                         style={'fontSize': '14px', 'fontStyle': 'italic'})
            ], style={'marginTop': '5px'}) if match['unmatched_skills'] else None,
            
            html.Div([
                html.Span("📋 Job requires: ", style={'fontWeight': 'bold', 'color': '#f39c12'}),
                html.Span(', '.join(missing_skills[:5]), style={'fontSize': '14px'}),
                html.Span(f" (+{len(missing_skills)-5} more)" if len(missing_skills) > 5 else "", 
                         style={'fontSize': '14px', 'fontStyle': 'italic'})
            ], style={'marginTop': '5px'}) if missing_skills else None
        ])
    ], style={'marginBottom': '15px', 'border': '1px solid #e0e0e0', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})

def display_strict_matches_paginated(matches, job_data, current_page=1):
    """Display strict match results with pagination"""
    if not matches:
        return dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.H4("No Perfect Matches Found", style={'marginTop': '20px', 'color': '#e74c3c'}),
                    html.P('No positions require all your selected skills. Check the "Extended Match" tab for partial matches.',
                           style={'marginTop': '10px', 'color': '#7f8c8d'})
                ], style={'textAlign': 'center', 'padding': '40px'})
            ])
        ])
    
    # Pagination settings
    items_per_page = 6
    total_pages = min((len(matches) - 1) // items_per_page + 1, 10)
    start_idx = (current_page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(matches), items_per_page * 10)
    
    # Summary card
    summary_card = dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H2(f"{min(len(matches), 60)}", className="text-success"),
                    html.P("Perfect Matches", className="text-muted")
                ], md=4),
                dbc.Col([
                    html.P("✅ All your skills will be utilized", className="mb-1", style={'fontSize': '16px'}),
                    html.P("✅ Education requirements met", className="mb-1", style={'fontSize': '16px'}),
                    html.P("✅ 100% skill utilization", className="mb-0", style={'fontSize': '16px'})
                ], md=8)
            ])
        ])
    ], style={'marginBottom': '20px', 'backgroundColor': '#d4edda'})
    
    # Job cards
    job_cards = []
    for match in matches[start_idx:end_idx]:
        job_cards.append(create_job_card(match, job_data, "success"))
    
    # Pagination
    pagination = dbc.Pagination(
        id={'type': 'pagination', 'index': 'strict'},
        active_page=current_page,
        max_value=total_pages,
        first_last=True,
        previous_next=True,
        size="md",
        style={'marginTop': '20px'}
    ) if total_pages > 1 else None
    
    return html.Div([
        summary_card,
        html.P(f"Showing {start_idx + 1}-{min(end_idx, len(matches))} of {min(60, len(matches))} results", 
               style={'textAlign': 'right', 'color': '#7f8c8d', 'fontSize': '14px'}),
        *job_cards,
        pagination
    ])

def display_extended_matches_paginated(matches_dict, job_data, current_page=1):
    """Display extended match results with pagination"""
    # Combine all matches
    all_matches = []
    for category, matches in matches_dict.items():
        for match in matches:
            match['category'] = category
            all_matches.append(match)
    
    # Sort by match ratio
    all_matches.sort(key=lambda x: x['match_ratio'], reverse=True)
    
    if not all_matches:
        return dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.H4("No Matches Found", style={'marginTop': '20px', 'color': '#e74c3c'}),
                    html.P("Try adjusting your search criteria", style={'marginTop': '10px', 'color': '#7f8c8d'})
                ], style={'textAlign': 'center', 'padding': '40px'})
            ])
        ])
    
    # Pagination settings
    items_per_page = 6
    total_pages = min((len(all_matches) - 1) // items_per_page + 1, 10)
    start_idx = (current_page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(all_matches), items_per_page * 10)
    
    # Summary cards
    summary_row = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(f"{min(len(all_matches), 60)}", className="text-primary"),
                    html.P("Total Matches", className="text-muted mb-0", style={'fontSize': '14px'})
                ], style={'padding': '15px'})
            ])
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(f"{len(matches_dict['excellent'])}", className="text-success"),
                    html.P("Excellent (80%+)", className="text-muted mb-0", style={'fontSize': '14px'})
                ], style={'padding': '15px'})
            ])
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(f"{len(matches_dict['good'])}", className="text-info"),
                    html.P("Good (60-79%)", className="text-muted mb-0", style={'fontSize': '14px'})
                ], style={'padding': '15px'})
            ])
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(f"{len(matches_dict['fair'])}", className="text-warning"),
                    html.P("Fair (40-59%)", className="text-muted mb-0", style={'fontSize': '14px'})
                ], style={'padding': '15px'})
            ])
        ], md=3)
    ], style={'marginBottom': '20px'})
    
    # Explanation
    explanation = dbc.Alert([
        html.H6("How matching works:", className="alert-heading"),
        html.P("The percentage shows how many of YOUR selected skills are needed by the job.", className="mb-1"),
        html.P("For example: 80% means 4 out of your 5 selected skills are required by this position.", className="mb-0")
    ], color="info", style={'marginBottom': '20px'})
    
    # Job cards
    job_cards = []
    for match in all_matches[start_idx:end_idx]:
        category = match['category']
        badge_color = 'success' if category == 'excellent' else 'info' if category == 'good' else 'warning'
        job_cards.append(create_job_card(match, job_data, badge_color))
    
    # Pagination
    pagination = dbc.Pagination(
        id={'type': 'pagination', 'index': 'extended'},
        active_page=current_page,
        max_value=total_pages,
        first_last=True,
        previous_next=True,
        size="md",
        style={'marginTop': '20px'}
    ) if total_pages > 1 else None
    
    return html.Div([
        summary_row,
        explanation,
        html.P(f"Showing {start_idx + 1}-{min(end_idx, len(all_matches))} of {min(60, len(all_matches))} results", 
               style={'textAlign': 'right', 'color': '#7f8c8d', 'fontSize': '14px'}),
        *job_cards,
        pagination
    ])

def load_skills_dictionary():
    """Load the skills categorization dictionary"""
    try:
        with open('skills_filter_dictionary.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Warning: skills_filter_dictionary.json not found. Using default categories.")
        return {
            'Programming Languages': ['Python', 'Java', 'JavaScript', 'C++', 'SQL', 'R', 'C#', 'Go', 'Ruby', 'PHP'],
            'Microsoft Office': ['Excel', 'Word', 'PowerPoint', 'Outlook', 'Access', 'OneNote', 'Microsoft Office'],
            'Data Science & Analytics': ['Tableau', 'Power BI', 'Machine Learning', 'Deep Learning', 'TensorFlow', 'PyTorch'],
            'Web Development': ['HTML', 'CSS', 'React', 'Node.js', 'Angular', 'Vue.js', 'Django', 'Flask'],
            'Cloud & DevOps': ['AWS', 'Azure', 'Docker', 'Kubernetes', 'Jenkins', 'Git', 'CI/CD'],
            'Design Software': ['Photoshop', 'Illustrator', 'Figma', 'Sketch', 'Adobe XD', 'Canva'],
            'CAD & Engineering': ['AutoCAD', 'SolidWorks', 'Revit', 'CATIA', 'ANSYS', 'MATLAB'],
            'Database': ['MySQL', 'PostgreSQL', 'MongoDB', 'Oracle', 'SQL Server', 'Redis'],
            'Operating Systems': ['Linux', 'Windows', 'macOS', 'Unix', 'Ubuntu', 'CentOS'],
            'CRM & Business Software': ['Salesforce', 'HubSpot', 'SAP', 'Dynamics 365', 'Jira', 'Asana']
        }

# Setup logging
logging.basicConfig(level=logging.INFO)
# ========================= DASHBOARD CREATION =========================

def create_combined_dashboard():
    """Create the combined dashboard application with all tabs"""
    # Load qualification data
    print("Loading qualification data...")
    quali_data = load_qualification_data()
    
    # Load job market data
    print("Loading job market data...")
    classified_tags, location_data, job_data = load_job_market_data()
    
    # Initialize Dash app with Bootstrap theme
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.title = "Job Market & Qualification Analysis Dashboard"
    
    # ========================= DATA PREPARATION FOR DASHBOARD =========================
    
    # === QUALIFICATION TAB DATA ===
    
    # Prepare skill category dropdown options for qualification tab
    skill_categories = ['All']
    if quali_data is not None and len(quali_data) > 0:
        tech_categories = quali_data['Tech_Skill_Category'].unique()
        skill_categories.extend([cat for cat in tech_categories if cat != 'None'])
    
    # Calculate percentage of jobs with experience requirements
    exp_required_pct = 0
    if quali_data is not None and len(quali_data) > 0:
        exp_required_pct = (quali_data['Has_Experience_Req'].sum() / len(quali_data) * 100)
    
    # === JOB MARKET TAB DATA ===
    
    # Get unique skill categories for the skills analysis tab
    job_skill_categories = []
    if classified_tags is not None:
        job_skill_categories = sorted(classified_tags['Category'].unique())
        if 'Uncategorized' in job_skill_categories:
            # Move 'Uncategorized' to the end
            job_skill_categories.remove('Uncategorized')
            job_skill_categories.append('Uncategorized')
    
    # Calculate category metrics for skills tab
    category_counts = None
    if classified_tags is not None:
        category_counts = classified_tags.groupby('Category').agg(
            Distinct_Tags=('Tag', 'count'),
            Total_Occurrences=('Count', 'sum')
        ).reset_index()
    
    # Get location data for location analysis
    location_counts = location_data.get('location_counts')
    all_locations = location_data.get('all_locations')
    
    # Calculate job metrics
    total_jobs = len(job_data) if job_data is not None else (
        location_counts['Count'].sum() if location_counts is not None else 0
    )
    
    total_locations = len(all_locations) if all_locations is not None else (
        len(location_counts) if location_counts is not None else 0
    )
    
    # Extract all unique locations from job data for time series tab
    all_location_options = []
    if job_data is not None and 'locations' in job_data.columns:
        all_location_set = set()
        for loc in job_data['locations'].dropna():
            individual_locs = extract_locations(loc)
            all_location_set.update(individual_locs)
        all_location_options = sorted(list(all_location_set))
    
    # Extract employers and industries from job data
    top_employers = []
    top_industries = []
    
    # Set the specific column names for employer and industry
    employer_column = 'companyName'  # Use companyName for employers
    industry_column = None  # Will try to detect industry column
    
    # Identify industry column if available
    if job_data is not None:
        # Check if the employer column exists
        if employer_column in job_data.columns:
            employer_counts = job_data[employer_column].value_counts().head(50)
            top_employers = employer_counts.index.tolist()
        else:
            print(f"Warning: '{employer_column}' column not found in job data")
            employer_column = None
                
        # Find industry column - could be 'industry', 'sector', etc.
        industry_candidates = ['industry', 'sector', 'field']
        for column in job_data.columns:
            if any(candidate in column.lower() for candidate in industry_candidates):
                industry_column = column
                break
        
        # If we found an industry column, get top industries
        if industry_column:
            industry_counts = job_data[industry_column].value_counts().head(20)
            top_industries = industry_counts.index.tolist()
        else:
            print("Warning: Could not identify industry column in job data")
    
    
     # === MAP DATA ===
    # Load your data
    map_df = pd.read_csv("dash_data/final_withcity_v3.csv")
    map_df.loc[map_df['Region'] == "Mexico City", 'Region'] = "MEXICO"
    map_df['City'] = map_df['City'].str.upper().str.strip()
    map_df['Region'] = map_df['Region'].str.upper().str.strip()
    map_df['Region'] = map_df['Region'].apply(normalize)

    print(map_df[map_df['Region'] == "Mexico City"])
    print(map_df[map_df['Region'] == "MEXICO"])

    province_name_map = {
    'Quebec': 'QUEBEC',
    'British Columbia': 'BRITISHCOLUMBIA',
    'Newfoundland And Labrador': 'NEWFOUNDLANDANDLABRADOR',
    'Newfoundland & Labrador': 'NEWFOUNDLANDANDLABRADOR',
    'Prince Edward Island': 'PRINCEEDWARDISLAND',
    'New Brunswick': 'NEWBRUNSWICK',
    'Nova Scotia': 'NOVASCOTIA',
    'Yukon': 'YUKON',
    'Ontario': 'ONTARIO',
    'Alberta': 'ALBERTA',
    'Saskatchewan': 'SASKATCHEWAN',
    'Manitoba': 'MANITOBA',
    'Northwest Territories': 'NORTHWESTTERRITORIES',
    'Nunavut': 'NUNAVUT'
}

  
    # Extract Month from "Month/Year" column (e.g., '25-Mar' → 'Mar')
    map_df['Month'] = map_df['Month/Year'].str[3:]


    logging.info(f"Rows before merge: {len(map_df)}")
    
    # Load mapping data
    mapping_df = pd.read_csv("uscities.csv")
    mapping_df['city'] = mapping_df['city'].str.upper().str.strip()
    mapping_df['County'] = mapping_df['county_name']
    mapping_df = mapping_df[['city', 'state_id', 'County']] 
    

    
    logging.info(f"Rows after merge: {len(map_df)}")
    logging.info(f"Duplicates after merge: {map_df.duplicated().sum()}")
    
    # Normalize Canadian Province names for consistent matching with GeoJSON
    canada_mask = map_df['Country'] == 'Canada'
    if canada_mask.any():
        map_df.loc[canada_mask, 'Region'] = (
            map_df.loc[canada_mask, 'Region']
            .map(province_name_map)
            .fillna(map_df.loc[canada_mask, 'Region'])
            .apply(normalize)
        )

    
    
    # Load All GeoJSON Files Dynamically
    
    geojson_files = {
        'USA_L1': 'gadm41_USA_1.json',
        'USA_L2': 'gadm41_USA_2.json',
        'CAN_L1': 'gadm41_CAN_1.json',
        'CAN_L3': 'gadm41_CAN_3.json',
        'ISR_L1': 'gadm41_ISR_1.json',
        'GBR_L1': 'gadm41_GBR_1.json',
        'GBR_L2': 'gadm41_GBR_2.json',
        'MEX_L1': 'gadm41_MEX_1.json',
        'MEX_L2': 'gadm41_MEX_2.json',
        'MAR_L1': 'gadm41_MAR_1.json',
        'MAR_L3': 'gadm41_MAR_3.json',
        'VNM_L1': 'gadm41_VNM_1.json',
        'VNM_L3': 'gadm41_VNM_3.json',
        'IDN_L1': 'gadm41_IDN_1.json',
        'IDN_L3': 'gadm41_IDN_3.json',
        'SGP_L1': 'gadm41_SGP_1.json',
        'CHN_L1': 'gadm41_CHN_1.json',
        'CHN_L3': 'gadm41_CHN_3.json',
    }
    
    
    # Load all GeoJSON files with proper keys and normalize properties
    geojson_data = {}
    for key, path in geojson_files.items():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                geojson = json.load(f)
    
                # ✅ Normalize relevant properties
                for feature in geojson.get('features', []):
                    if 'properties' in feature:
                        for name_key in ['NAME_1', 'NAME_2', 'NAME_3']:
                            if name_key in feature['properties']:
                                feature['properties'][name_key] = feature['properties'][name_key].upper()
    
                if key == 'CAN_L1':
                    for feature in geojson['features']:
                        name = normalize(feature['properties'].get('NAME_1', ''))
                        if name == 'QUABEC' or name == 'QUEBEC':
                            feature['properties']['NAME_1'] = 'QUEBEC'
                    
                geojson_data[key] = geojson
                logging.info(f"✅ Loaded and normalized geojson: {key}")
        except Exception as e:
            logging.warning(f"❌ Failed to load {key}: {e}")
        

    
    
    
    # 
    skills_dict = load_skills_dictionary()

    matcher_degree_options = []
    matcher_tech_options = []
    matcher_soft_options = []
    matcher_language_options = []

    if quali_data is not None and len(quali_data) > 0:
        # Degree options (保持不变)
        unique_degrees = quali_data['Degree_Clean'].unique()
        degree_order = [
            '1_No Requirement', '2_Student', '3_High School', 
            '4_Pursuing Bachelor\'s', '5_Bachelor\'s', 
            '6_Pursuing Master\'s', '6_Bachelor\'s/Master\'s', '7_Master\'s',
            '8_Pursuing PhD', '9_PhD', '10_Field Specific', '11_Other'
        ]
        matcher_degree_options = [
            {'label': display_degree_name(d), 'value': d} 
            for d in degree_order if d in unique_degrees
        ]
        
        # Technical skills options - 使用JSON文件的分类方式（只影响job matcher）
        skill_options = []
        for category, skills in skills_dict.items():
            skill_options.append({
                'label': html.Span([
                    html.Span(f"━━━ {category} ━━━", style={'fontWeight': 'bold', 'color': '#7f8c8d'})
                ]),
                'value': f'__category__{category}',
                'disabled': True
            })
            for skill in skills:
                skill_options.append({
                    'label': f"    {skill}",
                    'value': skill
                })
        matcher_tech_options = skill_options
        
        # Add normalized skills to quali_data if not already present
        if 'Technical_Skills_Normalized' not in quali_data.columns:
            quali_data['Technical_Skills_Normalized'] = quali_data['Technical_Skills_List'].apply(
                lambda skills: [normalize_skill_name(s) for s in skills]
            )
            
    # ========================= CREATE DASHBOARD LAYOUT =========================
    app.layout = html.Div(
        
        style={'fontFamily': 'Arial, sans-serif', 'maxWidth': '1200px', 'margin': '0 auto', 'padding': '20px'}, 
        children=[
            # Data refresh area
            html.Div([
                html.Div([
                    html.Button(
                        '🔄 Refresh Data', 
                        id='refresh-button',
                        n_clicks=0,
                        style={
                            'padding': '10px 20px', 
                            'backgroundColor': '#3498db', 
                            'color': 'white', 
                            'border': 'none', 
                            'borderRadius': '5px', 
                            'cursor': 'pointer',
                            'fontSize': '16px',
                            'marginRight': '15px'
                        }
                    ),
                    html.Span(
                        id='refresh-status',
                        children='Ready',
                        style={
                            'fontSize': '14px', 
                            'color': '#7f8c8d',
                            'marginRight': '15px'
                        }
                    ),
                    html.Span(
                        id='last-update-time',
                        children=f'Last Updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                        style={
                            'fontSize': '12px', 
                            'color': '#95a5a6'
                        }
                    )
                ], style={
                    'display': 'flex', 
                    'alignItems': 'center', 
                    'padding': '15px', 
                    'backgroundColor': '#f8f9fa', 
                    'borderRadius': '8px',
                    'marginBottom': '20px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                })
            ]),
            
            # Add hidden storage component to trigger data refresh
            dcc.Store(id='refresh-trigger', data=0),
            
            html.H1(
                "Job Market & Qualification Analysis Dashboard", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'}
            ),
            
            # HERE'S THE CORRECTED dcc.Tabs COMPONENT:
            dcc.Tabs(id='dashboard-tabs', children=[
                
                # Tab 1: Location Analysis
                dcc.Tab(label="Location Analysis", value="location-tab", children=[
                    html.Div([
                        # Location KPI summary cards
                        html.Div([
                            html.Div([
                                html.H3("Total Job Listings", style={'textAlign': 'center', 'color': '#7f8c8d'}),
                                html.H2(f"{total_jobs:,}", style={'textAlign': 'center', 'color': '#2c3e50'})
                            ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '5px', 
                                    'boxShadow': '2px 2px 2px lightgrey', 'flex': '1', 'margin': '10px'}),
                            
                            html.Div([
                                html.H3("Unique Locations", style={'textAlign': 'center', 'color': '#7f8c8d'}),
                                html.H2(f"{total_locations:,}", style={'textAlign': 'center', 'color': '#2c3e50'})
                            ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '5px', 
                                    'boxShadow': '2px 2px 2px lightgrey', 'flex': '1', 'margin': '10px'}),
                            
                            html.Div([
                                html.H3("Average Jobs Per Location", style={'textAlign': 'center', 'color': '#7f8c8d'}),
                                html.H2(f"{total_jobs / total_locations:.1f}" if total_locations > 0 else "N/A", 
                                        style={'textAlign': 'center', 'color': '#2c3e50'})
                            ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '5px', 
                                    'boxShadow': '2px 2px 2px lightgrey', 'flex': '1', 'margin': '10px'}),
                        ], style={'display': 'flex', 'flexWrap': 'wrap', 'marginBottom': '30px', 'marginTop': '20px'}),


                        
                        # Filters and Map
                        # html.Div([
                        #     html.H2("Country", style={'textAlign': 'left', 'color': '#2c3e50', 'marginBottom': '10px'}),
                        #     dcc.Dropdown(
                        #         id='country-dropdown',
                        #         options=[{'label': c, 'value': c} for c in countries] if countries else [],
                        #         value=countries[0] if countries else None
                        #     ),
                        #     dcc.Graph(id='map-chart', style={'height': '50vh'})
                        # ]),


                        html.Div([
                                html.H1("Country and Region Filter with City-Level Map"),
                            
                                html.Label("Job Category"),
                                dcc.Dropdown(
                                    id='category-dropdown_map',
                                    multi=True,
                                    placeholder="Select Job Category"
                                ),
                            
                                html.Label("Country"),
                                dcc.Dropdown(
                                    id='country-dropdown',
                                    multi=True,
                                    placeholder="Select Country"
                                ),
                            
                                html.Label("Region"),
                                dcc.Dropdown(
                                    id='region-dropdown',
                                    multi=True,
                                    placeholder="Select Region(s)"
                                ),
                            
                                html.Label("Year"),
                                dcc.Dropdown(
                                    id='year-dropdown',
                                    multi=True,
                                    placeholder="Select Year"
                                ),
                            
                                html.Label("Month"),
                                dcc.Dropdown(
                                    id='month-dropdown',
                                    multi=True,
                                    placeholder="Select Month"
                                ),
                            
                                dcc.Graph(id='choropleth-map')
                                    ]),







                        
                        # Top locations visualization
                        html.Div([
                            html.H2("Top Job Locations", 
                                    style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'}),
                            
                            # Location count slider
                            html.Div([
                                html.Label("Number of locations to display:", 
                                          style={'fontSize': '14px', 'marginRight': '10px'}),
                                dcc.Slider(
                                    id='location-count-slider',
                                    min=5,
                                    max=30,
                                    step=5,
                                    value=10,
                                    marks={i: str(i) for i in range(5, 35, 5)},
                                ),
                            ], style={'width': '80%', 'margin': '0 auto', 'marginBottom': '20px'}),
                            
                            # Location bar chart
                            dcc.Graph(id='top-locations-chart'),
                        ], style={'marginBottom': '30px'}),
                                            
                        # Location search
                        html.Div([
                            html.H2("Search Locations", 
                                    style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'}),
                            
                            html.Div([
                                dcc.Input(
                                    id='location-search',
                                    type='text',
                                    placeholder='Type location name to search...',
                                    style={'width': '70%', 'padding': '10px', 'marginRight': '10px'}
                                ),
                                html.Button(
                                    'Search', 
                                    id='location-search-button',
                                    style={'padding': '10px 15px', 'backgroundColor': '#3498db', 'color': 'white', 
                                          'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}
                                )
                            ], style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '20px'}),
                            
                            html.Div(id='location-search-results'),
                        ], style={'marginBottom': '30px'})
                    ])
                ]),
                
                # Tab 2: Job Trends
                dcc.Tab(label="Job Trends", value="trends-tab", children=[
                    html.Div([                          
                        # Time series by location
                        html.Div([
                            html.H2("Job Postings Over Time by Location", 
                                    style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'}),
                            
                            html.Div([
                                html.Label("Select Location:", style={'fontSize': '16px', 'marginRight': '10px'}),
                                dcc.Dropdown(
                                    id='location-time-dropdown',
                                    options=[{'label': 'All Locations', 'value': 'All Locations'}] + 
                                            [{'label': loc, 'value': loc} for loc in all_location_options],
                                    value='All Locations',
                                    style={'width': '100%'}
                                )
                            ], style={'width': '50%', 'margin': '0 auto', 'marginBottom': '20px'}),
                            
                            dcc.Graph(id='time-series-chart', style={'height': '700px'})
                        ], style={'marginBottom': '5px'}),
                        
                        # Monthly trend metrics
                        html.Div([
                            html.H2("Monthly Trends Analysis", 
                                    style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'}),
                            
                            html.Div(id='monthly-trend-metrics', style={'marginTop': '20px'}),
                        ], style={'marginBottom': '30px'})
                    ])
                ]),
                
                # Tab 3: Employer-Industry Analysis
                dcc.Tab(label="Employer & Industry", value="employer-tab", children=[
                    html.Div([
                        # Tab introduction and description
                        html.Div([
                            html.H2("Employer and Industry Analysis by Location", 
                                    style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
                            html.P("Explore how employers and industries are distributed across different locations.",
                                   style={'textAlign': 'center', 'marginBottom': '20px', 'fontSize': '16px'})
                        ]),
                        
                        # Location selector
                        html.Div([
                            html.Label("Select Location:", style={'fontSize': '16px', 'marginRight': '10px'}),
                            dcc.Dropdown(
                                id='employer-location-dropdown',
                                options=[{'label': 'All Locations', 'value': 'All Locations'}] + 
                                        [{'label': loc, 'value': loc} for loc in all_location_options],
                                value='All Locations',
                                style={'width': '100%'}
                            )
                        ], style={'width': '50%', 'margin': '0 auto', 'marginBottom': '20px'}),
                        
                        # Top employers section
                        html.Div([
                            html.H3("Top Employers in Selected Location", 
                                    style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'}),
                            
                            # Number of employers to display
                            html.Div([
                                html.Label("Number of employers to display:", 
                                          style={'fontSize': '14px', 'marginRight': '10px'}),
                                dcc.Slider(
                                    id='employer-count-slider',
                                    min=5,
                                    max=20,
                                    step=5,
                                    value=10,
                                    marks={i: str(i) for i in range(5, 25, 5)},
                                ),
                            ], style={'width': '50%', 'margin': '0 auto', 'marginBottom': '20px'}),
                            
                            # Employer bar chart
                            dcc.Graph(id='top-employers-chart', style={'height': '500px'}),
                        ], style={'marginBottom': '30px'}),
                        
                        # Top industries section
                        html.Div([
                            html.H3("Top Industries in Selected Location", 
                                    style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'}),
                            
                            # Industry pie chart
                            dcc.Graph(id='industry-pie-chart', style={'height': '500px'}),
                        ], style={'marginBottom': '30px'}),
                        
                        # Employer-Industry heatmap section
                        html.Div([
                            html.H3("Employer-Industry Distribution", 
                                    style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'}),
                            
                            dcc.Graph(id='employer-industry-heatmap', style={'height': '700px'}),
                        ], style={'marginBottom': '30px'}),
                    ])
                ]),
                
                # Tab 4: Skills Analysis
                dcc.Tab(label="Skills Analysis", value="skills-tab", children=[
                    # First row: KPI summary cards
                    html.Div([
                        html.Div([
                            html.H3("Total Unique Tags", style={'textAlign': 'center', 'color': '#7f8c8d'}),
                            html.H2(f"{len(classified_tags)}" if classified_tags is not None else "N/A", 
                                    style={'textAlign': 'center', 'color': '#2c3e50'})
                        ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '5px', 'boxShadow': '2px 2px 2px lightgrey', 'flex': '1', 'margin': '10px'}),
                        
                        html.Div([
                            html.H3("Total Categories", style={'textAlign': 'center', 'color': '#7f8c8d'}),
                            html.H2(f"{len(job_skill_categories)}" if job_skill_categories else "N/A", style={'textAlign': 'center', 'color': '#2c3e50'})
                        ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '5px', 'boxShadow': '2px 2px 2px lightgrey', 'flex': '1', 'margin': '10px'}),
                        
                        html.Div([
                            html.H3("Total Subcategories", style={'textAlign': 'center', 'color': '#7f8c8d'}),
                            html.H2(f"{len(classified_tags['Subcategory'].unique())}" if classified_tags is not None else "N/A", 
                                    style={'textAlign': 'center', 'color': '#2c3e50'})
                        ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '5px', 'boxShadow': '2px 2px 2px lightgrey', 'flex': '1', 'margin': '10px'}),
                        
                        html.Div([
                            html.H3("Total Occurrences", style={'textAlign': 'center', 'color': '#7f8c8d'}),
                            html.H2(f"{classified_tags['Count'].sum():,}" if classified_tags is not None else "N/A", 
                                    style={'textAlign': 'center', 'color': '#2c3e50'})
                        ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '5px', 'boxShadow': '2px 2px 2px lightgrey', 'flex': '1', 'margin': '10px'})
                    ], style={'display': 'flex', 'flexWrap': 'wrap', 'marginBottom': '30px', 'marginTop': '20px'}),
                    
                    # Second row: Main charts
                    html.Div([
                        # Left panel: Category distribution
                        html.Div([
                            html.H2("Distribution of Tags by Category", 
                                    style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'}),
                            dcc.Graph(id='category-distribution')
                        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                        
                        # Right panel: Category breakdown (pie chart)
                        html.Div([
                            html.H2("Category Breakdown", 
                                    style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'}),
                            dcc.Graph(id='category-pie')
                        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'})
                    ], style={'marginBottom': '30px'}),
                    
                    # Third row: Category selector and subcategory chart
                    html.Div([
                        html.H2("Subcategories Breakdown", 
                                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'}),
                        
                        html.Div([
                            html.Label("Select Category:", style={'fontSize': '16px', 'marginRight': '10px'}),
                            dcc.Dropdown(
                                id='category-dropdown',
                                options=[{'label': cat, 'value': cat} for cat in job_skill_categories] if job_skill_categories else [],
                                value=job_skill_categories[0] if job_skill_categories else None,
                                style={'width': '100%'}
                            )
                        ], style={'width': '50%', 'margin': '0 auto', 'marginBottom': '20px'}),
                        
                        dcc.Graph(id='subcategory-chart')
                    ], style={'marginBottom': '30px'}),
                    
                    # Fourth row: Top tags table
                    html.Div([
                        html.H2("Top Tags in Selected Category", 
                                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'}),
                        
                        html.Div(id='top-tags-table')
                    ], style={'marginBottom': '30px'})
                ]),
                
                # Tab 5: Qualification Analysis - MODIFIED VERSION
                dcc.Tab(label="Qualification Analysis", value="qualification-tab", children=[
                    html.Div([
                        # Keep the original summary statistics
                        html.Div([
                            html.Div([
                                html.H3("Require Degree", style={'textAlign': 'center', 'color': '#7f8c8d'}),
                                html.H2(f"{100 - (quali_data['Degree_Clean'].str.contains('No Requirement').sum() / len(quali_data) * 100):.1f}%" if quali_data is not None and len(quali_data) > 0 else "N/A", 
                                        style={'textAlign': 'center', 'color': '#2c3e50'})
                            ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '5px', 
                                    'boxShadow': '2px 2px 2px lightgrey', 'flex': '1', 'margin': '10px'}),
                            
                            html.Div([
                                html.H3("Require Tech Skills", style={'textAlign': 'center', 'color': '#7f8c8d'}),
                                html.H2(f"{100 - (quali_data['Tech_Skill_Category'].value_counts().get('None', 0) / len(quali_data) * 100):.1f}%" if quali_data is not None and len(quali_data) > 0 else "N/A", 
                                        style={'textAlign': 'center', 'color': '#2c3e50'})
                            ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '5px', 
                                    'boxShadow': '2px 2px 2px lightgrey', 'flex': '1', 'margin': '10px'}),
                            
                            html.Div([
                                html.H3("Require Soft Skills", style={'textAlign': 'center', 'color': '#7f8c8d'}),
                                html.H2(f"{100 - (quali_data['Soft_Skills_Keywords'].apply(lambda x: len(x) == 0).sum() / len(quali_data) * 100):.1f}%" if quali_data is not None and len(quali_data) > 0 else "N/A", 
                                        style={'textAlign': 'center', 'color': '#2c3e50'})
                            ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '5px', 
                                    'boxShadow': '2px 2px 2px lightgrey', 'flex': '1', 'margin': '10px'}),
                            
                            html.Div([
                                html.H3("Require Experience", style={'textAlign': 'center', 'color': '#7f8c8d'}),
                                html.H2(f"{exp_required_pct:.1f}%" if exp_required_pct > 0 else "N/A", 
                                        style={'textAlign': 'center', 'color': '#2c3e50'})
                            ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '5px', 
                                    'boxShadow': '2px 2px 2px lightgrey', 'flex': '1', 'margin': '10px'})
                        ], style={'display': 'flex', 'flexWrap': 'wrap', 'marginBottom': '30px', 'marginTop': '20px'}),

                        # NLP Extract Skill Test Section
                        html.Div([
                            html.H2("NLP Extract Skill（Test）", style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'}),
                        
                            html.Div([
                                dcc.Input(
                                    id='nlp-input-text',
                                    type='text',
                                    placeholder='Please enter a qualification text...',
                                    style={'width': '80%', 'padding': '10px', 'marginRight': '10px'}
                                ),
                                html.Button(
                                    'Extract classified content', 
                                    id='nlp-extract-button',
                                    style={'padding': '10px 20px', 'backgroundColor': '#27ae60', 'color': 'white',
                                           'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}
                                )
                            ], style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '20px'}),
                        
                            html.Div(id='nlp-output', style={'textAlign': 'center', 'fontSize': '18px', 'color': '#2980b9'})
                        ], style={'marginTop': '50px', 'padding': '30px', 'backgroundColor': '#fdfdfd', 'border': '1px solid #ddd', 'borderRadius': '10px'}),
                        
                        # Divider
                        html.Hr(style={'margin': '40px 0', 'border': '1px solid #e0e0e0'}),
                        
                        # NEW Job Matching System
                        html.Div([
                            dcc.Store(id='match-results-store'),
                            dcc.Store(id='current-page', data=1),
                            
                            html.H2("Smart Job Matching System", 
                                    style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
                            html.P("Select your qualifications to find matching job opportunities",
                                   style={'textAlign': 'center', 'marginBottom': '30px', 'fontSize': '16px', 'color': '#7f8c8d'}),
                            
                            # Input Section
                            html.Div([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H4("Your Qualifications", style={'marginBottom': '20px', 'color': '#34495e'}),
                                        
                                        # Education Level
                                        dbc.Row([
                                            dbc.Col([
                                                html.Label("Education Level", style={'fontWeight': 'bold', 'marginBottom': '10px'}),
                                                dcc.Dropdown(
                                                    id='degree-dropdown',
                                                    options=matcher_degree_options,
                                                    placeholder="Select your education level...",
                                                    style={'marginBottom': '20px'}
                                                )
                                            ], md=12)
                                        ]),
                                        
                                        # Technical Skills
                                        dbc.Row([
                                            dbc.Col([
                                                html.Label("Technical Skills", style={'fontWeight': 'bold', 'marginBottom': '10px'}),
                                                html.P("Select all skills you possess", 
                                                       style={'fontSize': '14px', 'color': '#7f8c8d', 'marginBottom': '10px'}),
                                                dcc.Dropdown(
                                                    id='tech-skills-dropdown',
                                                    options=matcher_tech_options,
                                                    value=[],
                                                    multi=True,
                                                    placeholder="Choose your technical skills...",
                                                    style={'marginBottom': '20px'}
                                                )
                                            ], md=12)
                                        ]),
                                        
                                        # Selected skills display
                                        html.Div(id='selected-skills-display', style={'marginBottom': '20px'}),
                                        
                                        # Search Button
                                        html.Div([
                                            dbc.Button(
                                                "🔍 Find Matching Jobs", 
                                                id='search-button',
                                                color="primary",
                                                size="lg",
                                                n_clicks=0,
                                                style={'width': '100%'}
                                            )
                                        ])
                                    ])
                                ], style={'marginBottom': '30px', 'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)'})
                            ], style={'maxWidth': '900px', 'margin': '0 auto'}),
                            
                            # Results Section
                            html.Div([
                                # Match type tabs
                                dbc.Tabs(
                                    id="match-tabs",
                                    active_tab="strict-match",
                                    children=[
                                        dbc.Tab(label="🎯 Perfect Match (All skills needed)", tab_id="strict-match"),
                                        dbc.Tab(label="📊 Extended Match (Partial skills match)", tab_id="extended-match"),
                                    ],
                                    style={'marginBottom': '20px'}
                                ),
                                
                                # Tab content
                                html.Div(id='tab-content')
                            ])
                        ])
                    ])
                ])
                
            ])  # THIS IS THE MISSING CLOSING BRACKET AND PARENTHESIS FOR dcc.Tabs!
        ]  # This closes the main app.layout children list
    )  # This closes the main app.layout html.Div
    
    #------- CALLBACKS FOR SKILLS TAB -------#
    # ===================  ===================
    @app.callback(
        [Output('refresh-trigger', 'data'),
         Output('refresh-status', 'children'),
         Output('refresh-status', 'style'),
         Output('last-update-time', 'children')],
        [Input('refresh-button', 'n_clicks')],
        prevent_initial_call=True
    )
    def refresh_data(n_clicks):
        if n_clicks and n_clicks > 0:
            try:
                # Show loading status
                status_text = '🔄 Loading data...'
                status_style = {'fontSize': '14px', 'color': '#f39c12', 'marginRight': '15px'}
                
                # Reload data
                nonlocal quali_data, classified_tags, location_data, job_data
                
                print("Refreshing qualification data...")
                quali_data = load_qualification_data()
                
                print("Refreshing job market data...")
                classified_tags, location_data, job_data = load_job_market_data()
                
                # Refresh successful
                status_text = '✅ Refresh successful'
                status_style = {'fontSize': '14px', 'color': '#27ae60', 'marginRight': '15px'}
                update_time = f'Last Updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
                
                return n_clicks, status_text, status_style, update_time
                
            except Exception as e:
                # Refresh failed
                status_text = f'❌ Refresh failed: {str(e)[:50]}'
                status_style = {'fontSize': '14px', 'color': '#e74c3c', 'marginRight': '15px'}
                update_time = f'Last Updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
                
                return n_clicks, status_text, status_style, update_time
        
        return 0, 'Ready', {'fontSize': '14px', 'color': '#7f8c8d', 'marginRight': '15px'}, f'Last Updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
            
    # Callback for initializing category distribution
    @app.callback(
        Output('category-distribution', 'figure'),
        [Input('dashboard-tabs', 'value'),  
         Input('refresh-trigger', 'data')]
    )
    def initialize_category_distribution(tab_value, refresh_trigger): 
        if category_counts is None:
            return go.Figure().update_layout(title="No category data available")
            
        fig = px.bar(
            category_counts.sort_values('Total_Occurrences', ascending=False),
            x='Category',
            y='Total_Occurrences',
            color='Category',
            labels={'Total_Occurrences': 'Total Occurrences', 'Category': 'Skill Category'},
            title='Distribution of Tags by Category',
            color_discrete_sequence=px.colors.qualitative.Bold,
            text='Total_Occurrences'
        )
        
        fig.update_layout(
            xaxis_title="Skill Category",
            yaxis_title="Total Occurrences",
            font=dict(family="Arial, sans-serif", size=12),
            hoverlabel=dict(font_size=12, font_family="Arial, sans-serif"),
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        fig.update_traces(
            texttemplate='%{text:,}',
            textposition='outside'
        )
        
        return fig
    
    # Callback for initializing category pie chart
    @app.callback(
        Output('category-pie', 'figure'),
        [Input('dashboard-tabs', 'value'),  
         Input('refresh-trigger', 'data')]
    )
    def initialize_category_distribution(tab_value, refresh_trigger): 
        if category_counts is None:
            return go.Figure().update_layout(title="No category data available")
            
        fig = px.pie(
            category_counts,
            values='Total_Occurrences',
            names='Category',
            title='Proportion of Tags by Category',
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        
        fig.update_layout(
            font=dict(family="Arial, sans-serif", size=12),
            hoverlabel=dict(font_size=12, font_family="Arial, sans-serif"),
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        fig.update_traces(
            textinfo='percent+label',
            textposition='inside',
            hoverinfo='label+percent+value'
        )
        
        return fig
    
    # Callback for updating subcategory chart based on category selection
    @app.callback(
        Output('subcategory-chart', 'figure'),
        [Input('category-dropdown', 'value'),
         Input('refresh-trigger', 'data')]
    )
    def update_subcategory_chart(selected_category,refresh_trigger):
        if not selected_category or classified_tags is None:
            return go.Figure()
        
        # Filter data for selected category
        filtered_data = classified_tags[classified_tags['Category'] == selected_category]
        
        # Group by subcategory
        subcategory_counts = filtered_data.groupby('Subcategory').agg(
            Total_Tags=('Tag', 'count'),
            Total_Occurrences=('Count', 'sum')
        ).reset_index().sort_values('Total_Occurrences', ascending=False)
        
        # Create the chart
        fig = px.bar(
            subcategory_counts,
            x='Subcategory',
            y='Total_Occurrences',
            color='Subcategory',
            labels={'Total_Occurrences': 'Total Occurrences', 'Subcategory': 'Subcategory'},
            title=f'Subcategories within {selected_category}',
            color_discrete_sequence=px.colors.qualitative.Pastel,
            text='Total_Occurrences'
        )
        
        fig.update_layout(
            xaxis_title="Subcategory",
            yaxis_title="Total Occurrences",
            xaxis_tickangle=-45,
            font=dict(family="Arial, sans-serif", size=12),
            hoverlabel=dict(font_size=12, font_family="Arial, sans-serif"),
            margin=dict(t=50, b=100, l=50, r=50),
            height=500
        )
        
        fig.update_traces(
            texttemplate='%{text:,}',
            textposition='outside'
        )
        
        return fig
    
    # Callback for updating top tags table based on category selection
    @app.callback(
        Output('top-tags-table', 'children'),
        [Input('category-dropdown', 'value'),
         Input('refresh-trigger', 'data')]
    )
    def update_top_tags_table(selected_category,refresh_trigger):
        if not selected_category or classified_tags is None:
            return html.Div("Please select a category")
        
        # Filter data for selected category
        filtered_data = classified_tags[classified_tags['Category'] == selected_category]
        
        # Get top 10 tags in this category
        top_tags = filtered_data.nlargest(10, 'Count')[['Tag', 'Count', 'Subcategory']]
        
        # Create a table
        table = html.Table([
            # Header
            html.Thead(html.Tr([
                html.Th("Tag", style={'padding': '10px', 'backgroundColor': '#f2f2f2', 'textAlign': 'left'}),
                html.Th("Count", style={'padding': '10px', 'backgroundColor': '#f2f2f2', 'textAlign': 'right'}),
                html.Th("Subcategory", style={'padding': '10px', 'backgroundColor': '#f2f2f2', 'textAlign': 'left'})
            ])),
            # Body
            html.Tbody([
                html.Tr([
                    html.Td(tag, style={'padding': '10px', 'borderBottom': '1px solid #ddd'}),
                    html.Td(f"{count:,}", style={'padding': '10px', 'textAlign': 'right', 'borderBottom': '1px solid #ddd'}),
                    html.Td(subcategory, style={'padding': '10px', 'borderBottom': '1px solid #ddd'})
                ]) for tag, count, subcategory in top_tags.itertuples(index=False)
            ])
        ], style={'width': '80%', 'margin': '0 auto', 'borderCollapse': 'collapse'})
        
        return table
    
    #------- CALLBACKS FOR LOCATION TAB -------#
    
    # Callback for map  
    # @app.callback(
    #     Output('map-chart', 'figure'),
    #     [Input('country-dropdown', 'value'),
    #      Input('refresh-trigger', 'data')]
    # )
    # Category dropdown
    @app.callback(
        Output('category-dropdown_map', 'options'),
        Output('category-dropdown_map', 'value'),
        [Input('category-dropdown_map', 'id'),
         Input('refresh-trigger', 'data')]
    )
    def update_category_options(_, __):
        categories = sorted(map_df['Job Category'].dropna().unique())
        options = [{'label': c, 'value': c} for c in categories]
        return options, []


    print("🧪 Job Categories:", map_df['Job Category'].dropna().unique())
    print(map_df.columns.tolist())
    
    @app.callback(
        Output('year-dropdown', 'options'),
        Output('year-dropdown', 'value'),
        [Input('year-dropdown', 'id'),
         Input('refresh-trigger', 'data')]
    )
    def update_year_options(_, __):
        years = sorted(map_df['Year'].dropna().unique())
        options = [{'label': str(y), 'value': y} for y in years]
        return options, []
    
    @app.callback(
        Output('month-dropdown', 'options'),
        Output('month-dropdown', 'value'),
        [Input('month-dropdown', 'id'),
         Input('refresh-trigger', 'data')]
    )
    def update_month_options(_, __):
        months = sorted(map_df['Month'].dropna().unique())
        options = [{'label': m, 'value': m} for m in months]
        return options, []

    
    # Country dropdown options
    @app.callback(
        Output('country-dropdown', 'options'),
        Output('country-dropdown', 'value'),
        [Input('country-dropdown', 'id'),
         Input('refresh-trigger', 'data')]
    )
    def update_country_options(_, __):
        countries = sorted(map_df['Country'].dropna().unique())
        options = [{'label': c, 'value': c} for c in countries]
        default_value = ['Canada'] if 'Canada' in countries else [countries[0]]
        return options, default_value
    
    # Region dropdown options
    @app.callback(
        Output('region-dropdown', 'options'),
        Output('region-dropdown', 'value'),
        [Input('country-dropdown', 'value'),
         Input('refresh-trigger', 'data')]
    )
    def update_region_options(selected_countries, _):
        if not selected_countries:
            return [], []
    
        if isinstance(selected_countries, str):
            selected_countries = [selected_countries]
        print("📍 Selected countries:", selected_countries)
        filtered = map_df[map_df['Country'].isin(selected_countries)]
        regions = sorted(filtered['Region'].dropna().unique())
    
        print("🌐 Available regions:", regions)
        for f in geojson_data['VNM_L1']['features']:
            print(f["properties"]["NAME_1"])
    
        return [{'label': r, 'value': r} for r in regions], []
  
    # Map update
    @app.callback(
        Output('choropleth-map', 'figure'),
        [Input('country-dropdown', 'value'),
        Input('region-dropdown', 'value'),
        Input('category-dropdown_map', 'value'),
        Input('year-dropdown', 'value'),
        Input('month-dropdown', 'value'),
        Input('refresh-trigger', 'data')]
    )
    


    # def update_main_charts(selected_country,refresh_trigger):
    #     # Map chart
    #     map_filtered = map_data[map_data['Country'] == selected_country]     
    #     geojson = geojson_map.get(selected_country)
    #     center = center_coords.get(selected_country, {"lat": 20, "lon": 0})
    #     zoom = zoom_levels.get(selected_country, 4) 
        
    #     if not map_data.empty and geojson:
    #         map_fig = px.choropleth_mapbox(
    #         map_filtered,
    #         geojson=geojson,
    #         locations='Province',
    #         color='Job Count',
    #         #color_continuous_scale=px.colors.sequential.OrRd,
    #         featureidkey='properties.NAME_1',
    #         center=center,
    #         mapbox_style="carto-positron",
    #         zoom=zoom
    #          )
    #             # animation_frame='Date',)
    #         map_fig.update_layout(
    #             title={
    #             'text': f"Job Count by Province in {selected_country}",
    #             'font': {'size': 24},
    #             'x': 0.5,
    #             'xanchor': 'center' }, height=900  )
    #     else:
    #         map_fig = px.scatter(title="No map data available for this selection.")

    #     return map_fig

    def update_map(selected_countries, selected_regions, selected_categories, selected_years, selected_months, _):
        if not selected_countries:
            return go.Figure()
    
        filtered_map = map_df.copy()
        filtered_map['City'] = filtered_map['City'].str.upper().str.strip()
        filtered_map['Region'] = filtered_map['Region'].apply(normalize)
        filtered_map['City'] = filtered_map['City'].apply(normalize)

        if selected_countries == ['Vietnam']:
            debug_df = filtered_map[['Country', 'Region', 'City', 'Job Category', 'Job Count', 'Month', 'Year']]
            print("✅ Debug - Canada only (first 20 rows):")
            print(debug_df.head(20))
            print("Rows in filtered data:", len(debug_df))
            print("Sum of Job Count in filtered data:", debug_df['Job Count'].sum())
    
        # Apply filters
        if selected_categories:
            filtered_map = filtered_map[filtered_map['Job Category'].isin(selected_categories)]
        if selected_countries:
            filtered_map = filtered_map[filtered_map['Country'].isin(selected_countries)]
        if selected_regions:
            selected_regions_normalized = [normalize(r) for r in selected_regions]
            filtered_map = filtered_map[filtered_map['Region'].isin(selected_regions_normalized)]
        if selected_years:
            filtered_map = filtered_map[filtered_map['Year'].isin(selected_years)]
        if selected_months:
            filtered_map = filtered_map[filtered_map['Month'].isin(selected_months)]
   
        if filtered_map.empty:
            return go.Figure().update_layout(title="No data available for selected filters.")
    
        # -------------------------------
        # ✅ Step 2: Dynamic Country Config
        # -------------------------------
        country_config = {
            'USA': {'region_level': 'L1', 'city_level': 'L2', 'region_key': 'NAME_1', 'city_key': 'NAME_2'},
            'CAN': {'region_level': 'L1', 'city_level': 'L3', 'region_key': 'NAME_1', 'city_key': 'NAME_3'},
            'ISR': {'region_level': 'L1', 'city_level': None, 'region_key': 'NAME_1'},
            'GBR': {'region_level': 'L1', 'city_level': None, 'region_key': 'NAME_1'},
            'MEX': {'region_level': 'L1', 'city_level': None, 'region_key': 'NAME_1'},
            'MAR': {'region_level': 'L1', 'city_level': 'L3', 'region_key': 'NAME_1', 'city_key': 'NAME_3'},
            'VNM': {'region_level': 'L1', 'city_level': 'L3', 'region_key': 'NAME_1', 'city_key': 'NAME_3'},
            'IDN': {'region_level': 'L1', 'city_level': 'L3', 'region_key': 'NAME_1', 'city_key': 'NAME_3'},
            'SGP': {'region_level': 'L1', 'city_level': None,   'region_key': 'NAME_1'},
            'CHN': {'region_level': 'L1', 'city_level': 'L3', 'region_key': 'NAME_1', 'city_key': 'NAME_3'}
        }
    
        country_name_to_iso3 = {
        'USA': 'USA',
        'Canada': 'CAN',
        'Israel': 'ISR',
        'United Kingdom': 'GBR',
        'Mexico': 'MEX',
        'Morocco': 'MAR',
        'Vietnam': 'VNM',
        'Indonesia': 'IDN',
        'Singapore': 'SGP',
        'China': 'CHN'
        }
    
        country_code = country_name_to_iso3.get(selected_countries[0])
        
        if not country_code:
            return go.Figure().update_layout(title="Country code not found.")
        
        config = country_config.get(country_code)
        if not config:
            return go.Figure().update_layout(title="No map configuration found for this country.")
    
        region_level = config['region_level']
        city_level = config.get('city_level')
        region_key = config['region_key']
        city_key = config.get('city_key')
    
        # -------------------------------
        # Region Selected → City-Level Map
        # -------------------------------
        if selected_regions and city_level:
            if country_code == 'USA':
                filtered_map = pd.merge(
                    filtered_map,
                    mapping_df,
                    how='left',
                    left_on='City',
                    right_on='city'
                )
                
                filtered_map = filtered_map[filtered_map['Region'].isin(selected_regions)]  # ✅ Ensure only selected state(s)
        
                # Combine County and State to make a unique key (used later to match with GeoJSON if it has state info)
                filtered_map['County_Key'] = filtered_map['County'].str.upper()  # Use only County name if geojson has NAME_2 = 'FRANKLIN'
        
                # Group by County and get Job Count
                grouped = filtered_map.groupby('County_Key', as_index=True)['Job Count'].sum()
                locations = grouped.index
                z_values = grouped.values
        
                featureidkey = f'properties.{city_key}'  # city_key is NAME_2
                geojson_key = f"{country_code}_{city_level}"
                geojson = geojson_data.get(geojson_key)
        
                # ✅ Filter geojson features to selected states only
                if geojson and selected_regions:
                    selected_regions_normalized = [normalize(r) for r in selected_regions]
                    geojson = {
                        "type": "FeatureCollection",
                        "features": [
                            f for f in geojson["features"]
                            if normalize(f["properties"].get(region_key, "")) in selected_regions_normalized
                        ]
                    }
        
            else:
                # Non-USA countries: use city-level
                grouped = filtered_map.groupby('City', as_index=False)['Job Count'].sum()
                locations = grouped['City'].str.upper()
                z_values = grouped['Job Count']
                featureidkey = f'properties.{city_key}'
                geojson_key = f"{country_code}_{city_level}"
                geojson = geojson_data.get(geojson_key)
        
            if not geojson:
                return go.Figure().update_layout(title="City-level GeoJSON not found.")
        
            fig = go.Figure(go.Choroplethmap(
                geojson=geojson,
                locations=locations,
                featureidkey=featureidkey,
                z=z_values,
                colorscale="Viridis",
                marker_line_width=0.5,
                marker_line_color='white',
                colorbar_title="Job Count",
                name="County Job Count" if country_code == 'USA' else "City Job Count"
            ))


            # 🔁 Load the correct region-level GeoJSON
            region_geojson_key = f"{country_code}_{region_level}"
            region_geojson = geojson_data.get(region_geojson_key)
            
            selected_region_features = []
            if region_geojson and selected_regions:
                selected_regions_normalized = [normalize(r) for r in selected_regions]
                selected_region_features = [
                    f for f in region_geojson["features"]
                    if normalize(f["properties"].get(region_key, "")) in selected_regions_normalized
                ]

            
            # --- Highlight selected regions ---
            if selected_region_features:
                highlighted_geojson = {
                    "type": "FeatureCollection",
                    "features": selected_region_features
                }

                bbox = compute_bounding_box(selected_region_features)
                zoom = compute_zoom(bbox)
            
                center = compute_center_from_geojson(selected_region_features)

        
            fig.add_trace(go.Choroplethmap(
                geojson=highlighted_geojson,
                locations=[f['properties'][region_key] for f in selected_region_features],
                z=[1] * len(selected_region_features),
                featureidkey=f'properties.{region_key}',
                colorscale=[[0, 'rgba(173, 216, 230, 0.15)'], [1, 'rgba(173, 216, 230, 0.15)']], 
                marker_line_width=1.2,
                marker_line_color='blue', 
                showscale=False,
                hoverinfo='skip',
                name='Highlighted Region'
            ))

     
            
            fig.update_layout(
                map=dict(
                    style="carto-positron",
                    center=  center,
                    zoom= zoom
                ),
                margin={"r": 0, "t": 30, "l": 0, "b": 0},
                title=f"Job Count in {', '.join(selected_regions)}"
            )
            return fig
    
        # -------------------------------
        # No Region → Region-Level Map
        # -------------------------------
        if selected_regions:
            selected_regions_normalized = [normalize(r) for r in selected_regions]
            filtered_map = filtered_map[filtered_map['Region'].apply(normalize).isin(selected_regions_normalized)]

        print(f"Total rows before aggregation: {len(filtered_map)}")
        print(f"Unique Region rows before aggregation: {filtered_map[['Country', 'Region']].drop_duplicates().shape[0]}")
        print("Sample:")
        print(filtered_map[['Country', 'Region', 'City', 'Job Category', 'Job Count']].head(10))

        
        filtered_regions = filtered_map.groupby(['Country', 'Region'], as_index=False).agg({'Job Count': 'sum'})

        print(filtered_regions.head(10))

        if filtered_regions.empty:
            return go.Figure().update_layout(title="No region data available.")
        
        locations = filtered_regions['Region'].str.upper()
        z_values = filtered_regions['Job Count']
        featureidkey = f'properties.{region_key}'
        geojson_key = f"{country_code}_{region_level}"
        geojson = geojson_data.get(geojson_key)
        if not geojson:
            return go.Figure().update_layout(title="Region-level GeoJSON not found.")
        
   
        # Filter GeoJSON features to only those present in filtered data regions, normalizing GeoJSON region names too
        geojson = filter_geojson_features(geojson, filtered_regions['Region'], region_key)

        # Normalize region names in GeoJSON to match filtered['Region']
        if geojson:
            region_names_in_data = filtered_regions['Region'].apply(normalize).unique()
            geojson = filter_geojson_features(geojson, filtered_regions['Region'], region_key)

        fig = go.Figure(go.Choroplethmap(
            geojson=geojson,
            locations=locations,
            featureidkey=featureidkey,
            z=z_values,
            colorscale="Viridis",
            marker_line_width=0.5,
            marker_line_color='white',
            colorbar_title="Job Count",
            name="City or Region Job Count"
        ))
    

        zoom = zoom_levels.get(selected_countries[0], 3.5)

        
        if 'Latitude' in filtered_map.columns and 'Longitude' in filtered_map.columns and not filtered_map[['Latitude', 'Longitude']].isnull().all().any():
            map_center = {
                "lat": filtered_map['Latitude'].mean(),
                "lon": filtered_map['Longitude'].mean()
            }
        else:
            fallback_centers = {
                'CAN': {"lat": 56.1304, "lon": -106.3468, "zoom": 3},
                'USA': {"lat": 37.0902, "lon": -95.7129, "zoom": 3},
                'GBR': {"lat": 55.3781, "lon": -3.4360, "zoom": 4},
                'ISR': {"lat": 31.0461, "lon": 34.8516, "zoom": 5},
                'MEX': {"lat": 23.6345, "lon": -102.5528, "zoom": 3},
                'MAR': {"lat": 31.7917, "lon": -7.0926, "zoom": 4},
                'VNM': {"lat": 14.0583, "lon": 108.2772, "zoom": 4},
                'IDN': {"lat": -0.7893, "lon": 113.9213, "zoom": 3},
                'SGP': {"lat": 1.3521, "lon": 103.8198, "zoom": 6},
                'CHN': {"lat": 35.8617, "lon": 104.1954, "zoom": 3}
            }
            default = fallback_centers.get(country_code, {"lat": 20, "lon": 0, "zoom": 2.5})
            map_center = {"lat": default["lat"], "lon": default["lon"]}
            map_zoom = default["zoom"]


        fig.update_layout(
            map=dict(
                style="carto-positron",
                center=map_center,
                zoom=map_zoom
            ),
            margin={"r": 0, "t": 30, "l": 0, "b": 0},
            title="Job Count by Region"
        )
                
        return fig   







    # Callback for updating top locations chart based on slider value
    @app.callback(
        Output('top-locations-chart', 'figure'),
        [Input('location-count-slider', 'value'),
         Input('refresh-trigger', 'data')]
    )
    def update_top_locations_chart(n_locations, refresh_trigger):
        if location_counts is None:
            return go.Figure()
        
        # Get top N locations
        top_n_locations = location_counts.head(n_locations)
        
        # Create bar chart
        fig = px.bar(
            top_n_locations,
            y='Location',
            x='Count',
            orientation='h',
            labels={'Count': 'Number of Job Listings', 'Location': 'Location'},
            title=f'Top {n_locations} Locations by Job Count',
            color='Count',
            color_continuous_scale=px.colors.sequential.Viridis,
            text='Count'
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},  # Sort bars by value
            font=dict(family="Arial, sans-serif", size=12),
            hoverlabel=dict(font_size=12, font_family="Arial, sans-serif"),
            margin=dict(t=50, b=50, l=150, r=50),  # Increase left margin for location names
            height=600
        )
        
        fig.update_traces(
            texttemplate='%{text:,}',
            textposition='outside'
        )
        
        return fig
    
    # Callback for updating location pie chart
    @app.callback(
        Output('location-pie-chart', 'figure'),
        [Input('location-count-slider', 'value'),
         Input('refresh-trigger', 'data')]
    )
    def update_location_pie_chart(n_locations, refresh_trigger):
        if location_counts is None:
            return go.Figure()
        
        # Get top N locations
        top_n_locations = location_counts.head(n_locations).copy()
        
        # Calculate other locations (not in top N)
        other_count = location_counts['Count'].sum() - top_n_locations['Count'].sum()
        
        # Add "Other Locations" to the dataframe
        if other_count > 0:
            other_df = pd.DataFrame({'Location': ['Other Locations'], 'Count': [other_count]})
            pie_data = pd.concat([top_n_locations, other_df], ignore_index=True)
        else:
            pie_data = top_n_locations
        
        # Create pie chart
        fig = px.pie(
            pie_data,
            values='Count',
            names='Location',
            title=f'Distribution of Jobs by Location (Top {n_locations} + Other)',
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        
        fig.update_layout(
            font=dict(family="Arial, sans-serif", size=12),
            hoverlabel=dict(font_size=12, font_family="Arial, sans-serif"),
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        fig.update_traces(
            textinfo='percent+label',
            textposition='inside',
            hoverinfo='label+percent+value'
        )
        
        return fig
    
    # Callback for location search
    @app.callback(
        Output('location-search-results', 'children'),
        [Input('location-search-button', 'n_clicks'),
         Input('refresh-trigger', 'data')],   
        [State('location-search', 'value')]     
    )
    def search_locations(n_clicks, refresh_trigger, search_term):
        if n_clicks is None or not search_term or location_counts is None:
            return html.Div("Enter a location name to search")
        
        # Search for matching locations (case insensitive)
        matches = location_counts[location_counts['Location'].str.contains(search_term, case=False, na=False)]
        
        if len(matches) == 0:
            return html.Div(f"No locations found matching '{search_term}'")
        
        # Create a table with the results
        table = html.Table([
            # Header
            html.Thead(html.Tr([
                html.Th("Location", style={'padding': '10px', 'backgroundColor': '#f2f2f2', 'textAlign': 'left'}),
                html.Th("Job Count", style={'padding': '10px', 'backgroundColor': '#f2f2f2', 'textAlign': 'right'}),
                html.Th("% of Total", style={'padding': '10px', 'backgroundColor': '#f2f2f2', 'textAlign': 'right'})
            ])),
            # Body
            html.Tbody([
                html.Tr([
                    html.Td(loc, style={'padding': '10px', 'borderBottom': '1px solid #ddd'}),
                    html.Td(f"{count:,}", style={'padding': '10px', 'textAlign': 'right', 'borderBottom': '1px solid #ddd'}),
                    html.Td(f"{(count / location_counts['Count'].sum()) * 100:.2f}%", 
                            style={'padding': '10px', 'textAlign': 'right', 'borderBottom': '1px solid #ddd'})
                ]) for loc, count in matches.itertuples(index=False)
            ])
        ], style={'width': '80%', 'margin': '0 auto', 'borderCollapse': 'collapse'})
        
        return html.Div([
            html.Div(f"Found {len(matches)} locations matching '{search_term}':", 
                    style={'textAlign': 'center', 'marginBottom': '10px'}),
            table
        ])
    
    #------- CALLBACKS FOR TIME SERIES TAB -------#
    
    # Function to get monthly counts for a specific location
    def get_monthly_counts(df, selected_location=None):
        # Filter dataframe based on selected location
        if selected_location and selected_location != "All Locations" and df is not None:
            mask = df['locations'].apply(
                lambda x: selected_location in extract_locations(x) if isinstance(x, str) else False
            )
            filtered_df = df[mask].copy()
        else:
            if df is not None:
                filtered_df = df.copy()
            else:
                return pd.DataFrame()  # Return empty dataframe if job_data is None
        
        # Process filtered data
        if 'postedDate' not in filtered_df.columns:
            return pd.DataFrame()
        
        # Drop rows with NaT values in postedDate
        filtered_df = filtered_df.dropna(subset=['postedDate'])
        
        # If dataframe is empty after filtering, return empty dataframe
        if len(filtered_df) == 0:
            return pd.DataFrame()
            
        # Create year_month column
        filtered_df['year_month'] = filtered_df['postedDate'].dt.strftime('%Y-%m')
        
        # Count listings by month
        monthly_counts = filtered_df['year_month'].value_counts().reset_index()
        monthly_counts.columns = ['year_month', 'count']
        monthly_counts = monthly_counts.sort_values('year_month')
        
        # Convert to datetime for proper ordering
        monthly_counts['date'] = pd.to_datetime(monthly_counts['year_month'] + '-01')
        
        return monthly_counts
    
    # Callback for updating time series chart based on location selection
    @app.callback(
        Output('time-series-chart', 'figure'),
        [Input('location-time-dropdown', 'value'),
         Input('refresh-trigger', 'data')]
    )
    def update_time_series_chart(selected_location,refresh_trigger):
        if job_data is None or 'postedDate' not in job_data.columns:
            return go.Figure().update_layout(title="Job data not available or missing date information")
        
        # Set up the figure
        fig = go.Figure()
        
        # Get the data for the selected location
        monthly_data = get_monthly_counts(job_data, selected_location)
        
        if len(monthly_data) == 0:
            return go.Figure().update_layout(title="No time series data available for the selected criteria")
        
        # Add trace for the selected location
        fig.add_trace(go.Scatter(
            x=monthly_data['date'],
            y=monthly_data['count'],
            mode='lines+markers+text',
            text=monthly_data['count'],
            textposition='top center',
            name=selected_location,
            line=dict(width=3)
        ))
        
        # Add range slider and time series features
        fig.update_layout(
            title={
                'text': f'Number of Job Listings by Month: {selected_location}',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            ),
            yaxis_title="Number of Listings",
            height=600
        )
        
        return fig
    
    # Callback for updating monthly trend metrics
    @app.callback(
        Output('monthly-trend-metrics', 'children'),
        [Input('location-time-dropdown', 'value'),
         Input('refresh-trigger', 'data')]
    )
    def update_monthly_trend_metrics(selected_location, refresh_trigger):
        if job_data is None or 'postedDate' not in job_data.columns:
            return html.Div("Job data not available or missing date information")
        
        # Get the data for the selected location
        monthly_data = get_monthly_counts(job_data, selected_location)
        
        if len(monthly_data) == 0:
            return html.Div("No time series data available for the selected criteria")
        
        # Calculate metrics
        try:
            total_jobs = monthly_data['count'].sum()
            avg_monthly = monthly_data['count'].mean()
            max_month = monthly_data.loc[monthly_data['count'].idxmax()]
            min_month = monthly_data.loc[monthly_data['count'].idxmin()]
            
            # Calculate growth rate (comparing last 3 months to previous 3 months)
            if len(monthly_data) >= 6:
                last_3_months = monthly_data.tail(3)['count'].sum()
                prev_3_months = monthly_data.tail(6).head(3)['count'].sum()
                growth_rate = ((last_3_months / prev_3_months) - 1) * 100 if prev_3_months > 0 else 0
            else:
                growth_rate = None
            
            # Create metrics display
            metrics_cards = html.Div([
                html.Div([
                    html.Div([
                        html.H3("Total Jobs", style={'textAlign': 'center', 'color': '#7f8c8d'}),
                        html.H2(f"{total_jobs:,}", style={'textAlign': 'center', 'color': '#2c3e50'})
                    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '5px', 
                            'boxShadow': '2px 2px 2px lightgrey', 'flex': '1', 'margin': '10px'}),
                    
                    html.Div([
                        html.H3("Avg. Monthly Jobs", style={'textAlign': 'center', 'color': '#7f8c8d'}),
                        html.H2(f"{avg_monthly:.1f}", style={'textAlign': 'center', 'color': '#2c3e50'})
                    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '5px', 
                            'boxShadow': '2px 2px 2px lightgrey', 'flex': '1', 'margin': '10px'}),
                    
                    html.Div([
                        html.H3("Peak Month", style={'textAlign': 'center', 'color': '#7f8c8d'}),
                        html.H2(f"{max_month['year_month']}", 
                                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '5px'}),
                        html.H3(f"({max_month['count']} jobs)", 
                                style={'textAlign': 'center', 'color': '#2c3e50', 'marginTop': '0px'})
                    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '5px', 
                            'boxShadow': '2px 2px 2px lightgrey', 'flex': '1', 'margin': '10px'}),
                    
                    html.Div([
                        html.H3("Recent Growth Rate", style={'textAlign': 'center', 'color': '#7f8c8d'}),
                        html.H2(f"{growth_rate:.1f}%" if growth_rate is not None else "N/A", 
                                style={'textAlign': 'center', 
                                      'color': '#27ae60' if growth_rate is not None and growth_rate > 0 else 
                                              '#e74c3c' if growth_rate is not None and growth_rate < 0 else '#2c3e50'})
                    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '5px', 
                            'boxShadow': '2px 2px 2px lightgrey', 'flex': '1', 'margin': '10px'})
                ], style={'display': 'flex', 'flexWrap': 'wrap'})
            ])
            
            return metrics_cards
            
        except Exception as e:
            return html.Div(f"Error calculating metrics: {str(e)}")
    
    #------- CALLBACKS FOR EMPLOYER-INDUSTRY TAB -------#
    
    # Function to get filtered data for a location
    def get_location_filtered_data(df, selected_location=None):
        if df is None:
            return pd.DataFrame()
            
        # Filter dataframe based on selected location
        if selected_location and selected_location != "All Locations":
            mask = df['locations'].apply(
                lambda x: selected_location in extract_locations(x) if isinstance(x, str) else False
            )
            filtered_df = df[mask].copy()
        else:
            filtered_df = df.copy()
            
        return filtered_df
    
    # Callback for updating top employers chart
    @app.callback(
        Output('top-employers-chart', 'figure'),
        [Input('employer-location-dropdown', 'value'),
         Input('employer-count-slider', 'value'),
         Input('refresh-trigger', 'data')]
    )
    def update_top_employers_chart(selected_location, n_employers, refresh_trigger):
        if job_data is None or employer_column is None or employer_column not in job_data.columns:
            return go.Figure().update_layout(
                title="Employer data not available", 
                annotations=[dict(
                    text=f"To use this feature, job data must include the '{employer_column}' column",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5
                )]
            )
        
        # Get filtered data for selected location
        filtered_data = get_location_filtered_data(job_data, selected_location)
        
        if len(filtered_data) == 0:
            return go.Figure().update_layout(title=f"No job listings found for {selected_location}")
        
        # Count employers
        employer_counts = filtered_data[employer_column].value_counts().head(n_employers)
        employer_df = pd.DataFrame({
            'Employer': employer_counts.index,
            'Count': employer_counts.values
        })
        
        # Create horizontal bar chart
        fig = px.bar(
            employer_df,
            y='Employer',
            x='Count',
            orientation='h',
            labels={'Count': 'Number of Job Listings', 'Employer': 'Employer'},
            title=f'Top {n_employers} Employers in {selected_location if selected_location != "All Locations" else "All Locations"}',
            color='Count',
            color_continuous_scale=px.colors.sequential.Blues,
            text='Count'
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},  # Sort bars by value
            font=dict(family="Arial, sans-serif", size=12),
            hoverlabel=dict(font_size=12, font_family="Arial, sans-serif"),
            margin=dict(t=50, b=50, l=200, r=50),  # Increase left margin for employer names
            height=500
        )
        
        fig.update_traces(
            texttemplate='%{text:,}',
            textposition='outside'
        )
        
        return fig
    
    # Callback for updating industry pie chart
    @app.callback(
        Output('industry-pie-chart', 'figure'),
        [Input('employer-location-dropdown', 'value'),
         Input('refresh-trigger', 'data')]
    )
    def update_industry_pie_chart(selected_location, refresh_trigger):
        if job_data is None or industry_column is None:
            return go.Figure().update_layout(
                title="Industry data not available", 
                annotations=[dict(
                    text="To use this feature, job data must include industry information",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5
                )]
            )
        
        # Get filtered data for selected location
        filtered_data = get_location_filtered_data(job_data, selected_location)
        
        if len(filtered_data) == 0:
            return go.Figure().update_layout(title=f"No job listings found for {selected_location}")
        
        # Count industries
        industry_counts = filtered_data[industry_column].value_counts().head(10)  # Top 10 industries
        industry_df = pd.DataFrame({
            'Industry': industry_counts.index,
            'Count': industry_counts.values
        })
        
        # Create pie chart
        fig = px.pie(
            industry_df,
            values='Count',
            names='Industry',
            title=f'Top Industries in {selected_location if selected_location != "All Locations" else "All Locations"}',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        fig.update_layout(
            font=dict(family="Arial, sans-serif", size=12),
            hoverlabel=dict(font_size=12, font_family="Arial, sans-serif"),
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        fig.update_traces(
            textinfo='percent+label',
            textposition='inside',
            hoverinfo='label+percent+value'
        )
        
        return fig
    
    # Callback for updating employer-industry heatmap
    @app.callback(
        Output('employer-industry-heatmap', 'figure'),
        [Input('employer-location-dropdown', 'value'),
         Input('refresh-trigger', 'data')]
    )
    def update_employer_industry_heatmap(selected_location, refresh_trigger):
        if job_data is None or employer_column is None or employer_column not in job_data.columns or industry_column is None:
            missing_column = "companyName" if employer_column not in job_data.columns else "industry"
            return go.Figure().update_layout(
                title="Employer-Industry data not available", 
                annotations=[dict(
                    text=f"To use this feature, job data must include both '{employer_column}' and industry columns (missing: {missing_column})",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5
                )]
            )
        
        # Get filtered data for selected location
        filtered_data = get_location_filtered_data(job_data, selected_location)
        
        if len(filtered_data) == 0:
            return go.Figure().update_layout(title=f"No job listings found for {selected_location}")
        
        # Get top 10 employers and industries
        top_employers = filtered_data[employer_column].value_counts().head(10).index.tolist()
        top_industries = filtered_data[industry_column].value_counts().head(10).index.tolist()
        
        # Filter data to only include top employers and industries
        heatmap_data = filtered_data[
            (filtered_data[employer_column].isin(top_employers)) & 
            (filtered_data[industry_column].isin(top_industries))
        ]
        
        if len(heatmap_data) == 0:
            return go.Figure().update_layout(title="Not enough data to generate heatmap")
        
        # Create a cross-tabulation of employer and industry
        heatmap_df = pd.crosstab(
            heatmap_data[employer_column], 
            heatmap_data[industry_column]
        )
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_df.values,
            x=heatmap_df.columns,
            y=heatmap_df.index,
            colorscale='Viridis',
            hoverongaps=False,
            text=heatmap_df.values,
            texttemplate="%{text}",
        ))
        
        fig.update_layout(
            title=f'Employer-Industry Distribution in {selected_location if selected_location != "All Locations" else "All Locations"}',
            font=dict(family="Arial, sans-serif", size=12),
            hoverlabel=dict(font_size=12, font_family="Arial, sans-serif"),
            xaxis_title="Industry",
            yaxis_title="Employer",
            margin=dict(t=50, b=100, l=200, r=50),
            height=700
        )
        
        # Rotate x-axis labels for better readability
        fig.update_xaxes(tickangle=45)
        
        return fig
    # Set default tab to be the first tab (Location Analysis)
    # This ensures the Skills Analysis tab isn't shown first
    app.layout.children[1].value = "location-tab"

    #------- CALLBACKS FOR QUALIFICATION TAB -------#
    
    # NLP Modle callback
    @app.callback(
        Output('nlp-output', 'children'),
        [Input('nlp-extract-button', 'n_clicks'),
         Input('refresh-trigger', 'data')],
        State('nlp-input-text', 'value')
    )
    def extract_degree_from_input(n_clicks, refresh_trigger, input_text):
        if n_clicks is None or not input_text:
            return "Please enter the text and click the button"

        result = extract_all_qualifications(input_text)

        degree_text = ", ".join(result['degree']) if result['degree'] else "Degree level not identified"
        tech_text = ", ".join(result['technical_skills']) if result['technical_skills'] else "No technical skills identified"
        soft_text = ", ".join(result['soft_skills']) if result['soft_skills'] else "Soft skills not identified"

        return html.Div([
            html.P(f"🎓 Degree: {degree_text}"),
            html.P(f"🛠️ technical skill: {tech_text}"),
            html.P(f"💬 soft skill: {soft_text}")
        ])
    
    # Job Matcher Callbacks
    
    # Callback for displaying selected skills
    @app.callback(
        Output('selected-skills-display', 'children'),
        Input('tech-skills-dropdown', 'value')
    )
    def display_selected_skills(selected_skills):
        if not selected_skills:
            return None
        
        # Filter out category headers
        actual_skills = [s for s in selected_skills if not s.startswith('__category__')]
        
        if not actual_skills:
            return None
        
        # Group skills by category
        skills_by_category = {}
        for category, skills in skills_dict.items():
            category_skills = [s for s in actual_skills if s in skills]
            if category_skills:
                skills_by_category[category] = category_skills
        
        # Display grouped skills
        skill_badges = []
        for category, skills in skills_by_category.items():
            skill_badges.append(
                html.Div([
                    html.Span(f"{category}: ", style={'fontWeight': 'bold', 'color': '#7f8c8d'}),
                    html.Span([
                        dbc.Badge(skill, color="info", className="me-1") 
                        for skill in skills
                    ])
                ], style={'marginBottom': '5px'})
            )
        
        return html.Div([
            html.Hr(),
            html.H6("Selected Skills:", style={'marginBottom': '10px'}),
            *skill_badges
        ])

    # Main callback for job matching
    @app.callback(
        [Output('match-results-store', 'data'),
         Output('tab-content', 'children'),
         Output('current-page', 'data')],
        [Input('search-button', 'n_clicks'),
         Input('match-tabs', 'active_tab'),
         Input({'type': 'pagination', 'index': ALL}, 'active_page')],
        [State('degree-dropdown', 'value'),
         State('tech-skills-dropdown', 'value'),
         State('match-results-store', 'data'),
         State('current-page', 'data')],
        prevent_initial_call=True
    )
    def update_job_matches(n_clicks, active_tab, page_clicks, selected_degree, selected_tech_skills, 
                          stored_results, current_page):
        ctx = callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Handle pagination
        if 'pagination' in trigger_id:
            if page_clicks and len(page_clicks) > 0:
                current_page = page_clicks[0]
        
        # If search button clicked, calculate new results
        if trigger_id == 'search-button':
            current_page = 1  # Reset to first page on new search
            
            if quali_data is None:
                return None, html.Div("Unable to load data", style={'textAlign': 'center', 'padding': '20px'}), current_page
            
            # Filter out category headers
            actual_skills = [s for s in selected_tech_skills if not s.startswith('__category__')] if selected_tech_skills else []
            
            if not selected_degree and not actual_skills:
                return None, dbc.Alert("Please select at least education level or technical skills", color="warning"), current_page
            
            # Calculate matches for all jobs
            results = {
                'strict_matches': [],
                'extended_matches': {'excellent': [], 'good': [], 'fair': []}
            }
            
            for idx, row in quali_data.iterrows():
                # Check degree requirement
                if selected_degree:
                    degree_hierarchy = {
                        '1_No Requirement': 0, '2_Student': 1, '3_High School': 2,
                        '4_Pursuing Bachelor\'s': 3, '5_Bachelor\'s': 4,
                        '6_Pursuing Master\'s': 5, '6_Bachelor\'s/Master\'s': 5.5,
                        '7_Master\'s': 6, '8_Pursuing PhD': 7, '9_PhD': 8,
                        '10_Field Specific': 4, '11_Other': 2
                    }
                    
                    user_level = degree_hierarchy.get(selected_degree, 0)
                    job_level = degree_hierarchy.get(row['Degree_Clean'], 0)
                    
                    if user_level < job_level:
                        continue  # Skip if education not met
                
                # Calculate skill match
                if actual_skills:
                    job_skills = row.get('Technical_Skills_Normalized', row['Technical_Skills_List'])
                    if not job_skills:
                        continue
                    
                    # Normalize user skills for better matching
                    user_skills_normalized = [normalize_skill_name(s) for s in actual_skills]
                    
                    # Find matches
                    matched_skills = [skill for skill in user_skills_normalized if skill in job_skills]
                    unmatched_skills = [skill for skill in user_skills_normalized if skill not in job_skills]
                    
                    match_ratio = len(matched_skills) / len(user_skills_normalized) if user_skills_normalized else 0
                    
                    job_info = {
                        'index': idx,
                        'degree': row['Degree_Clean'],
                        'job_skills': job_skills,
                        'matched_skills': matched_skills,
                        'unmatched_skills': unmatched_skills,
                        'match_ratio': match_ratio,
                        'experience_min': row['Experience_Years_Min'],
                        'experience_max': row['Experience_Years_Max'],
                        'soft_skills': row['Soft_Skills_Keywords'],
                        'language': row.get('Language', 'None')
                    }
                    
                    # Categorize match
                    if match_ratio == 1.0:  # 100% match - strict
                        results['strict_matches'].append(job_info)
                    elif match_ratio >= 0.8:  # 80%+ - excellent
                        results['extended_matches']['excellent'].append(job_info)
                    elif match_ratio >= 0.6:  # 60-79% - good
                        results['extended_matches']['good'].append(job_info)
                    elif match_ratio >= 0.4:  # 40-59% - fair
                        results['extended_matches']['fair'].append(job_info)
            
            stored_results = results
        
        # Display results based on active tab
        if not stored_results:
            return stored_results, html.Div("Please search first", style={'textAlign': 'center', 'padding': '40px'}), current_page
        
        if active_tab == "strict-match":
            return stored_results, display_strict_matches_paginated(stored_results['strict_matches'], job_data, current_page), current_page
        else:
            return stored_results, display_extended_matches_paginated(stored_results['extended_matches'], job_data, current_page), current_page
    
    return app

# ============================ MAIN CODE FOR EXECUTION ============================

def main():
    """Main function to run the combined dashboard"""
    # Create the combined dashboard
    app= create_combined_dashboard()
    
   
    port = int(os.environ.get("PORT", 10000))

    app.run(host="0.0.0.0", port=port, debug=False)

if __name__ == "__main__":
    main()

