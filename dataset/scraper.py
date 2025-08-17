import requests
import pandas as pd
from bs4 import BeautifulSoup as bs
import time
import re
from urllib.parse import urljoin, urlparse

# Configuration
pages = 5  # Number of pages to scrape
base_url = "https://www.jobberman.com.gh"

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

def clean_text(text):
    """Clean and normalize text content"""
    if not text:
        return None
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text.strip())
    return text if text else None

def extract_job_title_from_info(info_text):
    """Extract job title from the info text"""
    if not info_text:
        return None
    
    # Look for patterns like job titles at the beginning
    lines = [line.strip() for line in info_text.split('\n') if line.strip()]
    
    # Find the first substantial line that looks like a job title
    for line in lines:
        # Skip common navigation/UI elements
        skip_patterns = [
            'subscribe', 'log in', 'jobs in ghana', 'job function',
            'any job function', 'email address', 'password', 'keep me logged'
        ]
        
        if (len(line) > 5 and len(line) < 100 and 
            not any(skip.lower() in line.lower() for skip in skip_patterns) and
            not line.lower().startswith('job function') and
            not re.match(r'^\d+\s+jobs\s+found', line, re.IGNORECASE)):
            return clean_text(line)
    
    return None

def extract_company_from_info(info_text):
    """Extract company name from info text"""
    if not info_text:
        return None
    
    lines = [line.strip() for line in info_text.split('\n') if line.strip()]
    
    # Look for company indicators
    company_indicators = [
        'ltd', 'limited', 'company', 'corp', 'corporation', 'inc',
        'group', 'enterprise', 'services', 'solutions', 'technologies'
    ]
    
    for line in lines:
        # Skip very short or very long lines
        if len(line) < 3 or len(line) > 80:
            continue
            
        # Skip obvious non-company lines
        skip_patterns = [
            'subscribe', 'log in', 'jobs in ghana', 'job function',
            'full time', 'part time', 'confidential', 'accra', 'kumasi',
            'region', 'email', 'password'
        ]
        
        if any(skip.lower() in line.lower() for skip in skip_patterns):
            continue
        
        # Check if line contains company indicators or looks like a company name
        if (any(indicator in line.lower() for indicator in company_indicators) or
            (line.count(' ') >= 1 and line.count(' ') <= 4 and 
             line[0].isupper() and not line.endswith(':'))):
            return clean_text(line)
    
    return None

def extract_location_from_info(info_text):
    """Extract location from info text"""
    if not info_text:
        return None
    
    # Ghana locations and regions
    ghana_locations = [
        'accra', 'tema', 'kumasi', 'tamale', 'cape coast', 'takoradi',
        'ho', 'koforidua', 'sunyani', 'wa', 'bolgatanga',
        'greater accra', 'ashanti', 'northern', 'western', 'central',
        'eastern', 'volta', 'upper east', 'upper west', 'brong ahafo'
    ]
    
    info_lower = info_text.lower()
    for location in ghana_locations:
        if location in info_lower:
            # Extract the full location phrase
            pattern = rf'([^.\n]*{re.escape(location)}[^.\n]*(?:region)?)'
            match = re.search(pattern, info_text, re.IGNORECASE)
            if match:
                location_text = clean_text(match.group(1))
                # Clean up common patterns
                location_text = re.sub(r'^\s*&\s*', '', location_text)
                return location_text
    
    return None

def extract_job_type_from_info(info_text):
    """Extract job type from info text"""
    if not info_text:
        return "Full Time"
    
    job_types = ['full time', 'part time', 'contract', 'temporary', 'internship']
    
    for job_type in job_types:
        if job_type in info_text.lower():
            return job_type.title()
    
    return "Full Time"

def extract_salary_from_info(info_text):
    """Extract salary information from info text"""
    if not info_text:
        return None
    
    # Look for salary patterns
    salary_patterns = [
        r'GHS\s*[\d,]+(?:\s*-\s*[\d,]+)?',
        r'₵\s*[\d,]+(?:\s*-\s*[\d,]+)?',
        r'NGN\s*[\d,]+(?:\s*-\s*[\d,]+)?',
        r'₦\s*[\d,]+(?:\s*-\s*[\d,]+)?',
        r'\b[\d,]+\s*-\s*[\d,]+\s*(?:ghana cedis|cedis|ghs)\b'
    ]
    
    for pattern in salary_patterns:
        match = re.search(pattern, info_text, re.IGNORECASE)
        if match:
            return clean_text(match.group(0))
    
    # Check if salary is mentioned as confidential
    if 'confidential' in info_text.lower():
        return 'Confidential'
    
    return None

def extract_experience_and_qualifications(info_text, job_url=None):
    """Extract experience and qualification details"""
    details = []
    info_details = []
    
    if not info_text and not job_url:
        return details, info_details
    
    # If we have a job URL, try to get more details
    if job_url:
        try:
            response = requests.get(job_url, headers=headers, timeout=10)
            if response.status_code == 200:
                job_soup = bs(response.content, 'html.parser')
                job_text = job_soup.get_text()
                
                # Extract requirements and descriptions
                requirement_sections = []
                
                # Look for common requirement section indicators
                req_indicators = [
                    'requirements', 'qualifications', 'minimum qualification',
                    'experience required', 'job requirements', 'what we need'
                ]
                
                for indicator in req_indicators:
                    pattern = rf'{re.escape(indicator)}[:\s]*([^.]*(?:\.[^.]*{{0,3}})?)'
                    match = re.search(pattern, job_text, re.IGNORECASE)
                    if match:
                        requirement_sections.append(clean_text(match.group(1)))
                
                # Extract experience years
                exp_patterns = [
                    r'(\d+)\s*(?:\+)?\s*years?\s+(?:of\s+)?experience',
                    r'minimum\s+of\s+(\d+)\s*years?',
                ]
                
                for pattern in exp_patterns:
                    match = re.search(pattern, job_text, re.IGNORECASE)
                    if match:
                        years = match.group(1)
                        details.append(f'Experience Length: {years} years')
                        
                        # Determine experience level
                        years_int = int(years)
                        if years_int <= 2:
                            details.append('Experience Level: Entry level')
                        elif years_int <= 7:
                            details.append('Experience Level: Mid level')
                        else:
                            details.append('Experience Level: Executive level')
                        break
                
                # Extract education requirements
                edu_keywords = [
                    'bachelor', 'master', 'mba', 'phd', 'degree', 'diploma',
                    'hnd', 'ond', 'certificate', 'qualification'
                ]
                
                for keyword in edu_keywords:
                    if keyword.lower() in job_text.lower():
                        if keyword.lower() in ['bachelor', 'master', 'mba', 'phd', 'degree']:
                            details.append('Minimum Qualification: Degree')
                        elif keyword.lower() in ['hnd', 'ond']:
                            details.append('Minimum Qualification: HND')
                        elif keyword.lower() == 'diploma':
                            details.append('Minimum Qualification: Diploma')
                        break
                
                # Get job description sections
                if requirement_sections:
                    info_details.extend(requirement_sections)
                else:
                    # Fallback: get first few meaningful paragraphs
                    paragraphs = job_soup.find_all(['p', 'div'], string=True)
                    for p in paragraphs[:3]:
                        text = clean_text(p.get_text())
                        if text and len(text) > 50:
                            info_details.append(text)
                
                time.sleep(1)  # Rate limiting
                
        except Exception as e:
            print(f"Error fetching details from {job_url}: {e}")
    
    # Fallback: extract from info_text if available
    if not details and info_text:
        if 'experience' in info_text.lower():
            details.append('Experience Level: Mid level')  # Default assumption
        if any(edu in info_text.lower() for edu in ['degree', 'bachelor', 'master']):
            details.append('Minimum Qualification: Degree')
    
    return details, info_details

def extract_date_posted(info_text):
    """Extract date posted from info text"""
    if not info_text:
        return None
    
    # Look for date posted patterns
    # Match patterns like "2 days ago", "1 week ago"
    match = re.search(r'(\d+\s+(?:day|week|month)s?\s+ago)', info_text, re.I)
    return clean_text(match.group(1)) if match else None

def extract_industry(info_text):
    """Extract industry from info text"""
    if not info_text:
        return None
    match = re.search(r'Job Function[:\s]*([^\n]+)', info_text, re.I)
    if match:
        return clean_text(match.group(1))
    return None


def process_extracted_jobs(raw_jobs):
    """Process and clean the raw job data"""
    processed_jobs = []
    
    
    print(f"Processing {len(raw_jobs)} raw job entries...")
    
    for i, job in enumerate(raw_jobs):
        # Skip obvious non-jobs
        if not job.get('job_url') or 'oauth' in job.get('job_url', ''):
            continue
        
        info_text = job.get('info', [''])[0] if job.get('info') else ''
        
        # Extract clean data
        processed_job = {
            'name': None,
            'job_url': job.get('job_url'),
            'hiring_firm': None,
            'hiring_firm_url': None,
            'job_function': job.get('job_function'),
            'title': None,
            'date_posted': None,
            'Location': None,
            'Job type': 'Full Time',
            'Industry': None,
            'Salary': None,
            'details': [],
            'info': []
        }
        
        # Extract title (try multiple sources)
        title = (extract_job_title_from_info(info_text) or 
                job.get('name') or 
                job.get('title'))
        
        if title and 'subscribe' not in title.lower():
            processed_job['name'] = title
            processed_job['title'] = title
        
        # Extract company
        company = extract_company_from_info(info_text)
        if company:
            processed_job['hiring_firm'] = company
            # Create company search URL
            company_encoded = company.replace(' ', '%20')
            processed_job['hiring_firm_url'] = f"{base_url}/jobs?q={company_encoded}"
        
        # Extract other details
        processed_job['date_posted'] = extract_date_posted(info_text)
        processed_job['Industry'] = extract_industry(info_text)
        processed_job['Location'] = extract_location_from_info(info_text)
        processed_job['Job type'] = extract_job_type_from_info(info_text)
        processed_job['Salary'] = extract_salary_from_info(info_text)
        
        # Extract experience and qualifications (with job page details)
        details, info_details = extract_experience_and_qualifications(
            info_text, processed_job['job_url']
        )
        processed_job['details'] = details
        processed_job['info'] = info_details
        
        # Only add jobs with meaningful titles
        if processed_job['name'] and len(processed_job['name']) > 3:
            processed_jobs.append(processed_job)
            print(f"  ✓ Processed: {processed_job['name']} at {processed_job['hiring_firm']}")
        else:
            print(f"  ✗ Skipped job {i+1}: insufficient data")
    
    return processed_jobs

# Main execution
def main():
    print("Ghana Jobberman Job Scraper")
    print("=" * 40)
    
    all_jobs = []
    
    # Test connection
    try:
        test_response = requests.get(f"{base_url}/jobs", headers=headers, timeout=10)
        test_response.raise_for_status()
        print(f"✓ Connected successfully (Status: {test_response.status_code})")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return
    
    # Scrape multiple pages
    for page in range(1, pages + 1):
        print(f"\nScraping page {page}...")
        
        try:
            response = requests.get(f"{base_url}/jobs?page={page}", headers=headers, timeout=10)
            response.raise_for_status()
            soup = bs(response.content, 'html.parser')
            
            # Use the container-based extraction that worked in diagnostic
            containers = soup.find_all(['div', 'article', 'section'], 
                                     class_=re.compile(r'job|listing|card|item', re.I))
            
            page_jobs = []
            for container in containers:
                job_data = {
                    'name': None,
                    'job_url': None,
                    'hiring_firm': None,
                    'job_function': None,
                    'info': []
                }
                
                # Extract job URL
                link = container.find('a', href=True)
                if link:
                    href = link['href']
                    if href.startswith('/'):
                        job_data['job_url'] = urljoin(base_url, href)
                    elif href.startswith('http'):
                        job_data['job_url'] = href
                
                # Extract job function
                container_text = container.get_text()
                if 'Job Function' in container_text:
                    job_func_match = re.search(r'Job Function[:\s]*([^\n]+)', container_text)
                    if job_func_match:
                        job_data['job_function'] = f"Job Function: {job_func_match.group(1).strip()}"
                
                # Store raw container text for processing
                job_data['info'] = [container_text]
                
                # Only add if we have a URL
                if job_data['job_url']:
                    page_jobs.append(job_data)
            
            print(f"  Found {len(page_jobs)} raw job entries")
            all_jobs.extend(page_jobs)
            
            time.sleep(2)  # Rate limiting
            
        except Exception as e:
            print(f"  Error on page {page}: {e}")
            continue
    
    print(f"\nTotal raw jobs collected: {len(all_jobs)}")
    
    # Process and clean the jobs
    if all_jobs:
        processed_jobs = process_extracted_jobs(all_jobs)
        
        if processed_jobs:
            # Create DataFrame
            df = pd.DataFrame(processed_jobs)
            
            # Ensure column order
            desired_columns = [
                'name', 'job_url', 'hiring_firm', 'hiring_firm_url', 'job_function',
                'title', 'date_posted', 'Location', 'Job type', 'Industry',
                'Salary', 'details', 'info'
            ]
            
            for col in desired_columns:
                if col not in df.columns:
                    df[col] = None
            
            df = df[desired_columns]
            
            # Save to CSV
            df.to_csv("ghana_jobs_cleaned.csv", index=False)
            print(f"\n✓ Saved {len(df)} cleaned jobs to 'ghana_jobs_cleaned.csv'")
            
            # Save to JSON
            df.to_json("ghana_jobs_cleaned.json", orient='records', lines=True)
            print(f"\n✓ Saved {len(df)} cleaned jobs to 'ghana_jobs_cleaned.json'")
            
            # Show summary
            print(f"\nData Summary:")
            print(f"  Jobs with titles: {df['name'].notna().sum()}")
            print(f"  Jobs with companies: {df['hiring_firm'].notna().sum()}")
            print(f"  Jobs with locations: {df['Location'].notna().sum()}")
            print(f"  Jobs with salaries: {df['Salary'].notna().sum()}")
            
        else:
            print("❌ No jobs could be processed successfully")
    else:
        print("❌ No raw jobs were collected")

if __name__ == "__main__":
    main()