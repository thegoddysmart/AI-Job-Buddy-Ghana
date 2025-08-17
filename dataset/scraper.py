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

def extract_job_description(job_url):
    """Extract detailed job description from individual job page"""
    print(f"    Fetching details from: {job_url}")
    
    try:
        response = requests.get(job_url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = bs(response.content, 'html.parser')
        
        job_details = {
            'job_description': None,
            'requirements': None,
            'responsibilities': None,
            'qualifications': None,
            'benefits': None,
            'company_info': None,
            'application_deadline': None,
            'employment_type': None,
            'experience_required': None,
            'education_required': None,
            'skills_required': None,
            'salary_range': None,
            'location_details': None
        }
        
        # Extract main job description content
        description_selectors = [
            '.job-description',
            '.job-details',
            '#job-description',
            '[class*="description"]',
            '[class*="details"]',
            '.content',
            '.job-content',
            'main',
            '.main-content'
        ]
        
        description_content = ""
        for selector in description_selectors:
            elements = soup.select(selector)
            if elements:
                for elem in elements:
                    text = elem.get_text(separator='\n', strip=True)
                    if len(text) > 30:  # Only consider substantial content
                        description_content += text + "\n\n"
        
        # If no specific selectors work, try to find the main content area
        if not description_content:
            # Look for divs with substantial text content
            all_divs = soup.find_all(['div', 'section', 'article'])
            for div in all_divs:
                text = div.get_text(strip=True)
                if len(text) > 200 and any(keyword in text.lower() for keyword in 
                                         ['responsibilities', 'requirements', 'qualifications', 'description']):
                    description_content += text + "\n\n"
                    break
        
        job_details['job_description'] = clean_text(description_content) if description_content else None
        
        # Extract specific sections
        sections_map = {
            'responsibilities': ['responsibilities', 'duties', 'role', 'what you will do'],
            'requirements': ['requirements', 'must have', 'essential', 'required'],
            'qualifications': ['qualifications', 'education', 'academic', 'degree'],
            'benefits': ['benefits', 'what we offer', 'perks', 'package'],
            'skills_required': ['skills', 'competencies', 'abilities', 'technical'],
        }
        
        # Try to extract structured information
        all_text = soup.get_text().lower()
        
        for field, keywords in sections_map.items():
            for keyword in keywords:
                # Look for sections that start with these keywords
                pattern = rf'{keyword}[:\s]*([^\.]*(?:\.[^\.]*)*)'
                matches = re.findall(pattern, all_text, re.IGNORECASE | re.MULTILINE)
                if matches:
                    job_details[field] = clean_text(matches[0][:500])  # Limit to 500 chars
                    break
        
        # Extract specific job metadata
        metadata_patterns = {
            'salary_range': [
                r'salary[:\s]*([^\.]*(?:ghs|₵|confidential)[^\.]*)',
                r'compensation[:\s]*([^\.]*(?:ghs|₵|confidential)[^\.]*)',
                r'pay[:\s]*([^\.]*(?:ghs|₵|confidential)[^\.]*)'
            ],
            'employment_type': [
                r'employment type[:\s]*([^\.]*(?:full|part|contract|temporary)[^\.]*)',
                r'job type[:\s]*([^\.]*(?:full|part|contract|temporary)[^\.]*)'
            ],
            'experience_required': [
                r'experience[:\s]*([^\.]*\d+[^\.]*year[^\.]*)',
                r'(\d+\s*(?:\+)?\s*years?\s+(?:of\s+)?experience)'
            ],
            'application_deadline': [
                r'deadline[:\s]*([^\.]*\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}[^\.]*)',
                r'apply before[:\s]*([^\.]*\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}[^\.]*)'
            ]
        }
        
        for field, patterns in metadata_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, all_text, re.IGNORECASE)
                if match:
                    job_details[field] = clean_text(match.group(1)[:200])  # Limit to 200 chars
                    break
        
        # Extract company information
        company_selectors = [
            '.company-info',
            '.company-description',
            '[class*="company"]',
            '.employer-info'
        ]
        
        for selector in company_selectors:
            company_elem = soup.select_one(selector)
            if company_elem:
                company_text = company_elem.get_text(strip=True)
                if len(company_text) > 50:
                    job_details['company_info'] = clean_text(company_text[:300])
                    break
        
        # Add delay to be respectful
        time.sleep(1)
        
        return job_details
        
    except Exception as e:
        print(f"    ✗ Error fetching job details: {e}")
        return {
            'job_description': None,
            'requirements': None,
            'responsibilities': None,
            'qualifications': None,
            'benefits': None,
            'company_info': None,
            'application_deadline': None,
            'employment_type': None,
            'experience_required': None,
            'education_required': None,
            'skills_required': None,
            'salary_range': None,
            'location_details': None
        }

def extract_job_details_from_container(container):
    """Extract all possible job information from a container element"""
    job_data = {
        'name': None,
        'job_url': None,
        'hiring_firm': None,
        'hiring_firm_url': None,
        'job_function': None,
        'title': None,
        'date_posted': None,
        'Location': None,
        'Job type': 'Full Time',
        'Industry': None,
        'Salary': None,
        'details': [],
        'info': []
    }
    
    # Get all text content
    container_text = container.get_text()
    
    # Extract job URL first
    link = container.find('a', href=True)
    if link:
        href = link['href']
        if href.startswith('/'):
            job_data['job_url'] = urljoin(base_url, href)
        elif href.startswith('http'):
            job_data['job_url'] = href
    
    # Parse the structured text content
    lines = [line.strip() for line in container_text.split('\n') if line.strip()]
    
    # Variables to track what we've found
    found_title = False
    found_company = False
    potential_title = None
    potential_company = None
    
    for i, line in enumerate(lines):
        line_clean = clean_text(line)
        if not line_clean or len(line_clean) < 2:
            continue
        
        # Skip common UI elements
        skip_patterns = [
            'subscribe', 'log in', 'jobs in ghana', 'any job function',
            'email address', 'password', 'keep me logged', 'google-icon',
            'or continue with', 'forgot password'
        ]
        
        if any(skip.lower() in line_clean.lower() for skip in skip_patterns):
            continue
        
        # Extract job function
        if line_clean.startswith('Job Function') and ':' in line_clean:
            job_function = line_clean.split(':', 1)[1].strip()
            job_data['job_function'] = f"Job Function: {job_function}"
        
        # Look for job titles (usually the first substantial line)
        if (not found_title and 
            len(line_clean) > 10 and len(line_clean) < 100 and
            not line_clean.lower().startswith('job function') and
            not any(char in line_clean for char in ['@', 'http', 'www']) and
            not re.match(r'^\d+\s+jobs\s+found', line_clean, re.IGNORECASE)):
            
            # Check if this looks like a job title
            job_title_indicators = [
                'manager', 'officer', 'executive', 'coordinator', 'assistant',
                'specialist', 'analyst', 'developer', 'engineer', 'designer',
                'consultant', 'advisor', 'supervisor', 'lead', 'head',
                'director', 'administrator', 'technician', 'representative'
            ]
            
            if any(indicator in line_clean.lower() for indicator in job_title_indicators):
                potential_title = line_clean
                found_title = True
        
        # Look for company names (often after job title)
        if (found_title and not found_company and
            len(line_clean) > 5 and len(line_clean) < 80 and
            line_clean != potential_title):
            
            # Company name indicators
            company_indicators = [
                'limited', 'ltd', 'company', 'corp', 'corporation', 'inc',
                'group', 'enterprise', 'services', 'solutions', 'technologies',
                'bank', 'university', 'school', 'hospital', 'clinic'
            ]
            
            # Check for anonymous/generic employers
            generic_employers = [
                'anonymous employer', "jobberman's client", 'reputable',
                'confidential', 'client jobs', 'well known'
            ]
            
            if (any(indicator in line_clean.lower() for indicator in company_indicators) or
                any(generic in line_clean.lower() for generic in generic_employers)):
                potential_company = line_clean
                found_company = True
        
        # Extract location information
        ghana_locations = [
            'accra', 'tema', 'kumasi', 'tamale', 'cape coast', 'takoradi',
            'ho', 'koforidua', 'sunyani', 'wa', 'bolgatanga',
            'greater accra', 'ashanti', 'northern', 'western', 'central',
            'eastern', 'volta', 'upper east', 'upper west', 'brong ahafo',
            'region'
        ]
        
        if any(location in line_clean.lower() for location in ghana_locations):
            if not job_data['Location']:  # Only set if we haven't found one yet
                job_data['Location'] = line_clean
        
        # Extract job type
        job_types = ['full time', 'part time', 'contract', 'temporary', 'internship']
        for job_type in job_types:
            if job_type in line_clean.lower():
                job_data['Job type'] = job_type.title()
        
        # Extract salary information
        if not job_data['Salary']:
            salary_patterns = [
                r'GHS\s*[\d,]+(?:\s*-\s*[\d,]+)?',
                r'₵\s*[\d,]+(?:\s*-\s*[\d,]+)?',
                r'confidential'
            ]
            
            for pattern in salary_patterns:
                if re.search(pattern, line_clean, re.IGNORECASE):
                    job_data['Salary'] = line_clean
                    break
    
    # Set the extracted values
    if potential_title:
        job_data['name'] = potential_title
        job_data['title'] = potential_title
    
    if potential_company:
        job_data['hiring_firm'] = potential_company
        # Create company search URL
        company_encoded = potential_company.replace(' ', '%20')
        job_data['hiring_firm_url'] = f"{base_url}/jobs?q={company_encoded}"
    
    # Extract additional details from the container text
    details = []
    info_sections = []
    
    # Look for qualification keywords in the text
    qualification_keywords = {
        'degree': 'Minimum Qualification: Degree',
        'bachelor': 'Minimum Qualification: Degree', 
        'master': 'Minimum Qualification: Degree',
        'mba': 'Minimum Qualification: Degree',
        'hnd': 'Minimum Qualification: HND',
        'ond': 'Minimum Qualification: OND',
        'diploma': 'Minimum Qualification: Diploma',
        'certificate': 'Minimum Qualification: Certificate'
    }
    
    container_text_lower = container_text.lower()
    for keyword, detail in qualification_keywords.items():
        if keyword in container_text_lower:
            details.append(detail)
            break  # Only add one qualification
    
    # Look for experience level indicators
    experience_indicators = {
        'entry level': 'Experience Level: Entry level',
        'graduate': 'Experience Level: Entry level',
        'fresh': 'Experience Level: Entry level',
        'senior': 'Experience Level: Executive level',
        'executive': 'Experience Level: Executive level',
        'manager': 'Experience Level: Executive level',
        'director': 'Experience Level: Executive level'
    }
    
    for indicator, detail in experience_indicators.items():
        if indicator in container_text_lower:
            details.append(detail)
            break  # Only add one experience level
    
    # If no experience level found, default to mid level
    if not any('Experience Level' in detail for detail in details):
        details.append('Experience Level: Mid level')
    
    # Extract experience years if mentioned
    exp_patterns = [
        r'(\d+)\s*(?:\+)?\s*years?\s+(?:of\s+)?experience',
        r'minimum\s+of\s+(\d+)\s*years?',
    ]
    
    for pattern in exp_patterns:
        match = re.search(pattern, container_text_lower)
        if match:
            years = match.group(1)
            details.append(f'Experience Length: {years} years')
            break
    
    # Create meaningful info sections
    if job_data['name']:
        info_sections.append(f"Job Title: {job_data['name']}")
    if job_data['hiring_firm']:
        info_sections.append(f"Company: {job_data['hiring_firm']}")
    if job_data['Location']:
        info_sections.append(f"Location: {job_data['Location']}")
    if job_data['Salary']:
        info_sections.append(f"Salary: {job_data['Salary']}")
    
    # Add job function info if available
    if job_data['job_function']:
        info_sections.append(job_data['job_function'])
    
    # Set details and info
    job_data['details'] = details
    job_data['info'] = info_sections
    
    return job_data

def extract_jobs_from_page(soup):
    """Extract job information from the current page"""
    jobs_data = []
    
    # Find job containers using multiple strategies
    containers = []
    
    # Strategy 1: Look for elements with job-related classes
    job_containers = soup.find_all(['div', 'article', 'section'], 
                                  class_=re.compile(r'job|listing|card|item', re.I))
    containers.extend(job_containers)
    
    # Strategy 2: Look for elements containing job URLs
    link_containers = soup.find_all('div', string=re.compile(r'listings/', re.I))
    containers.extend(link_containers)
    
    # Strategy 3: Look for elements with substantial text content that might be jobs
    if not containers:
        all_divs = soup.find_all('div')
        for div in all_divs:
            text = div.get_text()
            # Check if this div contains job-like content
            if (len(text) > 100 and 
                ('job function' in text.lower() or 'apply' in text.lower()) and
                any(word in text.lower() for word in ['manager', 'officer', 'executive', 'coordinator'])):
                containers.append(div)
    
    print(f"Found {len(containers)} potential job containers")
    
    for i, container in enumerate(containers):
        try:
            job_data = extract_job_details_from_container(container)
            
            # Only add jobs with meaningful data
            if (job_data['job_url'] and 
                (job_data['name'] or job_data['job_function']) and
                'oauth' not in job_data['job_url']):
                
                jobs_data.append(job_data)
                print(f"  ✓ Extracted: {job_data['name']} at {job_data['hiring_firm']}")
            
        except Exception as e:
            print(f"  ✗ Error processing container {i+1}: {e}")
            continue
    
    return jobs_data

def main():
    print("Enhanced Ghana Jobberman Job Scraper with Job Description Details")
    print("=" * 60)
    
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
            
            # Extract jobs from the current page
            page_jobs = extract_jobs_from_page(soup)
            
            all_jobs.extend(page_jobs)
            print(f"  Found {len(page_jobs)} jobs on page {page}")
            
            time.sleep(2)  # Rate limiting
            
        except Exception as e:
            print(f"  Error on page {page}: {e}")
            continue
    
    print(f"\nTotal jobs collected: {len(all_jobs)}")
    
    if all_jobs:
        print(f"\nFetching detailed job descriptions...")
        
        # Fetch detailed job descriptions
        enhanced_jobs = []
        for i, job in enumerate(all_jobs[:10]):  # Limit to first 10 jobs for testing
            print(f"  Processing job {i+1}/{min(10, len(all_jobs))}: {job.get('name', 'Unknown')}")
            
            # Get job description details
            job_details = extract_job_description(job['job_url'])
            
            # Merge the job details
            enhanced_job = {**job, **job_details}
            enhanced_jobs.append(enhanced_job)
            
            # Add delay between requests
            time.sleep(2)
        
        # Create DataFrame
        df = pd.DataFrame(enhanced_jobs)
        
        # Ensure column order with new detailed columns
        desired_columns = [
            'name', 'job_url', 'hiring_firm', 'hiring_firm_url', 'job_function',
            'title', 'date_posted', 'Location', 'Job type', 'Industry',
            'Salary', 'details', 'info',
            # New detailed columns
            'job_description', 'requirements', 'responsibilities', 'qualifications',
            'benefits', 'company_info', 'application_deadline', 'employment_type',
            'experience_required', 'education_required', 'skills_required',
            'salary_range', 'location_details'
        ]
        
        for col in desired_columns:
            if col not in df.columns:
                df[col] = None
        
        df = df[desired_columns]
        
        # Remove duplicates based on job URL
        df = df.drop_duplicates(subset=['job_url'], keep='first')
        
        # Save to CSV
        df.to_csv("ghana_jobs_detailed.csv", index=False)
        print(f"\n✓ Saved {len(df)} detailed jobs to 'ghana_jobs.csv'")
        
        # Show enhanced summary statistics
        print(f"\nDetailed Data Quality Summary:")
        print(f"  Jobs with titles: {df['name'].notna().sum()} ({df['name'].notna().sum()/len(df)*100:.1f}%)")
        print(f"  Jobs with companies: {df['hiring_firm'].notna().sum()} ({df['hiring_firm'].notna().sum()/len(df)*100:.1f}%)")
        print(f"  Jobs with locations: {df['Location'].notna().sum()} ({df['Location'].notna().sum()/len(df)*100:.1f}%)")
        print(f"  Jobs with descriptions: {df['job_description'].notna().sum()} ({df['job_description'].notna().sum()/len(df)*100:.1f}%)")
        print(f"  Jobs with requirements: {df['requirements'].notna().sum()} ({df['requirements'].notna().sum()/len(df)*100:.1f}%)")
        print(f"  Jobs with responsibilities: {df['responsibilities'].notna().sum()} ({df['responsibilities'].notna().sum()/len(df)*100:.1f}%)")
        print(f"  Jobs with qualifications: {df['qualifications'].notna().sum()} ({df['qualifications'].notna().sum()/len(df)*100:.1f}%)")
        print(f"  Jobs with benefits: {df['benefits'].notna().sum()} ({df['benefits'].notna().sum()/len(df)*100:.1f}%)")
        
        # Show sample detailed data
        print(f"\nSample detailed job data:")
        for i, (_, row) in enumerate(df.head(2).iterrows()):
            print(f"\n--- Job {i+1} ---")
            print(f"Title: {row['name']}")
            print(f"Company: {row['hiring_firm']}")
            print(f"Location: {row['Location']}")
            print(f"Description (first 200 chars): {str(row['job_description'])[:200] if row['job_description'] else 'N/A'}...")
            print(f"Requirements: {str(row['requirements'])[:100] if row['requirements'] else 'N/A'}...")
            print(f"Benefits: {str(row['benefits'])[:100] if row['benefits'] else 'N/A'}...")
        
    else:
        print("❌ No jobs were collected")

if __name__ == "__main__":
    main()