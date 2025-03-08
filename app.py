import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from collections import Counter
import re
from urllib.parse import urlparse
import concurrent.futures
import time
import io
import random

# Set up page configuration
st.set_page_config(
    page_title="Website Keyword Extractor",
    page_icon="🔍",
    layout="wide"
)

# Add title and description
st.title("Website Keyword Extractor")
st.markdown("Extract the most common keywords or tags from your list of websites.")

# Define a simple list of stopwords
english_stopwords = set(['and', 'the', 'for', 'with', 'that', 'this', 'you', 'your', 'our', 'from', 
             'have', 'has', 'are', 'not', 'when', 'what', 'where', 'why', 'how', 'all',
             'been', 'being', 'both', 'but', 'by', 'can', 'could', 'did', 'do', 'does',
             'doing', 'down', 'each', 'few', 'more', 'most', 'off', 'on', 'once', 'only',
             'own', 'same', 'should', 'so', 'some', 'such', 'than', 'too', 'very', 'will'])

# Initialize session state variables
if 'results_df' not in st.session_state:
    st.session_state['results_df'] = None

# Function to normalize URLs
def normalize_url(url):
    # Add http:// prefix if missing
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    # Parse the URL to handle variations
    parsed = urlparse(url)
    
    # Remove 'www.' if present
    netloc = parsed.netloc
    if netloc.startswith('www.'):
        netloc = netloc[4:]
    
    # Return the normalized domain
    return netloc

# Function to extract keywords from a website with enhanced search capabilities
def extract_keywords_from_website(url, max_retries=3):
    try:
        # Add http:// prefix if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        # Rotate user agents to avoid blocking
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0'
        ]
        
        headers = {
            'User-Agent': random.choice(user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # Enhanced retry mechanism
        success = False
        error_msg = ""
        
        for attempt in range(max_retries):
            try:
                # Fetch the website with a timeout
                response = requests.get(url, headers=headers, timeout=15)
                response.raise_for_status()  # Raise exception for HTTP errors
                success = True
                break
            except requests.exceptions.RequestException as e:
                error_msg = str(e)
                # Wait before retrying
                time.sleep(1 * (attempt + 1))
        
        if not success:
            return f"Error: {error_msg}"
        
        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract keywords from different sources
        keywords = []
        
        # 1. Enhanced meta tag extraction (including OpenGraph and Twitter tags)
        for meta in soup.find_all('meta'):
            # Meta keywords
            if meta.get('name') == 'keywords' and meta.get('content'):
                keywords.extend([k.strip().lower() for k in meta.get('content').split(',')])
            
            # Meta description
            if meta.get('name') == 'description' and meta.get('content'):
                desc_words = re.findall(r'\b\w+\b', meta.get('content').lower())
                keywords.extend([word for word in desc_words if len(word) > 3])
                
            # OpenGraph tags
            if meta.get('property') and 'og:' in meta.get('property') and meta.get('content'):
                if 'title' in meta.get('property') or 'description' in meta.get('property'):
                    og_words = re.findall(r'\b\w+\b', meta.get('content').lower())
                    keywords.extend([word for word in og_words if len(word) > 3])
        
        # 2. Title tags
        if soup.title:
            title_words = re.findall(r'\b\w+\b', soup.title.text.lower())
            keywords.extend([word for word in title_words if len(word) > 3])
        
        # 3. Heading tags with priority (h1, h2, h3)
        for i, heading_tag in enumerate(['h1', 'h2', 'h3']):
            # Give more weight to h1 than h2, and h2 more than h3
            weight = 3 - i
            for heading in soup.find_all(heading_tag):
                heading_words = re.findall(r'\b\w+\b', heading.text.lower())
                filtered_words = [word for word in heading_words if len(word) > 3]
                # Add words multiple times based on weight
                keywords.extend(filtered_words * weight)
        
        # 4. Enhanced content extraction
        content_tags = ['article', 'main', 'section', 'div']
        content_classes = ['content', 'post', 'entry', 'article', 'main', 'blog']
        
        # Look for content in semantic tags first
        for tag in content_tags[:3]:  # article, main, section
            for content in soup.find_all(tag):
                content_words = re.findall(r'\b\w+\b', content.text.lower())
                keywords.extend([word for word in content_words if len(word) > 3])
        
        # Look for content in divs with specific classes
        for cls in content_classes:
            for content in soup.find_all('div', class_=re.compile(cls, re.I)):
                content_words = re.findall(r'\b\w+\b', content.text.lower())
                keywords.extend([word for word in content_words if len(word) > 3])
                
        # 5. Enhanced tag extraction
        # Look for tags in multiple places with various patterns
        tag_patterns = ['tag', 'category', 'topic', 'keyword', 'subject', 'label']
        
        # Check for elements with tag-related classes
        for pattern in tag_patterns:
            # Class contains pattern
            for tag in soup.find_all(class_=re.compile(pattern, re.I)):
                tag_text = tag.text.strip().lower()
                if tag_text and len(tag_text) > 2:
                    # Add tag multiple times to increase weight
                    keywords.extend([tag_text] * 3)  
            
            # ID contains pattern
            for tag in soup.find_all(id=re.compile(pattern, re.I)):
                tag_text = tag.text.strip().lower()
                if tag_text and len(tag_text) > 2:
                    keywords.extend([tag_text] * 3)
        
        # 6. Check for tags in URLs
        tag_url_patterns = [
            r'(?:tag|tags)[=/]([^/&?#]+)',
            r'(?:category|categories)[=/]([^/&?#]+)',
            r'(?:topic|topics)[=/]([^/&?#]+)',
            r'(?:keyword|keywords)[=/]([^/&?#]+)'
        ]
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            for pattern in tag_url_patterns:
                match = re.search(pattern, href)
                if match:
                    tag = match.group(1).replace('-', ' ').replace('_', ' ').replace('+', ' ').lower()
                    # Add URL tags with higher weight
                    keywords.extend([tag] * 2)
        
        # Count occurrences of each keyword
        keyword_counter = Counter(keywords)
        
        # Remove common English stop words and short words
        for word in list(keyword_counter.keys()):
            if word in english_stopwords or len(word) <= 2:
                del keyword_counter[word]
        
        # Get the 10 most common keywords
        most_common = keyword_counter.most_common(10)
        
        # Format as a string: "keyword1, keyword2, keyword3, ..."
        if most_common:
            return ', '.join([f"{k}" for k, _ in most_common])
        else:
            return "No keywords found"
            
    except Exception as e:
        return f"Error: {str(e)}"

# Function to extract keywords from a batch of websites
def process_websites_batch(websites):
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(websites)
    results = {}
    
    # Determine the number of workers
    max_workers = min(10, total)  # Limit to max 10 workers
    
    # Show status
    status_text.text(f"Processing {total} websites...")
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(extract_keywords_from_website, url): url for url in websites}
        
        for i, future in enumerate(concurrent.futures.as_completed(future_to_url)):
            url = future_to_url[future]
            try:
                keywords = future.result()
                results[url] = keywords
            except Exception as e:
                results[url] = f"Error: {str(e)}"
            
            # Update progress
            progress = (i + 1) / total
            progress_bar.progress(progress)
            status_text.text(f"Processed {i+1}/{total} websites ({int(progress*100)}%)")
    
    # Create a DataFrame from results
    df = pd.DataFrame(list(results.items()), columns=['Website', 'Top Keywords'])
    
    # Complete
    progress_bar.progress(1.0)
    status_text.text(f"Completed processing {total} websites!")
    
    return df

# Main app layout
uploaded_file = st.file_uploader("Upload a CSV or text file with website URLs (one per line)", type=["csv", "txt"])

if uploaded_file is not None:
    try:
        # Process the uploaded file
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Look for URL columns
            url_columns = [col for col in df.columns if any(kw in col.lower() for kw in ['url', 'website', 'site', 'link', 'domain'])]
            
            if url_columns:
                url_column = st.selectbox("Select the column containing website URLs:", url_columns)
            else:
                url_column = st.selectbox("Select the column containing website URLs:", df.columns)
            
            # Get the website URLs
            websites = df[url_column].dropna().tolist()
        else:
            # Read as text file
            content = uploaded_file.getvalue().decode("utf-8")
            websites = [line.strip() for line in content.split('\n') if line.strip()]
        
        st.write(f"Found {len(websites)} websites")
        
        # Show a sample
        if len(websites) > 5:
            with st.expander("View sample websites"):
                st.write(websites[:10])
        else:
            st.write("Websites:", websites)
        
        # Process button
        if st.button("Extract Keywords"):
            # Process websites and get results
            st.session_state['results_df'] = process_websites_batch(websites)
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.exception(e)

# Display results if available
if st.session_state['results_df'] is not None:
    st.subheader("Results")
    st.dataframe(st.session_state['results_df'])
    
    # Download options
    st.subheader("Download Results")
    col1, col2 = st.columns(2)
    
    # Download as CSV
    csv = st.session_state['results_df'].to_csv(index=False)
    col1.download_button(
        label="Download as CSV",
        data=csv,
        file_name="website_keywords.csv",
        mime="text/csv"
    )
    
    # Download as Excel
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        st.session_state['results_df'].to_excel(writer, index=False, sheet_name='Keywords')
    
    buffer.seek(0)
    col2.download_button(
        label="Download as Excel",
        data=buffer,
        file_name="website_keywords.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Help section
with st.expander("Help & Information"):
    st.write("""
    ### About This App
    
    This app extracts the most common keywords from websites by analyzing:
    - Meta keywords tags and OpenGraph metadata
    - Page titles and headings (with priority weighting)
    - Article and content sections
    - Tag elements and categories
    - URL patterns containing tag or category information
    
    ### Tips for Best Results
    
    - Ensure URLs are valid (http:// or https:// will be added if missing)
    - For CSV files, select the column containing website URLs
    - Some websites may block scraping attempts
    - Processing large lists may take several minutes
    """)
