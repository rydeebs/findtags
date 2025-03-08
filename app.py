import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from collections import Counter
import re
from urllib.parse import urlparse
import concurrent.futures
import time
import json
from datetime import datetime
import os
import random
import io

# Only try to download nltk data if nltk is installed
try:
    import nltk
    from nltk.corpus import stopwords
    
    # Download NLTK resources (run once)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Set up page configuration
st.set_page_config(
    page_title="Website Keyword Extractor",
    page_icon="🔍",
    layout="wide"
)

# Add title and description
st.title("Website Keyword Extractor")
st.markdown("Extract the 10 most common keywords or tags from your list of websites.")

# Cache management - store previously extracted keywords
@st.cache_data
def get_empty_cache():
    return {}

# We'll use session state for cache instead of file system
if "keyword_cache" not in st.session_state:
    st.session_state.keyword_cache = get_empty_cache()

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

# Define stopwords
if NLTK_AVAILABLE:
    english_stopwords = set(stopwords.words('english'))
else:
    # Fallback to a simple list of common stopwords if NLTK is not available
    english_stopwords = set(['and', 'the', 'for', 'with', 'that', 'this', 'you', 'your', 'our', 'from', 
                 'have', 'has', 'are', 'not', 'when', 'what', 'where', 'why', 'how', 'all',
                 'been', 'being', 'both', 'but', 'by', 'can', 'could', 'did', 'do', 'does',
                 'doing', 'down', 'each', 'few', 'more', 'most', 'off', 'on', 'once', 'only',
                 'own', 'same', 'should', 'so', 'some', 'such', 'than', 'too', 'very', 'will'])

# Function to extract keywords from a website with retry mechanism
def extract_keywords_from_website(url, retries=3, backoff_factor=0.5):
    # Normalize URL for display and caching
    normalized_url = normalize_url(url)
    
    # Check cache first
    if normalized_url in st.session_state.keyword_cache:
        return st.session_state.keyword_cache[normalized_url]
    
    for attempt in range(retries):
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
            
            # Fetch the website with a timeout
            response = requests.get(url, headers=headers, timeout=15)
            
            # Check if we got a successful response
            if response.status_code != 200:
                raise requests.exceptions.RequestException(f"HTTP Error: {response.status_code}")
            
            # Parse HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract keywords from different sources
            keywords = []
            
            # 1. Meta keywords
            meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
            if meta_keywords and meta_keywords.get('content'):
                keywords.extend([k.strip().lower() for k in meta_keywords.get('content').split(',')])
            
            # 2. Meta description (for potential keywords)
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc and meta_desc.get('content'):
                desc_words = re.findall(r'\b\w+\b', meta_desc.get('content').lower())
                keywords.extend([word for word in desc_words if len(word) > 3])
            
            # 3. Title tags
            if soup.title:
                title_words = re.findall(r'\b\w+\b', soup.title.text.lower())
                keywords.extend([word for word in title_words if len(word) > 3])
            
            # 4. Heading tags (h1, h2, h3)
            for heading in soup.find_all(['h1', 'h2', 'h3']):
                heading_words = re.findall(r'\b\w+\b', heading.text.lower())
                keywords.extend([word for word in heading_words if len(word) > 3])
            
            # 5. Article tags and main content
            for content in soup.find_all(['article', 'main', 'section']):
                content_words = re.findall(r'\b\w+\b', content.text.lower())
                keywords.extend([word for word in content_words if len(word) > 3])
                
            # 6. Tag elements and category elements
            tag_patterns = ['tag', 'category', 'topic', 'keyword', 'subject', 'label']
            for pattern in tag_patterns:
                # Look for elements with these classes or IDs
                for tag in soup.find_all(class_=re.compile(pattern, re.I)):
                    tag_text = tag.text.strip().lower()
                    if tag_text and len(tag_text) > 2:
                        keywords.append(tag_text)
                
                # Also check ID attributes
                for tag in soup.find_all(id=re.compile(pattern, re.I)):
                    tag_text = tag.text.strip().lower()
                    if tag_text and len(tag_text) > 2:
                        keywords.append(tag_text)
            
            # 7. Check for tags in URLs
            for link in soup.find_all('a', href=True):
                href = link['href']
                if 'tag' in href or 'category' in href or 'topic' in href:
                    # Extract the tag from the URL
                    tag_match = re.search(r'(?:tag|category|topic)[=/]([^/&?]+)', href)
                    if tag_match:
                        tag = tag_match.group(1).replace('-', ' ').replace('_', ' ').lower()
                        keywords.append(tag)
            
            # Count occurrences of each keyword
            keyword_counter = Counter(keywords)
            
            # Remove common English stop words
            for word in list(keyword_counter.keys()):
                if word in english_stopwords or len(word) <= 2:
                    del keyword_counter[word]
            
            # Get the 10 most common keywords
            most_common = keyword_counter.most_common(10)
            
            # Format as a string: "keyword1, keyword2, keyword3, ..."
            result = ', '.join([f"{k}" for k, _ in most_common]) if most_common else "No keywords found"
            
            # Save to cache
            st.session_state.keyword_cache[normalized_url] = result
            
            return result
            
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                # Wait with exponential backoff before retrying
                wait_time = backoff_factor * (2 ** attempt)
                time.sleep(wait_time)
                continue
            else:
                return f"Error: Connection failed after {retries} attempts"
        
        except Exception as e:
            return f"Error: {str(e)}"

# Function to process the list of websites
def process_websites(urls, max_workers=10, batch_size=100):
    results = {}
    total_urls = len(urls)
    
    # Create session state for tracking progress
    if 'processed_count' not in st.session_state:
        st.session_state.processed_count = 0
        st.session_state.start_time = time.time()
        st.session_state.results = {}
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    progress_counter = col1.empty()
    time_metric = col2.empty()
    completion_metric = col3.empty()
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Results area
    results_area = st.empty()
    
    # Calculate starting position for this batch
    start_idx = st.session_state.processed_count
    end_idx = min(start_idx + batch_size, total_urls)
    current_batch = urls[start_idx:end_idx]
    
    if not current_batch:
        status_text.text("All websites have been processed!")
        return st.session_state.results
    
    # Update status
    status_text.text(f"Processing batch {start_idx//batch_size + 1} of {(total_urls+batch_size-1)//batch_size}...")
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and map URLs to their futures
        future_to_url = {executor.submit(extract_keywords_from_website, url): url for url in current_batch}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                keywords = future.result()
                st.session_state.results[url] = keywords
                
                # Update progress
                st.session_state.processed_count += 1
                progress_percent = st.session_state.processed_count / total_urls
                progress_bar.progress(progress_percent)
                
                # Update metrics
                elapsed_time = time.time() - st.session_state.start_time
                estimated_total = elapsed_time / progress_percent if progress_percent > 0 else 0
                time_remaining = estimated_total - elapsed_time if estimated_total > 0 else 0
                
                progress_counter.metric("Processed", f"{st.session_state.processed_count}/{total_urls}")
                time_metric.metric("Time Elapsed", f"{int(elapsed_time/60)}m {int(elapsed_time%60)}s")
                completion_metric.metric("Est. Completion", f"{int(time_remaining/60)}m {int(time_remaining%60)}s")
                
                # Periodically update the displayed results
                if st.session_state.processed_count % 10 == 0 or st.session_state.processed_count == total_urls:
                    result_df = pd.DataFrame(list(st.session_state.results.items()), columns=['Website', 'Top Keywords'])
                    results_area.dataframe(result_df)
                    
            except Exception as e:
                st.session_state.results[url] = f"Error: {str(e)}"
                st.session_state.processed_count += 1
    
    # If there are more URLs to process, update status
    if st.session_state.processed_count < total_urls:
        status_text.text(f"Batch completed. {st.session_state.processed_count}/{total_urls} websites processed so far.")
        
        # Auto-continue button
        if st.button("Process Next Batch"):
            return process_websites(urls, max_workers, batch_size)
    else:
        # All done
        progress_bar.progress(1.0)
        status_text.text(f"Completed processing all {total_urls} websites!")
    
    return st.session_state.results

# Add tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["Extract Keywords", "Settings", "Help"])

with tab1:
    # File uploader widget
    uploaded_file = st.file_uploader("Upload a CSV or text file with website URLs (one per line)", type=["csv", "txt"])

    # Process the file when uploaded
    if uploaded_file is not None:
        try:
            # Determine file type and read accordingly
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                # Try to read CSV assuming different possible column names for URLs
                df = pd.read_csv(uploaded_file)
                
                # Look for columns that might contain URLs
                url_columns = [col for col in df.columns if any(kw in col.lower() for kw in ['url', 'website', 'site', 'link', 'domain'])]
                
                if url_columns:
                    # Let user select the column
                    url_column = st.selectbox("Select the column containing website URLs:", url_columns)
                    websites = df[url_column].dropna().tolist()
                else:
                    # If no obvious URL column, let user select
                    url_column = st.selectbox("Select the column containing website URLs:", df.columns)
                    websites = df[url_column].dropna().tolist()
            else:
                # For text files, read line by line
                content = uploaded_file.getvalue().decode("utf-8")
                websites = [line.strip() for line in content.split('\n') if line.strip()]
            
            # Show website count and sample
            st.write(f"Found {len(websites)} websites")
            if len(websites) > 5:
                with st.expander("View sample websites"):
                    st.write(websites[:10])
            else:
                st.write("Websites:", websites)
            
            # Add a button to start processing
            col1, col2 = st.columns(2)
            
            # Reset button to start fresh
            if col1.button("Reset Processing"):
                if 'processed_count' in st.session_state:
                    del st.session_state.processed_count
                    del st.session_state.start_time
                    del st.session_state.results
                st.experimental_rerun()
            
            # Start/continue processing button
            start_button = col2.button("Start/Continue Processing")
            
            # Get settings from the settings tab
            if 'max_workers' not in st.session_state:
                st.session_state.max_workers = 10
            if 'batch_size' not in st.session_state:
                st.session_state.batch_size = 100
            
            if start_button:
                # Process the websites and get results
                with st.spinner('Extracting keywords from websites...'):
                    results = process_websites(
                        websites, 
                        max_workers=st.session_state.max_workers, 
                        batch_size=st.session_state.batch_size
                    )
                
                # If processing is complete, offer downloads
                if 'processed_count' in st.session_state and st.session_state.processed_count == len(websites):
                    result_df = pd.DataFrame(list(results.items()), columns=['Website', 'Top Keywords'])
                    
                    # Add timestamp to filenames
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # Provide download buttons
                    col1, col2 = st.columns(2)
                    
                    # Download as CSV
                    csv = result_df.to_csv(index=False)
                    col1.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name=f"website_keywords_{timestamp}.csv",
                        mime="text/csv"
                    )
                    
                    # Download as Excel - Fixed the Excel writing method
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        result_df.to_excel(writer, index=False, sheet_name='Keywords')
                    
                    excel_data = buffer.getvalue()
                    col2.download_button(
                        label="Download Results as Excel",
                        data=excel_data,
                        file_name=f"website_keywords_{timestamp}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.exception(e)

with tab2:
    st.header("Settings")
    
    # Performance settings
    st.subheader("Performance Settings")
    
    # Adjust max workers (threads)
    st.session_state.max_workers = st.slider(
        "Max parallel requests", 
        min_value=1, 
        max_value=30, 
        value=st.session_state.get('max_workers', 10),
        help="Higher values process more websites simultaneously but may cause rate limiting"
    )
    
    # Batch size for processing
    st.session_state.batch_size = st.slider(
        "Batch size", 
        min_value=10, 
        max_value=500, 
        value=st.session_state.get('batch_size', 100),
        help="Number of websites to process in each batch"
    )
    
    # Cache management
    st.subheader("Cache Management")
    st.write(f"Cache contains data for {len(st.session_state.keyword_cache)} websites")
    
    if st.button("Clear Cache"):
        st.session_state.keyword_cache = get_empty_cache()
        st.success("Cache cleared successfully")
        st.experimental_rerun()

with tab3:
    st.header("Help & Information")
    
    st.subheader("About This App")
    st.markdown("""
    This app extracts the most common keywords from websites by analyzing:
    - Meta keywords tags
    - Meta descriptions
    - Page titles
    - Headings (H1, H2, H3)
    - Article and section content
    - Tag elements and categories
    - URL patterns containing tag or category information
    
    The app handles large lists of websites efficiently by:
    - Processing in parallel using multiple threads
    - Using a batch processing approach for large datasets
    - Caching results to avoid duplicate work
    - Implementing retry mechanisms for failed requests
    """)
    
    st.subheader("Tips for Best Results")
    st.markdown("""
    - **URL Format**: Ensure your URLs are valid. The app will attempt to add 'https://' if missing
    - **Processing Time**: Expect about 5-10 seconds per website on average
    - **Rate Limiting**: Some websites may block scraping attempts if too many requests are made
    - **Resume Processing**: If processing stops, you can continue where you left off
    - **Column Selection**: For CSV files, ensure you select the correct column containing URLs
    - **Performance Tuning**: Adjust the concurrent requests in Settings to balance speed vs. reliability
    """)
    
    st.subheader("Troubleshooting")
    st.markdown("""
    - **Connection Errors**: Reduce the "Max parallel requests" in the Settings tab
    - **Incomplete Results**: Some websites may block scraping - try lowering the parallel requests
    - **No Keywords Found**: The website might be using JavaScript to render content or has unique tag structures
    - **Processing Stops**: Use the "Continue Processing" button to resume where you left off
    - **Large Files**: Break very large lists into smaller batches of 1000-2000 websites
    """)
