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
import numpy as np

# Set up page configuration
st.set_page_config(
    page_title="Website Keyword & Status Analyzer",
    page_icon="🔍",
    layout="wide"
)

# Add title and description
st.title("Website Keyword & Status Analyzer")
st.markdown("Extract keywords, categorize websites, and check domain status")

# Define a simple list of stopwords
english_stopwords = set(['and', 'the', 'for', 'with', 'that', 'this', 'you', 'your', 'our', 'from', 
             'have', 'has', 'are', 'not', 'when', 'what', 'where', 'why', 'how', 'all',
             'been', 'being', 'both', 'but', 'by', 'can', 'could', 'did', 'do', 'does',
             'doing', 'down', 'each', 'few', 'more', 'most', 'off', 'on', 'once', 'only',
             'own', 'same', 'should', 'so', 'some', 'such', 'than', 'too', 'very', 'will',
             'about', 'after', 'all', 'also', 'an', 'any', 'as', 'at', 'back', 'be',
             'because', 'before', 'between', 'come', 'day', 'even', 'first', 'from', 'get',
             'give', 'go', 'good', 'have', 'he', 'her', 'him', 'his', 'how', 'i', 'if',
             'in', 'into', 'it', 'its', 'just', 'know', 'like', 'look', 'make', 'me',
             'most', 'my', 'new', 'no', 'not', 'now', 'of', 'on', 'one', 'only', 'or',
             'other', 'our', 'out', 'over', 'people', 'say', 'see', 'she', 'so', 'some',
             'take', 'than', 'that', 'the', 'their', 'them', 'then', 'there', 'these',
             'they', 'think', 'this', 'time', 'to', 'two', 'up', 'us', 'use', 'want',
             'way', 'we', 'well', 'what', 'when', 'which', 'who', 'will', 'with', 'would',
             'year', 'you', 'your'])

# Define category keyword associations
CATEGORY_KEYWORDS = {
    'Transportation': ['transport', 'logistics', 'shipping', 'delivery', 'freight', 'cargo', 'truck', 'fleet', 'bus', 'taxi', 'travel', 'transit', 'courier', 'train', 'rail', 'ship', 'vessel', 'aviation', 'airline', 'airport', 'vehicle', 'mobility'],
    
    'Supplement': ['supplement', 'vitamin', 'mineral', 'nutrition', 'dietary', 'protein', 'fitness', 'health', 'wellness', 'natural', 'organic', 'superfood', 'antioxidant', 'nutrition', 'amino', 'capsule', 'tablet', 'powder', 'extract', 'herb', 'omega'],
    
    'Industrial': ['industrial', 'manufacturing', 'factory', 'machinery', 'equipment', 'tool', 'hardware', 'construction', 'build', 'material', 'chemical', 'processing', 'production', 'supply', 'engineering', 'maintenance', 'safety', 'fabrication', 'metal', 'steel', 'operation'],
    
    'Home Goods': ['home', 'furniture', 'decor', 'kitchen', 'bathroom', 'bedroom', 'living', 'house', 'interior', 'decoration', 'appliance', 'domestic', 'housewares', 'furnishing', 'textile', 'bedding', 'linen', 'curtain', 'carpet', 'rug', 'lamp'],
    
    'Food & Beverage': ['food', 'beverage', 'drink', 'restaurant', 'cafe', 'catering', 'kitchen', 'chef', 'culinary', 'cuisine', 'gourmet', 'meal', 'recipe', 'ingredient', 'grocery', 'coffee', 'tea', 'wine', 'beer', 'juice', 'snack', 'bakery', 'dessert', 'vegetable', 'fruit', 'meat'],
    
    'Electronics': ['electronic', 'tech', 'technology', 'digital', 'computer', 'device', 'gadget', 'hardware', 'software', 'smartphone', 'tablet', 'laptop', 'gaming', 'appliance', 'audio', 'video', 'camera', 'television', 'wireless', 'battery', 'charger', 'cable', 'accessory'],
    
    'Packaging': ['packaging', 'package', 'container', 'box', 'wrap', 'packing', 'carton', 'bottle', 'label', 'seal', 'bag', 'pouch', 'crate', 'tape', 'film', 'foam', 'paper', 'plastic', 'glass', 'metal', 'shipping', 'storage', 'protective'],
    
    'Personal Care': ['beauty', 'cosmetic', 'makeup', 'skin', 'hair', 'nail', 'perfume', 'fragrance', 'hygiene', 'clean', 'care', 'wellness', 'grooming', 'spa', 'salon', 'lotion', 'cream', 'soap', 'shampoo', 'deodorant', 'toothpaste', 'brush'],
    
    'Jewelry': ['jewelry', 'jewellery', 'accessory', 'ring', 'necklace', 'bracelet', 'earring', 'watch', 'gold', 'silver', 'diamond', 'gem', 'gemstone', 'precious', 'metal', 'stone', 'pearl', 'crystal', 'design', 'fashion', 'luxury', 'handmade'],
    
    'Agriculture': ['agriculture', 'farm', 'farming', 'crop', 'livestock', 'animal', 'plant', 'seed', 'soil', 'harvest', 'garden', 'organic', 'greenhouse', 'irrigation', 'fertilizer', 'pesticide', 'dairy', 'poultry', 'grain', 'vegetable', 'fruit'],
    
    'Automotive': ['automotive', 'car', 'vehicle', 'truck', 'repair', 'maintenance', 'part', 'accessory', 'dealer', 'garage', 'engine', 'tire', 'wheel', 'battery', 'oil', 'fuel', 'transmission', 'brake', 'steering', 'suspension', 'exhaust'],
    
    'Footwear': ['shoe', 'footwear', 'boot', 'sneaker', 'sandal', 'slipper', 'heel', 'sole', 'insole', 'lace', 'leather', 'comfort', 'running', 'walking', 'sport', 'athletic', 'casual', 'formal', 'children', 'men', 'women'],
    
    'Entertainment': ['entertainment', 'event', 'show', 'performance', 'concert', 'theater', 'cinema', 'movie', 'film', 'music', 'game', 'sport', 'festival', 'party', 'fun', 'leisure', 'recreation', 'amusement', 'media', 'streaming', 'video'],
    
    'Clothing': ['clothing', 'apparel', 'fashion', 'wear', 'garment', 'textile', 'fabric', 'dress', 'shirt', 'pants', 'jacket', 'coat', 'suit', 'uniform', 'casual', 'formal', 'sportswear', 'underwear', 'accessory', 'designer', 'collection'],
    
    'Medical': ['medical', 'healthcare', 'health', 'hospital', 'clinic', 'doctor', 'physician', 'patient', 'treatment', 'therapy', 'medicine', 'pharmaceutical', 'device', 'equipment', 'diagnostic', 'surgery', 'care', 'wellness', 'dental', 'vision', 'prescription']
}

# Domain parking/placeholder patterns
PARKED_DOMAIN_PATTERNS = [
    'domain is for sale',
    'buy this domain',
    'purchase this domain',
    'domain may be for sale',
    'domain parking',
    'parked domain',
    'this web page is parked',
    'domain has been registered',
    'godaddy',
    'this website is temporarily unavailable',
    'website coming soon',
    'under construction',
    'site not found',
    'site currently unavailable',
    'account suspended',
    'namecheap',
    'hostgator',
    'bluehost',
    'domainname',
    'networksolutions',
    'pendingrenewaldeletion',
    'enom',
    'please check back soon'
]

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

# Function to check website status and detect parked domains
def check_website_status(url, response=None, html_content=None):
    """
    Check if a website is active and not a parked/placeholder domain.
    
    Args:
        url: The URL to check
        response: Optional HTTP response object if already fetched
        html_content: Optional HTML content if already parsed
    
    Returns:
        Tuple of (status_code, status_message, is_parked)
    """
    status_code = 0
    status_message = "Unknown"
    is_parked = False
    
    try:
        # If response not provided, fetch it
        if response is None:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            }
            
            # Normalize URL
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
                
            response = requests.get(url, headers=headers, timeout=10)
        
        status_code = response.status_code
        
        # Check HTTP status code
        if 200 <= status_code < 300:
            status_message = "Active"
        elif 300 <= status_code < 400:
            status_message = f"Redirect ({status_code})"
        elif status_code == 403:
            status_message = "Forbidden"
        elif status_code == 404:
            status_message = "Not Found"
        elif status_code == 500:
            status_message = "Server Error"
        else:
            status_message = f"HTTP {status_code}"
        
        # If we have a successful response, check if it's a parked domain
        if 200 <= status_code < 300:
            # Use provided HTML content or parse from response
            if html_content is None:
                html_content = response.text.lower()
            else:
                html_content = html_content.lower()
            
            # Check for common parking page indicators
            for pattern in PARKED_DOMAIN_PATTERNS:
                if pattern.lower() in html_content:
                    is_parked = True
                    status_message = "Parked Domain"
                    break
            
            # Check for very short content (often parking pages)
            if len(html_content) < 500 and not is_parked:
                # Look for minimal HTML structure
                if html_content.count('<') < 20:
                    is_parked = True
                    status_message = "Minimal Page"
            
            # Check for domain registrar or hosting placeholder
            registrar_patterns = ['godaddy', 'namecheap', 'hostgator', 'bluehost', 'domain', 'hosting']
            for pattern in registrar_patterns:
                if pattern in html_content and ('welcome' in html_content or 'domain' in html_content):
                    if not is_parked:  # Don't override other parked signals
                        is_parked = True
                        status_message = "Registrar Placeholder"
                        break
    
    except requests.exceptions.ConnectionError:
        status_code = -1
        status_message = "Connection Failed"
    except requests.exceptions.Timeout:
        status_code = -2
        status_message = "Timeout"
    except requests.exceptions.TooManyRedirects:
        status_code = -3
        status_message = "Too Many Redirects"
    except requests.exceptions.RequestException:
        status_code = -4
        status_message = "Request Failed"
    except Exception as e:
        status_code = -99
        status_message = f"Error: {str(e)[:30]}"
    
    return status_code, status_message, is_parked

# Function to categorize a website based on extracted keywords and meta descriptions
def categorize_website(keywords, meta_description=""):
    if not keywords or keywords == "No keywords found":
        return "Other", 0

    # Create a single text for analysis
    full_text = keywords.lower()
    if meta_description:
        full_text += " " + meta_description.lower()
    
    # Calculate scores for each category
    category_scores = {}
    for category, category_keywords in CATEGORY_KEYWORDS.items():
        score = 0
        for keyword in category_keywords:
            # Check if the keyword appears in the text
            matches = re.findall(r'\b' + re.escape(keyword) + r'\b', full_text)
            if matches:
                # Add weight based on the number of matches and the specificity of the keyword
                weight = 2 if len(keyword) > 5 else 1  # Longer keywords get more weight
                score += len(matches) * weight
        
        category_scores[category] = score
    
    # Find the category with the highest score
    max_score = max(category_scores.values()) if category_scores else 0
    
    if max_score == 0:
        return "Other", 0
    
    # In case of a tie, pick the one that appears first in the text
    max_categories = [cat for cat, score in category_scores.items() if score == max_score]
    if len(max_categories) > 1:
        for category in max_categories:
            for keyword in CATEGORY_KEYWORDS[category]:
                if keyword in full_text:
                    first_index = full_text.find(keyword)
                    category_scores[category] = (category_scores[category], -first_index)
        
        best_category = max(max_categories, key=lambda cat: category_scores[cat] if isinstance(category_scores[cat], tuple) else category_scores[cat])
    else:
        best_category = max_categories[0]
    
    confidence = min(max_score / 10, 1.0)  # Cap confidence at 1.0
    
    return best_category, confidence

# Function to extract keywords, meta information, and check status of a website
def extract_website_info(url, max_retries=3):
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
        response = None
        
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
        
        # Check website status
        if response:
            status_code, status_message, is_parked = check_website_status(url, response=response, html_content=response.text if success else None)
        else:
            status_code, status_message, is_parked = check_website_status(url)
        
        if not success:
            return {
                'keywords': f"Error: {error_msg}", 
                'meta_description': '', 
                'title': '', 
                'category': 'Other', 
                'confidence': 0,
                'status_code': status_code,
                'status': status_message,
                'is_parked': is_parked
            }
        
        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract meta description
        meta_description = ""
        meta_desc_tag = soup.find('meta', attrs={'name': 'description'})
        if meta_desc_tag and meta_desc_tag.get('content'):
            meta_description = meta_desc_tag.get('content').strip()
        
        # Extract page title
        page_title = ""
        if soup.title:
            page_title = soup.title.text.strip()
        
        # If the domain is parked, we'll return minimal information
        if is_parked:
            return {
                'keywords': "Parked Domain",
                'meta_description': meta_description,
                'title': page_title,
                'category': 'Other',
                'confidence': 0,
                'status_code': status_code,
                'status': status_message,
                'is_parked': is_parked
            }
        
        # Extract keywords from different sources
        keywords = []
        
        # 1. Enhanced meta tag extraction (including OpenGraph and Twitter tags)
        for meta in soup.find_all('meta'):
            # Meta keywords
            if meta.get('name') == 'keywords' and meta.get('content'):
                keywords.extend([k.strip().lower() for k in meta.get('content').split(',')])
            
            # Meta description (for keywords)
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
        content_classes = ['content', 'post', 'entry', 'article', 'main', 'blog', 'about', 'product', 'service']
        
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
        
        # Get the most common keywords (increase to 20 for better categorization)
        most_common = keyword_counter.most_common(20)
        
        # Format as a string: "keyword1, keyword2, keyword3, ..."
        keywords_str = ', '.join([f"{k}" for k, _ in most_common]) if most_common else "No keywords found"
        
        # Categorize the website
        category, confidence = categorize_website(keywords_str, meta_description)
        
        # Reduce to top 10 keywords for display
        top_10_keywords = ', '.join([f"{k}" for k, _ in most_common[:10]]) if most_common else "No keywords found"
        
        return {
            'keywords': top_10_keywords,
            'meta_description': meta_description,
            'title': page_title,
            'category': category,
            'confidence': confidence,
            'status_code': status_code,
            'status': status_message,
            'is_parked': is_parked
        }
            
    except Exception as e:
        return {
            'keywords': f"Error: {str(e)}",
            'meta_description': '',
            'title': '',
            'category': 'Other',
            'confidence': 0,
            'status_code': -99,
            'status': f"Error: {str(e)[:30]}",
            'is_parked': False
        }

# Function to process a batch of websites
def process_websites_batch(websites):
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(websites)
    results = []
    
    # Determine the number of workers
    max_workers = min(10, total)  # Limit to max 10 workers
    
    # Show status
    status_text.text(f"Processing {total} websites...")
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(extract_website_info, url): url for url in websites}
        
        for i, future in enumerate(concurrent.futures.as_completed(future_to_url)):
            url = future_to_url[future]
            try:
                info = future.result()
                results.append({
                    'Website': url,
                    'Status': info['status'],
                    'Is Parked': "Yes" if info['is_parked'] else "No",
                    'Top Keywords': info['keywords'],
                    'Meta Description': info['meta_description'],
                    'Page Title': info['title'],
                    'Category': info['category'],
                    'Confidence': f"{info['confidence']:.2f}"
                })
            except Exception as e:
                results.append({
                    'Website': url,
                    'Status': "Error",
                    'Is Parked': "Unknown",
                    'Top Keywords': f"Error: {str(e)}",
                    'Meta Description': '',
                    'Page Title': '',
                    'Category': 'Other',
                    'Confidence': '0.00'
                })
            
            # Update progress
            progress = (i + 1) / total
            progress_bar.progress(progress)
            status_text.text(f"Processed {i+1}/{total} websites ({int(progress*100)}%)")
    
    # Create a DataFrame from results
    df = pd.DataFrame(results)
    
    # Complete
    progress_bar.progress(1.0)
    status_text.text(f"Completed processing {total} websites!")
    
    return df

# Main app layout
uploaded_file = st.file_uploader("Upload a CSV or Excel file with website URLs", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        # Process the uploaded file
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
        else:
            # Read Excel file
            df = pd.read_excel(uploaded_file)
        
        # Look for URL columns
        url_columns = [col for col in df.columns if any(kw in col.lower() for kw in ['url', 'website', 'site', 'link', 'domain'])]
        
        if url_columns:
            url_column = st.selectbox("Select the column containing website URLs:", url_columns)
        else:
            url_column = st.selectbox("Select the column containing website URLs:", df.columns)
        
        # Get the website URLs
        websites = df[url_column].dropna().tolist()
        
        st.write(f"Found {len(websites)} websites")
        
        # Show a sample
        if len(websites) > 5:
            with st.expander("View sample websites"):
                st.write(websites[:10])
        else:
            st.write("Websites:", websites)
        
        # Process button
        if st.button("Analyze Websites"):
            # Process websites and get results
            st.session_state['results_df'] = process_websites_batch(websites)
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.exception(e)

# Display results if available
if st.session_state['results_df'] is not None:
    st.subheader("Analysis Results")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["All Results", "Active Sites", "Status Summary"])
    
    with tab1:
        # Show all results
        st.dataframe(st.session_state['results_df'])
    
    with tab2:
        # Show only active sites (not parked, with successful status)
        active_sites = st.session_state['results_df'][
            (st.session_state['results_df']['Is Parked'] == "No") & 
            (st.session_state['results_df']['Status'] == "Active")
        ]
        
        if len(active_sites) > 0:
            st.write(f"Found {len(active_sites)} active websites")
            st.dataframe(active_sites)
            
            # Display category distribution for active sites
            st.subheader("Category Distribution for Active Sites")
            active_category_counts = active_sites['Category'].value_counts()
            st.bar_chart(active_category_counts)
        else:
            st.warning("No active websites found")
    
    with tab3:
        # Show summary statistics
        col1, col2 = st.columns(2)
        
        with col1:
            # Status distribution
            st.subheader("Website Status Distribution")
            status_counts = st.session_state['results_df']['Status'].value_counts()
            st.dataframe(status_counts.reset_index().rename(columns={'index': 'Status', 'Status': 'Count'}))
        
        with col2:
            # Parked domain distribution
            st.subheader("Parked Domain Distribution")
            parked_counts = st.session_state['results_df']['Is Parked'].value_counts()
            st.dataframe(parked_counts.reset_index().rename(columns={'index': 'Is Parked', 'Is Parked': 'Count'}))
        
        # Category distribution
        st.subheader("Overall Category Distribution")
        category_counts = st.session_state['results_df']['Category'].value_counts()
        st.bar_chart(category_counts)
    
    # Download options
    st.subheader("Download Results")
    col1, col2 = st.columns(2)
    
    # Download as CSV
    csv = st.session_state['results_df'].to_csv(index=False)
    col1.download_button(
        label="Download as CSV",
        data=csv,
        file_name="website_analysis.csv",
        mime="text/csv"
    )
    
    # Download as Excel with multiple sheets
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        # All results sheet
        st.session_state['results_df'].to_excel(writer, index=False, sheet_name='All Websites')
        
        # Active sites sheet
        if len(active_sites) > 0:
            active_sites.to_excel(writer, index=False, sheet_name='Active Websites')
        
        # Category summary sheet
        category_summary = pd.DataFrame({
            'Category': category_counts.index,
            'Count': category_counts.values,
            'Percentage': (category_counts.values / len(st.session_state['results_df']) * 100).round(2)
        })
        category_summary.to_excel(writer, index=False, sheet_name='Category Summary')
        
        # Status summary sheet
        status_summary = pd.DataFrame({
            'Status': status_counts.index,
            'Count': status_counts.values,
            'Percentage': (status_counts.values / len(st.session_state['results_df']) * 100).round(2)
        })
        status_summary.to_excel(writer, index=False, sheet_name='Status Summary')
        
        # Create a separate worksheet for each category
        for category in st.session_state['results_df']['Category'].unique():
            if category == 'Other':
                continue  # Skip "Other" category to save space
                
            category_sites = st.session_state['results_df'][st.session_state['results_df']['Category'] == category]
            if len(category_sites) > 0:
                # Clean category name for worksheet name
                sheet_name = category[:31].replace(':', '').replace('\\', '').replace('/', '').replace('?', '').replace('*', '').replace('[', '').replace(']', '')
                category_sites.to_excel(writer, index=False, sheet_name=sheet_name)
    
    buffer.seek(0)
    col2.download_button(
        label="Download as Excel",
        data=buffer,
        file_name="website_analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
