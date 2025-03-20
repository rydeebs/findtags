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

# Improved function to check website status with better content detection
def check_website_status(url, response=None, html_content=None):
    """
    Enhanced check if a website is active with better handling of modern websites.
    
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
        # If response not provided, fetch it with multiple attempts and varying user agents
        if response is None:
            user_agents = [
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15',
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0',
                'Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148'
            ]
            
            # Try both HTTPS and HTTP if needed
            protocols = ['https://', 'http://']
            
            for protocol in protocols:
                # Normalize URL with current protocol
                if url.startswith('http'):
                    # Remove existing protocol
                    url_parts = url.split('://', 1)
                    if len(url_parts) > 1:
                        clean_url = url_parts[1]
                    else:
                        clean_url = url
                else:
                    clean_url = url
                
                current_url = f"{protocol}{clean_url}"
                
                # Try with different user agents
                for user_agent in user_agents:
                    headers = {
                        'User-Agent': user_agent,
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.5',
                        'Accept-Encoding': 'gzip, deflate, br',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1',
                        'Cache-Control': 'max-age=0'
                    }
                    
                    try:
                        # Use a session to handle cookies and redirects better
                        session = requests.Session()
                        response = session.get(current_url, headers=headers, timeout=15, allow_redirects=True)
                        
                        # If successful, break out of the loops
                        if response.status_code == 200:
                            break
                    except:
                        continue
                
                # If we got a valid response, break out of protocols loop
                if response and response.status_code == 200:
                    break
        
        # If we still don't have a response, raise an exception
        if response is None:
            raise requests.exceptions.RequestException("Failed to connect to website")
        
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
                html_content = response.text
            
            # Check if the page has actual content vs. being a parked domain
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements that don't contribute to visible content
            for script_or_style in soup(['script', 'style', 'noscript']):
                script_or_style.decompose()
            
            # Get the visible text
            page_text = soup.get_text().lower()
            
            # Check for common parking page indicators with exact phrases
            parking_phrases = [
                'domain is for sale',
                'buy this domain',
                'purchase this domain',
                'domain may be for sale',
                'domain parking',
                'parked domain',
                'this web page is parked',
                'this website is temporarily unavailable',
                'website coming soon',
                'under construction',
                'account suspended',
                'pendingrenewaldeletion'
            ]
            
            for phrase in parking_phrases:
                if phrase in page_text:
                    is_parked = True
                    status_message = "Parked Domain"
                    break
            
            # If not yet identified as parked, check for more conditions
            if not is_parked:
                # Check for proper HTML structure - real sites usually have substantial structure
                content_elements = soup.find_all(['div', 'section', 'article', 'main', 'header', 'footer', 'aside', 'nav'])
                
                # Check for common content indicators
                has_links = len(soup.find_all('a')) > 5
                has_images = len(soup.find_all('img')) > 0
                has_paragraphs = len(soup.find_all('p')) > 2
                has_headings = len(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])) > 0
                
                # More sophisticated content check
                is_real_site = (
                    has_links or 
                    has_images or 
                    has_paragraphs or 
                    has_headings or 
                    len(content_elements) > 5
                )
                
                # If the page fails most content checks, it might be parked
                if not is_real_site and len(page_text.strip()) < 1000:
                    # Do a final check for registrar placeholders
                    registrar_patterns = ['godaddy', 'namecheap', 'hostgator', 'bluehost', 'domain for sale']
                    
                    for pattern in registrar_patterns:
                        if pattern in page_text and ('welcome' in page_text or 'domain' in page_text):
                            is_parked = True
                            status_message = "Registrar Placeholder"
                            break
                            
                    # If still not marked as parked but has very little content
                    if not is_parked and len(page_text.strip()) < 200:
                        is_parked = True
                        status_message = "Minimal Page"
    
    except requests.exceptions.ConnectionError:
        status_code = -1
        status_message = "Connection Failed"
    except requests.exceptions.Timeout:
        status_code = -2
        status_message = "Timeout"
    except requests.exceptions.TooManyRedirects:
        status_code = -3
        status_message = "Too Many Redirects"
    except requests.exceptions.RequestException as e:
        status_code = -4
        status_message = f"Request Failed: {str(e)[:30]}"
    except Exception as e:
        status_code = -99
        status_message = f"Error: {str(e)[:30]}"
    
    return status_code, status_message, is_parked

# Enhanced function to categorize a website based on extracted keywords and meta descriptions
def categorize_website(keywords, meta_description="", url="", page_title=""):
    """
    Improved categorization function that uses multiple signals to determine the website category.
    
    Args:
        keywords: Extracted keywords string
        meta_description: Page meta description
        url: Website URL for domain-based hints
        page_title: Page title for additional context
        
    Returns:
        Tuple of (category, confidence)
    """
    if not keywords or keywords == "No keywords found":
        # Try to categorize based on URL and title alone
        if url or page_title:
            return categorize_by_url_and_title(url, page_title)
        return "Other", 0
    
    # Create a comprehensive text for analysis by combining all available data
    full_text = keywords.lower()
    if meta_description:
        full_text += " " + meta_description.lower()
    if page_title:
        full_text += " " + page_title.lower()
    if url:
        # Extract domain parts for additional signals
        domain_parts = extract_domain_parts(url)
        if domain_parts:
            full_text += " " + " ".join(domain_parts)
    
    # Split full text into individual words for more precise matching
    word_set = set(re.findall(r'\b\w+\b', full_text.lower()))
    
    # Calculate scores for each category
    category_scores = {}
    for category, category_keywords in CATEGORY_KEYWORDS.items():
        # Calculate score based on:
        # 1. Exact phrase matches (highest weight)
        # 2. Individual keyword matches
        # 3. Partial/substring matches (lower weight)
        
        # 1. First check for exact category phrases in the full text
        exact_match_score = 0
        for phrase in category_keywords:
            if len(phrase.split()) > 1:  # Multi-word phrases get higher weight
                if phrase in full_text:
                    exact_match_score += 5 * len(phrase.split())  # Higher weight for longer phrases
        
        # 2. Then check for individual keyword matches
        keyword_match_score = 0
        for keyword in category_keywords:
            # Check for whole word matches
            matches = re.findall(r'\b' + re.escape(keyword) + r'\b', full_text)
            if matches:
                # Add weight based on the number of matches and the specificity of the keyword
                weight = 3 if len(keyword) > 5 else 2  # Longer keywords get more weight
                keyword_match_score += len(matches) * weight
            
            # Check for word presence in set for faster matching
            if keyword in word_set:
                keyword_match_score += 2
                
        # 3. Check for partial/substring matches with lower weight
        partial_match_score = 0
        for keyword in category_keywords:
            if len(keyword) > 4:  # Only consider substantial keywords
                # Look for the keyword as part of compound words
                for word in word_set:
                    if len(word) > len(keyword) and keyword in word:
                        partial_match_score += 1
        
        # Combine scores with appropriate weighting
        total_score = exact_match_score + keyword_match_score + (partial_match_score * 0.5)
        
        # Apply industry-specific bonuses
        total_score = apply_industry_specific_rules(category, total_score, full_text, url)
        
        category_scores[category] = total_score
    
    # Find the category with the highest score
    max_score = max(category_scores.values()) if category_scores else 0
    
    # Only use "Other" if no category has a meaningful score
    if max_score < 2:
        # Try domain-based categorization as a last resort
        fallback_category, fallback_score = categorize_by_url_and_title(url, page_title)
        if fallback_score > 0:
            return fallback_category, fallback_score / 10  # Normalize confidence
        return "Other", 0
    
    # In case of a tie, use more sophisticated tie-breaking
    max_categories = [cat for cat, score in category_scores.items() if score == max_score]
    if len(max_categories) > 1:
        # Try to break ties using domain name and title
        if url or page_title:
            domain_category, domain_score = categorize_by_url_and_title(url, page_title)
            if domain_category in max_categories:
                best_category = domain_category
            else:
                # Otherwise use the category that appears earliest in the text
                best_category = find_earliest_category(max_categories, full_text)
        else:
            best_category = find_earliest_category(max_categories, full_text)
    else:
        best_category = max_categories[0]
    
    # Calculate confidence (0.0 to 1.0)
    confidence = min(max_score / 15, 1.0)  # Cap confidence at 1.0
    
    return best_category, confidence

# Helper function to extract meaningful parts from domain name
def extract_domain_parts(url):
    if not url:
        return []
    
    # Extract domain
    domain = ""
    if '://' in url:
        domain = url.split('://', 1)[1].split('/', 1)[0]
    else:
        domain = url.split('/', 1)[0]
    
    # Remove www. if present
    if domain.startswith('www.'):
        domain = domain[4:]
    
    # Extract parts
    parts = []
    
    # Split by dots and dashes
    for part in re.split(r'[.-]', domain):
        if len(part) > 2:  # Only meaningful parts
            # Remove common TLDs and domain words
            if part.lower() not in ['com', 'org', 'net', 'io', 'co', 'inc', 'llc', 'store', 'shop']:
                parts.append(part.lower())
    
    # Add compound words using camelCase and PascalCase detection
    for part in parts[:]:  # Work on a copy of the list
        # Find potential compound words
        compounds = re.findall(r'([a-z])([A-Z])', part)
        if compounds:
            # Split at capital letters
            words = re.sub(r'([a-z])([A-Z])', r'\1 \2', part).lower().split()
            parts.extend([w for w in words if len(w) > 2])
    
    return parts

# Helper function to find which category appears earliest in the text
def find_earliest_category(categories, text):
    first_index = float('inf')
    earliest_category = categories[0]  # Default
    
    for category in categories:
        for keyword in CATEGORY_KEYWORDS[category]:
            index = text.find(keyword)
            if index != -1 and index < first_index:
                first_index = index
                earliest_category = category
    
    return earliest_category

# Helper function for domain and title-based categorization
def categorize_by_url_and_title(url, title):
    if not url and not title:
        return "Other", 0
    
    # Create combined text from URL and title
    combined = ""
    domain_parts = extract_domain_parts(url)
    if domain_parts:
        combined += " ".join(domain_parts) + " "
    if title:
        combined += title.lower()
    
    # Convert to lowercase and create word set for efficient matching
    combined = combined.lower()
    word_set = set(re.findall(r'\b\w+\b', combined))
    
    # Calculate simple scores based on keyword presence
    scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword in combined:
                score += 2
            elif keyword in word_set:
                score += 1
        scores[category] = score
    
    # Get max score
    if not scores:
        return "Other", 0
        
    max_score = max(scores.values())
    if max_score == 0:
        return "Other", 0
        
    # Find category with max score
    max_categories = [cat for cat, score in scores.items() if score == max_score]
    return max_categories[0], max_score

# Apply special rules for specific industries to improve categorization
def apply_industry_specific_rules(category, score, text, url):
    
    # Footwear-specific bonuses
    if category == "Footwear":
        # Check for common footwear terms that may not be in the main keyword list
        footwear_indicators = ['shoes', 'footwear', 'sneaker', 'boot', 'sandal', 'trainer', 'running shoe']
        for indicator in footwear_indicators:
            if indicator in text:
                score += 3  # Strong bonus for shoe-specific terms
        
        # Check for sports + shoes combination
        if 'sport' in text and ('shoe' in text or 'footwear' in text):
            score += 4
            
        # Check for athletic terms often associated with footwear
        athletic_terms = ['running', 'training', 'workout', 'gym', 'fitness', 'athletics', 'sport']
        for term in athletic_terms:
            if term in text:
                score += 1.5  # Moderate bonus for athletic context
    
    # Electronics-specific bonuses
    elif category == "Electronics":
        tech_indicators = ['tech', 'electronic', 'device', 'gadget', 'computer', 'digital']
        for indicator in tech_indicators:
            if indicator in text:
                score += 2
    
    # Medical-specific bonuses
    elif category == "Medical":
        if 'health' in text and ('care' in text or 'medical' in text):
            score += 3
    
    # Food & Beverage bonuses
    elif category == "Food & Beverage":
        food_terms = ['recipe', 'ingredient', 'cook', 'kitchen', 'meal', 'restaurant', 'eat']
        for term in food_terms:
            if term in text:
                score += 2
    
    # Home Goods bonuses
    elif category == "Home Goods":
        home_terms = ['home', 'house', 'living', 'decor', 'furniture']
        for term in home_terms:
            if term in text:
                score += 2
    
    # Clothing bonuses
    elif category == "Clothing":
        clothing_terms = ['wear', 'apparel', 'clothing', 'fashion', 'style', 'outfit']
        for term in clothing_terms:
            if term in text:
                score += 2
    
    # URL-based bonuses
    if url:
        domain_lower = url.lower()
        if category == "Footwear" and ('shoe' in domain_lower or 'foot' in domain_lower):
            score += 4
        elif category == "Clothing" and ('apparel' in domain_lower or 'wear' in domain_lower or 'cloth' in domain_lower):
            score += 4
        elif category == "Electronics" and ('tech' in domain_lower or 'electronic' in domain_lower):
            score += 4
        elif category == "Food & Beverage" and ('food' in domain_lower or 'eat' in domain_lower or 'kitchen' in domain_lower):
            score += 4
    
    return score

# Enhanced function to extract website information with better content access
def extract_website_info(url, max_retries=3):
    try:
        # Format URL properly
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        # Remove trailing slashes for consistency
        url = url.rstrip('/')

        # Rotate user agents to avoid blocking
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0'
        ]
        
        # Enhanced headers to look more like a real browser
        headers = {
            'User-Agent': random.choice(user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://www.google.com/',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'cross-site',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        }
        
        # Create a session to handle cookies and redirects
        session = requests.Session()
        
        # Try both HTTPS and HTTP versions if needed
        protocols = ['https://', 'http://']
        success = False
        response = None
        final_url = None
        
        for protocol in protocols:
            # Skip if we already succeeded
            if success:
                break
                
            # Create protocol-specific URL
            if url.startswith('http'):
                # Use URL as is for first attempt
                if protocol == 'https://' and url.startswith('https://'):
                    current_url = url
                elif protocol == 'http://' and url.startswith('http://'):
                    current_url = url
                else:
                    # Remove existing protocol for second attempt
                    url_parts = url.split('://', 1)
                    if len(url_parts) > 1:
                        current_url = f"{protocol}{url_parts[1]}"
                    else:
                        current_url = f"{protocol}{url}"
            else:
                current_url = f"{protocol}{url}"
            
            # Try multiple user agents
            for attempt in range(max_retries):
                try:
                    # Use a different user agent for each retry
                    headers['User-Agent'] = random.choice(user_agents)
                    
                    # Fetch the website with a timeout, allowing redirects
                    response = session.get(current_url, headers=headers, timeout=15, allow_redirects=True)
                    
                    # Check if we got a successful response
                    if response.status_code == 200:
                        success = True
                        final_url = response.url  # Get final URL after possible redirects
                        break
                except requests.exceptions.RequestException:
                    # Wait before retrying
                    time.sleep(1 * (attempt + 1))
            
            # If protocol succeeded, break the loop
            if success:
                break
        
        # Check website status using our improved function
        if response:
            status_code, status_message, is_parked = check_website_status(url, response=response, html_content=response.text if success else None)
        else:
            status_code, status_message, is_parked = check_website_status(url)
        
        # If we couldn't connect at all
        if not success:
            return {
                'keywords': "Connection failed", 
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
        
        # 4. Enhanced content extraction - look in more places for content
        content_tags = ['article', 'main', 'section', 'div', 'p', 'span']
        content_classes = ['content', 'post', 'entry', 'article', 'main', 'blog', 'about', 'product', 
                          'service', 'description', 'text', 'body', 'page', 'container']
        
        # Look for content in semantic tags first
        for tag in content_tags[:4]:  # article, main, section, div
            for content in soup.find_all(tag):
                content_words = re.findall(r'\b\w+\b', content.text.lower())
                keywords.extend([word for word in content_words if len(word) > 3])
        
        # Look for content in divs with specific classes
        for cls in content_classes:
            for content in soup.find_all(['div', 'section'], class_=re.compile(cls, re.I)):
                content_words = re.findall(r'\b\w+\b', content.text.lower())
                keywords.extend([word for word in content_words if len(word) > 3])
        
        # 5. Look for text in paragraphs (often contains important content)
        for p in soup.find_all('p'):
            if len(p.text.strip()) > 20:  # Only meaningful paragraphs
                p_words = re.findall(r'\b\w+\b', p.text.lower())
                keywords.extend([word for word in p_words if len(word) > 3])
                
        # 6. Enhanced tag extraction
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
        
        # 7. Check for tags in URLs
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
        
        # Look for product names in the page (often good keywords for a business)
        product_patterns = soup.find_all(['div', 'section', 'article'], class_=re.compile('product|item|service', re.I))
        for product in product_patterns:
            product_name = product.find(['h2', 'h3', 'h4', 'strong', 'b'])
            if product_name:
                product_words = re.findall(r'\b\w+\b', product_name.text.lower())
                keywords.extend([word for word in product_words if len(word) > 3])
        
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
            'is_parked': is_parked,
            'final_url': final_url  # Include the final URL after redirects
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
            'is_parked': False,
            'final_url': None
        }

# NEW ENHANCED FUNCTIONS FOR BETTER URL-BASED CATEGORIZATION

# Enhanced domain tokenization
def extract_enhanced_domain_parts(domain):
    """
    Extract meaningful parts from domain name with improved tokenization
    
    Args:
        domain: Domain name (normalized)
        
    Returns:
        List of meaningful domain parts
    """
    if not domain:
        return []
    
    # Remove www. if present
    if domain.startswith('www.'):
        domain = domain[4:]
    
    # Extract parts
    parts = []
    
    # Step 1: Split by obvious separators
    for part in re.split(r'[.-]', domain):
        if len(part) > 2:  # Only meaningful parts
            # Remove common TLDs and domain words
            if part.lower() not in ['com', 'org', 'net', 'io', 'co', 'inc', 'llc', 'store', 'shop', 'app', 'tech', 'site', 'info', 'biz', 'online']:
                parts.append(part.lower())
    
    # Step 2: Detect and split compound words (camelCase and PascalCase)
    additional_parts = []
    for part in parts:
        # Find potential compound words
        compounds = re.findall(r'([a-z])([A-Z])', part)
        if compounds:
            # Split at capital letters
            words = re.sub(r'([a-z])([A-Z])', r'\1 \2', part).lower().split()
            additional_parts.extend([w for w in words if len(w) > 2])
    
    # Step 3: Dictionary-based word splitting for domains without separators
    # (This would require a dictionary - simplified version here)
    for part in parts[:]:  # Work on a copy
        if len(part) > 7:  # Only try to split longer words
            # Try to detect common embedded words
            for common_word in ['app', 'tech', 'shop', 'store', 'food', 'health', 'eco', 'bio', 'care', 'home', 'auto', 'web', 'cloud', 'soft', 'med', 'sport', 'fit', 'pet', 'edu', 'pro', 'office']:
                if common_word in part and part != common_word:
                    # Split around the common word
                    split_parts = part.split(common_word)
                    if all(len(sp) > 0 for sp in split_parts):  # Only if it results in meaningful splits
                        additional_parts.append(common_word)
                        additional_parts.extend([sp for sp in split_parts if len(sp) > 2])
    
    parts.extend(additional_parts)
    return list(set(parts))  # Remove duplicates

# Generate n-grams from domain parts
def generate_ngrams(parts, n=2):
    """
    Generate n-grams (pairs of words) from domain parts
    """
    ngrams = []
    for i in range(len(parts) - n + 1):
        ngram = " ".join(parts[i:i+n])
        ngrams.append(ngram)
    return ngrams

# Calculate base category score
def calculate_category_score(text, keywords):
    """
    Calculate basic score based on keyword matching
    """
    score = 0
    for keyword in keywords:
        # Exact match (higher weight)
        if keyword in text:
            # Weight by keyword specificity
            weight = 2
            # Higher weight for longer, more specific keywords
            if len(keyword) > 5:
                weight = 3
            score += weight
        
        # Partial match (lower weight)
        elif any(keyword in word for word in text.split()):
            score += 0.5
            
    return score

# Detect industry-specific patterns in domains
def detect_industry_patterns(domain, category):
    """
    Apply industry-specific pattern detection to domain names
    """
    domain_lower = domain.lower()
    
    score = 0
    
    # Industry-specific patterns
    if category == "Food & Beverage":
        food_patterns = ['food', 'eat', 'meal', 'dine', 'cook', 'chef', 'kitchen', 'cafe', 'restaurant', 'pizza', 'burger', 'grill', 'bakery', 'brew', 'coffee', 'tea', 'juice', 'fruit', 'veg', 'organic', 'farm', 'fresh', 'market', 'grocery', 'diet', 'nutrition']
        for pattern in food_patterns:
            if pattern in domain_lower:
                score += 3
    
    elif category == "Transportation":
        transport_patterns = ['transport', 'logistics', 'shipping', 'delivery', 'freight', 'cargo', 'truck', 'fleet', 'bus', 'taxi', 'travel', 'transit', 'courier', 'train', 'rail', 'ship', 'vessel', 'aviation', 'airline', 'airport', 'vehicle', 'mobility', 'move', 'flight', 'express']
        for pattern in transport_patterns:
            if pattern in domain_lower:
                score += 3
    
    elif category == "Supplement":
        supplement_patterns = ['supplement', 'vitamin', 'mineral', 'nutrition', 'dietary', 'protein', 'fitness', 'health', 'wellness', 'natural', 'organic', 'herb', 'omega', 'nutrient', 'biotech', 'life', 'body', 'active', 'energy', 'boost']
        for pattern in supplement_patterns:
            if pattern in domain_lower:
                score += 3
    
    elif category == "Electronics":
        tech_patterns = ['tech', 'electronic', 'digital', 'computer', 'device', 'gadget', 'software', 'hardware', 'app', 'web', 'net', 'cyber', 'data', 'mobile', 'phone', 'smart', 'robot', 'ai', 'code', 'dev', 'program']
        for pattern in tech_patterns:
            if pattern in domain_lower:
                score += 3
    
    elif category == "Medical":
        medical_patterns = ['med', 'health', 'care', 'clinic', 'hospital', 'doctor', 'therapy', 'pharma', 'drug', 'dental', 'vision', 'eye', 'cardio', 'neuro', 'ortho', 'rehab', 'physical', 'mental', 'psych', 'wellness']
        for pattern in medical_patterns:
            if pattern in domain_lower:
                score += 3
    
    elif category == "Clothing":
        clothing_patterns = ['wear', 'apparel', 'clothing', 'fashion', 'style', 'outfit', 'dress', 'shirt', 'pants', 'jacket', 'coat', 'suit', 'uniform', 'casual', 'formal', 'sport', 'design', 'boutique', 'textile', 'fabric']
        for pattern in clothing_patterns:
            if pattern in domain_lower:
                score += 3
    
    elif category == "Footwear":
        footwear_patterns = ['shoe', 'foot', 'boot', 'sneaker', 'sandal', 'heel', 'sole', 'insole', 'run', 'walk', 'comfort', 'sport', 'athletic', 'casual', 'formal']
        for pattern in footwear_patterns:
            if pattern in domain_lower:
                score += 3
    
    elif category == "Home Goods":
        home_patterns = ['home', 'house', 'living', 'furniture', 'decor', 'interior', 'kitchen', 'bath', 'bed', 'room', 'garden', 'appliance', 'domestic', 'design', 'comfort', 'light', 'lamp', 'rug', 'carpet']
        for pattern in home_patterns:
            if pattern in domain_lower:
                score += 3
    
    elif category == "Packaging":
        packaging_patterns = ['pack', 'box', 'container', 'wrap', 'packaging', 'ship', 'label', 'tape', 'carton', 'bottle', 'bag', 'pouch', 'film', 'plastic', 'paper', 'bundle']
        for pattern in packaging_patterns:
            if pattern in domain_lower:
                score += 3
                
    elif category == "Personal Care":
        care_patterns = ['beauty', 'skin', 'care', 'hair', 'makeup', 'cosmetic', 'salon', 'spa', 'groom', 'hygiene', 'perfume', 'fragrance', 'clean', 'lotion', 'cream']
        for pattern in care_patterns:
            if pattern in domain_lower:
                score += 3
                
    elif category == "Jewelry":
        jewelry_patterns = ['jewel', 'jewelry', 'gem', 'diamond', 'gold', 'silver', 'accessory', 'watch', 'ring', 'necklace', 'bracelet', 'earring', 'pearl', 'precious']
        for pattern in jewelry_patterns:
            if pattern in domain_lower:
                score += 3
                
    elif category == "Agriculture":
        agri_patterns = ['farm', 'agri', 'crop', 'garden', 'plant', 'grow', 'seed', 'harvest', 'organic', 'soil', 'food', 'produce', 'vegetable', 'fruit']
        for pattern in agri_patterns:
            if pattern in domain_lower:
                score += 3
                
    elif category == "Automotive":
        auto_patterns = ['auto', 'car', 'vehicle', 'motor', 'drive', 'truck', 'garage', 'repair', 'tire', 'wheel', 'engine', 'brake', 'part', 'dealer']
        for pattern in auto_patterns:
            if pattern in domain_lower:
                score += 3
                
    elif category == "Entertainment":
        entertainment_patterns = ['fun', 'game', 'play', 'entertainment', 'movie', 'film', 'music', 'theater', 'show', 'event', 'concert', 'ticket', 'festival', 'party', 'media', 'stream']
        for pattern in entertainment_patterns:
            if pattern in domain_lower:
                score += 3
                
    elif category == "Industrial":
        industrial_patterns = ['industry', 'industrial', 'factory', 'manufacture', 'machine', 'equipment', 'tool', 'part', 'material', 'production', 'supply', 'build', 'construct', 'steel', 'metal', 'product']
        for pattern in industrial_patterns:
            if pattern in domain_lower:
                score += 3
    
    return score

# Calculate score based on n-gram matching
def calculate_ngram_score(ngrams, keywords):
    """
    Calculate score based on n-gram matching with keywords
    """
    score = 0
    for ngram in ngrams:
        for keyword in keywords:
            if keyword in ngram:
                score += 2  # Higher weight for n-gram matches
    return score

# Apply context multipliers for better categorization
def get_context_multiplier(category, domain_parts):
    """
    Get context-based multiplier for category scores
    """
    # Default multiplier
    multiplier = 1.0
    
    # Strong industry indicators that deserve a boost
    strong_indicators = {
        "Food & Beverage": ['food', 'restaurant', 'cafe', 'catering', 'kitchen', 'chef', 'bakery', 'meal'],
        "Electronics": ['tech', 'electronic', 'digital', 'computer', 'mobile', 'smart'],
        "Medical": ['health', 'medical', 'clinic', 'hospital', 'pharma', 'therapy'],
        "Clothing": ['fashion', 'apparel', 'clothing', 'wear', 'outfit'],
        "Footwear": ['shoe', 'footwear', 'boot', 'sneaker'],
        "Supplement": ['supplement', 'vitamin', 'nutrition', 'wellness'],
        "Home Goods": ['home', 'furniture', 'decor', 'interior'],
        "Transportation": ['transport', 'logistics', 'delivery', 'shipping'],
        "Industrial": ['industrial', 'manufacturing', 'factory', 'equipment'],
        "Packaging": ['packaging', 'package', 'container', 'box'],
        "Personal Care": ['beauty', 'cosmetic', 'skincare', 'haircare'],
        "Jewelry": ['jewelry', 'accessory', 'gold', 'silver', 'diamond'],
        "Agriculture": ['farm', 'agriculture', 'crop', 'garden'],
        "Automotive": ['auto', 'car', 'vehicle', 'automotive'],
        "Entertainment": ['entertainment', 'game', 'movie', 'music']
    }
    
    # Check if any domain parts are strong indicators for this category
    if category in strong_indicators:
        for indicator in strong_indicators[category]:
            if indicator in domain_parts:
                multiplier = 1.5  # 50% boost
                break
    
    return multiplier

# Enhanced tie-breaking function
def break_category_tie(categories, domain):
    """
    Break ties between categories with equal scores
    """
    domain_lower = domain.lower()
    
    # Try to find the most specific match
    best_match = None
    best_match_score = 0
    
    for category in categories:
        # Get specific terms for this category
        specific_terms = CATEGORY_KEYWORDS.get(category, [])
        
        # Calculate how many specific terms appear in the domain
        match_score = 0
        for term in specific_terms:
            if term in domain_lower:
                # Give higher score to longer, more specific terms
                match_score += len(term)
        
        # Track the best match
        if match_score > best_match_score:
            best_match_score = match_score
            best_match = category
    
    # If we found a best match, return it
    if best_match:
        return best_match
    
    # Fallback: return the first category (arbitrary but consistent)
    return categories[0]

# Enhanced domain-based categorization function
def enhanced_categorize_by_url(url):
    """
    Enhanced function to categorize a website based solely on URL patterns
    
    Args:
        url: Website URL
        
    Returns:
        Tuple of (category, confidence)
    """
    if not url:
        return "Other", 0
    
    # Normalize URL and extract domain
    domain = normalize_url(url)
    
    # Extract meaningful parts from domain with improved tokenization
    domain_parts = extract_enhanced_domain_parts(domain)
    
    # Create a combined text from domain parts
    combined = " ".join(domain_parts)
    combined_lower = combined.lower()
    
    # Enhanced n-gram analysis (consider pairs of words)
    ngrams = generate_ngrams(domain_parts, 2)
    
    # Calculate scores with enhanced weighting
    category_scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        # Base score calculation
        base_score = calculate_category_score(combined_lower, keywords)
        
        # Enhanced pattern detection for industry-specific formats
        pattern_score = detect_industry_patterns(domain, category)
        
        # N-gram matching for multi-word terms
        ngram_score = calculate_ngram_score(ngrams, keywords)
        
        # Apply context multipliers
        context_multiplier = get_context_multiplier(category, domain_parts)
        
        # Combined weighted score
        total_score = (base_score * 1.0) + (pattern_score * 1.5) + (ngram_score * 1.2)
        total_score *= context_multiplier
        
        category_scores[category] = total_score
    
    # Get max score
    if not category_scores:
        return "Other", 0
        
    max_score = max(category_scores.values())
    if max_score == 0:
        return "Other", 0
        
    # Find category with max score
    max_categories = [cat for cat, score in category_scores.items() if score == max_score]
    
    # Tie-breaking with enhanced domain affinity
    if len(max_categories) > 1:
        best_category = break_category_tie(max_categories, domain)
    else:
        best_category = max_categories[0]
    
    # Calculate confidence (0.0 to 1.0)
    confidence = min(max_score / 15, 1.0)  # Cap confidence at 1.0
    
    return best_category, confidence

# Enhanced function to process a batch of websites
def enhanced_process_websites_batch(websites):
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
                
                # If keywords were not found or confidence is low, try enhanced URL-based categorization
                if info['keywords'] == "No keywords found" or info['confidence'] < 0.3:
                    enhanced_category, enhanced_confidence = enhanced_categorize_by_url(url)
                    
                    # Only use enhanced categorization if it has higher confidence
                    if enhanced_confidence > info['confidence']:
                        info['category'] = enhanced_category
                        info['confidence'] = enhanced_confidence
                
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
                # For errors, try URL-based categorization
                try:
                    url_category, url_confidence = enhanced_categorize_by_url(url)
                    results.append({
                        'Website': url,
                        'Status': "Error",
                        'Is Parked': "Unknown",
                        'Top Keywords': f"Error: {str(e)}",
                        'Meta Description': '',
                        'Page Title': '',
                        'Category': url_category,
                        'Confidence': f"{url_confidence:.2f}"
                    })
                except:
                    # Fallback if everything fails
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

# Original function to process a batch of websites (keeping for backward compatibility)
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
        
        # Add a toggle for using enhanced categorization
        use_enhanced = st.checkbox("Use Enhanced URL-based Categorization", value=True, 
                                 help="Turn on advanced domain analysis for better categorization when content can't be accessed")
        
        # Process button
        if st.button("Analyze Websites"):
            # Process websites and get results
            if use_enhanced:
                st.session_state['results_df'] = enhanced_process_websites_batch(websites)
            else:
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
