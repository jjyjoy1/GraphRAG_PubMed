#!/usr/bin/env python3
"""
PubMed Article Collector for Graph RAG Pipeline
Collects abstracts and metadata from PubMed using NCBI E-utilities API
"""

import requests
import xml.etree.ElementTree as ET
import json
import time
import csv
from typing import List, Dict, Optional
from dataclasses import dataclass
from urllib.parse import quote
import argparse
import logging

@dataclass
class PubMedArticle:
    """Data class to store PubMed article information"""
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    journal: str
    pub_date: str
    doi: Optional[str]
    keywords: List[str]
    mesh_terms: List[str]

class PubMedCollector:
    """
    Collects articles from PubMed using NCBI E-utilities API
    """
    
    def __init__(self, email: str, tool_name: str = "GraphRAG_Collector", api_key: str = None):
        """
        Initialize PubMed collector
        
        Args:
            email: Your email (required by NCBI)
            tool_name: Name of your tool (for NCBI logs)
            api_key: NCBI API key (optional, but recommended for higher rate limits)
        """
        self.email = email
        self.tool_name = tool_name
        self.api_key = api_key
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        
        # Rate limiting: 3 requests/second without API key, 10/second with API key
        self.delay = 0.34 if api_key else 0.34
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def search_pubmed(self, query: str, max_results: int = 1000, sort: str = "relevance") -> List[str]:
        """
        Search PubMed and return list of PMIDs
        
        Args:
            query: Search query (PubMed query syntax)
            max_results: Maximum number of results to return
            sort: Sort order (relevance, pub_date, etc.)
            
        Returns:
            List of PMIDs
        """
        search_url = f"{self.base_url}/esearch.fcgi"
        
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'xml',
            'sort': sort,
            'tool': self.tool_name,
            'email': self.email
        }
        
        if self.api_key:
            params['api_key'] = self.api_key
        
        try:
            self.logger.info(f"Searching PubMed with query: {query}")
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            pmids = [id_elem.text for id_elem in root.findall('.//Id')]
            
            self.logger.info(f"Found {len(pmids)} articles")
            return pmids
            
        except requests.RequestException as e:
            self.logger.error(f"Error searching PubMed: {e}")
            return []
        except ET.ParseError as e:
            self.logger.error(f"Error parsing XML response: {e}")
            return []
    
    def fetch_article_details(self, pmids: List[str]) -> List[PubMedArticle]:
        """
        Fetch detailed information for a list of PMIDs
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            List of PubMedArticle objects
        """
        articles = []
        batch_size = 200  # Process in batches to avoid overwhelming the API
        
        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i + batch_size]
            batch_articles = self._fetch_batch(batch_pmids)
            articles.extend(batch_articles)
            
            # Rate limiting
            time.sleep(self.delay)
            
            self.logger.info(f"Processed {min(i + batch_size, len(pmids))}/{len(pmids)} articles")
        
        return articles
    
    def _fetch_batch(self, pmids: List[str]) -> List[PubMedArticle]:
        """Fetch a batch of articles"""
        fetch_url = f"{self.base_url}/efetch.fcgi"
        
        params = {
            'db': 'pubmed',
            'id': ','.join(pmids),
            'retmode': 'xml',
            'tool': self.tool_name,
            'email': self.email
        }
        
        if self.api_key:
            params['api_key'] = self.api_key
        
        try:
            response = requests.get(fetch_url, params=params)
            response.raise_for_status()
            
            return self._parse_articles(response.content)
            
        except requests.RequestException as e:
            self.logger.error(f"Error fetching article details: {e}")
            return []
    
    def _parse_articles(self, xml_content: bytes) -> List[PubMedArticle]:
        """Parse XML content and extract article information"""
        articles = []
        
        try:
            root = ET.fromstring(xml_content)
            
            for article_elem in root.findall('.//PubmedArticle'):
                try:
                    article = self._parse_single_article(article_elem)
                    if article:
                        articles.append(article)
                except Exception as e:
                    self.logger.warning(f"Error parsing individual article: {e}")
                    continue
            
            return articles
            
        except ET.ParseError as e:
            self.logger.error(f"Error parsing XML content: {e}")
            return []
    
    def _parse_single_article(self, article_elem) -> Optional[PubMedArticle]:
        """Parse a single article element"""
        try:
            # Extract PMID
            pmid_elem = article_elem.find('.//PMID')
            pmid = pmid_elem.text if pmid_elem is not None else ""
            
            # Extract title
            title_elem = article_elem.find('.//ArticleTitle')
            title = title_elem.text if title_elem is not None else ""
            
            # Extract abstract
            abstract_parts = []
            for abstract_elem in article_elem.findall('.//AbstractText'):
                label = abstract_elem.get('Label', '')
                text = abstract_elem.text or ""
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
            abstract = " ".join(abstract_parts)
            
            # Extract authors
            authors = []
            for author_elem in article_elem.findall('.//Author'):
                last_name = author_elem.find('LastName')
                first_name = author_elem.find('ForeName')
                if last_name is not None and first_name is not None:
                    authors.append(f"{first_name.text} {last_name.text}")
            
            # Extract journal
            journal_elem = article_elem.find('.//Journal/Title')
            journal = journal_elem.text if journal_elem is not None else ""
            
            # Extract publication date
            pub_date_elem = article_elem.find('.//PubDate')
            pub_date = ""
            if pub_date_elem is not None:
                year = pub_date_elem.find('Year')
                month = pub_date_elem.find('Month')
                day = pub_date_elem.find('Day')
                
                date_parts = []
                if year is not None:
                    date_parts.append(year.text)
                if month is not None:
                    date_parts.append(month.text)
                if day is not None:
                    date_parts.append(day.text)
                pub_date = "-".join(date_parts)
            
            # Extract DOI
            doi = ""
            for id_elem in article_elem.findall('.//ArticleId'):
                if id_elem.get('IdType') == 'doi':
                    doi = id_elem.text
                    break
            
            # Extract keywords
            keywords = []
            for keyword_elem in article_elem.findall('.//Keyword'):
                if keyword_elem.text:
                    keywords.append(keyword_elem.text)
            
            # Extract MeSH terms
            mesh_terms = []
            for mesh_elem in article_elem.findall('.//MeshHeading/DescriptorName'):
                if mesh_elem.text:
                    mesh_terms.append(mesh_elem.text)
            
            return PubMedArticle(
                pmid=pmid,
                title=title,
                abstract=abstract,
                authors=authors,
                journal=journal,
                pub_date=pub_date,
                doi=doi,
                keywords=keywords,
                mesh_terms=mesh_terms
            )
            
        except Exception as e:
            self.logger.warning(f"Error parsing article: {e}")
            return None
    
    def save_to_json(self, articles: List[PubMedArticle], filename: str):
        """Save articles to JSON file"""
        data = []
        for article in articles:
            data.append({
                'pmid': article.pmid,
                'title': article.title,
                'abstract': article.abstract,
                'authors': article.authors,
                'journal': article.journal,
                'pub_date': article.pub_date,
                'doi': article.doi,
                'keywords': article.keywords,
                'mesh_terms': article.mesh_terms
            })
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved {len(articles)} articles to {filename}")
    
    def save_to_csv(self, articles: List[PubMedArticle], filename: str):
        """Save articles to CSV file"""
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['PMID', 'Title', 'Abstract', 'Authors', 'Journal', 
                           'Publication_Date', 'DOI', 'Keywords', 'MeSH_Terms'])
            
            for article in articles:
                writer.writerow([
                    article.pmid,
                    article.title,
                    article.abstract,
                    '; '.join(article.authors),
                    article.journal,
                    article.pub_date,
                    article.doi,
                    '; '.join(article.keywords),
                    '; '.join(article.mesh_terms)
                ])
        
        self.logger.info(f"Saved {len(articles)} articles to {filename}")

def build_gwas_query(disease: str, additional_terms: List[str] = None) -> str:
    """
    Build a PubMed query for GWAS studies of a specific disease
    
    Args:
        disease: Disease name (e.g., "breast cancer")
        additional_terms: Additional search terms
        
    Returns:
        Formatted PubMed query string
    """
    gwas_terms = [
        "genome-wide association study",
        "genome-wide association studies", 
        "GWAS",
        "genome-wide association",
        "genomewide association"
    ]
    
    # Combine GWAS terms with OR
    gwas_query = "(" + " OR ".join([f'"{term}"' for term in gwas_terms]) + ")"
    
    # Add disease term
    disease_query = f'"{disease}"'
    
    # Combine with AND
    full_query = f"{disease_query} AND {gwas_query}"
    
    # Add additional terms if provided
    if additional_terms:
        additional_query = " AND ".join([f'"{term}"' for term in additional_terms])
        full_query += f" AND ({additional_query})"
    
    return full_query

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Collect PubMed articles for Graph RAG pipeline')
    parser.add_argument('--email', required=True, help='Your email address (required by NCBI)')
    parser.add_argument('--query', required=True, help='PubMed search query')
    parser.add_argument('--max_results', type=int, default=1000, help='Maximum number of results')
    parser.add_argument('--output', default='pubmed_articles', help='Output filename (without extension)')
    parser.add_argument('--format', choices=['json', 'csv', 'both'], default='json', help='Output format')
    parser.add_argument('--api_key', help='NCBI API key (optional)')
    
    args = parser.parse_args()
    
    # Initialize collector
    collector = PubMedCollector(
        email=args.email,
        api_key=args.api_key
    )
    
    # Search and collect articles
    pmids = collector.search_pubmed(args.query, args.max_results)
    
    if not pmids:
        print("No articles found for the given query.")
        return
    
    articles = collector.fetch_article_details(pmids)
    
    if not articles:
        print("No article details could be retrieved.")
        return
    
    # Save results
    if args.format in ['json', 'both']:
        collector.save_to_json(articles, f"{args.output}.json")
    
    if args.format in ['csv', 'both']:
        collector.save_to_csv(articles, f"{args.output}.csv")
    
    print(f"Successfully collected {len(articles)} articles!")

# Example usage for breast cancer GWAS studies
if __name__ == "__main__":
    # Example: Collect breast cancer GWAS studies
    # Uncomment and modify the following code for interactive usage
    
    """
    # Initialize collector with your email
    collector = PubMedCollector(
        email="your.email@domain.com",  # Replace with your email
        api_key="your_api_key"  # Optional: get from https://www.ncbi.nlm.nih.gov/account/settings/
    )
    
    # Build query for breast cancer GWAS studies
    query = build_gwas_query("breast cancer")
    print(f"Search query: {query}")
    
    # Search and collect articles
    pmids = collector.search_pubmed(query, max_results=500)
    articles = collector.fetch_article_details(pmids)
    
    # Save results
    collector.save_to_json(articles, "breast_cancer_gwas.json")
    collector.save_to_csv(articles, "breast_cancer_gwas.csv")
    
    print(f"Collected {len(articles)} breast cancer GWAS articles")
    """
    
    # Run command-line interface
    main()

