#!/usr/bin/env python3
"""
Enhanced Biomedical Entity Extractor for Graph RAG Pipeline
Uses standardized ontologies and databases (PubTator, UMLS, NCBI)
"""

import json
import csv
import re
import requests
import time
from typing import List, Dict, Set, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import argparse
import logging
from pathlib import Path
from urllib.parse import quote
import xml.etree.ElementTree as ET

@dataclass
class StandardizedEntity:
    """Represents a standardized biomedical entity with database IDs"""
    text: str
    entity_type: str
    start_pos: int
    end_pos: int
    confidence: float
    source: str
    context: str = ""
    
    # Standardized identifiers
    database_id: Optional[str] = None  # Primary database ID
    umls_cui: Optional[str] = None     # UMLS Concept Unique Identifier
    ncbi_gene_id: Optional[str] = None # NCBI Gene ID
    mesh_id: Optional[str] = None      # MeSH term ID
    chebi_id: Optional[str] = None     # ChEBI ID for chemicals
    
    # Additional metadata
    preferred_name: Optional[str] = None  # Standardized name
    synonyms: List[str] = field(default_factory=list)
    semantic_types: List[str] = field(default_factory=list)
    attributes: Dict = field(default_factory=dict)

@dataclass
class StandardizedRelation:
    """Represents a standardized relationship with ontology support"""
    entity1_id: str
    entity2_id: str
    entity1_text: str
    entity2_text: str
    relation_type: str
    relation_id: Optional[str] = None  # Relation ontology ID
    confidence: float = 0.0
    evidence: str = ""
    source_pmid: str = ""
    semantic_relation: Optional[str] = None  # UMLS semantic relation

class EnhancedBiomedicalExtractor:
    """
    Enhanced biomedical entity extractor using standardized ontologies
    """
    
    def __init__(self, email: str, ncbi_api_key: str = None, use_pubtator: bool = True, 
                 use_umls: bool = True, use_ncbi: bool = True):
        """
        Initialize enhanced extractor with standardized resources
        
        Args:
            email: Email for NCBI services
            ncbi_api_key: NCBI API key for higher rate limits
            use_pubtator: Use PubTator Central for entity recognition
            use_umls: Use UMLS for concept normalization
            use_ncbi: Use NCBI databases for validation
        """
        self.email = email
        self.ncbi_api_key = ncbi_api_key
        self.use_pubtator = use_pubtator
        self.use_umls = use_umls
        self.use_ncbi = use_ncbi
        
        # API endpoints
        self.pubtator_url = "https://www.ncbi.nlm.nih.gov/research/pubtator-api"
        self.ncbi_base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.umls_base_url = "https://uts-ws.nlm.nih.gov/rest"
        
        # Rate limiting
        self.request_delay = 0.34  # Respect NCBI rate limits
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Initialize caches
        self.entity_cache = {}
        self.umls_cache = {}
        self.ncbi_cache = {}
        
        # Standardized entity type mappings
        self.standard_entity_types = {
            'Gene': 'GENE',
            'Disease': 'DISEASE', 
            'Chemical': 'CHEMICAL',
            'Species': 'SPECIES',
            'Mutation': 'VARIANT',
            'CellLine': 'CELL_LINE',
            'DNAMutation': 'VARIANT',
            'ProteinMutation': 'VARIANT',
            'SNP': 'VARIANT'
        }
        
        # UMLS semantic type mappings
        self.umls_semantic_types = {
            'T116': 'Amino Acid, Peptide, or Protein',
            'T028': 'Gene or Genome',
            'T047': 'Disease or Syndrome',
            'T121': 'Pharmacologic Substance',
            'T109': 'Organic Chemical',
            'T086': 'Nucleotide Sequence'
        }
    
    def extract_entities_from_pmids(self, pmids: List[str]) -> Dict[str, List[StandardizedEntity]]:
        """
        Extract standardized entities from PubMed articles using PMIDs
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            Dictionary mapping PMIDs to extracted entities
        """
        all_entities = {}
        
        if self.use_pubtator:
            self.logger.info("Using PubTator Central for entity extraction")
            pubtator_entities = self._extract_with_pubtator(pmids)
            
            for pmid, entities in pubtator_entities.items():
                all_entities[pmid] = entities
        
        # Enhance entities with additional standardized information
        for pmid, entities in all_entities.items():
            enhanced_entities = []
            for entity in entities:
                enhanced_entity = self._enhance_entity_with_databases(entity)
                enhanced_entities.append(enhanced_entity)
            all_entities[pmid] = enhanced_entities
        
        return all_entities
    
    def _extract_with_pubtator(self, pmids: List[str]) -> Dict[str, List[StandardizedEntity]]:
        """Extract entities using PubTator Central API"""
        entities_by_pmid = {}
        
        # Process PMIDs in batches
        batch_size = 100
        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i + batch_size]
            
            try:
                # PubTator API call
                pmid_string = ','.join(batch_pmids)
                url = f"{self.pubtator_url}/publications/export/biocjson"
                params = {'pmids': pmid_string}
                
                self.logger.info(f"Fetching entities for PMIDs {i+1}-{min(i+batch_size, len(pmids))}")
                
                response = requests.get(url, params=params)
                response.raise_for_status()
                
                # Parse PubTator response
                pubtator_data = response.json()
                batch_entities = self._parse_pubtator_response(pubtator_data)
                entities_by_pmid.update(batch_entities)
                
                time.sleep(self.request_delay)
                
            except requests.RequestException as e:
                self.logger.error(f"Error fetching from PubTator: {e}")
                continue
            except json.JSONDecodeError as e:
                self.logger.error(f"Error parsing PubTator response: {e}")
                continue
        
        return entities_by_pmid
    
    def _parse_pubtator_response(self, pubtator_data: Union[Dict, List]) -> Dict[str, List[StandardizedEntity]]:
        """Parse PubTator Central JSON response"""
        entities_by_pmid = {}
        
        # Handle both single document and multiple documents
        if isinstance(pubtator_data, dict):
            documents = [pubtator_data]
        elif isinstance(pubtator_data, list):
            documents = pubtator_data
        else:
            return entities_by_pmid
        
        for document in documents:
            pmid = str(document.get('pmid', ''))
            if not pmid:
                continue
            
            entities = []
            
            # Extract entities from annotations
            annotations = document.get('annotations', [])
            passages = document.get('passages', [])
            
            # Get full text for context
            full_text = ""
            for passage in passages:
                full_text += passage.get('text', '') + " "
            
            for annotation in annotations:
                try:
                    entity = StandardizedEntity(
                        text=annotation.get('text', ''),
                        entity_type=self.standard_entity_types.get(
                            annotation.get('infons', {}).get('type', ''), 
                            annotation.get('infons', {}).get('type', '')
                        ),
                        start_pos=annotation.get('locations', [{}])[0].get('offset', 0),
                        end_pos=annotation.get('locations', [{}])[0].get('offset', 0) + 
                               annotation.get('locations', [{}])[0].get('length', 0),
                        confidence=0.9,  # PubTator has high confidence
                        source="PubTator",
                        context=self._get_context(full_text, 
                                                annotation.get('locations', [{}])[0].get('offset', 0),
                                                annotation.get('locations', [{}])[0].get('offset', 0) + 
                                                annotation.get('locations', [{}])[0].get('length', 0)),
                        database_id=annotation.get('infons', {}).get('identifier', ''),
                        attributes={'pmid': pmid}
                    )
                    
                    # Add additional identifiers from PubTator
                    infons = annotation.get('infons', {})
                    if 'NCBI Gene' in infons:
                        entity.ncbi_gene_id = infons['NCBI Gene']
                    if 'MESH' in infons:
                        entity.mesh_id = infons['MESH']
                    
                    entities.append(entity)
                    
                except Exception as e:
                    self.logger.warning(f"Error parsing annotation: {e}")
                    continue
            
            entities_by_pmid[pmid] = entities
        
        return entities_by_pmid
    
    def _enhance_entity_with_databases(self, entity: StandardizedEntity) -> StandardizedEntity:
        """Enhance entity with additional database information"""
        
        # Get UMLS information
        if self.use_umls and entity.text:
            umls_info = self._get_umls_info(entity.text)
            if umls_info:
                entity.umls_cui = umls_info.get('cui')
                entity.preferred_name = umls_info.get('preferred_name')
                entity.synonyms = umls_info.get('synonyms', [])
                entity.semantic_types = umls_info.get('semantic_types', [])
        
        # Get NCBI Gene information for genes
        if self.use_ncbi and entity.entity_type == 'GENE' and entity.text:
            ncbi_info = self._get_ncbi_gene_info(entity.text)
            if ncbi_info:
                entity.ncbi_gene_id = ncbi_info.get('gene_id')
                entity.preferred_name = ncbi_info.get('gene_name')
                entity.synonyms.extend(ncbi_info.get('aliases', []))
        
        return entity
    
    def _get_umls_info(self, term: str) -> Optional[Dict]:
        """Get UMLS concept information"""
        if term in self.umls_cache:
            return self.umls_cache[term]
        
        try:
            # UMLS REST API search
            search_url = f"{self.umls_base_url}/search/current"
            params = {
                'string': term,
                'searchType': 'exact',
                'returnIdType': 'concept'
            }
            
            response = requests.get(search_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('result', {}).get('results', [])
                
                if results:
                    result = results[0]  # Take first result
                    umls_info = {
                        'cui': result.get('ui'),
                        'preferred_name': result.get('name'),
                        'semantic_types': [result.get('semanticType', {}).get('name', '')],
                        'synonyms': []
                    }
                    
                    # Cache result
                    self.umls_cache[term] = umls_info
                    return umls_info
            
            time.sleep(self.request_delay)
            
        except Exception as e:
            self.logger.warning(f"Error fetching UMLS info for {term}: {e}")
        
        self.umls_cache[term] = None
        return None
    
    def _get_ncbi_gene_info(self, gene_symbol: str) -> Optional[Dict]:
        """Get NCBI Gene database information"""
        if gene_symbol in self.ncbi_cache:
            return self.ncbi_cache[gene_symbol]
        
        try:
            # Search NCBI Gene database
            search_url = f"{self.ncbi_base_url}/esearch.fcgi"
            search_params = {
                'db': 'gene',
                'term': f"{gene_symbol}[Gene Name] AND Homo sapiens[Organism]",
                'retmax': 1,
                'retmode': 'json',
                'tool': 'GraphRAG_Extractor',
                'email': self.email
            }
            
            if self.ncbi_api_key:
                search_params['api_key'] = self.ncbi_api_key
            
            search_response = requests.get(search_url, params=search_params)
            search_data = search_response.json()
            
            gene_ids = search_data.get('esearchresult', {}).get('idlist', [])
            
            if gene_ids:
                # Fetch gene details
                fetch_url = f"{self.ncbi_base_url}/efetch.fcgi"
                fetch_params = {
                    'db': 'gene',
                    'id': gene_ids[0],
                    'retmode': 'xml',
                    'tool': 'GraphRAG_Extractor',
                    'email': self.email
                }
                
                if self.ncbi_api_key:
                    fetch_params['api_key'] = self.ncbi_api_key
                
                fetch_response = requests.get(fetch_url, params=fetch_params)
                
                # Parse XML response
                root = ET.fromstring(fetch_response.content)
                
                gene_info = {
                    'gene_id': gene_ids[0],
                    'gene_name': '',
                    'aliases': []
                }
                
                # Extract gene name and aliases
                for gene_ref in root.findall('.//Gene-ref'):
                    locus = gene_ref.find('.//Gene-ref_locus')
                    if locus is not None:
                        gene_info['gene_name'] = locus.text
                    
                    # Extract aliases
                    syn_elements = gene_ref.findall('.//Gene-ref_syn/Gene-ref_syn_E')
                    aliases = [syn.text for syn in syn_elements if syn.text]
                    gene_info['aliases'] = aliases
                
                # Cache result
                self.ncbi_cache[gene_symbol] = gene_info
                return gene_info
            
            time.sleep(self.request_delay)
            
        except Exception as e:
            self.logger.warning(f"Error fetching NCBI info for {gene_symbol}: {e}")
        
        self.ncbi_cache[gene_symbol] = None
        return None
    
    def extract_standardized_relations(self, entities_by_pmid: Dict[str, List[StandardizedEntity]]) -> List[StandardizedRelation]:
        """Extract standardized relationships between entities"""
        all_relations = []
        
        for pmid, entities in entities_by_pmid.items():
            # Co-occurrence based relations
            cooccurrence_relations = self._extract_cooccurrence_relations(entities, pmid)
            all_relations.extend(cooccurrence_relations)
            
            # Knowledge-based relations (if entities have database IDs)
            knowledge_relations = self._extract_knowledge_relations(entities, pmid)
            all_relations.extend(knowledge_relations)
        
        return all_relations
    
    def _extract_cooccurrence_relations(self, entities: List[StandardizedEntity], pmid: str) -> List[StandardizedRelation]:
        """Extract relationships based on entity co-occurrence"""
        relations = []
        
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                if entity1.entity_type == entity2.entity_type:
                    continue
                
                # Calculate distance
                distance = abs(entity1.start_pos - entity2.start_pos)
                
                if distance < 150:  # Close proximity
                    relation_type = self._determine_relation_type(entity1.entity_type, entity2.entity_type)
                    
                    if relation_type:
                        relation = StandardizedRelation(
                            entity1_id=entity1.database_id or entity1.text,
                            entity2_id=entity2.database_id or entity2.text,
                            entity1_text=entity1.text,
                            entity2_text=entity2.text,
                            relation_type=relation_type,
                            confidence=min(entity1.confidence, entity2.confidence) * 0.8,
                            evidence=f"Co-occurrence in PMID {pmid}",
                            source_pmid=pmid
                        )
                        relations.append(relation)
        
        return relations
    
    def _extract_knowledge_relations(self, entities: List[StandardizedEntity], pmid: str) -> List[StandardizedRelation]:
        """Extract relationships based on knowledge bases"""
        relations = []
        
        # This could be enhanced with specific knowledge base queries
        # For now, we'll use semantic type-based relationships
        
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                if entity1.umls_cui and entity2.umls_cui:
                    # Use UMLS semantic relationships
                    semantic_relation = self._get_umls_semantic_relation(
                        entity1.semantic_types, entity2.semantic_types
                    )
                    
                    if semantic_relation:
                        relation = StandardizedRelation(
                            entity1_id=entity1.umls_cui,
                            entity2_id=entity2.umls_cui,
                            entity1_text=entity1.text,
                            entity2_text=entity2.text,
                            relation_type=semantic_relation,
                            confidence=0.7,
                            evidence=f"UMLS semantic relationship in PMID {pmid}",
                            source_pmid=pmid,
                            semantic_relation=semantic_relation
                        )
                        relations.append(relation)
        
        return relations
    
    def _determine_relation_type(self, type1: str, type2: str) -> Optional[str]:
        """Determine relationship type based on standardized entity types"""
        relations_map = {
            ('GENE', 'DISEASE'): 'gene_disease_association',
            ('GENE', 'VARIANT'): 'gene_variant_association',
            ('VARIANT', 'DISEASE'): 'variant_disease_association',
            ('GENE', 'CHEMICAL'): 'gene_chemical_interaction',
            ('CHEMICAL', 'DISEASE'): 'chemical_disease_association',
            ('CHEMICAL', 'GENE'): 'chemical_gene_interaction',
        }
        
        return relations_map.get((type1, type2)) or relations_map.get((type2, type1))
    
    def _get_umls_semantic_relation(self, types1: List[str], types2: List[str]) -> Optional[str]:
        """Get UMLS semantic relationship between entity types"""
        # Simplified semantic relationship mapping
        semantic_relations = {
            ('Gene or Genome', 'Disease or Syndrome'): 'causally_related_to',
            ('Pharmacologic Substance', 'Disease or Syndrome'): 'treats',
            ('Amino Acid, Peptide, or Protein', 'Disease or Syndrome'): 'associated_with',
        }
        
        for type1 in types1:
            for type2 in types2:
                relation = semantic_relations.get((type1, type2)) or semantic_relations.get((type2, type1))
                if relation:
                    return relation
        
        return None
    
    def _get_context(self, text: str, start: int, end: int, window: int = 75) -> str:
        """Get surrounding context for an entity"""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]
    
    def process_pubmed_file(self, input_file: str) -> Dict:
        """
        Process PubMed articles and extract standardized entities
        
        Args:
            input_file: Path to JSON file from step 1
            
        Returns:
            Dictionary with standardized entities and relationships
        """
        self.logger.info(f"Processing PubMed file with standardized extraction: {input_file}")
        
        # Load articles
        with open(input_file, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        
        # Extract PMIDs
        pmids = [article['pmid'] for article in articles if article.get('pmid')]
        
        self.logger.info(f"Extracting entities for {len(pmids)} articles")
        
        # Extract standardized entities
        entities_by_pmid = self.extract_entities_from_pmids(pmids)
        
        # Extract relationships
        relations = self.extract_standardized_relations(entities_by_pmid)
        
        # Flatten entities for summary
        all_entities = []
        for pmid_entities in entities_by_pmid.values():
            all_entities.extend(pmid_entities)
        
        # Generate summary
        entity_counts = Counter(entity.entity_type for entity in all_entities)
        database_coverage = {
            'entities_with_database_id': sum(1 for e in all_entities if e.database_id),
            'entities_with_umls_cui': sum(1 for e in all_entities if e.umls_cui),
            'entities_with_ncbi_id': sum(1 for e in all_entities if e.ncbi_gene_id),
        }
        
        summary = {
            'total_articles_processed': len(articles),
            'total_entities_extracted': len(all_entities),
            'entity_type_counts': dict(entity_counts),
            'total_relations': len(relations),
            'database_coverage': database_coverage,
            'extraction_methods': {
                'pubtator': self.use_pubtator,
                'umls': self.use_umls,
                'ncbi': self.use_ncbi
            }
        }
        
        self.logger.info(f"Standardized extraction complete: {summary}")
        
        return {
            'entities_by_pmid': entities_by_pmid,
            'relations': relations,
            'summary': summary
        }
    
    def save_standardized_results(self, results: Dict, output_prefix: str):
        """Save standardized extraction results"""
        
        # Flatten entities
        all_entities = []
        for pmid_entities in results['entities_by_pmid'].values():
            all_entities.extend(pmid_entities)
        
        # Save entities with standardized information
        entities_data = []
        for entity in all_entities:
            entities_data.append({
                'text': entity.text,
                'entity_type': entity.entity_type,
                'confidence': entity.confidence,
                'source': entity.source,
                'context': entity.context,
                'database_id': entity.database_id,
                'umls_cui': entity.umls_cui,
                'ncbi_gene_id': entity.ncbi_gene_id,
                'mesh_id': entity.mesh_id,
                'preferred_name': entity.preferred_name,
                'synonyms': entity.synonyms,
                'semantic_types': entity.semantic_types,
                'attributes': entity.attributes
            })
        
        with open(f"{output_prefix}_standardized_entities.json", 'w', encoding='utf-8') as f:
            json.dump(entities_data, f, indent=2, ensure_ascii=False)
        
        # Save relations
        relations_data = []
        for relation in results['relations']:
            relations_data.append({
                'entity1_id': relation.entity1_id,
                'entity2_id': relation.entity2_id,
                'entity1_text': relation.entity1_text,
                'entity2_text': relation.entity2_text,
                'relation_type': relation.relation_type,
                'relation_id': relation.relation_id,
                'confidence': relation.confidence,
                'evidence': relation.evidence,
                'source_pmid': relation.source_pmid,
                'semantic_relation': relation.semantic_relation
            })
        
        with open(f"{output_prefix}_standardized_relations.json", 'w', encoding='utf-8') as f:
            json.dump(relations_data, f, indent=2, ensure_ascii=False)
        
        # Save as CSV for easy viewing
        with open(f"{output_prefix}_standardized_entities.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'text', 'entity_type', 'confidence', 'database_id', 'umls_cui', 
                'ncbi_gene_id', 'preferred_name', 'synonyms', 'semantic_types', 'pmid'
            ])
            
            for entity in all_entities:
                writer.writerow([
                    entity.text,
                    entity.entity_type,
                    entity.confidence,
                    entity.database_id,
                    entity.umls_cui,
                    entity.ncbi_gene_id,
                    entity.preferred_name,
                    '; '.join(entity.synonyms) if entity.synonyms else '',
                    '; '.join(entity.semantic_types) if entity.semantic_types else '',
                    entity.attributes.get('pmid', '')
                ])
        
        # Save summary
        with open(f"{output_prefix}_standardized_summary.json", 'w', encoding='utf-8') as f:
            json.dump(results['summary'], f, indent=2)
        
        self.logger.info(f"Standardized results saved with prefix: {output_prefix}")

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Extract standardized biomedical entities')
    parser.add_argument('--input', required=True, help='Input JSON file from step 1')
    parser.add_argument('--output', default='standardized_entities', help='Output file prefix')
    parser.add_argument('--email', required=True, help='Email for NCBI services')
    parser.add_argument('--ncbi_api_key', help='NCBI API key (recommended)')
    parser.add_argument('--use_pubtator', action='store_true', default=True, help='Use PubTator Central')
    parser.add_argument('--use_umls', action='store_true', default=True, help='Use UMLS')
    parser.add_argument('--use_ncbi', action='store_true', default=True, help='Use NCBI databases')
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: Input file {args.input} not found")
        return
    
    # Initialize enhanced extractor
    extractor = EnhancedBiomedicalExtractor(
        email=args.email,
        ncbi_api_key=args.ncbi_api_key,
        use_pubtator=args.use_pubtator,
        use_umls=args.use_umls,
        use_ncbi=args.use_ncbi
    )
    
    # Process file
    results = extractor.process_pubmed_file(args.input)
    
    # Save results
    extractor.save_standardized_results(results, args.output)
    
    print("Enhanced entity extraction complete!")
    print(f"Extracted {results['summary']['total_entities_extracted']} standardized entities")
    print(f"Database coverage: {results['summary']['database_coverage']}")
    print(f"Found {results['summary']['total_relations']} relationships")

if __name__ == "__main__":
    main()
