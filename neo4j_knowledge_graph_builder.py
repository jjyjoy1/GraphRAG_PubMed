#!/usr/bin/env python3
"""
Biomedical Graph RAG Query System
Complete pipeline for querying biomedical knowledge graphs with LLM integration
"""

import json
import re
import requests
from typing import List, Dict, Set, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import argparse
import logging
from pathlib import Path
from collections import defaultdict, Counter
import time

# Neo4j imports
try:
    from neo4j import GraphDatabase, basic_auth
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

# Optional LLM integrations
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

@dataclass
class QueryResult:
    """Represents a query result with context and answer"""
    query: str
    entities_found: List[Dict]
    graph_context: Dict
    answer: str
    confidence: float
    sources: List[str]
    execution_time: float
    reasoning_steps: List[str]

@dataclass
class GraphContext:
    """Represents graph context for LLM"""
    nodes: List[Dict]
    relationships: List[Dict]
    paths: List[List[Dict]]
    summary: str
    confidence_scores: List[float]

class BiomedicalGraphRAG:
    """
    Complete Biomedical Graph RAG System
    """
    
    def __init__(self, neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j", neo4j_password: str = "password",
                 neo4j_database: str = "neo4j",
                 llm_provider: str = "ollama", llm_model: str = "llama3.1",
                 llm_base_url: str = "http://localhost:11434"):
        """
        Initialize the Graph RAG system
        
        Args:
            neo4j_uri: Neo4j database URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            neo4j_database: Neo4j database name
            llm_provider: LLM provider (ollama, openai, etc.)
            llm_model: LLM model name
            llm_base_url: LLM API base URL
        """
        # Neo4j configuration
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.neo4j_database = neo4j_database
        self.driver = None
        
        # LLM configuration
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Query patterns for entity extraction
        self.entity_patterns = {
            'gene': [
                r'\b[A-Z][A-Z0-9]+\b',  # Gene symbols
                r'\b[A-Z]+[0-9]+[A-Z]*\b',
            ],
            'disease': [
                r'cancer\b',
                r'carcinoma\b', 
                r'tumor\b',
                r'disease\b',
                r'syndrome\b'
            ],
            'chemical': [
                r'drug\b',
                r'compound\b',
                r'therapy\b',
                r'treatment\b'
            ]
        }
        
        # Compile patterns
        self.compiled_patterns = {}
        for category, patterns in self.entity_patterns.items():
            self.compiled_patterns[category] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def connect_neo4j(self):
        """Connect to Neo4j database"""
        try:
            if not NEO4J_AVAILABLE:
                raise ImportError("Neo4j driver not available")
            
            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=basic_auth(self.neo4j_user, self.neo4j_password),
                encrypted=False
            )
            
            # Test connection
            with self.driver.session(database=self.neo4j_database) as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                if test_value == 1:
                    self.logger.info("Connected to Neo4j successfully")
                else:
                    raise ConnectionError("Neo4j connection test failed")
                    
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def disconnect_neo4j(self):
        """Disconnect from Neo4j"""
        if self.driver:
            self.driver.close()
            self.logger.info("Disconnected from Neo4j")
    
    def extract_entities_from_query(self, query: str) -> List[Dict]:
        """Extract potential biomedical entities from user query"""
        entities = []
        
        # Pattern-based extraction
        for category, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                matches = pattern.findall(query)
                for match in matches:
                    entities.append({
                        'text': match,
                        'category': category,
                        'method': 'pattern'
                    })
        
        # Simple keyword extraction for known biomedical terms
        biomedical_keywords = [
            'BRCA1', 'BRCA2', 'TP53', 'EGFR', 'KRAS', 'PIK3CA',
            'breast cancer', 'lung cancer', 'diabetes', 'alzheimer',
            'chemotherapy', 'radiotherapy', 'immunotherapy'
        ]
        
        query_lower = query.lower()
        for keyword in biomedical_keywords:
            if keyword.lower() in query_lower:
                entities.append({
                    'text': keyword,
                    'category': 'biomedical',
                    'method': 'keyword'
                })
        
        # Deduplicate entities
        unique_entities = []
        seen = set()
        for entity in entities:
            key = entity['text'].lower()
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        self.logger.info(f"Extracted {len(unique_entities)} entities from query")
        return unique_entities
    
    def find_entities_in_graph(self, extracted_entities: List[Dict]) -> List[Dict]:
        """Find extracted entities in the knowledge graph"""
        found_entities = []
        
        with self.driver.session(database=self.neo4j_database) as session:
            for entity in extracted_entities:
                # Search by exact match first
                exact_query = """
                MATCH (e:Entity) 
                WHERE toLower(e.primary_name) = toLower($text)
                   OR toLower($text) IN [syn IN e.synonyms | toLower(syn)]
                RETURN e, e.node_id as id, e.primary_name as name, 
                       e.entity_type as type, e.confidence as confidence
                LIMIT 5
                """
                
                exact_results = session.run(exact_query, text=entity['text'])
                
                for record in exact_results:
                    found_entity = dict(record['e'])
                    found_entity.update({
                        'original_query_text': entity['text'],
                        'match_type': 'exact',
                        'query_category': entity['category']
                    })
                    found_entities.append(found_entity)
                
                # If no exact matches, try fuzzy search
                if not list(session.run(exact_query, text=entity['text'])):
                    fuzzy_query = """
                    CALL db.index.fulltext.queryNodes('entity_search_index', $search_text)
                    YIELD node, score
                    WHERE score > 0.5
                    RETURN node as e, node.node_id as id, node.primary_name as name,
                           node.entity_type as type, score
                    LIMIT 3
                    """
                    
                    search_text = f"{entity['text']}~0.8"  # Fuzzy search with edit distance
                    fuzzy_results = session.run(fuzzy_query, search_text=search_text)
                    
                    for record in fuzzy_results:
                        found_entity = dict(record['e'])
                        found_entity.update({
                            'original_query_text': entity['text'],
                            'match_type': 'fuzzy',
                            'match_score': record['score'],
                            'query_category': entity['category']
                        })
                        found_entities.append(found_entity)
        
        self.logger.info(f"Found {len(found_entities)} entities in graph")
        return found_entities
    
    def get_local_context(self, entity_ids: List[str], max_hops: int = 2, 
                         min_confidence: float = 0.5) -> GraphContext:
        """Get local graph context around specific entities"""
        nodes = []
        relationships = []
        paths = []
        
        with self.driver.session(database=self.neo4j_database) as session:
            # Get multi-hop neighborhood
            context_query = f"""
            MATCH (start:Entity)
            WHERE start.node_id IN $entity_ids
            CALL {{
                WITH start
                MATCH path = (start)-[r*1..{max_hops}]-(connected)
                WHERE ALL(rel IN relationships(path) WHERE rel.confidence >= $min_confidence)
                RETURN path
                LIMIT 50
            }}
            UNWIND nodes(path) as node
            UNWIND relationships(path) as rel
            RETURN DISTINCT 
                node,
                startNode(rel) as source,
                endNode(rel) as target,
                rel,
                path
            """
            
            results = session.run(context_query, 
                                entity_ids=entity_ids, 
                                min_confidence=min_confidence)
            
            # Collect unique nodes and relationships
            node_map = {}
            rel_set = set()
            path_list = []
            
            for record in results:
                # Add node
                node = dict(record['node'])
                node_id = node['node_id']
                if node_id not in node_map:
                    node_map[node_id] = node
                
                # Add relationship
                rel = dict(record['rel'])
                source = dict(record['source'])
                target = dict(record['target'])
                
                rel_key = (source['node_id'], target['node_id'], rel.get('relationship_type', ''))
                if rel_key not in rel_set:
                    rel_set.add(rel_key)
                    relationships.append({
                        'source_id': source['node_id'],
                        'target_id': target['node_id'],
                        'source_name': source['primary_name'],
                        'target_name': target['primary_name'],
                        'relationship_type': rel.get('relationship_type', ''),
                        'confidence': rel.get('confidence', 0.0),
                        'evidence': rel.get('evidence', ''),
                        'source_pmid': rel.get('source_pmid', '')
                    })
                
                # Add path
                path_data = record['path']
                if path_data:
                    path_nodes = []
                    for path_node in path_data.nodes:
                        path_nodes.append({
                            'id': path_node['node_id'],
                            'name': path_node['primary_name'],
                            'type': path_node['entity_type']
                        })
                    if len(path_nodes) > 1:  # Only add meaningful paths
                        paths.append(path_nodes)
            
            nodes = list(node_map.values())
        
        # Generate summary
        summary = self._generate_context_summary(nodes, relationships, paths)
        
        # Calculate confidence scores
        confidence_scores = [rel['confidence'] for rel in relationships if rel['confidence'] > 0]
        
        return GraphContext(
            nodes=nodes,
            relationships=relationships,
            paths=paths,
            summary=summary,
            confidence_scores=confidence_scores
        )
    
    def get_global_context(self, query: str, limit: int = 100) -> GraphContext:
        """Get global graph context for broad queries"""
        nodes = []
        relationships = []
        
        with self.driver.session(database=self.neo4j_database) as session:
            # Search for relevant nodes based on query terms
            global_query = """
            CALL db.index.fulltext.queryNodes('entity_search_index', $search_terms)
            YIELD node, score
            WHERE score > 0.3
            WITH node, score
            ORDER BY score DESC
            LIMIT $limit
            
            OPTIONAL MATCH (node)-[r]-(connected)
            WHERE r.confidence > 0.7
            
            RETURN node, r, connected, score
            """
            
            # Extract key terms from query for search
            search_terms = ' OR '.join(query.split()[:5])  # Use first 5 words
            
            results = session.run(global_query, search_terms=search_terms, limit=limit)
            
            node_map = {}
            rel_set = set()
            
            for record in results:
                # Add main node
                node = dict(record['node'])
                node_id = node['node_id']
                if node_id not in node_map:
                    node['search_score'] = record['score']
                    node_map[node_id] = node
                
                # Add connected nodes and relationships
                if record['r'] and record['connected']:
                    rel = dict(record['r'])
                    connected = dict(record['connected'])
                    
                    # Add connected node
                    connected_id = connected['node_id']
                    if connected_id not in node_map:
                        node_map[connected_id] = connected
                    
                    # Add relationship
                    rel_key = (node_id, connected_id, rel.get('relationship_type', ''))
                    if rel_key not in rel_set:
                        rel_set.add(rel_key)
                        relationships.append({
                            'source_id': node_id,
                            'target_id': connected_id,
                            'source_name': node['primary_name'],
                            'target_name': connected['primary_name'],
                            'relationship_type': rel.get('relationship_type', ''),
                            'confidence': rel.get('confidence', 0.0),
                            'evidence': rel.get('evidence', ''),
                            'source_pmid': rel.get('source_pmid', '')
                        })
            
            nodes = list(node_map.values())
        
        # Generate summary
        summary = self._generate_context_summary(nodes, relationships, [])
        confidence_scores = [rel['confidence'] for rel in relationships if rel['confidence'] > 0]
        
        return GraphContext(
            nodes=nodes,
            relationships=relationships,
            paths=[],
            summary=summary,
            confidence_scores=confidence_scores
        )
    
    def _generate_context_summary(self, nodes: List[Dict], relationships: List[Dict], 
                                 paths: List[List[Dict]]) -> str:
        """Generate a summary of the graph context"""
        # Count entity types
        entity_counts = Counter(node.get('entity_type', 'Unknown') for node in nodes)
        
        # Count relationship types
        rel_counts = Counter(rel.get('relationship_type', 'Unknown') for rel in relationships)
        
        # Build summary
        summary_parts = []
        
        if entity_counts:
            entity_summary = ', '.join([f"{count} {etype.lower()}s" for etype, count in entity_counts.most_common(3)])
            summary_parts.append(f"Entities: {entity_summary}")
        
        if rel_counts:
            rel_summary = ', '.join([f"{count} {rtype.lower()}" for rtype, count in rel_counts.most_common(3)])
            summary_parts.append(f"Relationships: {rel_summary}")
        
        if paths:
            summary_parts.append(f"Pathways: {len(paths)} connection paths found")
        
        return '. '.join(summary_parts)
    
    def format_context_for_llm(self, context: GraphContext, query: str) -> str:
        """Format graph context for LLM consumption"""
        formatted_context = []
        
        # Add context summary
        formatted_context.append("BIOMEDICAL KNOWLEDGE GRAPH CONTEXT:")
        formatted_context.append(f"Summary: {context.summary}")
        formatted_context.append("")
        
        # Add key entities
        if context.nodes:
            formatted_context.append("KEY ENTITIES:")
            high_confidence_nodes = sorted(
                [node for node in context.nodes if node.get('confidence', 0) > 0.7],
                key=lambda x: x.get('confidence', 0),
                reverse=True
            )[:10]  # Top 10 most confident entities
            
            for node in high_confidence_nodes:
                entity_info = f"- {node['primary_name']} ({node['entity_type']})"
                if node.get('preferred_name') and node['preferred_name'] != node['primary_name']:
                    entity_info += f" [also known as: {node['preferred_name']}]"
                if node.get('umls_cui'):
                    entity_info += f" [UMLS: {node['umls_cui']}]"
                formatted_context.append(entity_info)
            formatted_context.append("")
        
        # Add key relationships
        if context.relationships:
            formatted_context.append("KEY RELATIONSHIPS:")
            high_confidence_rels = sorted(
                [rel for rel in context.relationships if rel.get('confidence', 0) > 0.7],
                key=lambda x: x.get('confidence', 0),
                reverse=True
            )[:15]  # Top 15 most confident relationships
            
            for rel in high_confidence_rels:
                rel_info = f"- {rel['source_name']} {rel['relationship_type']} {rel['target_name']}"
                if rel.get('confidence'):
                    rel_info += f" (confidence: {rel['confidence']:.2f})"
                if rel.get('source_pmid'):
                    rel_info += f" [PMID: {rel['source_pmid']}]"
                formatted_context.append(rel_info)
            formatted_context.append("")
        
        # Add connection paths for complex queries
        if context.paths and len(context.paths) > 0:
            formatted_context.append("CONNECTION PATHS:")
            for i, path in enumerate(context.paths[:5]):  # Show top 5 paths
                path_str = " → ".join([f"{node['name']} ({node['type']})" for node in path])
                formatted_context.append(f"- Path {i+1}: {path_str}")
            formatted_context.append("")
        
        # Add statistical context
        if context.confidence_scores:
            avg_confidence = sum(context.confidence_scores) / len(context.confidence_scores)
            formatted_context.append(f"EVIDENCE QUALITY: Average confidence score: {avg_confidence:.2f}")
            formatted_context.append("")
        
        return '\n'.join(formatted_context)
    
    def call_local_llm(self, prompt: str, max_tokens: int = 1000) -> Dict:
        """Call local LLM (Ollama or other local providers)"""
        try:
            if self.llm_provider.lower() == "ollama":
                # Ollama API call
                url = f"{self.llm_base_url}/api/generate"
                payload = {
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.1,  # Low temperature for factual responses
                        "top_p": 0.9
                    }
                }
                
                response = requests.post(url, json=payload, timeout=120)
                response.raise_for_status()
                
                result = response.json()
                return {
                    "response": result.get("response", ""),
                    "success": True,
                    "model": self.llm_model,
                    "provider": "ollama"
                }
            
            else:
                # Generic OpenAI-compatible API
                url = f"{self.llm_base_url}/v1/completions"
                headers = {"Content-Type": "application/json"}
                payload = {
                    "model": self.llm_model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.1
                }
                
                response = requests.post(url, json=payload, headers=headers, timeout=120)
                response.raise_for_status()
                
                result = response.json()
                return {
                    "response": result["choices"][0]["text"],
                    "success": True,
                    "model": self.llm_model,
                    "provider": self.llm_provider
                }
                
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            return {
                "response": f"Error calling LLM: {str(e)}",
                "success": False,
                "error": str(e)
            }
    
    def generate_biomedical_prompt(self, query: str, context: str) -> str:
        """Generate biomedical-specific prompt for the LLM"""
        prompt = f"""You are a biomedical AI assistant with access to a comprehensive knowledge graph of scientific literature. 

USER QUESTION: {query}

RELEVANT BIOMEDICAL KNOWLEDGE:
{context}

INSTRUCTIONS:
1. Answer the question using ONLY the information provided in the knowledge graph context above
2. Be specific and cite evidence where possible (mention PMID numbers when available)
3. If the knowledge graph doesn't contain enough information to answer the question, clearly state this
4. For gene-disease associations, mention confidence scores when provided
5. Structure your answer clearly with key findings first
6. Include relevant biomedical identifiers (UMLS CUIs, Gene IDs) when discussing entities

ANSWER:"""
        
        return prompt
    
    def query(self, user_query: str, search_strategy: str = "auto") -> QueryResult:
        """
        Main query function for the Graph RAG system
        
        Args:
            user_query: User's question
            search_strategy: "local", "global", or "auto"
            
        Returns:
            QueryResult with answer and context
        """
        start_time = time.time()
        reasoning_steps = []
        
        self.logger.info(f"Processing query: {user_query}")
        reasoning_steps.append(f"Query received: {user_query}")
        
        # Step 1: Extract entities from query
        extracted_entities = self.extract_entities_from_query(user_query)
        reasoning_steps.append(f"Extracted {len(extracted_entities)} potential entities")
        
        # Step 2: Find entities in graph
        found_entities = self.find_entities_in_graph(extracted_entities)
        reasoning_steps.append(f"Found {len(found_entities)} entities in knowledge graph")
        
        # Step 3: Determine search strategy
        if search_strategy == "auto":
            if found_entities:
                search_strategy = "local"
                reasoning_steps.append("Using local search strategy (specific entities found)")
            else:
                search_strategy = "global"
                reasoning_steps.append("Using global search strategy (no specific entities found)")
        
        # Step 4: Get graph context
        if search_strategy == "local" and found_entities:
            entity_ids = [entity['node_id'] for entity in found_entities[:5]]  # Limit to top 5
            graph_context = self.get_local_context(entity_ids)
            reasoning_steps.append(f"Retrieved local context: {len(graph_context.nodes)} nodes, {len(graph_context.relationships)} relationships")
        else:
            graph_context = self.get_global_context(user_query)
            reasoning_steps.append(f"Retrieved global context: {len(graph_context.nodes)} nodes, {len(graph_context.relationships)} relationships")
        
        # Step 5: Format context for LLM
        formatted_context = self.format_context_for_llm(graph_context, user_query)
        reasoning_steps.append("Formatted context for LLM")
        
        # Step 6: Generate biomedical prompt
        prompt = self.generate_biomedical_prompt(user_query, formatted_context)
        reasoning_steps.append("Generated biomedical prompt")
        
        # Step 7: Call LLM
        llm_response = self.call_local_llm(prompt)
        reasoning_steps.append(f"LLM response received (success: {llm_response['success']})")
        
        # Step 8: Extract sources
        sources = []
        for rel in graph_context.relationships:
            if rel.get('source_pmid'):
                sources.append(rel['source_pmid'])
        sources = list(set(sources))  # Remove duplicates
        
        # Step 9: Calculate confidence
        if graph_context.confidence_scores:
            confidence = sum(graph_context.confidence_scores) / len(graph_context.confidence_scores)
        else:
            confidence = 0.5  # Default confidence
        
        execution_time = time.time() - start_time
        reasoning_steps.append(f"Query completed in {execution_time:.2f} seconds")
        
        return QueryResult(
            query=user_query,
            entities_found=found_entities,
            graph_context={
                'nodes_count': len(graph_context.nodes),
                'relationships_count': len(graph_context.relationships),
                'summary': graph_context.summary
            },
            answer=llm_response['response'] if llm_response['success'] else "Error generating response",
            confidence=confidence,
            sources=sources,
            execution_time=execution_time,
            reasoning_steps=reasoning_steps
        )
    
    def batch_query(self, queries: List[str], search_strategy: str = "auto") -> List[QueryResult]:
        """Process multiple queries in batch"""
        results = []
        for i, query in enumerate(queries):
            self.logger.info(f"Processing batch query {i+1}/{len(queries)}")
            result = self.query(query, search_strategy)
            results.append(result)
        return results
    
    def get_system_stats(self) -> Dict:
        """Get system statistics"""
        stats = {}
        
        with self.driver.session(database=self.neo4j_database) as session:
            # Graph statistics
            stats['total_nodes'] = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
            stats['total_relationships'] = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
            
            # Entity type distribution
            entity_types = session.run("""
                MATCH (n:Entity) 
                RETURN n.entity_type as type, count(n) as count 
                ORDER BY count DESC
            """)
            stats['entity_types'] = {record['type']: record['count'] for record in entity_types}
            
            # Relationship type distribution
            rel_types = session.run("""
                MATCH ()-[r]->() 
                RETURN type(r) as rel_type, count(r) as count 
                ORDER BY count DESC
            """)
            stats['relationship_types'] = {record['rel_type']: record['count'] for record in rel_types}
            
            # High confidence entities
            high_conf_entities = session.run("""
                MATCH (n:Entity) 
                WHERE n.confidence > 0.9 
                RETURN count(n) as count
            """).single()["count"]
            stats['high_confidence_entities'] = high_conf_entities
        
        # System configuration
        stats['llm_config'] = {
            'provider': self.llm_provider,
            'model': self.llm_model,
            'base_url': self.llm_base_url
        }
        
        return stats

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Biomedical Graph RAG Query System')
    parser.add_argument('--query', help='Single query to process')
    parser.add_argument('--batch_file', help='File containing multiple queries (one per line)')
    parser.add_argument('--neo4j_uri', default='bolt://localhost:7687', help='Neo4j URI')
    parser.add_argument('--neo4j_user', default='neo4j', help='Neo4j username')
    parser.add_argument('--neo4j_password', default='password', help='Neo4j password')
    parser.add_argument('--llm_provider', default='ollama', help='LLM provider')
    parser.add_argument('--llm_model', default='llama3.1', help='LLM model')
    parser.add_argument('--llm_url', default='http://localhost:11434', help='LLM base URL')
    parser.add_argument('--search_strategy', choices=['local', 'global', 'auto'], default='auto')
    parser.add_argument('--output', help='Output file for results')
    parser.add_argument('--stats', action='store_true', help='Show system statistics')
    parser.add_argument('--interactive', action='store_true', help='Start interactive mode')
    
    args = parser.parse_args()
    
    # Initialize Graph RAG system
    graph_rag = BiomedicalGraphRAG(
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        llm_base_url=args.llm_url
    )
    
    try:
        # Connect to Neo4j
        graph_rag.connect_neo4j()
        
        # Show statistics if requested
        if args.stats:
            stats = graph_rag.get_system_stats()
            print("\n" + "="*60)
            print("BIOMEDICAL GRAPH RAG SYSTEM STATISTICS")
            print("="*60)
            print(f"Total Nodes: {stats['total_nodes']:,}")
            print(f"Total Relationships: {stats['total_relationships']:,}")
            print(f"High Confidence Entities: {stats['high_confidence_entities']:,}")
            
            print("\nEntity Types:")
            for etype, count in list(stats['entity_types'].items())[:5]:
                print(f"  {etype}: {count:,}")
            
            print("\nRelationship Types:")
            for rtype, count in list(stats['relationship_types'].items())[:5]:
                print(f"  {rtype}: {count:,}")
            
            print(f"\nLLM Configuration:")
            print(f"  Provider: {stats['llm_config']['provider']}")
            print(f"  Model: {stats['llm_config']['model']}")
            print(f"  URL: {stats['llm_config']['base_url']}")
            print("")
        
        # Process queries
        results = []
        
        if args.interactive:
            # Interactive mode
            print("Biomedical Graph RAG Interactive Mode")
            print("Type 'quit' to exit")
            print("-" * 40)
            
            while True:
                user_input = input("\nEnter your biomedical question: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if user_input:
                    result = graph_rag.query(user_input, args.search_strategy)
                    
                    print(f"\nAnswer:")
                    print(result.answer)
                    print(f"\nConfidence: {result.confidence:.2f}")
                    print(f"Execution Time: {result.execution_time:.2f}s")
                    if result.sources:
                        print(f"Sources: {', '.join(result.sources[:5])}")
                    
                    results.append(result)
        
        elif args.query:
            # Single query
            result = graph_rag.query(args.query, args.search_strategy)
            
            print(f"\nQuery: {result.query}")
            print(f"Answer: {result.answer}")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Execution Time: {result.execution_time:.2f}s")
            print(f"Entities Found: {len(result.entities_found)}")
            if result.sources:
                print(f"Sources: {', '.join(result.sources[:10])}")
            
            results.append(result)
        
        elif args.batch_file:
            # Batch processing
            with open(args.batch_file, 'r') as f:
                queries = [line.strip() for line in f if line.strip()]
            
            print(f"Processing {len(queries)} queries...")
            results = graph_rag.batch_query(queries, args.search_strategy)
            
            for i, result in enumerate(results):
                print(f"\nQuery {i+1}: {result.query}")
                print(f"Answer: {result.answer[:200]}...")
                print(f"Confidence: {result.confidence:.2f}")
        
        # Save results if requested
        if args.output and results:
            output_data = []
            for result in results:
                output_data.append({
                    'query': result.query,
                    'answer': result.answer,
                    'confidence': result.confidence,
                    'execution_time': result.execution_time,
                    'entities_found': len(result.entities_found),
                    'sources': result.sources,
                    'reasoning_steps': result.reasoning_steps
                })
            
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"\nResults saved to {args.output}")
    
    except Exception as e:
        print(f"Error: {e}")
    finally:
        graph_rag.disconnect_neo4j()

if __name__ == "__main__":
    main()

