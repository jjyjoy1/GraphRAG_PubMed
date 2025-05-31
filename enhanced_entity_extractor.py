#!/usr/bin/env python3
"""
Neo4j Knowledge Graph Builder for Biomedical Graph RAG Pipeline
Creates a standardized biomedical knowledge graph from extracted entities
"""

import json
import csv
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass
import argparse
import logging
from pathlib import Path
from collections import defaultdict, Counter
import re

# Neo4j imports
try:
    from neo4j import GraphDatabase, basic_auth
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("Warning: neo4j package not available. Install with: pip install neo4j")

@dataclass
class GraphNode:
    """Represents a node in the knowledge graph"""
    node_id: str
    primary_name: str
    entity_type: str
    properties: Dict[str, Any]
    labels: List[str]

@dataclass
class GraphRelationship:
    """Represents a relationship in the knowledge graph"""
    source_id: str
    target_id: str
    relationship_type: str
    properties: Dict[str, Any]

class BiomedicalKnowledgeGraphBuilder:
    """
    Builds a Neo4j knowledge graph from standardized biomedical entities
    """
    
    def __init__(self, uri: str = "bolt://localhost:7687", 
                 username: str = "neo4j", password: str = "password",
                 database: str = "neo4j"):
        """
        Initialize the knowledge graph builder
        
        Args:
            uri: Neo4j database URI
            username: Neo4j username
            password: Neo4j password  
            database: Neo4j database name
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Entity type to Neo4j label mapping
        self.entity_labels = {
            'GENE': ['Entity', 'Gene'],
            'DISEASE': ['Entity', 'Disease'], 
            'CHEMICAL': ['Entity', 'Chemical'],
            'VARIANT': ['Entity', 'Variant'],
            'SPECIES': ['Entity', 'Species'],
            'CELL_LINE': ['Entity', 'CellLine'],
            'PATHWAY': ['Entity', 'Pathway'],
            'MEASUREMENT': ['Entity', 'Measurement']
        }
        
        # Relationship type standardization
        self.relationship_types = {
            'gene_disease_association': 'ASSOCIATED_WITH',
            'gene_variant_association': 'HAS_VARIANT', 
            'variant_disease_association': 'ASSOCIATED_WITH',
            'gene_chemical_interaction': 'INTERACTS_WITH',
            'chemical_disease_association': 'TREATS',
            'chemical_gene_interaction': 'TARGETS',
            'causally_related_to': 'CAUSES',
            'treats': 'TREATS',
            'associated_with': 'ASSOCIATED_WITH'
        }
        
        # Statistics tracking
        self.stats = {
            'nodes_created': 0,
            'relationships_created': 0,
            'duplicates_merged': 0,
            'errors': 0
        }
    
    def connect(self):
        """Connect to Neo4j database"""
        try:
            if not NEO4J_AVAILABLE:
                raise ImportError("Neo4j driver not available")
            
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=basic_auth(self.username, self.password),
                encrypted=False  # Set to True for production
            )
            
            # Test connection
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                if test_value == 1:
                    self.logger.info("Successfully connected to Neo4j")
                else:
                    raise ConnectionError("Neo4j connection test failed")
                    
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def disconnect(self):
        """Disconnect from Neo4j database"""
        if self.driver:
            self.driver.close()
            self.logger.info("Disconnected from Neo4j")
    
    def clear_database(self):
        """Clear all data from the database (use with caution!)"""
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
            self.logger.info("Database cleared")
    
    def create_indexes(self):
        """Create indexes for efficient querying"""
        indexes = [
            # Primary identifier indexes
            "CREATE INDEX entity_id_index IF NOT EXISTS FOR (e:Entity) ON (e.node_id)",
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.primary_name)",
            
            # Database identifier indexes
            "CREATE INDEX umls_cui_index IF NOT EXISTS FOR (e:Entity) ON (e.umls_cui)",
            "CREATE INDEX ncbi_gene_id_index IF NOT EXISTS FOR (e:Entity) ON (e.ncbi_gene_id)",
            "CREATE INDEX mesh_id_index IF NOT EXISTS FOR (e:Entity) ON (e.mesh_id)",
            "CREATE INDEX database_id_index IF NOT EXISTS FOR (e:Entity) ON (e.database_id)",
            
            # Type-specific indexes
            "CREATE INDEX gene_index IF NOT EXISTS FOR (g:Gene) ON (g.primary_name)",
            "CREATE INDEX disease_index IF NOT EXISTS FOR (d:Disease) ON (d.primary_name)",
            "CREATE INDEX chemical_index IF NOT EXISTS FOR (c:Chemical) ON (c.primary_name)",
            
            # Full-text search indexes
            "CREATE FULLTEXT INDEX entity_search_index IF NOT EXISTS FOR (e:Entity) ON EACH [e.primary_name, e.synonyms_text, e.preferred_name]",
            
            # Relationship indexes
            "CREATE INDEX relationship_type_index IF NOT EXISTS FOR ()-[r]-() ON (r.relationship_type)",
            "CREATE INDEX relationship_confidence_index IF NOT EXISTS FOR ()-[r]-() ON (r.confidence)"
        ]
        
        with self.driver.session(database=self.database) as session:
            for index_query in indexes:
                try:
                    session.run(index_query)
                    self.logger.info(f"Created index: {index_query.split('FOR')[0].split('IF NOT EXISTS')[0].strip()}")
                except Exception as e:
                    self.logger.warning(f"Index creation failed: {e}")
    
    def create_constraints(self):
        """Create uniqueness constraints"""
        constraints = [
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.node_id IS UNIQUE",
            "CREATE CONSTRAINT gene_unique IF NOT EXISTS FOR (g:Gene) REQUIRE g.node_id IS UNIQUE",
            "CREATE CONSTRAINT disease_unique IF NOT EXISTS FOR (d:Disease) REQUIRE d.node_id IS UNIQUE",
            "CREATE CONSTRAINT chemical_unique IF NOT EXISTS FOR (c:Chemical) REQUIRE c.node_id IS UNIQUE"
        ]
        
        with self.driver.session(database=self.database) as session:
            for constraint_query in constraints:
                try:
                    session.run(constraint_query)
                    self.logger.info(f"Created constraint: {constraint_query.split('FOR')[0].split('IF NOT EXISTS')[0].strip()}")
                except Exception as e:
                    self.logger.warning(f"Constraint creation failed: {e}")
    
    def load_standardized_entities(self, entities_file: str) -> List[Dict]:
        """Load standardized entities from Step 2"""
        self.logger.info(f"Loading entities from {entities_file}")
        
        with open(entities_file, 'r', encoding='utf-8') as f:
            entities = json.load(f)
        
        self.logger.info(f"Loaded {len(entities)} entities")
        return entities
    
    def load_standardized_relations(self, relations_file: str) -> List[Dict]:
        """Load standardized relationships from Step 2"""
        self.logger.info(f"Loading relationships from {relations_file}")
        
        with open(relations_file, 'r', encoding='utf-8') as f:
            relations = json.load(f)
        
        self.logger.info(f"Loaded {len(relations)} relationships")
        return relations
    
    def process_entities_to_nodes(self, entities: List[Dict]) -> List[GraphNode]:
        """Convert standardized entities to graph nodes"""
        nodes = []
        entity_map = {}  # For deduplication
        
        for entity in entities:
            try:
                # Generate unique node ID
                node_id = self._generate_node_id(entity)
                
                # Check for duplicates
                if node_id in entity_map:
                    # Merge duplicate entity information
                    existing_node = entity_map[node_id]
                    existing_node.properties = self._merge_entity_properties(
                        existing_node.properties, 
                        self._extract_entity_properties(entity)
                    )
                    self.stats['duplicates_merged'] += 1
                    continue
                
                # Create new node
                node = GraphNode(
                    node_id=node_id,
                    primary_name=entity.get('text', ''),
                    entity_type=entity.get('entity_type', 'UNKNOWN'),
                    properties=self._extract_entity_properties(entity),
                    labels=self._get_node_labels(entity.get('entity_type', 'UNKNOWN'))
                )
                
                nodes.append(node)
                entity_map[node_id] = node
                
            except Exception as e:
                self.logger.warning(f"Error processing entity {entity.get('text', 'unknown')}: {e}")
                self.stats['errors'] += 1
        
        self.logger.info(f"Processed {len(nodes)} unique nodes")
        return nodes
    
    def process_relations_to_edges(self, relations: List[Dict], existing_nodes: Set[str]) -> List[GraphRelationship]:
        """Convert standardized relationships to graph edges"""
        edges = []
        
        for relation in relations:
            try:
                # Generate node IDs for source and target
                source_id = self._generate_relation_node_id(relation, 'entity1')
                target_id = self._generate_relation_node_id(relation, 'entity2')
                
                # Skip if nodes don't exist in the graph
                if source_id not in existing_nodes or target_id not in existing_nodes:
                    continue
                
                # Standardize relationship type
                rel_type = self.relationship_types.get(
                    relation.get('relation_type', '').lower(),
                    'RELATED_TO'
                )
                
                edge = GraphRelationship(
                    source_id=source_id,
                    target_id=target_id,
                    relationship_type=rel_type,
                    properties=self._extract_relation_properties(relation)
                )
                
                edges.append(edge)
                
            except Exception as e:
                self.logger.warning(f"Error processing relationship: {e}")
                self.stats['errors'] += 1
        
        self.logger.info(f"Processed {len(edges)} relationships")
        return edges
    
    def create_nodes_in_neo4j(self, nodes: List[GraphNode]):
        """Create nodes in Neo4j database"""
        self.logger.info("Creating nodes in Neo4j...")
        
        batch_size = 1000
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]
            
            with self.driver.session(database=self.database) as session:
                # Prepare batch data
                batch_data = []
                for node in batch:
                    node_data = {
                        'node_id': node.node_id,
                        'primary_name': node.primary_name,
                        'entity_type': node.entity_type,
                        **node.properties
                    }
                    batch_data.append(node_data)
                
                # Create nodes with MERGE to handle duplicates
                query = """
                UNWIND $batch AS nodeData
                MERGE (e:Entity {node_id: nodeData.node_id})
                SET e += nodeData
                """
                
                # Add specific labels
                for node in batch:
                    if node.labels:
                        label_str = ':'.join(node.labels[1:])  # Skip 'Entity' base label
                        if label_str:
                            query += f"\nSET e:{label_str}"
                
                session.run(query, batch=batch_data)
                self.stats['nodes_created'] += len(batch)
            
            self.logger.info(f"Created {min(i + batch_size, len(nodes))}/{len(nodes)} nodes")
    
    def create_relationships_in_neo4j(self, edges: List[GraphRelationship]):
        """Create relationships in Neo4j database"""
        self.logger.info("Creating relationships in Neo4j...")
        
        batch_size = 1000
        for i in range(0, len(edges), batch_size):
            batch = edges[i:i + batch_size]
            
            with self.driver.session(database=self.database) as session:
                # Group relationships by type for efficiency
                relationships_by_type = defaultdict(list)
                for edge in batch:
                    relationships_by_type[edge.relationship_type].append(edge)
                
                for rel_type, rel_batch in relationships_by_type.items():
                    # Prepare batch data
                    batch_data = []
                    for edge in rel_batch:
                        edge_data = {
                            'source_id': edge.source_id,
                            'target_id': edge.target_id,
                            **edge.properties
                        }
                        batch_data.append(edge_data)
                    
                    # Create relationships
                    query = f"""
                    UNWIND $batch AS relData
                    MATCH (source:Entity {{node_id: relData.source_id}})
                    MATCH (target:Entity {{node_id: relData.target_id}})
                    MERGE (source)-[r:{rel_type}]->(target)
                    SET r += relData
                    """
                    
                    session.run(query, batch=batch_data)
                    self.stats['relationships_created'] += len(rel_batch)
            
            self.logger.info(f"Created {min(i + batch_size, len(edges))}/{len(edges)} relationships")
    
    def _generate_node_id(self, entity: Dict) -> str:
        """Generate unique node ID for an entity"""
        # Priority order for ID generation
        id_fields = [
            'umls_cui',
            'ncbi_gene_id', 
            'mesh_id',
            'database_id'
        ]
        
        for field in id_fields:
            if entity.get(field):
                return f"{field}:{entity[field]}"
        
        # Fallback to normalized text
        text = entity.get('text', '').strip().lower()
        entity_type = entity.get('entity_type', 'UNKNOWN')
        return f"text:{entity_type}:{text}"
    
    def _generate_relation_node_id(self, relation: Dict, entity_key: str) -> str:
        """Generate node ID for relationship endpoint"""
        # Try to use standardized IDs first
        if f"{entity_key}_id" in relation:
            return relation[f"{entity_key}_id"]
        
        # Fallback to text-based ID
        text = relation.get(f"{entity_key}_text", '').strip().lower()
        return f"text:UNKNOWN:{text}"
    
    def _extract_entity_properties(self, entity: Dict) -> Dict[str, Any]:
        """Extract properties for a graph node"""
        properties = {}
        
        # Core properties
        properties['confidence'] = entity.get('confidence', 0.0)
        properties['source'] = entity.get('source', '')
        properties['context'] = entity.get('context', '')[:500]  # Truncate long context
        
        # Standardized identifiers
        if entity.get('umls_cui'):
            properties['umls_cui'] = entity['umls_cui']
        if entity.get('ncbi_gene_id'):
            properties['ncbi_gene_id'] = entity['ncbi_gene_id']
        if entity.get('mesh_id'):
            properties['mesh_id'] = entity['mesh_id']
        if entity.get('database_id'):
            properties['database_id'] = entity['database_id']
        
        # Names and synonyms
        if entity.get('preferred_name'):
            properties['preferred_name'] = entity['preferred_name']
        
        synonyms = entity.get('synonyms', [])
        if synonyms:
            properties['synonyms'] = synonyms
            properties['synonyms_text'] = ' | '.join(synonyms)  # For full-text search
        
        semantic_types = entity.get('semantic_types', [])
        if semantic_types:
            properties['semantic_types'] = semantic_types
            properties['semantic_types_text'] = ' | '.join(semantic_types)
        
        # Attributes from original extraction
        attributes = entity.get('attributes', {})
        for key, value in attributes.items():
            if key in ['pmid', 'journal', 'pub_date', 'authors']:
                properties[key] = value
        
        return properties
    
    def _extract_relation_properties(self, relation: Dict) -> Dict[str, Any]:
        """Extract properties for a graph relationship"""
        properties = {}
        
        # Core relationship properties
        properties['confidence'] = relation.get('confidence', 0.0)
        properties['evidence'] = relation.get('evidence', '')[:500]  # Truncate
        properties['source_pmid'] = relation.get('source_pmid', '')
        properties['relationship_type'] = relation.get('relation_type', '')
        
        if relation.get('semantic_relation'):
            properties['semantic_relation'] = relation['semantic_relation']
        
        if relation.get('relation_id'):
            properties['relation_id'] = relation['relation_id']
        
        return properties
    
    def _get_node_labels(self, entity_type: str) -> List[str]:
        """Get Neo4j labels for entity type"""
        return self.entity_labels.get(entity_type, ['Entity', 'Unknown'])
    
    def _merge_entity_properties(self, existing: Dict, new: Dict) -> Dict:
        """Merge properties from duplicate entities"""
        merged = existing.copy()
        
        # Merge confidence scores (take maximum)
        if 'confidence' in new:
            merged['confidence'] = max(
                existing.get('confidence', 0.0),
                new.get('confidence', 0.0)
            )
        
        # Merge synonyms
        existing_synonyms = set(existing.get('synonyms', []))
        new_synonyms = set(new.get('synonyms', []))
        all_synonyms = list(existing_synonyms | new_synonyms)
        if all_synonyms:
            merged['synonyms'] = all_synonyms
            merged['synonyms_text'] = ' | '.join(all_synonyms)
        
        # Merge other fields (prefer non-empty values)
        merge_fields = ['preferred_name', 'umls_cui', 'ncbi_gene_id', 'mesh_id']
        for field in merge_fields:
            if new.get(field) and not existing.get(field):
                merged[field] = new[field]
        
        return merged
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the created graph"""
        with self.driver.session(database=self.database) as session:
            # Node counts
            node_counts = {}
            for entity_type in self.entity_labels.keys():
                if entity_type != 'UNKNOWN':
                    label = entity_type.title()
                    result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                    node_counts[entity_type] = result.single()["count"]
            
            # Total nodes and relationships
            total_nodes = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
            total_rels = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
            
            # Relationship type counts
            rel_type_counts = {}
            rel_types_result = session.run("""
                MATCH ()-[r]->() 
                RETURN type(r) as rel_type, count(r) as count 
                ORDER BY count DESC
            """)
            for record in rel_types_result:
                rel_type_counts[record["rel_type"]] = record["count"]
            
            # Database coverage
            umls_coverage = session.run("MATCH (n:Entity) WHERE n.umls_cui IS NOT NULL RETURN count(n) as count").single()["count"]
            ncbi_coverage = session.run("MATCH (n:Entity) WHERE n.ncbi_gene_id IS NOT NULL RETURN count(n) as count").single()["count"]
            
            statistics = {
                'total_nodes': total_nodes,
                'total_relationships': total_rels,
                'node_counts_by_type': node_counts,
                'relationship_counts_by_type': rel_type_counts,
                'database_coverage': {
                    'entities_with_umls_cui': umls_coverage,
                    'entities_with_ncbi_id': ncbi_coverage,
                    'umls_coverage_percent': round(umls_coverage / total_nodes * 100, 2) if total_nodes > 0 else 0,
                    'ncbi_coverage_percent': round(ncbi_coverage / total_nodes * 100, 2) if total_nodes > 0 else 0
                },
                'creation_stats': self.stats
            }
            
            return statistics
    
    def export_graph_sample(self, output_file: str, limit: int = 100):
        """Export a sample of the graph for visualization"""
        with self.driver.session(database=self.database) as session:
            # Get a connected subgraph sample
            query = """
            MATCH (n)-[r]-(m)
            WITH n, r, m, rand() as random
            ORDER BY random
            LIMIT $limit
            RETURN n, r, m
            """
            
            result = session.run(query, limit=limit)
            
            # Convert to JSON format for visualization
            nodes = {}
            edges = []
            
            for record in result:
                # Add nodes
                for node_key in ['n', 'm']:
                    node = record[node_key]
                    node_id = node['node_id']
                    if node_id not in nodes:
                        nodes[node_id] = {
                            'id': node_id,
                            'label': node['primary_name'],
                            'type': node['entity_type'],
                            'properties': dict(node)
                        }
                
                # Add edge
                rel = record['r']
                edges.append({
                    'source': record['n']['node_id'],
                    'target': record['m']['node_id'],
                    'type': rel.type,
                    'properties': dict(rel)
                })
            
            # Save to file
            graph_data = {
                'nodes': list(nodes.values()),
                'edges': edges
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Exported graph sample to {output_file}")
    
    def build_graph_from_files(self, entities_file: str, relations_file: str):
        """Build complete knowledge graph from standardized files"""
        self.logger.info("Starting knowledge graph construction...")
        
        # Load data
        entities = self.load_standardized_entities(entities_file)
        relations = self.load_standardized_relations(relations_file)
        
        # Process entities to nodes
        nodes = self.process_entities_to_nodes(entities)
        
        # Get set of existing node IDs for relationship validation
        existing_node_ids = {node.node_id for node in nodes}
        
        # Process relationships to edges
        edges = self.process_relations_to_edges(relations, existing_node_ids)
        
        # Create graph structure
        self.create_constraints()
        self.create_indexes()
        
        # Create nodes and relationships
        self.create_nodes_in_neo4j(nodes)
        self.create_relationships_in_neo4j(edges)
        
        self.logger.info("Knowledge graph construction completed!")
        
        # Return statistics
        return self.get_graph_statistics()

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Build Neo4j knowledge graph from standardized entities')
    parser.add_argument('--entities', required=True, help='Standardized entities JSON file from step 2')
    parser.add_argument('--relations', required=True, help='Standardized relations JSON file from step 2')
    parser.add_argument('--neo4j_uri', default='bolt://localhost:7687', help='Neo4j database URI')
    parser.add_argument('--username', default='neo4j', help='Neo4j username')
    parser.add_argument('--password', default='password', help='Neo4j password')
    parser.add_argument('--database', default='neo4j', help='Neo4j database name')
    parser.add_argument('--clear_db', action='store_true', help='Clear database before building')
    parser.add_argument('--export_sample', help='Export graph sample for visualization')
    parser.add_argument('--stats_output', help='Save statistics to JSON file')
    
    args = parser.parse_args()
    
    # Check input files
    if not Path(args.entities).exists():
        print(f"Error: Entities file {args.entities} not found")
        return
    
    if not Path(args.relations).exists():
        print(f"Error: Relations file {args.relations} not found")
        return
    
    # Initialize graph builder
    builder = BiomedicalKnowledgeGraphBuilder(
        uri=args.neo4j_uri,
        username=args.username,
        password=args.password,
        database=args.database
    )
    
    try:
        # Connect to Neo4j
        builder.connect()
        
        # Clear database if requested
        if args.clear_db:
            print("Clearing existing data...")
            builder.clear_database()
        
        # Build graph
        statistics = builder.build_graph_from_files(args.entities, args.relations)
        
        # Print statistics
        print("\n" + "="*50)
        print("KNOWLEDGE GRAPH STATISTICS")
        print("="*50)
        print(f"Total Nodes: {statistics['total_nodes']}")
        print(f"Total Relationships: {statistics['total_relationships']}")
        print(f"Creation Errors: {statistics['creation_stats']['errors']}")
        print(f"Duplicates Merged: {statistics['creation_stats']['duplicates_merged']}")
        
        print("\nNode Types:")
        for node_type, count in statistics['node_counts_by_type'].items():
            print(f"  {node_type}: {count}")
        
        print("\nRelationship Types:")
        for rel_type, count in statistics['relationship_counts_by_type'].items():
            print(f"  {rel_type}: {count}")
        
        print(f"\nDatabase Coverage:")
        print(f"  UMLS Coverage: {statistics['database_coverage']['umls_coverage_percent']}%")
        print(f"  NCBI Coverage: {statistics['database_coverage']['ncbi_coverage_percent']}%")
        
        # Save statistics
        if args.stats_output:
            with open(args.stats_output, 'w') as f:
                json.dump(statistics, f, indent=2)
            print(f"\nStatistics saved to {args.stats_output}")
        
        # Export sample for visualization
        if args.export_sample:
            builder.export_graph_sample(args.export_sample)
            print(f"Graph sample exported to {args.export_sample}")
        
        print("\nKnowledge graph construction completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        builder.disconnect()

if __name__ == "__main__":
    main()



