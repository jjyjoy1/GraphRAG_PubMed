# Microsoft Graph RAG Setup Guide with OpenAI and PDFs

## Prerequisites

### 1. Environment Setup
```bash
# Install required packages
pip install graphrag
pip install openai
pip install pypdf2 or pymupdf  # for PDF processing
pip install python-dotenv
```

### 2. API Keys Required
- OpenAI API key (for LLM and embeddings)
- Optional: Azure OpenAI if using Azure services

## Step 1: Project Structure

Create your project directory structure:
```
graph_rag_project/
├── input/
│   ├── pdfs/           # Your PDF files go here
│   └── processed/      # Processed text files
├── output/             # Graph RAG outputs
├── .env               # Environment variables
├── settings.yaml      # Configuration file
└── process_pdfs.py    # PDF processing script
```

## Step 2: Environment Configuration

Create a `.env` file:
```env
GRAPHRAG_API_KEY=your_openai_api_key_here
GRAPHRAG_API_BASE=https://api.openai.com/v1
GRAPHRAG_API_TYPE=openai
GRAPHRAG_API_VERSION=2023-05-15
```

## Step 3: PDF Processing Script

Create `process_pdfs.py`:
```python
import os
import PyPDF2
# or alternatively: import fitz  # PyMuPDF
from pathlib import Path

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
    return text

def process_pdfs_to_text(input_dir, output_dir):
    """Convert all PDFs in input directory to text files"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for pdf_file in input_path.glob("*.pdf"):
        print(f"Processing: {pdf_file.name}")
        text = extract_text_from_pdf(pdf_file)
        
        # Save as text file
        output_file = output_path / f"{pdf_file.stem}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Saved: {output_file}")

if __name__ == "__main__":
    process_pdfs_to_text("input/pdfs", "input/processed")
```

## Step 4: Graph RAG Configuration

Create `settings.yaml`:
```yaml
llm:
  api_key: ${GRAPHRAG_API_KEY}
  type: openai_chat
  model: gpt-4  # or gpt-3.5-turbo for cost efficiency
  model_supports_json: true
  max_tokens: 4000
  temperature: 0.1
  top_p: 1.0
  n: 1

parallelization:
  stagger: 0.3
  num_threads: 50

async_mode: threaded

embeddings:
  async_mode: threaded
  llm:
    api_key: ${GRAPHRAG_API_KEY}
    type: openai_embedding
    model: text-embedding-ada-002
    max_tokens: 8191

input:
  type: file
  file_type: text
  base_dir: "input/processed"
  file_encoding: utf-8
  file_pattern: ".*\\.txt$"

cache:
  type: file
  base_dir: "cache"

storage:
  type: file
  base_dir: "output"

chunk:
  size: 300
  overlap: 100
  group_by_columns: [id]

embed_graph:
  enabled: false

umap:
  enabled: false

snapshots:
  graphml: false
  raw_entities: false
  top_level_nodes: false

entity_extraction:
  prompt: "prompts/entity_extraction.txt"
  entity_types: [person, organization, location, event, concept, technology]
  max_gleanings: 1

summarize_descriptions:
  prompt: "prompts/summarize_descriptions.txt"
  max_length: 500

claim_extraction:
  enabled: true
  prompt: "prompts/claim_extraction.txt"
  description: "Any claims or facts that could be relevant to information discovery"
  max_gleanings: 1

community_report:
  prompt: "prompts/community_report.txt"
  max_length: 2000
  max_input_length: 8000

cluster_graph:
  max_cluster_size: 10

embed_graph:
  enabled: false

local_search:
  text_unit_prop: 0.5
  community_prop: 0.1
  conversation_history_max_turns: 5
  top_k_mapped_entities: 10
  top_k_relationships: 10
  max_tokens: 12000

global_search:
  max_tokens: 12000
  data_max_tokens: 12000
  map_max_tokens: 1000
  reduce_max_tokens: 2000
  concurrency: 32
```

## Step 5: Running the Pipeline

### Initialize Graph RAG
```bash
# Initialize the project
python -m graphrag.index --init --root ./

# This creates default prompt files in prompts/ directory
```

### Process Your Data
```bash
# First, convert PDFs to text
python process_pdfs.py

# Run the indexing pipeline
python -m graphrag.index --root ./
```

### Query the Knowledge Graph
```python
# query_example.py
import asyncio
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_communities,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore

async def main():
    # Setup LLM
    llm = ChatOpenAI(
        api_key="your_openai_api_key",
        model="gpt-4",
        api_type=OpenaiApiType.OpenAI,
        max_retries=20,
    )

    # Load indexed data
    entities = read_indexer_entities("./output/artifacts", entity_table="create_final_entities", entity_key_col="title")
    relationships = read_indexer_relationships("./output/artifacts")
    reports = read_indexer_reports("./output/artifacts", community_table="create_final_communities")
    text_units = read_indexer_text_units("./output/artifacts")

    # Setup vector store
    vector_store = LanceDBVectorStore(collection_name="default-entity-description")
    vector_store.connect(db_uri="./output/lancedb")

    # Setup search engine
    context_builder = LocalSearchMixedContext(
        community_reports=reports,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        entity_text_embeddings=vector_store,
        embedding_vectorstore_key=EntityVectorStoreKey.TITLE,
        text_embedder=None,  # We'll use the default
    )

    search_engine = LocalSearch(
        llm=llm,
        context_builder=context_builder,
        token_encoder=None,  # We'll use the default
        response_type="multiple paragraphs",
    )

    # Perform search
    result = await search_engine.asearch("What are the main topics discussed in the documents?")
    print(result.response)

if __name__ == "__main__":
    asyncio.run(main())
```

## Step 6: Advanced Configuration Options

### Custom Prompts
You can customize the extraction prompts in the `prompts/` directory:
- `entity_extraction.txt` - Controls what entities are extracted
- `summarize_descriptions.txt` - How entity descriptions are summarized
- `claim_extraction.txt` - What claims/facts are identified
- `community_report.txt` - How community summaries are generated

### Performance Tuning
- Adjust `chunk.size` and `chunk.overlap` based on your content
- Modify `parallelization.num_threads` based on your system
- Use `gpt-3.5-turbo` instead of `gpt-4` for faster/cheaper processing
- Enable `cache` to avoid reprocessing

### Memory and Cost Optimization
```yaml
# For large document sets, consider:
llm:
  model: gpt-3.5-turbo  # More cost-effective
  max_tokens: 2000      # Reduce for cost savings

chunk:
  size: 200            # Smaller chunks for better granularity
  overlap: 50          # Reduce overlap to save on processing

parallelization:
  num_threads: 10      # Reduce if hitting rate limits
```

## Step 7: Usage Examples

### Local Search (Specific Questions)
```python
result = await search_engine.asearch("What are the key findings about climate change?")
```

### Global Search (Broad Topics)
```python
# For global search, use GlobalSearch instead of LocalSearch
from graphrag.query.structured_search.global_search.search import GlobalSearch

global_search = GlobalSearch(
    llm=llm,
    context_builder=global_context_builder,
    token_encoder=token_encoder,
    max_data_tokens=12000,
)

result = await global_search.asearch("Summarize all major themes across the documents")
```

## Troubleshooting

### Common Issues:
1. **PDF Text Extraction**: Some PDFs may have poor text extraction. Consider using OCR for scanned documents
2. **Rate Limits**: Adjust `stagger` and `num_threads` if hitting OpenAI rate limits
3. **Memory Issues**: Reduce chunk sizes or process documents in batches
4. **Empty Results**: Check that PDF text extraction worked and files are in the correct format

### Monitoring Progress:
- Check `output/artifacts/` for generated files
- Review logs for any processing errors
- Verify text extraction quality from PDF files

This setup will create a knowledge graph from your PDF documents and allow you to perform both local (specific) and global (broad) queries using OpenAI's language models.