# graphrag_search.py

import os
import logging
import pandas as pd
import tiktoken
from typing import AsyncGenerator

from dotenv import load_dotenv

from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.structured_search.local_search.mixed_context import (
    LocalSearchMixedContext,
)
from graphrag.query.structured_search.global_search.community_context import (
    GlobalCommunityContext,
)
from graphrag.query.indexer_adapters import (
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.input.loaders.dfs import store_entity_semantic_embeddings
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.query.structured_search.global_search.search import GlobalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore

# Configure logging
logger = logging.getLogger(__name__)

class GraphRAGSearchEngine:
    def __init__(
        self,
        input_dir,
        lancedb_uri,
        api_key,
        api_base,
        llm_model,
        embedding_model,
        search_type='local',  # Added parameter
        api_type=OpenaiApiType.AzureOpenAI,
        community_level=2,
        collection_name="entity_description_embeddings",
        entity_table="create_final_nodes",
        entity_embedding_table="create_final_entities",
        relationship_table="create_final_relationships",
        community_report_table="create_final_community_reports",
        text_unit_table="create_final_text_units",
        local_context_params=None,
        global_context_builder_params=None,  # Added
        global_search_params=None,           # Added
        llm_params=None,
        response_type="multiple paragraphs",
        embedding_vectorstore_key=EntityVectorStoreKey.ID,
        token_encoder_name="cl100k_base",
    ):
        self.input_dir = input_dir
        self.lancedb_uri = lancedb_uri
        self.api_key = api_key
        self.api_base = api_base
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.api_type = api_type
        self.community_level = community_level
        self.collection_name = collection_name
        self.entity_table = entity_table
        self.entity_embedding_table = entity_embedding_table
        self.relationship_table = relationship_table
        self.community_report_table = community_report_table
        self.text_unit_table = text_unit_table
        self.embedding_vectorstore_key = embedding_vectorstore_key
        self.token_encoder_name = token_encoder_name

        self.search_type = search_type

        # Assign context builder params based on search type
        if self.search_type == 'local':
            if local_context_params is None:
                self.local_context_params = {
                    # Default local context params
                    "text_unit_prop": 0.5,
                    "community_prop": 0.1,
                    "conversation_history_max_turns": 5,
                    "conversation_history_user_turns_only": True,
                    "top_k_mapped_entities": 10,
                    "top_k_relationships": 10,
                    "include_entity_rank": True,
                    "include_relationship_weight": True,
                    "include_community_rank": False,
                    "return_candidate_context": False,
                    "embedding_vectorstore_key": self.embedding_vectorstore_key,
                    "max_tokens": 12_000,
                }
            else:
                self.local_context_params = local_context_params
        elif self.search_type == 'global':
            if global_context_builder_params is None:
                self.global_context_builder_params = {
                    # Default global context builder params
                    "use_community_summary": False,
                    "shuffle_data": True,
                    "include_community_rank": True,
                    "min_community_rank": 0,
                    "community_rank_name": "rank",
                    "include_community_weight": True,
                    "community_weight_name": "occurrence weight",
                    "normalize_community_weight": True,
                    "max_tokens": 12_000,
                    "context_name": "Reports",
                }
            else:
                self.global_context_builder_params = global_context_builder_params

            if global_search_params is None:
                self.global_search_params = {
                    "max_data_tokens": 12_000,
                    "map_llm_params": {
                        "max_tokens": 1000,
                        "temperature": 0.0,
                        "response_format": {"type": "json_object"},
                    },
                    "reduce_llm_params": {
                        "max_tokens": 2000,
                        "temperature": 0.0,
                    },
                    "allow_general_knowledge": False,
                    "json_mode": True,
                    "concurrent_coroutines": 32,
                }
            else:
                self.global_search_params = global_search_params
        else:
            raise ValueError(f"Invalid search_type: {self.search_type}")

        if llm_params is None:
            self.llm_params = {
                "max_tokens": 2_000,
                "temperature": 0.0,
            }
        else:
            self.llm_params = llm_params

        self.response_type = response_type

        # Initialize components
        self.load_data()
        self.initialize_vectorstore()
        self.initialize_llm()
        self.initialize_context_builder()
        self.initialize_search_engine()

    def load_data(self):
        logger.info("Loading entities...")
        # Load entities
        entity_df = pd.read_parquet(f"{self.input_dir}/{self.entity_table}.parquet")
        entity_embedding_df = pd.read_parquet(f"{self.input_dir}/{self.entity_embedding_table}.parquet")
        self.entities = read_indexer_entities(entity_df, entity_embedding_df, self.community_level)
        logger.info(f"Loaded {len(self.entities)} entities.")

        # Load relationships
        logger.info("Loading relationships...")
        relationship_df = pd.read_parquet(f"{self.input_dir}/{self.relationship_table}.parquet")
        self.relationships = read_indexer_relationships(relationship_df)
        logger.info(f"Loaded {len(self.relationships)} relationships.")

        # Load reports
        logger.info("Loading reports...")
        report_df = pd.read_parquet(f"{self.input_dir}/{self.community_report_table}.parquet")
        self.reports = read_indexer_reports(report_df, entity_df, self.community_level)
        logger.info(f"Loaded {len(self.reports)} reports.")

        # Load text units
        logger.info("Loading text units...")
        text_unit_df = pd.read_parquet(f"{self.input_dir}/{self.text_unit_table}.parquet")
        self.text_units = read_indexer_text_units(text_unit_df)
        logger.info(f"Loaded {len(self.text_units)} text units.")

    def initialize_vectorstore(self):
        logger.info("Initializing the description embedding store...")

        # Ensure the lancedb directory exists
        if not os.path.exists(self.lancedb_uri):
            os.makedirs(self.lancedb_uri)

        # Initialize the description embedding store
        self.description_embedding_store = LanceDBVectorStore(
            collection_name=self.collection_name,
        )
        self.description_embedding_store.connect(db_uri=self.lancedb_uri)
        # Store entity semantic embeddings
        logger.info("Storing entity semantic embeddings...")
        store_entity_semantic_embeddings(
            entities=self.entities, vectorstore=self.description_embedding_store
        )
        logger.info("Entity semantic embeddings stored.")

    def initialize_llm(self):
        logger.info("Initializing the LLM...")
        # Initialize the LLM
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            api_version="2024-02-15-preview",
            api_base=self.api_base,
            model=self.llm_model,
            api_type=self.api_type,
            max_retries=20,
        )

        # Initialize the token encoder
        logger.info("Initializing the token encoder...")
        self.token_encoder = tiktoken.get_encoding(self.token_encoder_name)

        # Initialize the text embedder
        logger.info("Initializing the text embedder...")
        self.text_embedder = OpenAIEmbedding(
            api_key=self.api_key,
            api_version="2024-02-15-preview",
            api_base=self.api_base,
            api_type=self.api_type,
            model=self.embedding_model,
            deployment_name=self.embedding_model,
            max_retries=20,
        )
        logger.info("LLM and embeddings initialized.")

    def initialize_context_builder(self):
        logger.info("Initializing the context builder...")
        if self.search_type == 'local':
            self.context_builder = LocalSearchMixedContext(
                community_reports=self.reports,
                text_units=self.text_units,
                entities=self.entities,
                relationships=self.relationships,
                entity_text_embeddings=self.description_embedding_store,
                embedding_vectorstore_key=self.embedding_vectorstore_key,
                text_embedder=self.text_embedder,
                token_encoder=self.token_encoder,
            )
        elif self.search_type == 'global':
            self.context_builder = GlobalCommunityContext(
                community_reports=self.reports,
                entities=self.entities,
                token_encoder=self.token_encoder,
            )
        else:
            raise ValueError(f"Invalid search_type: {self.search_type}")
        logger.info("Context builder initialized.")

    def initialize_search_engine(self):
        logger.info("Initializing the search engine...")
        if self.search_type == 'local':
            self.search_engine = LocalSearch(
                llm=self.llm,
                context_builder=self.context_builder,
                token_encoder=self.token_encoder,
                llm_params=self.llm_params,
                context_builder_params=self.local_context_params,
                response_type=self.response_type,
            )
        elif self.search_type == 'global':
            self.search_engine = GlobalSearch(
                llm=self.llm,
                context_builder=self.context_builder,
                token_encoder=self.token_encoder,
                max_data_tokens=self.global_search_params.get('max_data_tokens', 12000),
                map_llm_params=self.global_search_params.get('map_llm_params', {}),
                reduce_llm_params=self.global_search_params.get('reduce_llm_params', {}),
                allow_general_knowledge=self.global_search_params.get('allow_general_knowledge', False),
                json_mode=self.global_search_params.get('json_mode', True),
                context_builder_params=self.global_context_builder_params,
                concurrent_coroutines=self.global_search_params.get('concurrent_coroutines', 32),
                response_type=self.response_type,
            )
        else:
            raise ValueError(f"Invalid search_type: {self.search_type}")
        logger.info("Search engine initialized.")


    # for synchronous queries
    async def run_query(self, query):
        logger.info(f"Running query: {query}")
        result = await self.search_engine.asearch(query)
        logger.info("Query completed.")
        return result.response
    
    async def run_query_stream(self, query: str) -> AsyncGenerator[str, None]:
            logger.info(f"Running query: {query}")
            # Enable streaming in the LLM
            self.llm.streaming = True
            self.search_engine.llm.streaming = True

            # Run the query and get an asynchronous generator
            generator = self.search_engine.astream_search(query)

            # Consume the first yielded item (context_records)
            try:
                context_records = await generator.__anext__()
            except StopAsyncIteration:
                # Handle the case where the generator is empty
                logger.error("The generator didn't yield any items.")
                return

            # Optionally, you can log or process context_records here
            # logger.info(f"Context Records: {context_records}")

            # Yield the rest of the response chunks to the client
            async for chunk in generator:
                logger.info(f"Received chunk: {chunk}")
                yield chunk

            logger.info("Query completed.")