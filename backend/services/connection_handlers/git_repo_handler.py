import asyncio
import os
import re
import logging
from typing import Dict, Any, Type, Tuple, List, Optional, Literal
import inspect

# Third-party imports
from gitingest import ingest_async
from pydantic import BaseModel, HttpUrl, Field
from mcp import StdioServerParameters # Import needed for type hint
from grpc.aio import AioRpcError # Import for specific exception handling
from httpx import HTTPStatusError # Import for specific exception handling

# Qdrant Client imports for direct interaction
from qdrant_client import AsyncQdrantClient, models

# Local imports
from backend.db.models import Connection
from .base import MCPConnectionHandler # Changed base class
from backend.core.logging import get_logger

logger = get_logger(__name__)

# Added constant: safe chunk length (~512 tokens ≈ 2 048 chars) for embedding models
DEFAULT_CHUNK_SIZE_CHARS = 2048

# Basic sanitization: replace common URL characters with hyphens
def sanitize_url_for_collection(url: str) -> str:
    # Remove protocol
    sanitized = re.sub(r'^https?:\/\/', '', url)
    # Replace non-alphanumeric characters (except hyphen) with hyphen
    sanitized = re.sub(r'[^\w\-]+', '-', sanitized)
    # Remove leading/trailing hyphens
    sanitized = sanitized.strip('-')
    # Limit length if necessary (Qdrant might have limits)
    return sanitized[:63] # Example limit

# Pydantic model for connection creation/testing
class GitRepoConnectionCreate(BaseModel):
    type: Literal["git_repo"] = "git_repo" # Added type discriminator
    repo_url: HttpUrl = Field(..., description="URL of the Git repository")

class GitRepoConnectionHandler(MCPConnectionHandler): # Changed base class
    """Handles indexing data from a Git repository using gitingest and Qdrant."""

    # --- Implementation of MCPConnectionHandler ABC ---

    def get_connection_type(self) -> str:
        """Return the unique string identifier for this connection type."""
        return "git_repo"

    def get_create_model(self) -> Type[BaseModel]:
        """Return the Pydantic model class used for validating connection creation data."""
        return GitRepoConnectionCreate

    # Optional: Provide separate models if update/test differs significantly
    def get_update_model(self) -> Type[BaseModel]:
        # For now, assume update uses the same model as create
        return GitRepoConnectionCreate

    def get_test_model(self) -> Type[BaseModel]:
         # For now, assume test uses the same model as create
        return GitRepoConnectionCreate

    async def prepare_config(self, connection_id: Optional[str], input_data: BaseModel, existing_config: Optional[Dict] = None) -> Dict[str, Any]:
        """Prepares the configuration dictionary for DB storage."""
        if not isinstance(input_data, GitRepoConnectionCreate):
            raise ValueError("Invalid input data type for GitRepo connection")

        repo_url = str(input_data.repo_url) # Ensure it's a string
        collection_name = f"git-repo-{sanitize_url_for_collection(repo_url)}"
        logger.info(f"Prepared config for {connection_id or 'new connection'}: collection_name={collection_name}")
        # Store both the original URL and the derived collection name
        return {"repo_url": repo_url, "collection_name": collection_name}

    async def test_connection(self, config_to_test: Dict[str, Any]) -> Tuple[bool, str]:
        """Tests the connection by attempting to run gitingest briefly."""
        repo_url = config_to_test.get("repo_url")
        if not repo_url:
            return False, "Missing 'repo_url' in configuration for testing."
        try:
            # Try a quick ingest - might be slow, consider a lighter check if possible
            logger.info(f"Testing connection for {repo_url}...")
            # We don't need the full result, just check if it runs without error
            # Note: Removed max_files=1 as it's not a valid parameter for ingest_async
            await ingest_async(repo_url) # Check if basic call succeeds
            logger.info(f"Connection test successful for {repo_url}")
            return True, "Successfully connected to repository (basic check)."
        except Exception as e:
            logger.error(f"Connection test failed for {repo_url}: {e}", exc_info=True)
            return False, f"Failed to access repository: {str(e)}"

    def get_stdio_params(self, db_config: Dict[str, Any]) -> StdioServerParameters:
        """Return the StdioServerParameters for the Qdrant MCP that stores this
        repository's vectors.  We launch the same *shared* `mcp-server-qdrant`
        binary used elsewhere (via *uvx*). This instance communicates via stdio
        and is typically used for querying by agents.

        It connects to the Qdrant database specified by `SHERLOG_QDRANT_DB_URL`.
        The collection name is provided dynamically from the db_config.

        Expected keys in ``db_config`` (populated in ``prepare_config``):
        • ``collection_name`` – the name that ingest_async wrote into Qdrant.
        Optionally the caller/environment may provide:
        • ``qdrant_db_url`` – override for the Qdrant DB URL.
        • ``embedding_model`` – embedding model to use (default from env or hardcoded).
        """

        collection_name = db_config.get("collection_name")
        if not collection_name:
            raise ValueError("GitRepoConnectionHandler.get_stdio_params: 'collection_name' missing from db_config")

        # Resolve QDRANT DB URL - use config override, then env var
        qdrant_db_url = db_config.get("qdrant_db_url") or os.getenv("SHERLOG_QDRANT_DB_URL")
        if not qdrant_db_url:
            logger.error("SHERLOG_QDRANT_DB_URL environment variable is not set, and 'qdrant_db_url' is not in db_config. Cannot configure Qdrant connection for stdio MCP server.")
            raise ValueError("Qdrant DB URL configuration is missing for stdio MCP server.")

        # Resolve EMBEDDING MODEL - use config override, then env var, then default
        embedding_model_from_config = db_config.get("embedding_model")
        embedding_model_env = os.getenv("SHERLOG_QDRANT_EMBEDDING_MODEL")
        embedding_model = embedding_model_from_config or embedding_model_env or "sentence-transformers/all-MiniLM-L6-v2"

        logger.info(f"Configuring stdio mcp-server-qdrant: QDRANT_URL='{qdrant_db_url}', COLLECTION_NAME='{collection_name}', EMBEDDING_MODEL='{embedding_model}'")

        env = os.environ.copy()
        # Ensure QDRANT_URL is set for mcp-server-qdrant
        env.update(
            {
                "QDRANT_URL": qdrant_db_url,
                "COLLECTION_NAME": collection_name, # Set specific collection for this stdio instance
                "EMBEDDING_MODEL": embedding_model,
            }
        )
        # Remove potentially conflicting local path setting if it exists
        if "QDRANT_LOCAL_PATH" in env:
            logger.warning("Removing conflicting QDRANT_LOCAL_PATH from environment for stdio mcp-server-qdrant as QDRANT_URL is being used.")
            del env["QDRANT_LOCAL_PATH"]


        return StdioServerParameters(
            command="uvx",
            args=["mcp-server-qdrant"], # Relying on environment variables for configuration
            env=env,
        )

    def get_sensitive_fields(self) -> List[str]:
        """Return sensitive fields (none for this handler)."""
        return [] # No sensitive fields like API keys here

    async def execute_tool_call(self, tool_name: str, tool_args: Dict[str, Any], db_config: Dict[str, Any], correlation_id: Optional[str] = None) -> Any:
        """This handler does not directly execute MCP tool calls via stdio."""
        # Agents will call the qdrant-mcp server directly using its URL
        raise NotImplementedError("GitRepo handler does not execute tool calls directly.")

    # --- Indexing Logic (To be triggered separately after connection creation) ---
    async def post_create_actions(self, connection_config: Any) -> None:
        """
        Fetches data using gitingest and indexes it directly into Qdrant
        using the qdrant-client library.
        This is called after the connection is successfully created and saved.
        """
        connection_id = getattr(connection_config, 'id', 'UnknownID')
        connection_name = getattr(connection_config, 'name', 'UnknownName')
        config_dict = getattr(connection_config, 'config', {})

        repo_url = config_dict.get("repo_url")
        collection_name = config_dict.get("collection_name")
        # Get Qdrant DB URL from environment
        qdrant_db_url = os.getenv("SHERLOG_QDRANT_DB_URL")
        # Get embedding model from environment or use default
        embedding_model_name = os.getenv("SHERLOG_QDRANT_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        embedding_dim = 384  # Dimension for all-MiniLM-L6-v2  # noqa: F841

        log_prefix = f"[GitRepoDirectIndexing: {connection_id}]"
        logger.info(f"{log_prefix} START: Initiating post-create indexing. Repo: {repo_url}, Collection: {collection_name}, Qdrant: {qdrant_db_url}, Model: {embedding_model_name}")

        if not repo_url or not collection_name:
            logger.error(f"{log_prefix} ERROR: Missing 'repo_url' ({repo_url}) or 'collection_name' ({collection_name}) in config. Aborting indexing.")
            # TODO: Update connection status
            return

        if not qdrant_db_url:
            logger.error(f"{log_prefix} ERROR: SHERLOG_QDRANT_DB_URL environment variable not set. Aborting indexing.")
            # TODO: Update connection status
            return

        try:
            # Initialize Async Qdrant Client
            # Using prefer_grpc=True for potentially faster uploads if gRPC port (6334) is available
            # url parameter handles http/https schemes
            qdrant_client = AsyncQdrantClient(
                url=qdrant_db_url, 
                prefer_grpc=True, 
                timeout=60 # Changed float to int
            )
            logger.info(f"{log_prefix} Qdrant client initialized for URL: {qdrant_db_url}")

            # 1. Ensure fresh collection with correct configuration
            try:
                # Attempt to delete collection if exists to ensure fresh parameters
                await qdrant_client.delete_collection(collection_name=collection_name)
                logger.info(f"{log_prefix} Deleted existing collection '{collection_name}' to ensure fresh configuration.")
            except Exception as del_exc:
                # If collection didn't exist, that's fine; ignore NOT_FOUND errors
                try:
                    from grpc.aio import AioRpcError as _DelAioRpcError  # local import to check error type
                    if isinstance(del_exc, _DelAioRpcError):
                        if "NOT_FOUND" in str(del_exc.code()).upper():
                            logger.info(f"{log_prefix} Collection '{collection_name}' did not exist prior to creation.")
                        else:
                            logger.warning(f"{log_prefix} Unexpected error deleting collection '{collection_name}': {del_exc}")
                    else:
                        logger.info(f"{log_prefix} Collection '{collection_name}' may not exist yet: {del_exc}")
                except Exception:
                    logger.debug(f"{log_prefix} Non-gRPC error during collection delete: {del_exc}")

            # NOTE: We intentionally do NOT pre-create the collection here.
            # The FastEmbed mix-in (qdrant_client.add) will automatically create
            # a collection with the correct **named-vector** configuration that
            # matches the selected embedding model.  Pre-creating the collection
            # with an *unnamed* vector configuration causes the well-known
            # "Collection have incompatible vector params" assertion error.  By
            # letting FastEmbed handle creation we avoid that mismatch entirely.

            # Ensure client knows the embedding model for fastembed
            try:
                if hasattr(qdrant_client, "set_model"):
                    set_model_callable = getattr(qdrant_client, "set_model")  # type: ignore[attr-defined]
                    if inspect.iscoroutinefunction(set_model_callable):
                        await set_model_callable(embedding_model_name)
                    else:
                        set_model_callable(embedding_model_name)
                    logger.info(f"{log_prefix} Embedding model set on Qdrant client: {embedding_model_name}")
            except Exception:
                # Any issue with setting model is non-fatal for indexing, just continue.
                logger.debug(f"{log_prefix} qdrant_client.set_model not available or failed; proceeding without explicit model set.")

            # 2. Run gitingest
            logger.info(f"{log_prefix} Starting gitingest for URL: {repo_url}")
            summary, tree, content_dict = await ingest_async(repo_url)
            # Ensure content_dict is processed correctly, handling potential string format
            logger.info(f"{log_prefix} Gitingest processing completed.") # Simplified log

            # Use a single client instance for all operations
            # Removed 'async with' as AsyncQdrantClient doesn't support it directly

            # 3. Index Summary and Tree using client.add
            docs_to_add = []
            metadata_list = []

            if summary:
                logger.info(f"{log_prefix} Preparing summary for indexing.")
                docs_to_add.append(summary)
                metadata_list.append({'type': 'summary', 'repo_url': repo_url, 'connection_id': connection_id})

            if tree:
                logger.info(f"{log_prefix} Preparing file tree for indexing.")
                docs_to_add.append(tree)
                metadata_list.append({'type': 'tree', 'repo_url': repo_url, 'connection_id': connection_id})
            
            if docs_to_add:
                logger.info(f"{log_prefix} Indexing summary and/or tree ({len(docs_to_add)} items).")
                await qdrant_client.add(  # Rely on auto-generated IDs
                    collection_name=collection_name,
                    documents=docs_to_add,
                    metadata=metadata_list,
                )

            # 4. Process and Index File Contents
            files_to_index = {} # Initialize an empty dictionary for files

            if isinstance(content_dict, dict) and content_dict:
                logger.info(f"{log_prefix} Gitingest returned a dictionary with {len(content_dict)} files.")
                files_to_index = content_dict
            elif isinstance(content_dict, str) and content_dict.strip():
                logger.info(f"{log_prefix} Gitingest returned a single string. Attempting to parse files using '=== FILE: ... ===' pattern...")
                # Attempt parsing based on the user's format. Lines look like:
                # ======== FILE: path/to/file.py ========
                # We'll match any line that starts with ≥4 '=' followed by optional spaces, the literal 'FILE:',
                # then capture the filename until the end of line. Another '=' marker line (any length) follows.
                file_pattern = re.compile(r"^={4,}\s*FILE:\s*(.+?)\s*\n={4,}.*\n", re.MULTILINE)
                matches_iter = list(file_pattern.finditer(content_dict))
                
                parsed_files_count = 0
                for idx, match in enumerate(matches_iter):
                    filename = match.group(1).strip()
                    start_content = match.end()
                    end_content = matches_iter[idx + 1].start() if idx + 1 < len(matches_iter) else len(content_dict)

                    content = content_dict[start_content:end_content].strip()

                    if filename and content:
                        files_to_index[filename] = content
                        parsed_files_count += 1

                if parsed_files_count > 0:
                     logger.info(f"{log_prefix} Successfully parsed {parsed_files_count} files from the string content.")
                else:
                     logger.warning(
                         f"{log_prefix} String content found, but failed to parse files using the expected pattern. "
                         "Falling back to indexing the raw string in chunks."
                     )

                     # --- Fallback: Index raw string in chunks ---
                     raw_text: str = content_dict.strip()
                     if raw_text:
                         # Split the raw text into safe-sized chunks
                         chunk_size = DEFAULT_CHUNK_SIZE_CHARS
                         raw_chunks = [
                             raw_text[i : i + chunk_size]
                             for i in range(0, len(raw_text), chunk_size)
                         ]

                         chunk_metadata = [
                             {
                                 "type": "raw_chunk",
                                 "repo_url": repo_url,
                                 "connection_id": connection_id,
                                 "chunk_index": idx,
                             }
                             for idx in range(len(raw_chunks))
                         ]

                         try:
                             logger.info(
                                 f"{log_prefix} Indexing {len(raw_chunks)} raw text chunks as fallback."  # noqa: E501
                             )
                             await qdrant_client.add(
                                 collection_name=collection_name,
                                 documents=raw_chunks,
                                 metadata=chunk_metadata,
                             )
                             logger.info(
                                 f"{log_prefix} Successfully indexed raw text chunks fallback."
                             )
                         except Exception as raw_err:
                             logger.error(
                                 f"{log_prefix} Error indexing raw text chunks fallback: {raw_err}",
                                 exc_info=True,
                             )
                     # --- End fallback ---
            elif not content_dict:
                 logger.info(f"{log_prefix} No file content found in content_dict (it was empty or None).")
            else: # Handle other unexpected types
                 logger.error(f"{log_prefix} ERROR: Expected content_dict to be a dictionary or string, but got {type(content_dict)}. Skipping file content indexing.")


            # Proceed with indexing using the files_to_index dictionary
            if files_to_index:
                file_items = list(files_to_index.items())
                file_count = len(file_items)
                batch_size = 100 # Adjust batch size as needed
                logger.info(f"{log_prefix} Starting indexing of {file_count} file contents in batches of {batch_size}.")

                for i in range(0, file_count, batch_size):
                    batch_items = file_items[i:min(i + batch_size, file_count)]
                    batch_docs = [item[1] for item in batch_items if item[1]] # Ensure content is not empty
                    if not batch_docs: # Check if the filtered list is empty
                        logger.warning(f"{log_prefix} Skipping empty or filtered batch {i//batch_size + 1}.")
                        continue

                    batch_metadata = [
                        {'type': 'file', 'file_path': item[0], 'repo_url': repo_url, 'connection_id': connection_id}
                        for item in batch_items if item[1] # Ensure metadata matches filtered docs
                    ]

                    logger.info(f"{log_prefix} Indexing file batch {i//batch_size + 1}/{(file_count + batch_size - 1)//batch_size} ({len(batch_docs)} files)...")
                    try:
                        await qdrant_client.add(  # Use qdrant_client directly; auto-generated IDs
                            collection_name=collection_name,
                            documents=batch_docs,
                            metadata=batch_metadata,
                        )
                        logger.info(f"{log_prefix} Successfully indexed file batch {i//batch_size + 1}.")
                    except Exception as batch_err:
                        logger.error(f"{log_prefix} Error indexing file batch {i//batch_size + 1}: {batch_err}", exc_info=True)
                        # Optionally: Decide whether to continue with next batch or stop

                    await asyncio.sleep(0.1) # Small delay between batches

                logger.info(f"{log_prefix} Finished indexing {file_count} file contents.")
            # This case now covers when no files were found or parsed successfully
            else:
                 logger.info(f"{log_prefix} No file contents available for indexing after processing content_dict.")

            logger.info(f"{log_prefix} SUCCESS: Successfully completed all indexing operations for connection {connection_id} into collection {collection_name}")
            # TODO: Update connection status to indicate successful indexing

        except Exception as e:
            logger.exception(f"{log_prefix} FAIL: Error during post-create indexing for connection {connection_id} (Repo URL: {repo_url}): {e}")
            # TODO: Update connection status to indicate indexing failure
        finally:
            # Ensure client is closed if initialized
            if 'qdrant_client' in locals() and qdrant_client:
                try:
                    await qdrant_client.close()
                    logger.info(f"{log_prefix} Qdrant client closed.")
                except Exception as e_close:
                    logger.error(f"{log_prefix} Error closing Qdrant client: {e_close}", exc_info=True)


    async def delete_connection_data(self, connection: Connection) -> None:
        """
        Deletes data points associated with this connection_id from the Qdrant collection.
        It does not delete the collection itself, as other connections or data might reside
        if the collection naming/strategy changes in the future.
        """
        config_dict = getattr(connection, 'config', {})
        collection_name = config_dict.get("collection_name")
        qdrant_db_url = os.getenv("SHERLOG_QDRANT_DB_URL")
        connection_id_str = str(connection.id) # Ensure it's a string for matching

        log_prefix = f"[GitRepoDeleteData: {connection_id_str}]"
        logger.info(f"{log_prefix} Initiating deletion of points for connection ID '{connection_id_str}' from collection: {collection_name}")

        if not collection_name:
            logger.error(f"{log_prefix} ERROR: Missing 'collection_name' in connection config. Aborting deletion.")
            return
        if not qdrant_db_url:
            logger.error(f"{log_prefix} ERROR: SHERLOG_QDRANT_DB_URL environment variable not set. Aborting deletion.")
            return

        qdrant_client = None # Initialize to None for finally block
        try:
            qdrant_client = AsyncQdrantClient(url=qdrant_db_url, prefer_grpc=True, timeout=60)
            logger.info(f"{log_prefix} Qdrant client initialized for URL: {qdrant_db_url}")

            # Check if collection exists before attempting to delete points
            try:
                await qdrant_client.get_collection(collection_name=collection_name)
                logger.info(f"{log_prefix} Collection '{collection_name}' found.")
            except Exception as e_get_coll:
                 is_not_found_error = False
                 if isinstance(e_get_coll, AioRpcError):
                     # Now e_get_coll is known to be AioRpcError
                     rpc_error: AioRpcError = e_get_coll 
                     status_code_enum_str = str(rpc_error.code()).upper()
                     if "NOT_FOUND" in status_code_enum_str:
                         is_not_found_error = True
                 elif isinstance(e_get_coll, HTTPStatusError):
                     # Now e_get_coll is known to be HTTPStatusError
                     http_error: HTTPStatusError = e_get_coll
                     if http_error.response.status_code == 404:
                         is_not_found_error = True
                 
                 if is_not_found_error:
                    logger.warning(f"{log_prefix} Collection '{collection_name}' not found. No points to delete.")
                    return # Nothing to do if collection doesn't exist
                 else:
                    logger.error(f"{log_prefix} Error checking collection '{collection_name}': {e_get_coll}. Aborting deletion.", exc_info=True)
                    return


            points_selector = models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="connection_id", # Assumes 'connection_id' is stored in metadata
                            match=models.MatchValue(value=connection_id_str),
                        )
                    ]
                )
            )
            
            logger.info(f"{log_prefix} Deleting points with connection_id '{connection_id_str}' from '{collection_name}'.")
            await qdrant_client.delete(
                collection_name=collection_name,
                points_selector=points_selector
            )
            logger.info(f"{log_prefix} Successfully submitted request to delete points for connection_id '{connection_id_str}' from collection '{collection_name}'.")
            # Note: Deletion is async in Qdrant. For immediate confirmation, one might need to check point counts or use `wait=True` if available.

        except Exception as e:
            logger.error(f"{log_prefix} Error during point deletion for connection {connection_id_str} from collection '{collection_name}': {e}", exc_info=True)
        finally:
            if qdrant_client:
                try:
                    await qdrant_client.close()
                    logger.info(f"{log_prefix} Qdrant client closed.")
                except Exception as e_close:
                    logger.error(f"{log_prefix} Error closing Qdrant client: {e_close}", exc_info=True)

# ---------------------------------------------------------------------------
# Register handler instance so it is discoverable via the registry at import
# time (mirrors behaviour of filesystem handler registration).
# ---------------------------------------------------------------------------

from backend.services.connection_handlers.registry import register_handler  # noqa: E402

register_handler(GitRepoConnectionHandler())
