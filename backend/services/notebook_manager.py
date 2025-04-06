"""
Notebook Manager Service
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Union
from uuid import UUID, uuid4

from backend.config import get_settings
from backend.core.notebook import Notebook, NotebookMetadata

# Configure logging
logger = logging.getLogger(__name__)

# Make sure correlation_id is set for all logs
class CorrelationIdFilter(logging.Filter):
    def filter(self, record):
        # Instead of requiring correlation_id, just add it if missing
        if not hasattr(record, 'correlation_id'):
            record.correlation_id = 'N/A'
        return True

# Add the filter to our logger
logger.addFilter(CorrelationIdFilter())

# Singleton instance
_notebook_manager_instance = None


def get_notebook_manager():
    """Get the singleton NotebookManager instance"""
    global _notebook_manager_instance
    if _notebook_manager_instance is None:
        _notebook_manager_instance = NotebookManager()
    return _notebook_manager_instance


class NotebookManager:
    """
    Manages notebook instances and their persistence
    """
    def __init__(self):
        correlation_id = str(uuid4())
        self.notebooks: Dict[UUID, Notebook] = {}
        self.settings = get_settings()
        self.notify_callback: Optional[Callable] = None
        
        # Load notebooks from storage
        self._load_notebooks()
    
    def set_notify_callback(self, callback: Callable) -> None:
        """
        Set the callback function for notifying clients of changes
        
        Args:
            callback: The callback function
        """
        self.notify_callback = callback
    
    def create_notebook(self, name: str, description: Optional[str] = None, metadata: Optional[Dict] = None) -> Notebook:
        """
        Create a new notebook
        
        Args:
            name: The name of the notebook
            description: Optional description
            metadata: Optional metadata
            
        Returns:
            The created notebook
        """
        correlation_id = str(uuid4())
        notebook_id = uuid4()
        logger.info(f"Creating new notebook with ID {notebook_id} and name '{name}'", extra={'correlation_id': correlation_id})
        
        notebook = Notebook(
            id=notebook_id,
            metadata=NotebookMetadata(
                title=name,
                description=description or "",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
        )
        
        # Update metadata if provided
        if metadata:
            logger.debug(f"Updating metadata for notebook {notebook_id}", extra={'correlation_id': correlation_id})
            notebook.update_metadata(metadata)
        
        self.notebooks[notebook_id] = notebook
        self.save_notebook(notebook_id)
        logger.info(f"Successfully created and saved notebook {notebook_id}", extra={'correlation_id': correlation_id})
        
        return notebook
    
    def get_notebook(self, notebook_id: UUID) -> Notebook:
        """
        Get a notebook by ID
        
        Args:
            notebook_id: The ID of the notebook
            
        Returns:
            The notebook
            
        Raises:
            KeyError: If the notebook is not found
        """
        if notebook_id not in self.notebooks:
            logger.error(f"Notebook {notebook_id} not found")
            raise KeyError(f"Notebook {notebook_id} not found")
        logger.debug(f"Retrieved notebook {notebook_id}")
        return self.notebooks[notebook_id]
    
    def list_notebooks(self) -> List[Notebook]:
        """
        Get a list of all notebooks
        
        Returns:
            List of notebooks
        """
        return list(self.notebooks.values())
    
    def save_notebook(self, notebook_id: UUID) -> None:
        """
        Save a notebook to storage
        
        Args:
            notebook_id: The ID of the notebook
            
        Raises:
            KeyError: If the notebook is not found
        """
        if notebook_id not in self.notebooks:
            logger.error(f"Cannot save notebook {notebook_id}: not found")
            raise KeyError(f"Notebook {notebook_id} not found")
        
        notebook = self.notebooks[notebook_id]
        logger.info(f"Saving notebook {notebook_id} to storage")
        
        # Update the updated_at timestamp
        notebook.metadata.updated_at = datetime.now(timezone.utc)
        
        # Save to storage based on storage type
        storage_type = self.settings.notebook_storage_type
        
        if storage_type == "file":
            self._save_notebook_to_file(notebook)
        elif storage_type == "s3":
            asyncio.create_task(self._save_notebook_to_s3(notebook))
        else:
            logger.warning(f"No persistent storage configured for notebook {notebook_id}")
            # Default to in-memory only (no persistence)
            pass
    
    def delete_notebook(self, notebook_id: UUID) -> None:
        """
        Delete a notebook
        
        Args:
            notebook_id: The ID of the notebook
            
        Raises:
            KeyError: If the notebook is not found
        """
        if notebook_id not in self.notebooks:
            logger.error(f"Cannot delete notebook {notebook_id}: not found")
            raise KeyError(f"Notebook {notebook_id} not found")
        
        logger.info(f"Deleting notebook {notebook_id}")
        
        # Delete from storage based on storage type
        storage_type = self.settings.notebook_storage_type
        
        if storage_type == "file":
            self._delete_notebook_from_file(notebook_id)
        elif storage_type == "s3":
            asyncio.create_task(self._delete_notebook_from_s3(notebook_id))
        
        # Remove from memory
        del self.notebooks[notebook_id]
        logger.info(f"Successfully deleted notebook {notebook_id}")
    
    def _load_notebooks(self) -> None:
        """Load notebooks from storage"""
        correlation_id = str(uuid4())
        storage_type = self.settings.notebook_storage_type
        logger.info(f"Loading notebooks from storage type: {storage_type}", extra={'correlation_id': correlation_id})
        
        if storage_type == "file":
            self._load_notebooks_from_file()
        elif storage_type == "s3":
            asyncio.create_task(self._load_notebooks_from_s3())
        else:
            logger.warning("No persistent storage configured, starting with empty notebook set", 
                          extra={'correlation_id': correlation_id})
            # In-memory only, nothing to load
            pass
    
    def _save_notebook_to_file(self, notebook: Notebook) -> None:
        """
        Save a notebook to a file
        
        Args:
            notebook: The notebook to save
        """
        directory = self.settings.notebook_file_storage_dir
        os.makedirs(directory, exist_ok=True)
        
        file_path = os.path.join(directory, f"{notebook.id}.json")
        logger.debug(f"Saving notebook {notebook.id} to file: {file_path}")
        
        try:
            with open(file_path, "w") as f:
                json.dump(notebook.serialize(), f, indent=2)
            logger.debug(f"Successfully saved notebook {notebook.id} to file")
        except Exception as e:
            logger.error(f"Error saving notebook {notebook.id} to file: {e}")
            raise
    
    def _load_notebooks_from_file(self) -> None:
        """Load notebooks from files"""
        correlation_id = str(uuid4())
        directory = self.settings.notebook_file_storage_dir
        logger.info(f"Loading notebooks from directory: {directory}", extra={'correlation_id': correlation_id})
        
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created notebook storage directory: {directory}", extra={'correlation_id': correlation_id})
            return
        
        for filename in os.listdir(directory):
            if not filename.endswith('.json'):
                continue
            
            file_path = os.path.join(directory, filename)
            logger.debug(f"Loading notebook from file: {file_path}")
            
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                
                notebook_id = UUID(data.get("id"))
                logger.debug(f"Creating notebook instance for ID: {notebook_id}")
                
                # Create a notebook instance
                notebook = Notebook(
                    id=notebook_id,
                    metadata=NotebookMetadata(
                        title=data.get("name", "Untitled"),
                        description=data.get("description", ""),
                        created_at=datetime.fromisoformat(data.get("created_at", datetime.now(timezone.utc).isoformat())),
                        updated_at=datetime.fromisoformat(data.get("updated_at", datetime.now(timezone.utc).isoformat()))
                    )
                )
                
                # Update metadata if provided
                if data.get("metadata"):
                    notebook.update_metadata(data.get("metadata"))
                
                # Load cells
                for cell_data in data.get("cells", []):
                    notebook.create_cell(
                        cell_type=cell_data["type"],
                        content=cell_data["content"],
                        **cell_data.get("metadata", {})
                    )
                
                # Load dependencies
                for dependency in data.get("dependencies", []):
                    dependent_id = UUID(dependency[0])
                    dependency_id = UUID(dependency[1])
                    notebook.add_dependency(dependent_id, dependency_id)
                
                # Add to notebooks dict
                self.notebooks[notebook_id] = notebook
                logger.debug(f"Successfully loaded notebook {notebook_id}")
            
            except Exception as e:
                logger.error(f"Error loading notebook from {file_path}: {e}")
    
    def _delete_notebook_from_file(self, notebook_id: UUID) -> None:
        """
        Delete a notebook file
        
        Args:
            notebook_id: The ID of the notebook
        """
        directory = self.settings.notebook_file_storage_dir
        file_path = os.path.join(directory, f"{notebook_id}.json")
        logger.debug(f"Deleting notebook file: {file_path}")
        
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.debug(f"Successfully deleted notebook file for {notebook_id}")
            except Exception as e:
                logger.error(f"Error deleting notebook file {file_path}: {e}")
                raise
        else:
            logger.warning(f"Notebook file not found for deletion: {file_path}")
    
    async def _save_notebook_to_s3(self, notebook: Notebook) -> None:
        """
        Save a notebook to S3
        
        Args:
            notebook: The notebook to save
        """
        try:
            import aioboto3
            
            bucket = self.settings.notebook_s3_bucket
            key = f"{self.settings.notebook_s3_prefix}/{notebook.id}.json"
            logger.debug(f"Saving notebook {notebook.id} to S3: {bucket}/{key}")
            
            session = aioboto3.Session(
                aws_access_key_id=self.settings.aws_access_key_id,
                aws_secret_access_key=self.settings.aws_secret_access_key,
                region_name=self.settings.aws_region
            )
            
            s3 = session.client("s3")
            await s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=json.dumps(notebook.serialize(), indent=2),
                ContentType="application/json"
            )
            logger.debug(f"Successfully saved notebook {notebook.id} to S3")
        
        except Exception as e:
            logger.error(f"Error saving notebook {notebook.id} to S3: {e}")
            raise
    
    async def _load_notebooks_from_s3(self) -> None:
        """Load notebooks from S3"""
        try:
            import aioboto3
            
            bucket = self.settings.notebook_s3_bucket
            prefix = self.settings.notebook_s3_prefix
            logger.info(f"Loading notebooks from S3: {bucket}/{prefix}")
            
            session = aioboto3.Session(
                aws_access_key_id=self.settings.aws_access_key_id,
                aws_secret_access_key=self.settings.aws_secret_access_key,
                region_name=self.settings.aws_region
            )
            
            s3 = session.client("s3")
            # List objects in the bucket with the prefix
            response = await s3.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix
            )
            
            # Get each notebook file
            for obj in response.get("Contents", []):
                key = obj["Key"]
                
                if not key.endswith(".json"):
                    continue
                
                logger.debug(f"Loading notebook from S3: {bucket}/{key}")
                
                # Get the notebook file
                response = await s3.get_object(
                    Bucket=bucket,
                    Key=key
                )
                
                # Read the data
                data = await response["Body"].read()
                data = json.loads(data)
                
                notebook_id = UUID(data.get("id"))
                logger.debug(f"Creating notebook instance for ID: {notebook_id}")
                
                # Create a notebook instance
                notebook = Notebook(
                    id=notebook_id,
                    metadata=NotebookMetadata(
                        title=data.get("name", "Untitled"),
                        description=data.get("description", ""),
                        created_at=datetime.fromisoformat(data.get("created_at", datetime.now(timezone.utc).isoformat())),
                        updated_at=datetime.fromisoformat(data.get("updated_at", datetime.now(timezone.utc).isoformat()))
                    )
                )
                
                # Update metadata if provided
                if data.get("metadata"):
                    notebook.update_metadata(data.get("metadata"))
                
                # Load cells
                for cell_data in data.get("cells", []):
                    notebook.create_cell(
                        cell_type=cell_data["type"],
                        content=cell_data["content"],
                        **cell_data.get("metadata", {})
                    )
                
                # Load dependencies
                for dependency in data.get("dependencies", []):
                    dependent_id = UUID(dependency[0])
                    dependency_id = UUID(dependency[1])
                    notebook.add_dependency(dependent_id, dependency_id)
                
                # Add to notebooks dict
                self.notebooks[notebook_id] = notebook
                logger.debug(f"Successfully loaded notebook {notebook_id} from S3")
        
        except Exception as e:
            logger.error(f"Error loading notebooks from S3: {e}")
            raise
    
    async def _delete_notebook_from_s3(self, notebook_id: UUID) -> None:
        """
        Delete a notebook from S3
        
        Args:
            notebook_id: The ID of the notebook
        """
        try:
            import aioboto3
            
            bucket = self.settings.notebook_s3_bucket
            key = f"{self.settings.notebook_s3_prefix}/{notebook_id}.json"
            logger.debug(f"Deleting notebook from S3: {bucket}/{key}")
            
            session = aioboto3.Session(
                aws_access_key_id=self.settings.aws_access_key_id,
                aws_secret_access_key=self.settings.aws_secret_access_key,
                region_name=self.settings.aws_region
            )
            
            s3 = session.client("s3")
            await s3.delete_object(
                Bucket=bucket,
                Key=key
            )
            logger.debug(f"Successfully deleted notebook {notebook_id} from S3")
        
        except Exception as e:
            logger.error(f"Error deleting notebook {notebook_id} from S3: {e}")
            raise