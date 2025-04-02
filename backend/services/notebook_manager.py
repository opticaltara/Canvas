"""
Notebook Manager Service
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Union
from uuid import UUID, uuid4

from backend.config import get_settings
from backend.core.notebook import Notebook

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
        notebook_id = uuid4()
        notebook = Notebook(
            id=notebook_id,
            name=name,
            description=description or "",
            metadata=metadata or {},
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat()
        )
        
        self.notebooks[notebook_id] = notebook
        self.save_notebook(notebook_id)
        
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
            raise KeyError(f"Notebook {notebook_id} not found")
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
            raise KeyError(f"Notebook {notebook_id} not found")
        
        notebook = self.notebooks[notebook_id]
        
        # Update the updated_at timestamp
        notebook.updated_at = datetime.utcnow().isoformat()
        
        # Save to storage based on storage type
        storage_type = self.settings.notebook_storage_type
        
        if storage_type == "file":
            self._save_notebook_to_file(notebook)
        elif storage_type == "s3":
            asyncio.create_task(self._save_notebook_to_s3(notebook))
        else:
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
            raise KeyError(f"Notebook {notebook_id} not found")
        
        # Delete from storage based on storage type
        storage_type = self.settings.notebook_storage_type
        
        if storage_type == "file":
            self._delete_notebook_from_file(notebook_id)
        elif storage_type == "s3":
            asyncio.create_task(self._delete_notebook_from_s3(notebook_id))
        
        # Remove from memory
        del self.notebooks[notebook_id]
    
    def _load_notebooks(self) -> None:
        """Load notebooks from storage"""
        storage_type = self.settings.notebook_storage_type
        
        if storage_type == "file":
            self._load_notebooks_from_file()
        elif storage_type == "s3":
            asyncio.create_task(self._load_notebooks_from_s3())
        else:
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
        
        with open(file_path, "w") as f:
            json.dump(notebook.serialize(), f, indent=2)
    
    def _load_notebooks_from_file(self) -> None:
        """Load notebooks from files"""
        directory = self.settings.notebook_file_storage_dir
        
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            return
        
        for filename in os.listdir(directory):
            if not filename.endswith('.json'):
                continue
            
            file_path = os.path.join(directory, filename)
            
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                
                notebook_id = UUID(data.get("id"))
                
                # Create a notebook instance
                notebook = Notebook(
                    id=notebook_id,
                    name=data.get("name", "Untitled"),
                    description=data.get("description", ""),
                    metadata=data.get("metadata", {}),
                    created_at=data.get("created_at", datetime.utcnow().isoformat()),
                    updated_at=data.get("updated_at", datetime.utcnow().isoformat())
                )
                
                # Load cells
                for cell_data in data.get("cells", []):
                    notebook.add_cell_from_dict(cell_data)
                
                # Load dependencies
                for dependency in data.get("dependencies", []):
                    dependent_id = UUID(dependency[0])
                    dependency_id = UUID(dependency[1])
                    notebook.add_dependency(dependent_id, dependency_id)
                
                # Add to notebooks dict
                self.notebooks[notebook_id] = notebook
            
            except Exception as e:
                print(f"Error loading notebook from {file_path}: {e}")
    
    def _delete_notebook_from_file(self, notebook_id: UUID) -> None:
        """
        Delete a notebook file
        
        Args:
            notebook_id: The ID of the notebook
        """
        directory = self.settings.notebook_file_storage_dir
        file_path = os.path.join(directory, f"{notebook_id}.json")
        
        if os.path.exists(file_path):
            os.remove(file_path)
    
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
            
            session = aioboto3.Session(
                aws_access_key_id=self.settings.aws_access_key_id,
                aws_secret_access_key=self.settings.aws_secret_access_key,
                region_name=self.settings.aws_region
            )
            
            async with session.client("s3") as s3:
                await s3.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=json.dumps(notebook.serialize(), indent=2),
                    ContentType="application/json"
                )
        
        except Exception as e:
            print(f"Error saving notebook to S3: {e}")
    
    async def _load_notebooks_from_s3(self) -> None:
        """Load notebooks from S3"""
        try:
            import aioboto3
            
            bucket = self.settings.notebook_s3_bucket
            prefix = self.settings.notebook_s3_prefix
            
            session = aioboto3.Session(
                aws_access_key_id=self.settings.aws_access_key_id,
                aws_secret_access_key=self.settings.aws_secret_access_key,
                region_name=self.settings.aws_region
            )
            
            async with session.client("s3") as s3:
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
                    
                    # Get the notebook file
                    response = await s3.get_object(
                        Bucket=bucket,
                        Key=key
                    )
                    
                    # Read the data
                    data = await response["Body"].read()
                    data = json.loads(data)
                    
                    notebook_id = UUID(data.get("id"))
                    
                    # Create a notebook instance
                    notebook = Notebook(
                        id=notebook_id,
                        name=data.get("name", "Untitled"),
                        description=data.get("description", ""),
                        metadata=data.get("metadata", {}),
                        created_at=data.get("created_at", datetime.utcnow().isoformat()),
                        updated_at=data.get("updated_at", datetime.utcnow().isoformat())
                    )
                    
                    # Load cells
                    for cell_data in data.get("cells", []):
                        notebook.add_cell_from_dict(cell_data)
                    
                    # Load dependencies
                    for dependency in data.get("dependencies", []):
                        dependent_id = UUID(dependency[0])
                        dependency_id = UUID(dependency[1])
                        notebook.add_dependency(dependent_id, dependency_id)
                    
                    # Add to notebooks dict
                    self.notebooks[notebook_id] = notebook
        
        except Exception as e:
            print(f"Error loading notebooks from S3: {e}")
    
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
            
            session = aioboto3.Session(
                aws_access_key_id=self.settings.aws_access_key_id,
                aws_secret_access_key=self.settings.aws_secret_access_key,
                region_name=self.settings.aws_region
            )
            
            async with session.client("s3") as s3:
                await s3.delete_object(
                    Bucket=bucket,
                    Key=key
                )
        
        except Exception as e:
            print(f"Error deleting notebook from S3: {e}")