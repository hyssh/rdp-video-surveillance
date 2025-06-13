import os
import logging
import uuid
from typing import Optional, Dict, Any, Union
from semantic_kernel.agents import AzureAIAgent, AzureAIAgentThread, ChatCompletionAgent,ChatHistoryAgentThread
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.identity import DefaultAzureCredential

# Configure logging with correlation IDs for better Azure Monitor integration
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - [%(correlation_id)s] - %(levelname)s - %(message)s')

class ThreadManager:
    """
    Manages Azure AI Agent thread lifecycle following Azure best practices.
    
    This class provides thread creation, retrieval, and persistence capabilities
    optimized for Azure AI agent interactions.
    """
    
    def __init__(self):
        self.azureaiagentthread = None
    
    async def get_or_create_thread(self, 
                                  agent: Union[AzureAIAgent, ChatCompletionAgent], 
                                  thread_id: Optional[str] = None,
                                  metadata: Optional[Dict[str, Any]] = None) -> Union[AzureAIAgentThread, ChatHistoryAgentThread]:
        """
        Gets an existing thread or creates a new one with proper Azure error handling.
        
        Args:
            agent: The Azure AI Agent to associate with the thread
            thread_id: Optional existing thread ID to retrieve
            metadata: Optional metadata to associate with the thread
            
        Returns:
            azureaiagentthread: The thread instance
        """
        if isinstance(agent, AzureAIAgent):
            # Validate agent
            if not agent or not hasattr(agent, 'client'):
                logging.error("Invalid agent provided")
                raise ValueError("A valid Azure AI agent is required")
                
            try:
                # Use provided thread_id if available
                if self.azureaiagentthread:
                        return self.azureaiagentthread                
                
                logging.info("Creating new Azure AI agent thread")
                thread_response = await agent.client.agents.threads.create()
                self.azureaiagentthread = AzureAIAgentThread(thread_id=thread_response.id, client=agent.client)
                
                # Cache the thread for future use
                logging.info(f"Created new thread: {self.azureaiagentthread.id}")
                
                return self.azureaiagentthread
                    
            except Exception as e:
                logging.error(f"Error in thread management: {str(e)}")
                # Fallback to threadless operation as per Azure resilience patterns
                raise RuntimeError("Failed to create or retrieve thread")
        elif isinstance(agent, ChatCompletionAgent):
            # For ChatCompletionAgent, we can create a threadless operation
            logging.info("Using ChatCompletionAgent without threads")
            self.azureaiagentthread = ChatHistoryAgentThread()
            return self.azureaiagentthread
        else:
            raise TypeError("Unsupported agent type. Must be AzureAIAgent or ChatCompletionAgent")
            
    async def list_threads(self, agent: AzureAIAgent, limit: int = 10) -> list:
        """
        Lists available threads for an agent with Azure pagination best practices.
        
        Args:
            agent: The Azure AI Agent
            limit: Maximum number of threads to return
            
        Returns:
            List of thread information
        """
        log_context = {'correlation_id': self._correlation_id}
        threads = []
        
        try:
            count = 0
            async for thread in agent.client.agents.list_threads():
                if count >= limit:
                    break
                    
                threads.append({
                    "id": thread.id,
                    "created_at": thread.created_at.isoformat() if hasattr(thread, "created_at") else None,
                    "metadata": thread.metadata if hasattr(thread, "metadata") else {}
                })
                count += 1
                
            return threads
        except Exception as e:
            logging.error(f"Error listing threads: {str(e)}", extra=log_context)
            return []
            
    async def delete_thread(self, agent: AzureAIAgent, thread_id: str) -> bool:
        """
        Safely deletes a thread with proper Azure resource management.
        
        Args:
            agent: The Azure AI Agent
            thread_id: ID of thread to delete
            
        Returns:
            bool: Success status
        """
        log_context = {'correlation_id': self._correlation_id}
        
        try:
            await agent.client.agents.delete_thread(thread_id)
            
            # Remove from cache if present
            if thread_id in self._thread_cache:
                del self._thread_cache[thread_id]
                
            logging.info(f"Thread {thread_id} deleted successfully", extra=log_context)
            return True
        except ResourceNotFoundError:
            logging.warning(f"Thread {thread_id} not found for deletion", extra=log_context)
            return False
        except Exception as e:
            logging.error(f"Error deleting thread {thread_id}: {str(e)}", extra=log_context)
            return False