from typing import List

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    """Represents a single message in a chat conversation."""
    model_config = ConfigDict(extra="forbid")

    role: str = Field(..., description="The role of the message sender (e.g., 'user' or 'assistant')")
    content: str = Field(..., description="The text content of the message")


class ChatRequest(BaseModel):
    """Request payload for the /chat endpoint."""
    model_config = ConfigDict(extra="forbid")

    messages: List[Message] = Field(..., description="A list of messages representing the conversation history")


class Recommendation(BaseModel):
    """Represents a single recommended catalog item."""
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Name or title of the recommended test/assessment")
    url: str = Field(..., description="URL to the assessment (must start with 'https://www.shl.com/')")
    test_type: str = Field(..., description="The type or category of the test")


class ChatResponse(BaseModel):
    """Response payload returned by the /chat endpoint."""
    model_config = ConfigDict(extra="forbid")

    reply: str = Field(..., description="The textual response from the agent")
    recommendations: List[Recommendation] = Field(..., description="A list of matching recommendations")
    end_of_conversation: bool = Field(..., description="Flag indicating if the recommendation flow has concluded")
