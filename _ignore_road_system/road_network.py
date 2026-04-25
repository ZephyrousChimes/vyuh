from typing import Dict
from pydantic import BaseModel


class Node(BaseModel):
    """
    TODO: Add Geographic details, if necessary
    """
    id: str


class Road(BaseModel):
    id: str
    src: Node
    dst: Node



class RoadNetwork(BaseModel):
    pass

