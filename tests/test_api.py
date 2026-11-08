from fastapi.testclient import TestClient
from api.server import app
import pytest
from . import conftest
import requests

client=TestClient(app=app)



    
    
