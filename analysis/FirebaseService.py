# This file contains all files related to firebase
import firebase_admin
from firebase_admin import credentials, db
import requests

# CREDENTIALS_PATH = r"E:\PRIVATE_KEY\worldsg.json"
CREDENTIALS_PATH = "/data/rohith/ag/PRIVATE_KEY/worldsg.json"
DATABASE_URL = "https://worldsg-default-rtdb.firebaseio.com"

# Initialize Firebase Admin SDK with service account
if not firebase_admin._apps:
    cred = credentials.Certificate(CREDENTIALS_PATH)
    firebase_admin.initialize_app(cred, {
        "databaseURL": DATABASE_URL
    })


class FirebaseService:
    """Firebase database service for storing and retrieving annotations."""

    def __init__(self):
        self.db = db.reference()

    def set_data(self, path: str, data: dict):
        """Set data at a specific path."""
        return self.db.child(path).set(data)

    def get_data(self, path: str):
        """Get data from a specific path."""
        return self.db.child(path).get()

    def update_data(self, path: str, data: dict):
        """Update data at a specific path."""
        return self.db.child(path).update(data)

    def delete_data(self, path: str):
        """Delete data at a specific path."""
        return self.db.child(path).delete()

    def push_data(self, path: str, data: dict):
        """Push data to create a new child with auto-generated key."""
        return self.db.child(path).push(data)

    def get_keys(self, path: str):
        """
        Get only the keys (child names) at a path without fetching full data.
        Uses Firebase REST API with shallow=true to avoid payload too large errors.
        """
        # Get the access token from the Admin SDK for authenticated requests
        access_token = firebase_admin.get_app().credential.get_access_token().access_token
        url = f"{DATABASE_URL}/{path}.json?shallow=true"
        headers = {"Authorization": f"Bearer {access_token}"}
        try:
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if data and isinstance(data, dict):
                    return list(data.keys())
                return []
            else:
                print(f"Firebase shallow query failed: {response.status_code}")
                return []
        except Exception as e:
            print(f"Error fetching keys from Firebase: {e}")
            return []



if __name__ == "__main__":
    db_service = FirebaseService()
    print("Firebase service initialized successfully")

    # Test connection
    try:
        test_data = db_service.get_data("worldframe_obb/world")
        print(f"Connection test: {'Data exists' if test_data else 'No data yet'}")
    except Exception as e:
        print(f"Connection error: {e}")
