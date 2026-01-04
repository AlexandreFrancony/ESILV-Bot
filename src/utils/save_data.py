import json
import os
from datetime import datetime

CONTACTS_FILE = "data/contacts.json"

def save_contact(data: dict):
    """
    Sauvegarde un nouveau contact dans le fichier JSON.
    
    Args:
        data: Dictionnaire avec first_name, last_name, email, etc.
    """
    data["timestamp"] = datetime.now().isoformat()
    
    os.makedirs("data", exist_ok=True)
    
    if os.path.exists(CONTACTS_FILE):
        with open(CONTACTS_FILE, "r", encoding="utf-8") as f:
            contacts = json.load(f)
    else:
        contacts = []
    
    contacts.append(data)
    
    with open(CONTACTS_FILE, "w", encoding="utf-8") as f:
        json.dump(contacts, f, indent=2, ensure_ascii=False)
    
    print(f"Contact sauvegardé: {data['email']}")

def get_all_contacts():
    """Récupère tous les contacts"""
    if os.path.exists(CONTACTS_FILE):
        with open(CONTACTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []