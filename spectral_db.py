# spectraldb.py
"""
SpectralDB: The autonomous memory core for AI collaboration systems.
A persistent knowledge engine that ingests, relates, and evolves information.
Divine temporal knowledge evolution through harmonic development.
"""

import sqlite3
import json
import uuid
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import logging
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DivinePantheon:
    """The 7 Gods governing the cognitive spectrum"""
    
    GODS = [
        # Band 1: Kairos (Infrared) - Primordial chaos, opportune emergence
        {"name": "Kairos", "band": "Atoms", "color": "Infrared", "freq_low": 32, "freq_high": 64,
         "role": "Primordial chaos, opportune emergence", "cognitive_level": "C1-C2",
         "description": "Source of all atomic primitives", "word_range": (1, 5)},
        
        # Band 2: Hermes (Orange) - Messenger, circulation, flow
        {"name": "Hermes", "band": "Molecules", "color": "Orange", "freq_low": 64, "freq_high": 128,
         "role": "Messenger, circulation, flow, linking domains", "cognitive_level": "C2-C3", 
         "description": "Bridges and distributes memory", "word_range": (6, 25)},
        
        # Band 3: Apollo (Yellow) - Light, pattern, clarity, symbolic insight
        {"name": "Apollo", "band": "Concepts", "color": "Yellow", "freq_low": 128, "freq_high": 256,
         "role": "Light, pattern, clarity, symbolic insight", "cognitive_level": "C3-C4",
         "description": "Synthesizes patterns and musicality", "word_range": (26, 500)},
        
        # Band 4: Demeter (Green) - Life, growth, nurture, natural cycles
        {"name": "Demeter", "band": "Thoughts", "color": "Green", "freq_low": 256, "freq_high": 512,
         "role": "Life, growth, nurture, natural cycles", "cognitive_level": "C4-C5",
         "description": "Expands and organizes growth cycles", "word_range": (501, 1600)},
        
        # Band 5: Poseidon (Blue) - Depth, fluidity, change, tides
        {"name": "Poseidon", "band": "Ideas", "color": "Blue", "freq_low": 512, "freq_high": 1024,
         "role": "Depth, fluidity, change, tides", "cognitive_level": "C5-C6",
         "description": "Dynamic transformation and adaptation", "word_range": (1601, 3200)},
        
        # Band 6: Artemis (Purple) - Threshold, reflection, liminality
        {"name": "Artemis", "band": "Essays", "color": "Purple", "freq_low": 1024, "freq_high": 2048,
         "role": "Threshold, reflection, liminality, silent synthesis", "cognitive_level": "C6-C7",
         "description": "Reflective, liminal synthesis", "word_range": (3201, 6400)},
        
        # Band 7: Athena (Violet) - Wisdom, strategy, reflective synthesis
        {"name": "Athena", "band": "Documents", "color": "Violet", "freq_low": 2048, "freq_high": 4096,
         "role": "Wisdom, strategy, reflective synthesis", "cognitive_level": "C7+",
         "description": "Curation and complex concept bridging", "word_range": (6401, float('inf'))}
    ]

    @classmethod
    def get_god_by_band(cls, band_name: str) -> Optional[Dict]:
        """Get god properties by band name"""
        for god in cls.GODS:
            if god['band'].lower() == band_name.lower():
                return god.copy()
        return None

    @classmethod
    def get_god_by_frequency(cls, frequency: float) -> Optional[Dict]:
        """Determine which god governs a given frequency"""
        for god in cls.GODS:
            if god['freq_low'] <= frequency <= god['freq_high']:
                return god.copy()
        return None

    @classmethod
    def get_god_by_word_count(cls, word_count: int) -> Optional[Dict]:
        """Get governing god by word count"""
        for god in cls.GODS:
            min_words, max_words = god['word_range']
            if min_words <= word_count <= max_words:
                return god.copy()
        return None

    @classmethod
    def get_all_gods(cls) -> List[Dict]:
        """Get all gods in the pantheon"""
        return [god.copy() for god in cls.GODS]

class SpectralBandIndex:
    """Spectral 7-band classification system with divine governance"""
    
    @classmethod
    def get_band(cls, word_count: int) -> Dict[str, Any]:
        """Returns band properties for given word count with divine governance"""
        god = DivinePantheon.get_god_by_word_count(word_count)
        if god:
            return {
                "name": god['band'],
                "freq_low": god['freq_low'],
                "freq_high": god['freq_high'],
                "color": god['color'],
                "cognitive_level": god['cognitive_level'],
                "description": god['description'],
                "governing_god": god['name'],
                "role": god['role'],
                "word_range": god['word_range']
            }
        return {"name": "Unknown", "freq_low": 0, "freq_high": 0, "color": "Black", 
                "cognitive_level": "C0", "description": "Unclassified", "governing_god": "Unknown"}

    @classmethod
    def get_frequency(cls, word_count: int) -> float:
        """Returns precise frequency based on word count within band"""
        band = cls.get_band(word_count)
        if band['name'] == "Unknown":
            return 0
        
        # Calculate proportional frequency within band
        band_range = band['freq_high'] - band['freq_low']
        word_range = band['word_range'][1] - band['word_range'][0]
        
        if word_range == 0:  # Avoid division by zero for infinite max
            return band['freq_low']
        
        position = (word_count - band['word_range'][0]) / word_range
        return band['freq_low'] + (position * band_range)

    @classmethod
    def get_musical_note(cls, frequency: float) -> str:
        """Convert frequency to musical note approximation with bounds checking"""
        if frequency <= 0:
            return "C0"
        
        try:
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            note_number = 12 * math.log2(frequency / 440.0) + 49
            octave = int(note_number // 12)
            
            # Get the fractional part and convert to note index (0-11)
            note_index = int(round(note_number - (octave * 12)))
            
            # Handle edge case where note_index might be 12 (should be 0 of next octave)
            if note_index == 12:
                note_index = 0
                octave += 1
            
            # Ensure note_index is within valid range
            note_index = max(0, min(11, note_index))
        
            return f"{note_names[note_index]}{octave}"
        except (ValueError, ZeroDivisionError):
            return "C0"

    @classmethod
    def get_band_by_name(cls, band_name: str) -> Optional[Dict[str, Any]]:
        """Get band properties by band name"""
        god = DivinePantheon.get_god_by_band(band_name)
        if god:
            return {
                "name": god['band'],
                "freq_low": god['freq_low'],
                "freq_high": god['freq_high'],
                "color": god['color'],
                "cognitive_level": god['cognitive_level'],
                "description": god['description'],
                "governing_god": god['name'],
                "role": god['role'],
                "word_range": god['word_range']
            }
        return None

class SpectralDB:
    """SpectralDB - Autonomous memory core for AI collaboration systems"""
    
    def __init__(self, db_path: str = "spectral_memory.db"):
        self.db_path = Path(db_path)
        self.init_database()
    
    def init_database(self):
        """Initialize SpectralDB with divine architecture"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create main memories table with divine governance
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    word_count INTEGER,
                    band_name TEXT,
                    frequency REAL,
                    musical_note TEXT,
                    color TEXT,
                    cognitive_level TEXT,
                    description TEXT,
                    content_hash TEXT UNIQUE,
                    source TEXT,
                    tags TEXT,
                    parent_ids TEXT,
                    synthesis_method TEXT,
                    harmonic_interval TEXT,
                    governing_god TEXT,
                    divine_role TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create relationships table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memory_relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parent_id TEXT,
                    child_id TEXT,
                    relationship_type TEXT,
                    interval TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (parent_id) REFERENCES memories (id),
                    FOREIGN KEY (child_id) REFERENCES memories (id)
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_governing_god ON memories (governing_god)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_band_name ON memories (band_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_frequency ON memories (frequency)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_cognitive_level ON memories (cognitive_level)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_musical_note ON memories (musical_note)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_content_hash ON memories (content_hash)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON memories (created_at)')
            
            conn.commit()
    
    def calculate_word_count(self, text: str) -> int:
        """Calculate precise word count"""
        words = re.findall(r'\b\w+\b', text.lower())
        return len(words)
    
    def create_content_hash(self, text: str) -> str:
        """Create hash for duplicate detection"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
    
    def ingest_memory(self, content: str, source: str = "", tags: List[str] = None, 
                     description: str = "", parent_ids: List[str] = None, 
                     synthesis_method: str = "", harmonic_interval: str = "") -> str:
        """
        Ingest memory with divine band classification
        Returns the ID of the ingested memory (or existing memory if duplicate)
        """
        content = content.strip()
        if not content:
            raise ValueError("Content cannot be empty")
        
        word_count = self.calculate_word_count(content)
        band_info = SpectralBandIndex.get_band(word_count)
        frequency = SpectralBandIndex.get_frequency(word_count)
        musical_note = SpectralBandIndex.get_musical_note(frequency)
        content_hash = self.create_content_hash(content)
        memory_id = str(uuid.uuid4())
        
        tags_json = json.dumps(tags or [])
        parent_ids_json = json.dumps(parent_ids or [])
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    INSERT INTO memories (
                        id, content, word_count, band_name, frequency, musical_note,
                        color, cognitive_level, description, content_hash, source, 
                        tags, parent_ids, synthesis_method, harmonic_interval,
                        governing_god, divine_role
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    memory_id, content, word_count, band_info['name'], frequency,
                    musical_note, band_info['color'], band_info['cognitive_level'],
                    description or band_info['description'], content_hash, source,
                    tags_json, parent_ids_json, synthesis_method, harmonic_interval,
                    band_info['governing_god'], band_info['role']
                ))
                
                # Add relationships if parents exist
                if parent_ids:
                    for parent_id in parent_ids:
                        cursor.execute('''
                            INSERT INTO memory_relationships (parent_id, child_id, relationship_type, interval)
                            VALUES (?, ?, ?, ?)
                        ''', (parent_id, memory_id, 'synthesis', harmonic_interval or ''))
                
                conn.commit()
                logger.info(f"Ingested {band_info['governing_god']} memory: {content[:50]}...")
                return memory_id
                
            except sqlite3.IntegrityError:
                # Duplicate content detected, return existing ID
                cursor.execute('SELECT id FROM memories WHERE content_hash = ?', (content_hash,))
                result = cursor.fetchone()
                return result[0] if result else memory_id

    def get_memory(self, memory_id: str) -> Optional[Dict]:
        """Get memory by ID"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM memories WHERE id = ?', (memory_id,))
            result = cursor.fetchone()
            return dict(result) if result else None

    def search_memories(self, search_term: str, limit: int = 10) -> List[Dict]:
        """Search memory content"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM memories 
                WHERE content LIKE ? 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (f'%{search_term}%', limit))
            
            return [dict(row) for row in cursor.fetchall()]

    def get_memories_by_band(self, band_name: str, limit: int = None) -> List[Dict]:
        """Get memories by band name"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = 'SELECT * FROM memories WHERE band_name = ? ORDER BY created_at DESC'
            params = [band_name]
            
            if limit:
                query += ' LIMIT ?'
                params.append(limit)
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_memories_by_god(self, god_name: str, limit: int = None) -> List[Dict]:
        """Get memories by governing god"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = 'SELECT * FROM memories WHERE governing_god = ? ORDER BY created_at DESC'
            params = [god_name]
            
            if limit:
                query += ' LIMIT ?'
                params.append(limit)
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_recent_memories(self, limit: int = 10) -> List[Dict]:
        """Get most recent memories"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM memories 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (limit,))
            
            return [dict(row) for row in cursor.fetchall()]

    def get_memory_lineage(self, memory_id: str) -> Dict:
        """Get full lineage (parents and children) of a memory"""
        memory = self.get_memory(memory_id)
        if not memory:
            return {}
        
        # Get parents
        parents = []
        if memory.get('parent_ids'):
            parent_ids = json.loads(memory['parent_ids'])
            for parent_id in parent_ids:
                parent = self.get_memory(parent_id)
                if parent:
                    parents.append(parent)
        
        # Get children
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT m.* FROM memories m
                JOIN memory_relationships r ON m.id = r.child_id
                WHERE r.parent_id = ? AND r.relationship_type = 'synthesis'
            ''', (memory_id,))
            
            children = [dict(row) for row in cursor.fetchall()]
        
        return {
            'memory': memory,
            'parents': parents,
            'children': children
        }

    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            stats = {}
            cursor.execute('SELECT COUNT(*) FROM memories')
            stats['total_memories'] = cursor.fetchone()[0]
            
            cursor.execute('SELECT band_name, COUNT(*) FROM memories GROUP BY band_name')
            stats['band_distribution'] = dict(cursor.fetchall())
            
            cursor.execute('SELECT governing_god, COUNT(*) FROM memories GROUP BY governing_god')
            stats['god_distribution'] = dict(cursor.fetchall())
            
            return stats

# Example usage
if __name__ == "__main__":
    # Simple test
    db = SpectralDB()
    
    # Ingest some memories
    test_memories = [
        "Hello world",
        "The quick brown fox jumps over the lazy dog",
        "Artificial intelligence is the simulation of human intelligence processes by machines, especially computer systems.",
        "The concept of the divine pantheon governs the spectral bands of knowledge, from atomic primitives to strategic wisdom."
    ]
    
    for memory in test_memories:
        memory_id = db.ingest_memory(memory, source="test")
        print(f"Ingested: {memory[:30]}... (ID: {memory_id})")
    
    # Show stats
    stats = db.get_database_stats()
    print(f"\nDatabase stats: {stats}")
