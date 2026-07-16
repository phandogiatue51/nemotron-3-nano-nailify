# src/vector_store.py

from typing import Dict, List

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from .api_client import fetch_products


class NailVectorStore:
    def __init__(self, persist_directory: str = "./data/vector_db"):
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name="nail_components",
            metadata={"hnsw:space": "cosine"},
        )
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self._initialize_data()

    def _initialize_data(self):
        """Create sample data for RAG context when the collection is empty."""
        if self.collection.count() != 0:
            return

        sample_components = [
            {
                "id": "comp_001",
                "name": "Nail Gems",
                "description": "Small rhinestones for nail art decoration that create a glamorous look.",
                "category": "nail_art",
                "colors": ["#FFFFFF", "#FFD700", "#C0C0C0"],
                "difficulty": 2,
                "occasion": "Party, Wedding, Event",
            },
            {
                "id": "comp_002",
                "name": "Stamping Plate",
                "description": "A tool for creating intricate patterns and designs on nails.",
                "category": "nail_art",
                "colors": ["#000000", "#FFFFFF", "#FF0000"],
                "difficulty": 3,
                "occasion": "Casual, Professional",
            },
            {
                "id": "comp_003",
                "name": "Matte Top Coat",
                "description": "A finish that reduces shine and creates a matte look.",
                "category": "finish",
                "colors": ["#FFFFFF"],
                "difficulty": 1,
                "occasion": "Everyday",
            },
            {
                "id": "comp_004",
                "name": "Glitter Powder",
                "description": "Adds sparkle and shimmer to nail designs.",
                "category": "nail_art",
                "colors": ["#FFD700", "#FF69B4", "#00CED1"],
                "difficulty": 2,
                "occasion": "Party, Event",
            },
            {
                "id": "comp_005",
                "name": "Nail Decals",
                "description": "Pre-designed nail stickers for quick nail art.",
                "category": "nail_art",
                "colors": ["#FF1493", "#4169E1", "#FFA500"],
                "difficulty": 1,
                "occasion": "Casual, Event",
            },
            {
                "id": "comp_006",
                "name": "Chrome Powder",
                "description": "Creates a metallic mirror-like finish on nails.",
                "category": "finish",
                "colors": ["#C0C0C0", "#FFD700", "#FF0000"],
                "difficulty": 3,
                "occasion": "Event, Wedding",
            },
        ]

        documents = [component["description"] for component in sample_components]
        embeddings = self.embedding_model.encode(documents).tolist()
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=sample_components,
            ids=[component["id"] for component in sample_components],
        )
        print(f"Initialized {len(sample_components)} sample components")

    def add_component(self, component: Dict):
        """Add one component to the vector database."""
        self._safe_add(component["id"], component["description"], component)

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for relevant components."""
        query_embedding = self.embedding_model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["metadatas", "documents", "distances"],
        )

        components = []
        if results["metadatas"]:
            for index, metadata in enumerate(results["metadatas"][0]):
                components.append(
                    {
                        **metadata,
                        "distance": results["distances"][0][index],
                        "relevance_score": 1 - results["distances"][0][index],
                    }
                )
        return components

    def search_by_category(self, category: str, query: str = "", top_k: int = 5):
        """Search components by category."""
        if query:
            return self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where={"category": category},
            )
        return self.collection.get(where={"category": category})

    def sync_with_api(self):
        """Sync product data from Nailify APIs into the vector store."""
        print("Syncing product data from APIs...")
        products = fetch_products()
        components = products["components"]
        shapes = products["shapes"]
        surfaces = products["surfaces"]

        self._add_components_to_db(components)
        self._add_shapes_to_db(shapes)
        self._add_surfaces_to_db(surfaces)
        print(f"Sync completed. Total vector-store items: {self.collection.count()}")

    def _safe_add(self, item_id: str, document: str, metadata: Dict):
        embedding = self.embedding_model.encode([document]).tolist()
        self.collection.upsert(
            embeddings=embedding,
            documents=[document],
            metadatas=[metadata],
            ids=[item_id],
        )

    def _add_components_to_db(self, components: List[Dict]):
        for component in components:
            item_id = f"component_{component.get('componentId')}"
            description = (
                f"Component: {component.get('name')}. "
                f"Type: {component.get('componentType', 'Unknown')}. "
                f"Price: {component.get('price', 0)} VND."
            )
            metadata = {
                "id": item_id,
                "component_id": component.get("componentId"),
                "name": component.get("name"),
                "category": component.get("componentType", "component"),
                "description": description,
                "image_url": component.get("imageUrl"),
                "price": component.get("price", 0),
                "source": "api",
            }
            self._safe_add(item_id, description, metadata)

    def _add_shapes_to_db(self, shapes: List[Dict]):
        for shape in shapes:
            item_id = f"shape_{shape.get('nailShapeId')}"
            description = f"Nail shape: {shape.get('name')}."
            metadata = {
                "id": item_id,
                "nail_shape_id": shape.get("nailShapeId"),
                "name": shape.get("name"),
                "category": "nail_shape",
                "description": description,
                "image_url": shape.get("imageUrl"),
                "source": "api",
            }
            self._safe_add(item_id, description, metadata)

    def _add_surfaces_to_db(self, surfaces: List[Dict]):
        for surface in surfaces:
            item_id = f"surface_{surface.get('nailSurfaceId')}"
            description = (
                f"Nail surface: {surface.get('name')}. "
                f"Price: {surface.get('price', 0)} VND. "
                f"Duration: {surface.get('duration', 0)} minutes."
            )
            metadata = {
                "id": item_id,
                "nail_surface_id": surface.get("nailSurfaceId"),
                "name": surface.get("name"),
                "category": "nail_surface",
                "description": description,
                "shader_params": surface.get("shaderParam"),
                "lightness_offset": surface.get("lightnessOffset", 0),
                "saturation_offset": surface.get("saturationOffset", 0),
                "hue_offset": surface.get("hueOffset", 0),
                "price": surface.get("price", 0),
                "duration": surface.get("duration", 0),
                "source": "api",
            }
            self._safe_add(item_id, description, metadata)
