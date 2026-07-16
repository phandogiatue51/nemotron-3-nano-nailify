# src/rag_engine.py

import json
import os
import random
import re
from datetime import datetime
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

from .api_client import fetch_products
from .prompt_templates import RAG_PROMPT_TEMPLATE, SYSTEM_PROMPT
from .vector_store import NailVectorStore


class NailRAGEngine:
    def __init__(self, config: Dict):
        load_dotenv()
        self.config = config
        self.provider = os.getenv("LLM_PROVIDER", "ollama").lower()
        self.ollama_url = os.getenv("OLLAMA_GENERATE_URL", "http://localhost:11434/api/generate")
        self.openrouter_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.model = os.getenv("OPENROUTER_MODEL") if self.provider == "openrouter" else None
        self.model = self.model or config.get("ollama", {}).get("model", "deepseek-r1:7b")
        self.temperature = config.get("ollama", {}).get("temperature", 0.3)
        self.top_p = config.get("ollama", {}).get("top_p", 0.9)
        self.max_tokens = config.get("ollama", {}).get("max_tokens", 2000)
        self.vector_store = NailVectorStore(
            config.get("chromadb", {}).get("persist_directory", "./data/vector_db")
        )

        # Lazy-load caches to reduce startup memory usage
        self.components_cache: Optional[List[Dict]] = None
        self.shapes_cache: Optional[List[Dict]] = None
        self.surfaces_cache: Optional[List[Dict]] = None

        print(f"RAG engine initialized with provider: {self.provider}, model: {self.model}")

    def recommend(self, customer_data: Dict, use_vision: bool = False) -> Dict:
        retriever_query = self._build_retriever_query(customer_data)
        retrieved_components = self.vector_store.search(retriever_query, top_k=3)
        components_text = self._format_components_info(retrieved_components)

        prompt = RAG_PROMPT_TEMPLATE.format(
            occupation=customer_data.get("occupation", "N/A"),
            skin_tone=customer_data.get("skinTone", "N/A"),
            skin_shade=customer_data.get("skinShade", "N/A"),
            hand_shape=customer_data.get("handShape", "N/A"),
            nail_condition=customer_data.get("nailCondition", "Healthy"),
            loyalty_point=customer_data.get("loyaltyPoint", 0),
            preferred_colors=", ".join(customer_data.get("preferredColors", ["N/A"])),
            preferred_styles=", ".join(customer_data.get("preferredStyles", ["N/A"])),
            preferred_occasions=", ".join(customer_data.get("preferredOccasions", ["N/A"])),
            preferred_nail_shape_id=customer_data.get("preferredNailShapeId", "N/A"),
            retrieved_context=components_text,
            real_shapes=self._format_real_shapes(),
            real_surfaces=self._format_real_surfaces(),
            real_components=self._format_real_components(),
        )

        try:
            response = self._call_llm(prompt)
            result = self._parse_response(response, customer_data)
            result = self._apply_fixed_nail_shape(result, customer_data)
            result = self._apply_random_variation(result, customer_data)
            result = self._apply_valid_colors(result, customer_data)
            result = self._map_to_real_ids(result)
            return self._format_api_response(result)
        except Exception as exc:
            print(f"Recommendation failed, using fallback: {exc}")
            return self._fallback_with_real_data(customer_data)

    def _load_api_data(self) -> None:
        """Lazy load API data from cache if not already loaded."""
        if self.components_cache is not None:
            return  # Already loaded

        print("Loading product data from Nailify APIs...")
        try:
            products = fetch_products()
        except Exception as exc:
            print(f"Could not load product data: {exc}")
            products = {"components": [], "shapes": [], "surfaces": []}

        self.components_cache = products["components"]
        self.shapes_cache = products["shapes"]
        self.surfaces_cache = products["surfaces"]
        print(
            "Loaded "
            f"{len(self.components_cache)} components, "
            f"{len(self.shapes_cache)} nail shapes, "
            f"{len(self.surfaces_cache)} nail surfaces"
        )

    def _get_components_cache(self) -> List[Dict]:
        """Get components cache, lazy loading if needed."""
        if self.components_cache is None:
            self._load_api_data()
        return self.components_cache or []

    def _get_shapes_cache(self) -> List[Dict]:
        """Get shapes cache, lazy loading if needed."""
        if self.shapes_cache is None:
            self._load_api_data()
        return self.shapes_cache or []

    def _get_surfaces_cache(self) -> List[Dict]:
        """Get surfaces cache, lazy loading if needed."""
        if self.surfaces_cache is None:
            self._load_api_data()
        return self.surfaces_cache or []

    def _build_retriever_query(self, data: Dict) -> str:
        parts = []
        if data.get("preferredStyles"):
            parts.append(f"nail art style: {', '.join(data['preferredStyles'])}")
        if data.get("preferredOccasions"):
            parts.append(f"occasion: {', '.join(data['preferredOccasions'])}")
        if data.get("preferredColors"):
            parts.append(f"colors: {', '.join(data['preferredColors'])}")
        return " ".join(parts) if parts else "nail design components"

    def _format_components_info(self, components: List[Dict]) -> str:
        if not components:
            return "No matching vector-store components found."

        lines = ["Closest vector-store matches:"]
        for index, comp in enumerate(components, 1):
            colors = comp.get("colors", [])
            color_text = ", ".join(colors) if isinstance(colors, list) else str(colors)
            lines.extend(
                [
                    f"{index}. {comp.get('name', 'Unknown')}",
                    f"   Description: {comp.get('description', 'N/A')}",
                    f"   Category: {comp.get('category', 'N/A')}",
                    f"   Colors: {color_text}",
                    f"   Similarity: {comp.get('relevance_score', 0):.2f}",
                ]
            )
        return "\n".join(lines)

    def _format_real_components(self) -> str:
        components_cache = self._get_components_cache()
        if not components_cache:
            return "No API components are available."

        grouped: Dict[str, List[Dict]] = {}
        for component in components_cache:
            grouped.setdefault(component.get("componentType", "Other"), []).append(component)

        lines = ["API components available:"]
        for component_type, components in grouped.items():
            lines.append(f"{component_type}:")
            for component in components[:8]:
                lines.append(
                    f"- {component.get('name')} "
                    f"(id: {component.get('componentId')}, price: {component.get('price', 0)} VND)"
                )
        return "\n".join(lines)

    def _format_real_shapes(self) -> str:
        shapes_cache = self._get_shapes_cache()
        if not shapes_cache:
            return "No API nail shapes are available."
        return "\n".join(
            f"- {shape.get('name')} (id: {shape.get('nailShapeId')})"
            for shape in shapes_cache
        )

    def _format_real_surfaces(self) -> str:
        surfaces_cache = self._get_surfaces_cache()
        if not surfaces_cache:
            return "No API nail surfaces are available."
        return "\n".join(
            f"- {surface.get('name')} "
            f"(id: {surface.get('nailSurfaceId')}, price: {surface.get('price', 0)} VND)"
            for surface in surfaces_cache
        )

    def _call_llm(self, prompt: str) -> Dict:
        if self.provider == "openrouter":
            return self._call_openrouter(prompt)
        return self._call_ollama(prompt)

    def _call_ollama(self, prompt: str) -> Dict:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": False,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "format": "json",
        }
        response = requests.post(self.ollama_url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()

    def _call_openrouter(self, prompt: str) -> Dict:
        if not self.openrouter_api_key or self.openrouter_api_key == "sk-or-v1-replace_me":
            raise RuntimeError("OPENROUTER_API_KEY is not configured")

        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", ""),
            "X-Title": os.getenv("OPENROUTER_APP_TITLE", "Nailify RAG"),
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        response = requests.post(self.openrouter_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"].get("content", "{}")
        return {"response": content}

    def _parse_response(self, response: Dict, customer_data: Dict) -> Dict:
        raw_response = response.get("response", "{}")
        json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        result = json.loads(json_match.group(0) if json_match else raw_response)

        for field in ["nail_shape", "surface", "colors", "components"]:
            if field not in result:
                result[field] = self._fallback_value(field, customer_data)

        result["_metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "user_id": customer_data.get("userId"),
            "confidence": result.get("confidence_score", 0.7),
        }
        return result

    def _map_to_real_ids(self, result: Dict) -> Dict:
        shapes_cache = self._get_shapes_cache()
        surfaces_cache = self._get_surfaces_cache()
        components_cache = self._get_components_cache()
        
        shape = self._find_by_name(shapes_cache, result.get("nail_shape"), "nailShapeId")
        if shape:
            result["nail_shape_id"] = shape.get("nailShapeId")

        surface = self._find_by_name(surfaces_cache, result.get("surface"), "nailSurfaceId")
        if surface:
            result["surface_id"] = surface.get("nailSurfaceId")
            result["surface_params"] = {
                "shaderParam": surface.get("shaderParam"),
                "lightnessOffset": surface.get("lightnessOffset"),
                "saturationOffset": surface.get("saturationOffset"),
                "hueOffset": surface.get("hueOffset"),
            }

        component_ids = []
        for name in result.get("components", []):
            component = self._find_by_name(components_cache, name, "componentId", partial=True)
            if component:
                component_ids.append(component.get("componentId"))
        result["component_ids"] = component_ids
        return result

    def _format_api_response(self, result: Dict) -> Dict:
        shapes_cache = self._get_shapes_cache()
        surfaces_cache = self._get_surfaces_cache()
        components_cache = self._get_components_cache()
        
        nail_shape = self._find_by_id(shapes_cache, result.get("nail_shape_id"), "nailShapeId")
        nail_surface = self._find_by_id(surfaces_cache, result.get("surface_id"), "nailSurfaceId")
        components = [
            component
            for component_id in result.get("component_ids", [])
            if (component := self._find_by_id(components_cache, component_id, "componentId"))
        ]

        component_ids = [component.get("componentId") for component in components]
        nail_shape_id = nail_shape.get("nailShapeId") if nail_shape else result.get("nail_shape_id")
        nail_surface_id = nail_surface.get("nailSurfaceId") if nail_surface else result.get("surface_id")

        return {
            "nailShapeId": nail_shape_id,
            "nailSurfaceId": nail_surface_id,
            "componentIds": component_ids,
            "nailShape": nail_shape or {
                "nailShapeId": nail_shape_id,
                "name": result.get("nail_shape"),
                "imageUrl": None,
            },
            "nailSurface": nail_surface or {
                "nailSurfaceId": nail_surface_id,
                "name": result.get("surface"),
                "shaderParam": None,
                "lightnessOffset": None,
                "saturationOffset": None,
                "hueOffset": None,
                "price": 0,
                "duration": None,
            },
            "components": components,
            "colors": result.get("colors", []),
        }

    def _apply_fixed_nail_shape(self, result: Dict, customer_data: Dict) -> Dict:
        """Keep the customer's preferred nail shape fixed across randomizations."""
        preferred_shape_id = customer_data.get("preferredNailShapeId")

        if preferred_shape_id is not None:
            shapes_cache = self._get_shapes_cache()
            for shape in shapes_cache:
                if shape.get("nailShapeId") == preferred_shape_id:
                    result["nail_shape"] = shape.get("name")
                    result["nail_shape_id"] = preferred_shape_id
                    return result

        return result

    def _apply_valid_surface(self, result: Dict, customer_data: Dict) -> Dict:
        """Ensure surface is one of the API nail surfaces, never a color/code."""
        surfaces_cache = self._get_surfaces_cache()
        if not surfaces_cache:
            return result

        surface_name = result.get("surface")
        if self._find_by_name(surfaces_cache, surface_name, "nailSurfaceId"):
            return result

        preferred_styles = {
            str(style).lower()
            for style in customer_data.get("preferredStyles", [])
        }
        selected_surface = None
        for surface in surfaces_cache:
            surface_name_lower = str(surface.get("name", "")).lower()
            if surface_name_lower in preferred_styles:
                selected_surface = surface
                break

        if not selected_surface:
            selected_surface = surfaces_cache[0]

        result["surface"] = selected_surface.get("name")
        result["surface_id"] = selected_surface.get("nailSurfaceId")
        return result

    def _apply_random_variation(self, result: Dict, customer_data: Dict) -> Dict:
        """Re-roll valid design choices while keeping the fixed nail shape."""
        rng = random.SystemRandom()
        surface = self._choose_random_surface(customer_data, rng)
        if surface:
            result["surface"] = surface.get("name")
            result["surface_id"] = surface.get("nailSurfaceId")
        else:
            result = self._apply_valid_surface(result, customer_data)

        components = self._choose_random_components(customer_data, rng)
        result["components"] = [component.get("name") for component in components]
        result["component_ids"] = [component.get("componentId") for component in components]

        colors = self._build_random_colors(customer_data, rng)
        if colors:
            result["colors"] = colors

        return result

    def _build_random_colors(self, customer_data: Dict, rng: random.SystemRandom) -> List[str]:
        preferred_colors = self._extract_hex_colors(customer_data.get("preferredColors", []))
        if not preferred_colors:
            fallback_palettes = [
                ["#F5F5DC", "#D4A5A5"],
                ["#FFFFFF", "#F8C8DC"],
                ["#E8D5C4", "#B76E79"],
            ]
            return rng.choice(fallback_palettes)

        selected = [rng.choice(preferred_colors)]
        target_count = rng.choice([2, 2, 3])
        skin_tone = str(customer_data.get("skinTone", "")).lower()

        warm_accents = ["#FFFFFF", "#FFF8E7", "#F7E7CE", "#EAD7C3", "#B76E79", "#D4AF37"]
        cool_accents = ["#FFFFFF", "#F8F7FF", "#D8BFD8", "#C8A2C8", "#C0C0C0", "#E5E7EB"]
        neutral_accents = ["#FFFFFF", "#F5F5F5", "#EDE7E3", "#D9D9D9", "#C8B6A6"]
        if "warm" in skin_tone:
            accent_pool = warm_accents + neutral_accents
        elif "cool" in skin_tone:
            accent_pool = cool_accents + neutral_accents
        else:
            accent_pool = warm_accents + cool_accents + neutral_accents

        # Mix preferred colors with compatible accents so repeated retries vary.
        candidate_pool = [color for color in preferred_colors if color not in selected] + accent_pool
        rng.shuffle(candidate_pool)
        for color in candidate_pool:
            normalized = color.upper()
            if len(selected) >= target_count:
                break
            if normalized not in selected:
                selected.append(normalized)

        rng.shuffle(accent_pool)
        for color in accent_pool:
            if len(selected) >= target_count:
                break
            normalized = color.upper()
            if normalized not in selected:
                selected.append(normalized)

        rng.shuffle(selected)
        return selected[:target_count]

    def _choose_random_surface(self, customer_data: Dict, rng: random.SystemRandom) -> Optional[Dict]:
        surfaces_cache = self._get_surfaces_cache()
        if not surfaces_cache:
            return None

        styles = {str(style).lower() for style in customer_data.get("preferredStyles", [])}
        occasions = {str(occasion).lower() for occasion in customer_data.get("preferredOccasions", [])}
        occupation = str(customer_data.get("occupation", "")).lower()

        weighted_surfaces = []
        for surface in surfaces_cache:
            name = str(surface.get("name", "")).lower()
            weight = 3

            if name in styles:
                weight += 5
            if "minimalist" in styles or "solid" in styles or "office" in occupation:
                if name in {"matte", "glossy"}:
                    weight += 4
                if name in {"holographic", "cat eyes", "chrome"}:
                    weight -= 1
            if "party" in occasions:
                if name in {"chrome", "holographic", "cat eyes"}:
                    weight += 3
            if "daily" in occasions or "commute" in occasions:
                if name in {"matte", "glossy"}:
                    weight += 3

            weighted_surfaces.extend([surface] * max(1, weight))

        return rng.choice(weighted_surfaces)

    def _choose_random_components(self, customer_data: Dict, rng: random.SystemRandom) -> List[Dict]:
        components_cache = self._get_components_cache()
        if not components_cache:
            return []

        styles = {str(style).lower() for style in customer_data.get("preferredStyles", [])}
        occasions = {str(occasion).lower() for occasion in customer_data.get("preferredOccasions", [])}
        complexity = str(customer_data.get("preferredComplexity", "")).lower()
        occupation = str(customer_data.get("occupation", "")).lower()

        if complexity == "simple" or "minimalist" in styles or "solid" in styles or "office" in occupation:
            count = rng.choice([0, 1])
        elif "party" in occasions or "pattern" in styles or "geometric" in styles:
            count = rng.choice([1, 2, 3])
        else:
            count = rng.choice([0, 1, 2])

        if count == 0:
            return []

        components = list(components_cache)
        rng.shuffle(components)
        return components[: min(count, len(components))]

    def _apply_valid_colors(self, result: Dict, customer_data: Dict) -> Dict:
        """Keep 2-3 valid hex colors and require at least one preferred color."""
        preferred_colors = self._extract_hex_colors(customer_data.get("preferredColors", []))
        result_colors = self._extract_hex_colors(result.get("colors", []))

        if preferred_colors and not any(color in preferred_colors for color in result_colors):
            result_colors.insert(0, random.SystemRandom().choice(preferred_colors))

        if len(result_colors) < 2:
            for color in preferred_colors:
                if len(result_colors) >= 2:
                    break
                if color not in result_colors:
                    result_colors.append(color)

        if not result_colors:
            result_colors = ["#FFFFFF", "#F5F5DC"]

        result["colors"] = result_colors[:3] if len(result_colors) >= 2 else (result_colors + ["#FFFFFF"])[:2]
        return result

    def _extract_hex_colors(self, colors) -> List[str]:
        if not isinstance(colors, list):
            colors = [colors]

        extracted = []
        for color in colors:
            for match in re.findall(r"#[0-9a-fA-F]{6}", str(color)):
                normalized = match.upper()
                if normalized not in extracted:
                    extracted.append(normalized)
        return extracted

    def _find_by_name(
        self, items: List[Dict], name: Optional[str], id_key: str, partial: bool = False
    ) -> Optional[Dict]:
        if not name:
            return None
        name_lower = str(name).lower()
        for item in items:
            item_name = str(item.get("name", "")).lower()
            if item_name == name_lower or (partial and item_name in name_lower):
                return item
        return None

    def _find_by_id(self, items: List[Dict], item_id, id_key: str) -> Optional[Dict]:
        if item_id is None:
            return None
        for item in items:
            if item.get(id_key) == item_id:
                return item
        return None

    def _fallback_with_real_data(self, customer_data: Dict) -> Dict:
        shapes_cache = self._get_shapes_cache()
        surfaces_cache = self._get_surfaces_cache()
        components_cache = self._get_components_cache()
        
        preferred_shape = self._preferred_shape(customer_data)
        shape = preferred_shape or (shapes_cache[0] if shapes_cache else {"name": "Ballerina", "nailShapeId": None})
        surface = surfaces_cache[0] if surfaces_cache else {"name": "Glossy", "nailSurfaceId": None}
        components = components_cache[:3]

        result = {
            "nail_shape": shape.get("name"),
            "nail_shape_id": shape.get("nailShapeId"),
            "surface": surface.get("name"),
            "surface_id": surface.get("nailSurfaceId"),
            "colors": self._extract_hex_colors(customer_data.get("preferredColors", []))[:3] or ["#FF69B4", "#FFFFFF"],
            "components": [component.get("name") for component in components],
            "component_ids": [component.get("componentId") for component in components],
        }
        return self._format_api_response(result)

    def _preferred_shape(self, customer_data: Dict) -> Optional[Dict]:
        preferred_shape_id = customer_data.get("preferredNailShapeId")

        if preferred_shape_id is not None:
            shapes_cache = self._get_shapes_cache()
            for shape in shapes_cache:
                if shape.get("nailShapeId") == preferred_shape_id:
                    return shape

        return None

    def _fallback_value(self, field: str, customer_data: Dict):
        fallbacks = {
            "surface": "Glossy",
            "colors": customer_data.get("preferredColors", ["#FF69B4", "#FFFFFF", "#C0C0C0"]),
            "components": ["Base Coat", "Top Coat"]
        }
        return fallbacks.get(field, "N/A")

    def chat_interaction(self, customer_data: Dict, user_query: str) -> str:
        context = self.recommend(customer_data)
        chat_prompt = f"""
Based on this customer profile and recommendation:

{json.dumps(context, indent=2, ensure_ascii=False)}

Customer question: {user_query}

Answer in English. Be practical, concise, and specific.
"""
        response = self._call_llm(chat_prompt)
        return response.get("response", "Sorry, I could not answer that question.")
