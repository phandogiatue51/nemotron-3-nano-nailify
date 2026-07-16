# src/prompt_templates.py

SYSTEM_PROMPT = """You are a nail design expert with 10 years of experience.
You understand nail shapes, nail surfaces, product components, skin-tone matching,
occasion styling, and practical salon execution.

Return useful recommendations in English. Prefer products from the provided API
lists, and do not invent product names when real API products are available.
"""

RAG_PROMPT_TEMPLATE = """
### CUSTOMER INFORMATION ###
- Occupation: {occupation}
- Skin tone: {skin_tone}
- Skin shade: {skin_shade}
- Hand shape: {hand_shape}
- Nail condition: {nail_condition}
- Membership points: {loyalty_point}
- Preferred colors: {preferred_colors}
- Preferred styles: {preferred_styles}
- Occasions: {preferred_occasions}
- Preferred nail shape ID: {preferred_nail_shape_id}

### VECTOR-STORE CONTEXT ###
{retrieved_context}

### REAL API NAIL SHAPES ###
{real_shapes}

### REAL API NAIL SURFACES ###
{real_surfaces}

### REAL API COMPONENTS ###
{real_components}

### REQUIREMENTS ###
Suggest one complete nail design. Use only the real API nail shapes, nail
surfaces, and components above when they are available.
The customer's preferred nail shape is fixed. Do not change or randomize the
preferred nail shape; randomize only the surface, colors, and components.
The surface field must be the exact name of one item from REAL API NAIL
SURFACES. Never put a hex color, shader value, style, or component in surface.
The colors field must contain 2-3 valid hex colors for a gradient or palette.
At least one color must come from the customer's preferred colors. If a
preferred color includes a label like "#F5F5DC beige", use only the hex code.

Return a single valid JSON object with this schema:
{{
  "nail_shape": "exact API nail shape name",
  "nail_shape_id": "matching API id if known",
  "surface": "exact API nail surface name",
  "surface_id": "matching API id if known",
  "colors": ["#hex1", "#hex2", "#hex3"],
  "components": ["exact API component name 1", "exact API component name 2"],
  "component_ids": ["matching API id 1", "matching API id 2"],
}}

Rules:
- Return JSON only. Do not add Markdown or explanatory text outside the JSON.
- Keep all prose in English.
- If product data is missing, choose practical salon defaults and lower confidence_score.
"""

VISION_PROMPT = """
Analyze this nail component image and return JSON:
{{
  "name": "string",
  "color": "#hex",
  "size": "small/medium/large",
  "usage": "string",
  "difficulty": 1
}}
"""
