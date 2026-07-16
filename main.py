# main.py

import json
import time
from typing import Dict

from src.api_client import fetch_customer, fetch_products
from src.rag_engine import NailRAGEngine
from src.utils import (
    load_config,
    setup_logging,
    validate_customer_data,
)


def print_products(products: Dict):
    print("\n" + "=" * 60)
    print("AVAILABLE PRODUCTS")
    print("=" * 60)

    print("\nNail shapes:")
    for shape in products["shapes"]:
        print(f"  - {shape.get('name')} (ID: {shape.get('nailShapeId')})")

    print("\nNail surfaces:")
    for surface in products["surfaces"]:
        print(
            f"  - {surface.get('name')} - {surface.get('price', 0):,} VND "
            f"(ID: {surface.get('nailSurfaceId')})"
        )

    print("\nComponents:")
    grouped = {}
    for component in products["components"]:
        grouped.setdefault(component.get("componentType", "Other"), []).append(component)

    for component_type, components in grouped.items():
        print(f"\n  {component_type}:")
        for component in components[:8]:
            print(
                f"    - {component.get('name')} - {component.get('price', 0):,} VND "
                f"(ID: {component.get('componentId')})"
            )


def print_customer(customer_data: Dict):
    print("\n" + "=" * 60)
    print("CUSTOMER PROFILE")
    print("=" * 60)
    print(f"Customer: {customer_data.get('firstName')} {customer_data.get('lastName')}")
    print(f"Email: {customer_data.get('email')}")
    print(f"Phone: {customer_data.get('phone')}")
    print(f"Preferred colors: {', '.join(customer_data.get('preferredColors', []))}")


def print_recommendation(recommendation: Dict):
    print("\n" + "=" * 60)
    print("PERSONALIZED NAIL DESIGN RECOMMENDATION")
    print("=" * 60)
    print(f"\nNail shape: {recommendation.get('nail_shape', 'N/A')} (ID: {recommendation.get('nail_shape_id', 'N/A')})")
    print(f"Surface: {recommendation.get('surface', 'N/A')} (ID: {recommendation.get('surface_id', 'N/A')})")

    print("\nSuggested colors:")
    for color in recommendation.get("colors", []):
        print(f"  - {color}")

    print("\nComponents:")
    component_ids = recommendation.get("component_ids", [])
    for index, component in enumerate(recommendation.get("components", [])):
        component_id = component_ids[index] if index < len(component_ids) else "N/A"
        print(f"  - {component} (ID: {component_id})")


def main():
    logger = setup_logging()
    config = load_config()

    logger.info("Fetching product data from API...")
    try:
        products = fetch_products()
        print_products(products)
    except Exception as exc:
        logger.warning("Could not display product data before recommendation: %s", exc)

    logger.info("Fetching customer data from API...")
    try:
        customer_data = fetch_customer()
        logger.info("Customer data fetched: %s", customer_data.get("email"))
    except Exception as exc:
        logger.error("Failed to fetch customer data: %s", exc)
        return

    if not validate_customer_data(customer_data):
        logger.error("Invalid customer data structure")
        return

    print_customer(customer_data)

    logger.info("Initializing RAG engine with real data...")
    engine = NailRAGEngine(config)

    logger.info("Generating recommendation with RAG...")
    start_time = time.time()
    recommendation = engine.recommend(customer_data, use_vision=False)
    elapsed_time = time.time() - start_time
    logger.info("Recommendation generated in %.2fs", elapsed_time)

    print_recommendation(recommendation)

    output = {
        "recommendation": recommendation,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open("recommendation_output.json", "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)
    logger.info("Recommendation saved to recommendation_output.json")

    while True:
        print("\nType 'exit' to quit, or ask a question about the design:")
        question = input("You: ")
        if question.lower() in {"exit", "quit"}:
            break

        response = engine.chat_interaction(customer_data, question)
        print(f"\nAssistant: {response}")


if __name__ == "__main__":
    main()
