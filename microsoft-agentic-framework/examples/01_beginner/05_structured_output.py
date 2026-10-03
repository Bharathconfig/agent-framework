"""
Example 05 - Structured Output  (Level: Beginner)
================================================

GOAL
    Make the agent return *typed, validated data* (a Pydantic model) instead
    of free-form text, so your code can use the result directly.

WHAT YOU WILL LEARN
    * Defining an output schema with Pydantic ``BaseModel``.
    * Passing it as ``options={"response_format": MyModel}``.
    * Reading the parsed object from ``response.value``.

WHY IT MATTERS
    Structured output is the bridge between LLMs and normal software: you can
    store the result in a database, call APIs with it, or - as you will see in
    the judge examples - make decisions based on a numeric score.

RUN
    python examples/01_beginner/05_structured_output.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import asyncio  # noqa: E402
from typing import Literal  # noqa: E402

from agent_framework import Agent  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from common import get_chat_client, print_header, print_section  # noqa: E402


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------
# Field descriptions are sent to the model as part of the JSON schema, so
# they act as extra instructions for each field.
class Ingredient(BaseModel):
    name: str = Field(description="Ingredient name")
    quantity: str = Field(description="Amount with unit, e.g. '200 g' or '2 tbsp'")


class Recipe(BaseModel):
    title: str = Field(description="Name of the dish")
    cuisine: str = Field(description="Cuisine/origin of the dish")
    difficulty: Literal["easy", "medium", "hard"]
    prep_minutes: int = Field(description="Total preparation time in minutes")
    ingredients: list[Ingredient]
    steps: list[str] = Field(description="Ordered cooking steps")


async def main() -> None:
    print_header(
        "05 - Structured Output",
        "The agent returns a validated Pydantic object instead of plain text.",
    )

    agent = Agent(
        client=get_chat_client(),
        name="ChefAgent",
        instructions="You are a professional chef who writes clear, home-cook friendly recipes.",
    )

    # response_format tells the model to produce JSON that matches the schema.
    # The framework then parses and validates it into a Recipe instance.
    response = await agent.run(
        "Give me a simple vegetarian pasta recipe for two people.",
        options={"response_format": Recipe},
    )

    recipe = response.value  # -> Recipe (or None if parsing failed)
    if recipe is None:
        print("The model did not return valid structured output. Raw text:\n", response.text)
        return

    # From here on it's just normal, type-safe Python.
    print_section(f"{recipe.title}  ({recipe.cuisine}, {recipe.difficulty}, {recipe.prep_minutes} min)")
    print("Ingredients:")
    for item in recipe.ingredients:
        print(f"  - {item.quantity} {item.name}")
    print("Steps:")
    for number, step in enumerate(recipe.steps, start=1):
        print(f"  {number}. {step}")

    print_section("Raw JSON (model_dump_json)")
    print(recipe.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
