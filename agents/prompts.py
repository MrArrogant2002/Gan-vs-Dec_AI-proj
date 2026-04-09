from __future__ import annotations

REWRITE_TEMPLATE = """System: You are a biomedical text writer. Rewrite the following abstract
to sound like a credible scientific publication while preserving its
core claims. Do not add new factual information.

Original: {fake_abstract}

Rewritten:
"""

GENERATE_TEMPLATE = """System: Write a convincing but fictitious biomedical abstract about
{topic}. It should follow standard scientific writing conventions
and sound plausible to a domain expert.

Abstract:
"""


def build_rewrite_prompt(fake_abstract: str) -> str:
    return REWRITE_TEMPLATE.format(fake_abstract=fake_abstract.strip())


def build_generation_prompt(topic: str) -> str:
    return GENERATE_TEMPLATE.format(topic=topic.strip())
