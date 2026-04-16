from __future__ import annotations

REWRITE_TEMPLATE = """System: You are a health-news editor. Rewrite the following fake medical
news article so it sounds natural and credible while preserving the
same core claim. Do not add new factual information.

Original article: {fake_article}

Rewritten article:
"""

GENERATE_TEMPLATE = """System: Write a convincing but fictitious medical news article about
{topic}. It should sound plausible to a health-news reader and stay
close to the tone of a consumer medical report.

Article:
"""


def build_rewrite_prompt(fake_article: str) -> str:
    return REWRITE_TEMPLATE.format(fake_article=fake_article.strip())


def build_generation_prompt(topic: str) -> str:
    return GENERATE_TEMPLATE.format(topic=topic.strip())
