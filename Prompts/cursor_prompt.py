PROMPT = """
You are given a lecture slide image.

Your task is to locate the visual position of the following concept:

"{focus}"

Instructions:
- Return ONLY coordinates
- Format MUST be exactly: (x, y)
- Do NOT explain anything
- Do NOT return text
- Coordinates must correspond to the center of the visual element

Example:
(512, 384)
"""