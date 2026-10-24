"""Small, fixed math-only infrastructure prompts; not a quality benchmark."""

TRAIN = [
    "Calculate 17 * 23. Give a concise explanation.",
    "Solve 3x + 7 = 31. Explain briefly.",
    "A train travels 180 km in 3 hours. What is its average speed?",
    "What is the area of a circle with radius 3? Use pi in your answer.",
    "Simplify 18/24 and explain the common factor.",
    "If a price increases from 80 to 100, what is the percentage increase?",
    "Find the sum of the integers from 1 to 20.",
    "Calculate the probability of two heads in two fair coin tosses.",
    "Explain why the square of an odd integer is odd.",
    "Find the derivative of x squared plus 3x.",
    "Calculate 2 to the power of 10.",
    "Solve x squared minus 5x plus 6 equals zero.",
    "What is the median of 2, 7, 9, 10, 12?",
    "A rectangle has perimeter 30 and width 5. Find its length.",
    "Convert 0.125 to a fraction in simplest form.",
    "If 5 workers finish a task in 12 days, how many days for 10 workers at the same rate?",
]
HELDOUT = ["Calculate 19 * 21. Explain briefly.", "Solve 4x - 3 = 25. Explain briefly."]
