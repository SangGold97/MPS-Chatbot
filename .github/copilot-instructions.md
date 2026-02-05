# Python Project Coding Standards

## Code Style
- Follow PEP 8 strictly; use `black` formatter with 79-char line length
- Use type hints for all function signatures and return types

## Naming Conventions
- Use meaningful variable and function names
- `snake_case` for functions, variables, modules
- `PascalCase` for classes; `UPPER_SNAKE_CASE` for constants

## Best Practices
- Write clean, readable, and well-documented code
- Write docstrings (Google style) for all public functions and classes, be short and concise
- Use try-except blocks for risky operations
- When use terminal command, use `conda activate mps` to activate the conda environment
- Do not use pytest
- Follow PROJECT_OVERVIEW.md for project-specific guidelines

## Code Generation Rules
- Add logging by using loguru library, not `print()` statements
- Keep functions under 20 lines; extract logic into helper functions
- Generate code shortly, optimization, and focus to the requirements of the user
- Generate code as required, without affecting the logic of other functions, methods, or classes.
- All block code (code snippet) in all functions or methods should be separated by a new line and have comments (with #) before each of them.