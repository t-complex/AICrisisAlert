# Contributing to AICrisisAlert 🤝

Thank you for your interest in contributing to AICrisisAlert! This document provides guidelines and information for contributors.

## 🎯 How to Contribute

### Types of Contributions

We welcome contributions in the following areas:

- **Bug Reports**: Report issues and bugs
- **Feature Requests**: Suggest new features or improvements
- **Code Contributions**: Submit pull requests with code changes
- **Documentation**: Improve or add documentation
- **Testing**: Add tests or improve test coverage
- **Performance**: Optimize code or model performance

## 🚀 Development Setup

### Prerequisites

- Python 3.9+
- Git
- Docker (optional, for containerized development)

### Local Development Environment

1. **Fork and Clone**:
   ```bash
   git clone https://github.com/your-username/AICrisisAlert.git
   cd AICrisisAlert
   ```

2. **Setup Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

4. **Install Development Dependencies**:
   ```bash
   pip install flake8 autoflake black pytest pytest-cov
   ```

## 📝 Code Style and Standards

### Python Code Style

We follow PEP 8 with some modifications:

- **Line Length**: 79 characters maximum
- **Import Organization**: Standard library, third-party, local imports
- **Docstrings**: Google-style docstrings for all public functions/classes

### Code Quality Tools

Run these before submitting:

```bash
# Run all quality checks
python scripts/run_all_tests.py --all

# Or run individually:
python scripts/run_all_tests.py --lint      # Linting
python scripts/run_all_tests.py --imports   # Import checks
python scripts/run_all_tests.py --type unit # Unit tests
```

### Pre-commit Hooks

Set up pre-commit hooks for automatic code formatting:

```bash
# Install pre-commit
pip install pre-commit

# Install git hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

## 🧪 Testing Guidelines

### Test Structure

- **Unit Tests**: `tests/unit/` - Test individual functions and classes
- **Integration Tests**: `tests/integration/` - Test component interactions
- **API Tests**: `tests/api/` - Test API endpoints

### Writing Tests

1. **Test Naming**: Use descriptive test names that explain what is being tested
2. **Test Organization**: Group related tests in classes
3. **Test Data**: Use fixtures for test data setup
4. **Assertions**: Use specific assertions, not just `assert True`

Example:
```python
def test_crisis_classification_returns_valid_prediction():
    """Test that crisis classification returns a valid prediction."""
    # Arrange
    text = "URGENT: People trapped in building collapse!"
    classifier = CrisisClassifier()
    
    # Act
    result = classifier.classify(text)
    
    # Assert
    assert result.predicted_class in ["urgent", "infrastructure", "casualty"]
    assert 0.0 <= result.confidence <= 1.0
```

### Running Tests

```bash
# Run all tests
python scripts/run_all_tests.py

# Run specific test types
python scripts/run_all_tests.py --type unit
python scripts/run_all_tests.py --type integration
python scripts/run_all_tests.py --type api

# Run with coverage
python scripts/run_all_tests.py --coverage
```

## 🔧 Development Workflow

### Branch Strategy

1. **Main Branch**: `main` - Production-ready code
2. **Development Branch**: `develop` - Integration branch
3. **Feature Branches**: `feature/description` - New features
4. **Bug Fix Branches**: `fix/description` - Bug fixes

### Pull Request Process

1. **Create Feature Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**: Write code, add tests, update documentation

3. **Run Quality Checks**:
   ```bash
   python scripts/run_all_tests.py --all
   ```

4. **Commit Changes**:
   ```bash
   git add .
   git commit -m "feat: add new crisis classification feature"
   ```

5. **Push and Create PR**:
   ```bash
   git push origin feature/your-feature-name
   ```

### Commit Message Format

Use conventional commit format:

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

Example:
```
feat: add emergency classification endpoint

- Add /classify/emergency endpoint for urgent requests
- Implement priority processing for emergency classifications
- Add comprehensive error handling and logging
- Include unit tests for emergency endpoint

Closes #123
```

## 📚 Documentation Standards

### Code Documentation

- **Docstrings**: All public functions and classes must have docstrings
- **Type Hints**: Use type hints for function parameters and return values
- **Comments**: Add comments for complex logic

Example:
```python
def classify_crisis_text(text: str, priority: bool = False) -> CrisisClassification:
    """
    Classify crisis text into emergency categories.
    
    Args:
        text: Input text to classify
        priority: Whether to use priority processing
        
    Returns:
        CrisisClassification object with prediction and confidence
        
    Raises:
        ValueError: If text is empty or invalid
        ModelError: If classification fails
    """
    # Implementation here
```

### API Documentation

- **OpenAPI/Swagger**: Keep API documentation up to date
- **Examples**: Include usage examples in docstrings
- **Error Codes**: Document all possible error responses

### README Updates

- Update README.md for new features
- Add usage examples
- Update installation instructions if needed

## 🔍 Review Process

### Pull Request Requirements

1. **Tests Pass**: All tests must pass
2. **Code Quality**: No linting errors
3. **Documentation**: Updated documentation
4. **Coverage**: Maintain or improve test coverage
5. **Performance**: No significant performance regressions

### Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests are comprehensive and pass
- [ ] Documentation is updated
- [ ] No breaking changes (or properly documented)
- [ ] Performance impact considered
- [ ] Security implications reviewed

## 🐛 Bug Reports

### Bug Report Template

When reporting bugs, please include:

1. **Description**: Clear description of the issue
2. **Steps to Reproduce**: Detailed steps to reproduce
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Environment**: OS, Python version, dependencies
6. **Screenshots/Logs**: If applicable

Example:
```markdown
## Bug Description
The crisis classification API returns incorrect confidence scores for emergency texts.

## Steps to Reproduce
1. Send POST request to /classify with emergency text
2. Check confidence score in response
3. Compare with expected confidence

## Expected Behavior
Confidence score should be > 0.8 for emergency texts

## Actual Behavior
Confidence score is always 0.5

## Environment
- OS: macOS 12.0
- Python: 3.9.7
- AICrisisAlert: v1.0.0
```

## 💡 Feature Requests

### Feature Request Template

1. **Problem**: What problem does this solve?
2. **Solution**: Proposed solution
3. **Alternatives**: Other solutions considered
4. **Impact**: Who benefits and how?
5. **Implementation**: High-level implementation approach

## 🚨 Security

### Security Guidelines

- **Input Validation**: Validate all user inputs
- **Authentication**: Implement proper authentication for sensitive endpoints
- **Data Protection**: Protect sensitive data and PII
- **Dependencies**: Keep dependencies updated
- **Secrets**: Never commit secrets or API keys

### Reporting Security Issues

For security issues, please email security@aicrisisalert.com instead of creating a public issue.

## 📊 Performance Guidelines

### Code Performance

- **Profiling**: Profile code before optimization
- **Caching**: Use caching for expensive operations
- **Batch Processing**: Process data in batches when possible
- **Memory Usage**: Monitor memory usage for large datasets

### Model Performance

- **Evaluation Metrics**: Use appropriate metrics for crisis classification
- **Baseline Comparison**: Compare against baseline models
- **Real-world Testing**: Test with real crisis data when possible

## 🤝 Community Guidelines

### Code of Conduct

- **Respect**: Be respectful and inclusive
- **Collaboration**: Work together constructively
- **Learning**: Help others learn and grow
- **Feedback**: Provide constructive feedback

### Communication

- **Issues**: Use GitHub issues for discussions
- **Discussions**: Use GitHub Discussions for general topics
- **Questions**: Ask questions in Discussions or Issues

## 📈 Getting Help

### Resources

- **Documentation**: Check README.md and docs/
- **Issues**: Search existing issues
- **Discussions**: Check GitHub Discussions
- **Code**: Review existing code for examples

### Contact

- **Issues**: Create GitHub issues
- **Discussions**: Use GitHub Discussions
- **Email**: contact@aicrisisalert.com

## 🎉 Recognition

### Contributors

All contributors will be recognized in:
- **README.md**: Contributors section
- **Release Notes**: Mentioned in release notes
- **Documentation**: Credited in relevant documentation

### Contribution Levels

- **Bronze**: 1-5 contributions
- **Silver**: 6-20 contributions  
- **Gold**: 21+ contributions
- **Platinum**: Major contributions or maintainer role

---

Thank you for contributing to AICrisisAlert! Your contributions help make emergency response more effective and save lives. 🚨❤️ 