# Real-Time Finance AI - Testing Guide

## Quick Start Testing

### 1. Validate Setup
Check that your environment is configured correctly:
```bash
python tests/test_setup.py
```

This verifies:
- ✅ Environment variables are set
- ✅ Required packages are installed
- ✅ API clients can be imported

---

### 2. Test API Clients
Run integration tests with real API calls:
```bash
python tests/test_api_clients.py
```

This tests:
- 📈 **Ninja API** - Stock prices, company info, crypto
- 📊 **FRED API** - GDP, inflation, unemployment, interest rates
- ✅ **S&P 500 Validator** - Ticker validation with fuzzy matching
- 📰 **NewsAPI** - Company news (if API key configured)

---

### 3. Run Interactive Demo
See all capabilities in action:
```bash
python tests/demo.py
```

This demonstrates:
- S&P 500 ticker validation with suggestions
- Real-time market data for multiple stocks
- Economic indicators summary
- Full company analysis with news

---

## Individual Component Testing

### Test Ninja API
```bash
python src/workflow_1/api_clients/ninja_api.py
```

### Test FRED API
```bash
python src/workflow_1/api_clients/fred_api.py
```

### Test S&P 500 Validator
```bash
python src/workflow_1/utils/sp500_validator.py
```

### Test NewsAPI
```bash
python src/workflow_1/api_clients/news_api.py
```

### Test Configuration
```bash
python src/workflow_1/config.py
```

---

## Expected Output

### Successful Setup Validation:
```
✓ NINJA_API_KEY                 - Required for stock/crypto data
✓ GOOGLE_CLOUD_PROJECT          - Required for GCP services
✓ FRED_API_KEY                  - Recommended for economic data
...
Summary: 10/10 checks passed
✅ Setup validation complete!
```

### Successful API Tests:
```
✅ PASS - Ninja API
✅ PASS - FRED API
✅ PASS - S&P 500 Validator
✅ PASS - NewsAPI

4/4 test suites passed
🎉 All tests passed!
```

---

## Troubleshooting

### Missing Environment Variables
```
⚠️  Missing required environment variables:
   - NINJA_API_KEY
   - GOOGLE_CLOUD_PROJECT
```
**Fix:** Set these in your `.env` file

### Import Errors
```
❌ ModuleNotFoundError: No module named 'requests'
```
**Fix:** Install dependencies:
```bash
python -m venv venv
venv\Scripts\activate
pip install -e .
```

### API Errors
```
❌ Ninja API error: 403 Forbidden
```
**Fix:** Check that your API key is valid and not expired

---

## What Gets Tested

| Component | What's Tested |
|-----------|--------------|
| **Ninja API** | Stock prices, company profiles, crypto prices, forex rates |
| **FRED API** | GDP, CPI, unemployment, interest rates |
| **S&P 500 Validator** | List loading, ticker validation, fuzzy matching |
| **NewsAPI** | Article fetching, fallback mode |
| **Config** | Environment variable loading, validation |

---

## Next Steps After Testing

Once all tests pass, you're ready to:
1. ✅ Build the 7 specialized agents
2. ✅ Create the root orchestrator
3. ✅ Implement data ingestion for RAG
4. ✅ Add Arize observability

The foundation is solid - time to build the agents! 🚀
