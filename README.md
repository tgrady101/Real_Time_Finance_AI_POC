# Real-Time Financial Markets Intelligence Chatbot

S&P 500-focused AI chatbot with multi-agent architecture for comprehensive financial market analysis.

## Features

- **7 Specialized Agents**: Market Data, Fundamentals, Economic, News Sentiment, Portfolio, Data Store (RAG), Root Orchestrator
- **Real-Time Data**: Ninja API (stocks, crypto, forex), FRED (economic indicators), NewsAPI
- **Deep Analysis**: Vertex AI Data Store with earnings call transcripts and SEC 10-K/10-Q filings
- **S&P 500 Scope**: 500 companies with guaranteed comprehensive data availability
- **Observability**: Arize AI for LLM tracing and agent monitoring

## Quick Start

### 1. Prerequisites

- Python 3.10+
- Google Cloud Project with billing enabled
- Terraform installed
- API Keys: Ninja API, FRED, NewsAPI (optional), Arize

### 2. Setup Infrastructure

```bash
# Provision GCP resources with Terraform
cd terraform
terraform init
terraform plan
terraform apply
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 5. Load Data (Optional but Recommended)

```bash
# Load S&P 500 earnings calls and SEC filings
python src/workflow_1/ingestion/earnings_call_ingestion.py
python src/workflow_1/ingestion/sec_filings_ingestion.py
```

### 6. Run the Chatbot

```bash
python src/workflow_1/main.py
```

## Architecture

```
Root Orchestrator
├── Market Data Agent (Ninja API)
├── Fundamentals Agent (Ninja API)
├── Economic Agent (FRED API)
├── News Sentiment Agent (NewsAPI + Gemini)
├── Portfolio Agent (Analysis & Recommendations)
└── Data Store Agent (Vertex AI RAG)
```

## Example Queries

- "What's Apple's current stock price and P/E ratio?"
- "Show me the latest inflation and unemployment rates"
- "What did Tesla's CEO say about production in the last earnings call?"
- "Compare AAPL and MSFT fundamentals and suggest which is better for growth"
- "Analyze my portfolio: 60% SPY, 30% QQQ, 10% BND"

## Project Structure

```
Real_Time_Finance_AI_POC/
├── terraform/              # Infrastructure as code
├── src/
│   └── workflow_1/        # Agent application code
│       ├── agents/        # 7 specialized agents
│       ├── api_clients/   # API wrappers
│       ├── ingestion/     # Data loading scripts
│       ├── observability/ # Arize integration
│       └── utils/         # S&P 500 validation
├── requirements.txt
└── .env
```

## Development

See [implementation_plan.md](.gemini/antigravity/brain/2fc45e52-b636-4cc2-a5c0-49cd8b239915/implementation_plan.md) for detailed architecture and development guide.

## License

MIT
