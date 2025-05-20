# Optimized Vericore Miner

## Overview

This is an optimized miner implementation for the Vericore subnet that is designed to achieve higher validator scores. The miner enhances evidence collection, source validation, and response format based on the validator's scoring mechanisms.

## Key Optimizations

- **Diverse Source Collection**: Searches multiple services to find the highest quality evidence
- **Content Verification**: Checks that snippets actually exist on the source page
- **Context Expansion**: Expands snippets to provide fuller context
- **Relationship Detection**: Filters out neutral content to focus on clear entailment or contradiction
- **Semantic Similarity**: Ensures high semantic similarity between statements and evidence
- **Domain Authority**: Prioritizes reputable domains and avoids newly registered sites
- **Source Diversity**: Provides evidence from multiple domains rather than a single source

## Setup Instructions

### Prerequisites

In addition to the standard Vericore requirements, the optimized miner requires:

- Python 3.10 or higher
- Transformer models (sentence-transformers, transformers)
- PyTorch
- BeautifulSoup4 for HTML parsing
- python-whois for domain verification

### Installation

1. Install dependencies:

```bash
pip install sentence-transformers transformers torch beautifulsoup4 python-whois httpx
```

2. Set up environment variables:

```bash
# At least one of these is required
export PERPLEXITY_API_KEY=<your_perplexity_api_key>
export PERPLEXICA_URL=<your_perplexica_url>

# Optional for additional search services
export SERP_API_KEY=<your_serp_api_key>
```

### Running the Optimized Miner

```bash
python -m miner.optimized.miner --wallet.name <wallet_name> --wallet.hotkey <hotkey_name> --axon.ip=<EXTERNAL_IP> --axon.port 8901 --netuid <netuid>
```

For local development or testing:

```bash
python -m miner.optimized.miner --wallet.name <wallet_name> --wallet.hotkey <hotkey_name> --axon.ip=<EXTERNAL_IP> --axon.port 8901 --netuid <netuid> --subtensor.network ws://127.0.0.1:9944
```

## Scoring Mechanism Alignment

This miner is specifically designed to maximize scores based on the validator's scoring criteria:

1. **High Statement Relevance**: Ensures snippets have a strong semantic connection to the statement
2. **Clear Evidence Relationship**: Prioritizes snippets that clearly support or contradict statements
3. **Source Verification**: Validates that snippets exist in the original source
4. **Domain Trustworthiness**: Avoids newly registered domains and prioritizes reputable sources
5. **Context Preservation**: Ensures snippets maintain sufficient context to be meaningful

## Monitoring Performance

The optimized miner provides detailed logging about:
- Search results across different services
- NLP model scores for relationship detection
- Semantic similarity measurements
- Source verification results
- Domain authority factors

Monitor the logs to understand how your miner is performing and which sources are providing the highest quality evidence.