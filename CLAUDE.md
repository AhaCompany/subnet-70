# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Vericore is a Bittensor subnet for fact-checking and verification at scale. It processes statements and returns evidence-based validation through relevant quotes and source materials that either support or contradict input claims.

## Project Structure

The codebase is organized into three main components:

1. **Miner**: Implementations of miners that fetch information from different sources
   - `perplexica/miner.py` - Miner using Perplexica for search
   - `perplexity/miner.py` - Miner using Perplexity API for search

2. **Validator**: Components for validating miner responses and evaluating quality
   - `api_server.py` - API server for receiving statements
   - `validator_daemon.py` - Daemon for handling axons/server tasks
   - `snippet_validator.py` - Core validator logic
   - `snippet_fetcher.py` - Fetches source material for validation
   - `quality_model.py` - Measures corroboration/refutation quality
   - `context_similarity_validator.py` - Validates context similarity
   - `domain_validator.py` - Handles domain validation

3. **Shared**: Common utilities and protocols
   - `veridex_protocol.py` - Protocol for communication between validator and miner
   - Various logging and environment utilities

## Core Commands

### Environment Setup

Create and activate a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Running Tests

Run unit tests:

```bash
python -m unittest discover tests/unit_tests
```

Run individual test:

```bash
python -m unittest tests/unit_tests/snippet_validator_test.py
```

Manual tests require specific setup and human verification:

```bash
python -m tests.manual.test_fetch_requests
python -m tests.manual.test_snippet_fetcher
python -m tests.manual.test_snippet_validator
```

### Running the Validator

Start the API server:

```bash
python -m validator.api_server --wallet.name <wallet_name> --wallet.hotkey <hotkey_name> --netuid 70 --axon.ip=<EXTERNAL_IP>
```

Run the validator daemon:

```bash
python -m validator.validator_daemon --wallet.name <wallet_name> --wallet.hotkey <hotkey_name> --netuid 70
```

For local blockchain testing, add:
```bash
--subtensor.network ws://127.0.0.1:9944
```

### Running the Miner

Perplexity miner (requires API key):

```bash
# Set the API key
export PERPLEXITY_API_KEY=<your_api_key>

# Run the miner
python -m miner.perplexity.miner --wallet.name <wallet_name> --wallet.hotkey <hotkey_name> --axon.ip=<EXTERNAL_IP> --axon.port 8901 --netuid 70
```

Perplexica miner (requires local Perplexica installation):

```bash
# Set the Perplexica URL
export PERPLEXICA_URL=<your_perplexica_url>

# Run the miner
python -m miner.perplexica.miner --wallet.name <wallet_name> --wallet.hotkey <hotkey_name> --axon.ip=<EXTERNAL_IP> --axon.port 8901 --netuid 70
```

## Key Architecture Concepts

1. **Validator-Miner Communication**: 
   - Protocol defined in `veridex_protocol.py`
   - Validators send statements to miners for verification
   - Miners return evidence with source attribution

2. **Validation Process**:
   - `snippet_validator.py` is the core component that evaluates miner responses
   - `context_similarity_validator.py` ensures context relevance
   - Quality scoring via `quality_model.py` and `similarity_quality_model.py`

3. **Source Verification**:
   - `snippet_fetcher.py` retrieves and validates source content
   - `domain_validator.py` verifies domain credibility

4. **Logging System**:
   - Structured logging via `log_data.py` and `proxy_log_handler.py`
   - Logs stored at `~/.bittensor/wallets/<wallet.name>/<wallet.hotkey>/netuid<netuid>/<miner or validator>/`

## Recent Development Focus

Recent commits indicate work on:
- Context similarity scoring logic
- Miner selection improvements
- Score averaging mechanisms

When making changes to scoring or selection logic, ensure proper testing as these are critical system components.