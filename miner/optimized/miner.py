import os
import time
import argparse
import traceback
import bittensor as bt
import json
import re
import requests
from typing import Tuple, List, Dict, Any, Optional
import logging
from urllib.parse import urlparse
from datetime import datetime
import whois
import httpx
from bs4 import BeautifulSoup

from dotenv import load_dotenv

from shared.log_data import LoggerType
from shared.proxy_log_handler import register_proxy_log_handler
from shared.veridex_protocol import VericoreSynapse, SourceEvidence

# Import transformer models for similarity and entailment detection
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# "openai" client for perplexity and other services
from openai import OpenAI

# debug
bt.logging.set_trace()

load_dotenv()

class OptimizedMiner:
    def __init__(self):
        self.config = self.get_config()
        self.setup_bittensor_objects()
        self.setup_logging()

        # Load API keys
        self.perplexity_api_key = os.environ.get("PERPLEXITY_API_KEY", "")
        self.perplexica_url = os.environ.get("PERPLEXICA_URL", "")
        self.serp_api_key = os.environ.get("SERP_API_KEY", "")  # Optional for additional sources
        
        # Set up clients
        self.setup_clients()
        
        # Set up models
        self.setup_models()
        
        # Domain authority data - sample trusted domains
        self.trusted_domains = [
            "wikipedia.org", "gov", "edu", "nature.com", "science.org", 
            "scholar.google.com", "nytimes.com", "reuters.com", "bbc.com",
            "washingtonpost.com", "economist.com", "scientificamerican.com",
            "ncbi.nlm.nih.gov", "who.int", "un.org", "europa.eu",
            "apnews.com", "bloomberg.com", "ft.com"
        ]
        
        # Cache for URL content to avoid duplicate fetches
        self.content_cache = {}

    def setup_models(self):
        """Initialize NLP models for similarity and relationship detection"""
        try:
            # Sentence similarity model (same as used by validator)
            self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Natural language inference model for contradiction/entailment
            self.nli_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
            self.nli_model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
            
            bt.logging.info("NLP models loaded successfully")
        except Exception as e:
            bt.logging.error(f"Error loading NLP models: {e}")
            # Fallback to None if models fail to load
            self.similarity_model = None
            self.nli_model = None
            self.nli_tokenizer = None

    def setup_clients(self):
        """Set up API clients for search services"""
        # Perplexity client
        if self.perplexity_api_key:
            self.perplexity_client = OpenAI(
                api_key=self.perplexity_api_key,
                base_url="https://api.perplexity.ai"
            )
        else:
            self.perplexity_client = None
            bt.logging.warning("No PERPLEXITY_API_KEY found. Perplexity search disabled.")

        # HTTPX client for fetching web content
        self.http_client = httpx.Client(
            follow_redirects=True,
            timeout=10.0,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
            }
        )

    def get_config(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--custom", default="my_custom_value", help="Adds a custom value.")
        parser.add_argument("--netuid", type=int, default=1, help="Subnet UID.")
        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.wallet.add_args(parser)
        bt.axon.add_args(parser)

        config = bt.config(parser)
        config.full_path = os.path.expanduser(
            "{}/{}/{}/netuid{}/{}".format(
                config.logging.logging_dir,
                config.wallet.name,
                config.wallet.hotkey_str,
                config.netuid,
                "optimized_miner",
            )
        )
        os.makedirs(config.full_path, exist_ok=True)
        return config

    def setup_logging(self):
        bt.logging(config=self.config, logging_dir=self.config.full_path)
        bt.logging.info(
            f"Running optimized miner for subnet: {self.config.netuid} on network: {self.config.subtensor.network} with config:"
        )
        bt.logging.info(self.config)

    def setup_proxy_logger(self):
        bt_logger = logging.getLogger("bittensor")
        register_proxy_log_handler(bt_logger, LoggerType.Miner, self.wallet)

    def setup_bittensor_objects(self):
        bt.logging.info("Setting up Bittensor objects.")
        self.wallet = bt.wallet(config=self.config)
        bt.logging.info(f"Wallet: {self.wallet}")

        self.subtensor = bt.subtensor(config=self.config)
        bt.logging.info(f"Subtensor: {self.subtensor}")

        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        bt.logging.info(f"Metagraph: {self.metagraph}")

        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(
                f"\nYour miner: {self.wallet} is not registered.\nRun 'btcli register' and try again."
            )
            exit()
        else:
            self.my_subnet_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
            bt.logging.info(f"Miner on uid: {self.my_subnet_uid}")

    def blacklist_fn(self, synapse: VericoreSynapse) -> Tuple[bool, str]:
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            bt.logging.trace(f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}")
            return True, None
        bt.logging.trace(f"Not blacklisting recognized hotkey {synapse.dendrite.hotkey}")
        return False, None

    def veridex_forward(self, synapse: VericoreSynapse) -> VericoreSynapse:
        """
        Optimized handler for Veridex requests. This implements the full pipeline:
        1. Diversify search sources
        2. Expand context
        3. Filter for relevance
        4. Verify sources
        5. Return optimized evidence
        """
        bt.logging.info(f"{synapse.request_id} | Received Veridex request")
        statement = synapse.statement
        
        # 1. Get diverse results from multiple sources
        raw_results = self.diversify_sources(statement)
        bt.logging.info(f"{synapse.request_id} | Found {len(raw_results)} initial results")
        
        if not raw_results:
            synapse.veridex_response = []
            return synapse
        
        # 2. Process and optimize all results
        optimized_evidence = self.optimize_results(statement, raw_results)
        bt.logging.info(f"{synapse.request_id} | Optimized to {len(optimized_evidence)} quality evidence items")
        
        synapse.veridex_response = optimized_evidence
        bt.logging.info(f"{synapse.request_id} | Miner returns {len(optimized_evidence)} evidence items for statement: '{statement}'.")
        return synapse

    def diversify_sources(self, statement: str) -> List[Dict[str, str]]:
        """
        Get results from multiple sources to increase diversity and quality
        """
        all_results = []
        
        # 1. Try Perplexity (primary source)
        if self.perplexity_client:
            perplexity_results = self.call_perplexity_ai(statement)
            if perplexity_results:
                all_results.extend(perplexity_results)
                bt.logging.info(f"Found {len(perplexity_results)} results from Perplexity")
        
        # 2. Try Perplexica if configured
        if self.perplexica_url:
            perplexica_results = self.call_perplexica(statement)
            if perplexica_results:
                all_results.extend(perplexica_results)
                bt.logging.info(f"Found {len(perplexica_results)} results from Perplexica")
        
        # 3. Try additional specialized searches for high-quality domains
        try:
            additional_results = self.search_reputable_sources(statement)
            if additional_results:
                all_results.extend(additional_results)
                bt.logging.info(f"Found {len(additional_results)} results from reputable sources")
        except Exception as e:
            bt.logging.error(f"Error in reputable sources search: {e}")
        
        # Filter out duplicates (same URL)
        seen_urls = set()
        unique_results = []
        
        for result in all_results:
            url = result.get('url', '').strip()
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)
        
        # Initial filter for HTTPS only
        https_results = [r for r in unique_results if r.get('url', '').startswith('https://')]
        
        # Prioritize by domain authority (if we have enough results)
        if len(https_results) > 5:
            return self.rank_by_domain_authority(https_results)
        
        return https_results

    def optimize_results(self, statement: str, raw_results: List[Dict[str, str]]) -> List[SourceEvidence]:
        """
        Process raw results to optimize for validator scoring
        """
        optimized_evidence = []
        
        for result in raw_results:
            try:
                url = result.get('url', '').strip()
                snippet = result.get('snippet', '').strip()
                
                if not url or not snippet:
                    continue
                
                # 1. Skip non-HTTPS URLs
                if not url.startswith('https://'):
                    continue
                
                # 2. Check domain reputation
                domain = urlparse(url).netloc
                if self.is_recently_registered_domain(domain):
                    bt.logging.info(f"Skipping recently registered domain: {domain}")
                    continue
                
                # 3. Expand the context to get more complete information
                expanded_snippet = self.expand_context(url, snippet)
                if not expanded_snippet:
                    expanded_snippet = snippet
                
                # 4. Check statement relationship (only include definitive ones)
                if self.nli_model and self.nli_tokenizer:
                    relationship = self.check_statement_relationship(statement, expanded_snippet)
                    if relationship == "neutral":
                        bt.logging.info(f"Skipping neutral result for: {url}")
                        continue
                
                # 5. Verify content exists on the page
                if not self.verify_snippet_exists(url, expanded_snippet):
                    bt.logging.info(f"Snippet not found in page: {url}")
                    continue
                
                # 6. Check context similarity score
                if self.similarity_model:
                    similarity = self.check_context_similarity(statement, expanded_snippet)
                    if similarity < 0.4:  # Lower threshold for initial filtering
                        bt.logging.info(f"Low similarity score ({similarity}) for: {url}")
                        continue
                
                # All checks passed, add to evidence
                ev = SourceEvidence(url=url, excerpt=expanded_snippet)
                optimized_evidence.append(ev)
                
                # Limit total evidence items to avoid overwhelming the validator
                if len(optimized_evidence) >= 10:
                    break
                    
            except Exception as e:
                bt.logging.error(f"Error processing result {result.get('url', 'unknown')}: {e}")
                continue
        
        return optimized_evidence

    def call_perplexity_ai(self, statement: str) -> List[Dict[str, str]]:
        """
        1) Provide system & user messages optimized for fact-checking
        2) Parse JSON from the response -> [ {url, snippet}, ... ]
        """
        system_content = """
You are an API that fact checks statements with high accuracy. 

Rules:
1. Return the response **only as a JSON array**.
2. The response **must be a valid JSON array**, formatted as:
   ```json
   [{"url": "<source url>", "snippet": "<snippet that directly agrees with or contradicts statement>"}]
3. Do not include any introductory text, explanations, or additional commentary.
4. Do not add any labels, headers, or markdown formatting—only return the JSON array.
5. Only use HTTPS URLs.
6. Do not shorten the snippet provided. Do not add '...' in between segments. Return the entire snippet.
7. Ensure that snippets are long enough to provide full context (at least 2-3 sentences).
8. Prioritize specific, fact-based sources from reputable websites (news organizations, academic sources, government websites).

Steps:
1. Find sources / text segments that either contradict or agree with the user provided statement.
2. Pick and extract the segments that most strongly agree or contradict the statement.
3. Do not return urls or segments that do not directly support or disagree with the statement.
4. Do not change any text in the segments (must return an exact html text match), but make sure to include enough context around the fact.
5. Create the json object for each source and statement and add them only INTO ONE array.

Response MUST returned as a json array. If it isn't returned as json object the response MUST BE EMPTY.
"""
        user_content = f"Return snippets that strongly agree with or reject the following statement:\n{statement}"

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        raw_text = None
        try:
            response = self.perplexity_client.chat.completions.create(
                model="sonar-pro",
                messages=messages,
                stream=False
            )
            if not hasattr(response, "choices") or len(response.choices) == 0:
                bt.logging.warn(f"Perplexity returned no choices: {response}")
                return []
            raw_text = response.choices[0].message.content.strip()
            raw_text = raw_text.removeprefix("```json").removesuffix("```").strip()

            data = json.loads(raw_text)
            if not isinstance(data, list):
                bt.logging.warn(f"Perplexity response is not a list: {data}")
                return []
            return data
        except Exception as e:
            if raw_text is not None:
                bt.logging.error(f"Raw Text of AI Response: {raw_text}")

            bt.logging.error(f"Error calling Perplexity AI: {e}")
            return []

    def call_perplexica(self, statement: str) -> List[Dict[str, str]]:
        """
        Call Perplexica API with optimized prompts for fact-checking
        """
        if not self.perplexica_url:
            return []
            
        endpoint_url = self.perplexica_url
        
        system_content = """
You are an API that fact checks statements with high accuracy. 

Rules:
1. Return the response **only as a JSON array**.
2. The response **must be a valid JSON array**, formatted as:
   ```json
   [{"url": "<source url>", "snippet": "<snippet that directly agrees with or contradicts statement>"}]
3. Do not include any introductory text, explanations, or additional commentary.
4. Do not add any labels, headers, or markdown formatting—only return the JSON array.
5. Only use HTTPS URLs.
6. Do not shorten the snippet provided. Do not add '...' in between segments. Return the entire snippet.
7. Ensure that snippets are long enough to provide full context (at least 2-3 sentences).
8. Prioritize specific, fact-based sources from reputable websites (news organizations, academic sources, government websites).

Steps:
1. Find sources / text segments that either contradict or agree with the user provided statement.
2. Pick and extract the segments that most strongly agree or contradict the statement.
3. Do not return urls or segments that do not directly support or disagree with the statement.
4. Do not change any text in the segments (must return an exact html text match), but make sure to include enough context around the fact.
5. Create the json object for each source and statement and add them only INTO ONE array.

Response MUST returned as a json array. If it isn't returned as json object the response MUST BE EMPTY.
"""
        user_content = f"Return snippets that strongly agree with or reject the following statement:\n{statement}"

        perplexica_search_object = {
            "chatModel": {
                "provider": "openai",
                "model": "gpt-4o-mini"
            },
            "optimizationMode": "speed",
            "focusMode": "webSearch",
            "query": f"{statement}",
            "history": [
                ["system", system_content],
                ["human", user_content],
            ]
        }
        
        headers = {"Content-Type": "application/json"}
        
        try:
            response = requests.post(
                endpoint_url, json=perplexica_search_object, timeout=60, headers=headers
            )
            
            response_data = response.json()
            raw_text = response_data.get("message", "").strip()
            raw_text = raw_text.replace("```json", "").replace("```", "").strip()
            
            data = json.loads(raw_text)
            if not isinstance(data, list):
                bt.logging.warn(f"Perplexica response is not a list: {data}")
                return []
            return data
        except Exception as e:
            bt.logging.error(f"Error calling Perplexica: {e}")
            return []

    def search_reputable_sources(self, statement: str) -> List[Dict[str, str]]:
        """
        Targeted search for high-quality academic and trusted sources
        Uses an additional prompt to focus on specific reputable domains
        """
        # If we don't have Perplexity available, skip this
        if not self.perplexity_client:
            return []
            
        try:
            system_content = """
You are an API that fact checks statements with a focus on reliable, authoritative sources.

Rules:
1. Return the response **only as a JSON array**.
2. The response **must be a valid JSON array**, formatted as:
   ```json
   [{"url": "<source url>", "snippet": "<snippet that directly agrees with or contradicts statement>"}]
3. ONLY focus on searching these specific types of domains:
   - .gov domains (government sources)
   - .edu domains (educational institutions)
   - Scientific journals and publications
   - Major reputable news organizations (NY Times, Washington Post, Reuters, AP, etc.)
   - Fact-checking organizations
   - International organizations (WHO, UN, etc.)
4. Provide full, contextual snippets (not just short quotes).
5. Only use HTTPS URLs.

Response MUST returned as a json array. If it isn't returned as json object the response MUST BE EMPTY.
"""
            user_content = f"Find authoritative sources that strongly agree with or reject the following statement:\n{statement}"

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]

            response = self.perplexity_client.chat.completions.create(
                model="sonar-pro",
                messages=messages,
                stream=False
            )
            raw_text = response.choices[0].message.content.strip()
            raw_text = raw_text.removeprefix("```json").removesuffix("```").strip()

            data = json.loads(raw_text)
            if not isinstance(data, list):
                return []
            return data
        except Exception as e:
            bt.logging.error(f"Error in reputable sources search: {e}")
            return []

    def fetch_page_content(self, url: str) -> Optional[str]:
        """
        Fetch the content of a web page with caching
        """
        # Check cache first
        if url in self.content_cache:
            return self.content_cache[url]
            
        try:
            response = self.http_client.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove scripts, styles, and other non-content elements
            for element in soup(["script", "style", "meta", "noscript", "header", "footer", "nav"]):
                element.decompose()
                
            # Get text content
            text = soup.get_text(separator='\n', strip=True)
            
            # Clean up extra whitespace
            text = re.sub(r'\n+', '\n', text)
            text = re.sub(r'\s+', ' ', text)
            
            # Store in cache
            self.content_cache[url] = text
            
            return text
        except Exception as e:
            bt.logging.error(f"Error fetching page content for {url}: {e}")
            return None

    def expand_context(self, url: str, snippet: str) -> Optional[str]:
        """
        Expand the context around the snippet to provide more complete information
        """
        page_content = self.fetch_page_content(url)
        if not page_content:
            return None
            
        # Find the snippet in the page content
        snippet_index = page_content.find(snippet)
        if snippet_index == -1:
            # Try a simplified version (removing extra spaces)
            simplified_snippet = re.sub(r'\s+', ' ', snippet).strip()
            simplified_content = re.sub(r'\s+', ' ', page_content).strip()
            snippet_index = simplified_content.find(simplified_snippet)
            if snippet_index == -1:
                return None
                
        # Extract an expanded context (300 chars before and after)
        start = max(0, snippet_index - 300)
        end = min(len(page_content), snippet_index + len(snippet) + 300)
        expanded_snippet = page_content[start:end]
        
        # Trim to complete sentences
        expanded_snippet = self.trim_to_whole_sentences(expanded_snippet)
        
        return expanded_snippet

    def trim_to_whole_sentences(self, text: str) -> str:
        """
        Trim text to whole sentences
        """
        # Find first sentence boundary
        start_match = re.search(r'[.!?]\s+[A-Z]', text[:100])
        if start_match:
            start = start_match.end() - 1
            text = text[start:]
        
        # Find last sentence boundary
        end_match = re.search(r'[.!?]\s+[A-Z]', text[-100:])
        if end_match:
            end = len(text) - 100 + end_match.start() + 1
            text = text[:end]
        
        return text

    def check_statement_relationship(self, statement: str, snippet: str) -> str:
        """
        Check if snippet contradicts, is neutral to, or entails the statement
        """
        if not self.nli_model or not self.nli_tokenizer:
            return "unknown"
            
        try:
            # Truncate if needed to fit model context
            max_length = self.nli_tokenizer.model_max_length - 20
            inputs = self.nli_tokenizer(statement, snippet, truncation=True, max_length=max_length, return_tensors="pt")
            
            # Get prediction
            with torch.no_grad():
                outputs = self.nli_model(**inputs)
            
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
            
            # Get label
            label_map = {0: "contradiction", 1: "neutral", 2: "entailment"}
            prediction_idx = probabilities.argmax().item()
            prediction = label_map[prediction_idx]
            
            # Get probabilities
            contra_prob = probabilities[0].item()
            neutral_prob = probabilities[1].item()
            entail_prob = probabilities[2].item()
            
            # Calculate local score similar to validator
            local_score = (contra_prob + entail_prob) - neutral_prob
            
            bt.logging.info(f"NLI Scores - Contradiction: {contra_prob:.3f}, Neutral: {neutral_prob:.3f}, Entailment: {entail_prob:.3f}, Local Score: {local_score:.3f}")
            
            return prediction
        except Exception as e:
            bt.logging.error(f"Error in statement relationship check: {e}")
            return "unknown"

    def check_context_similarity(self, statement: str, snippet: str) -> float:
        """
        Check semantic similarity between statement and snippet
        Returns similarity score (0-1)
        """
        if not self.similarity_model:
            return 0.5  # Default middle value if model not available
            
        try:
            # Encode texts
            statement_embedding = self.similarity_model.encode(statement, convert_to_tensor=True)
            snippet_embedding = self.similarity_model.encode(snippet, convert_to_tensor=True)
            
            # Calculate cosine similarity
            similarity = util.pytorch_cos_sim(statement_embedding, snippet_embedding).item()
            
            return similarity
        except Exception as e:
            bt.logging.error(f"Error in context similarity check: {e}")
            return 0.5

    def verify_snippet_exists(self, url: str, snippet: str) -> bool:
        """
        Verify if the snippet actually exists in the page content
        """
        page_content = self.fetch_page_content(url)
        if not page_content:
            return False
            
        # Check if snippet is in the page
        if snippet in page_content:
            return True
            
        # Try more flexible matching
        simplified_snippet = re.sub(r'\s+', ' ', snippet).strip()
        simplified_content = re.sub(r'\s+', ' ', page_content).strip()
        
        # Check for substantial overlap
        snippet_words = set(simplified_snippet.lower().split())
        if len(snippet_words) < 10:
            return False  # Too short to verify
            
        # Count word overlap
        found_words = 0
        for word in snippet_words:
            if word in simplified_content.lower():
                found_words += 1
                
        # If more than 80% of words match, consider it found
        return found_words / len(snippet_words) > 0.8

    def is_recently_registered_domain(self, domain: str) -> bool:
        """
        Check if a domain was registered recently (less than 6 months ago)
        """
        try:
            # First, check if it's a subdomain of a trusted domain
            for trusted in self.trusted_domains:
                if trusted in domain:
                    return False  # It's a trusted domain
                    
            # Try to get WHOIS information
            domain_info = whois.whois(domain)
            
            # Check creation date
            if domain_info.creation_date:
                # Handle both single date and list of dates
                creation_date = domain_info.creation_date
                if isinstance(creation_date, list):
                    creation_date = creation_date[0]
                    
                # Calculate domain age
                domain_age = (datetime.now() - creation_date).days
                
                # Consider domains less than 180 days (6 months) old as "recently registered"
                return domain_age < 180
        except Exception:
            # If WHOIS lookup fails, we can't determine, so default to false
            pass
            
        return False

    def rank_by_domain_authority(self, results: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Rank results by domain authority and diversity
        """
        scored_results = []
        for result in results:
            url = result.get('url', '')
            domain = urlparse(url).netloc
            
            # Base score
            domain_score = 0
            
            # Check against trusted domains
            for trusted in self.trusted_domains:
                if trusted in domain:
                    domain_score += 5
                    break
                    
            # Favor academic and government domains
            if domain.endswith('.edu'):
                domain_score += 3
            elif domain.endswith('.gov'):
                domain_score += 3
            elif domain.endswith('.org'):
                domain_score += 2
                
            # Add the score to the result
            scored_results.append((result, domain_score))
            
        # Sort by score (highest first)
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Extract just the results
        ranked_results = [item[0] for item in scored_results]
        
        # Add domain diversity - alternate between high-scoring domains
        # to avoid all results from the same domain
        seen_domains = set()
        diverse_results = []
        
        for result in ranked_results:
            url = result.get('url', '')
            domain = urlparse(url).netloc
            
            # If we already have 2 results from this domain, skip unless we have few results
            if domain in seen_domains and seen_domains.count(domain) >= 2 and len(diverse_results) > 5:
                continue
                
            diverse_results.append(result)
            seen_domains.add(domain)
            
        return diverse_results

    def setup_axon(self):
        self.axon = bt.axon(wallet=self.wallet, config=self.config)
        bt.logging.info(f"Attaching forward function to axon" )
        self.axon.attach(
            forward_fn=self.veridex_forward,
            blacklist_fn=self.blacklist_fn,
        )
        bt.logging.info(f"Serving axon on network: {self.config.subtensor.network} netuid: {self.config.netuid}")
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        bt.logging.info(f"Axon: {self.axon}")

        bt.logging.info(f"Starting axon server on port: {self.config.axon.port}")
        self.axon.start()

    def run(self):
        bt.logging.info("Setting up axon")
        self.setup_axon()

        bt.logging.info("Setting up proxy logger")
        self.setup_proxy_logger()

        bt.logging.info("Starting main loop")
        step = 0
        while True:
            try:
                if step % 60 == 0:
                    self.metagraph.sync()
                    log = (f"Block: {self.metagraph.block.item()} | "
                           f"Incentive: {self.metagraph.I[self.my_subnet_uid]} | ")
                    bt.logging.info(log)
                step += 1
                time.sleep(1)
            except KeyboardInterrupt:
                self.axon.stop()
                bt.logging.success("Miner killed by keyboard interrupt.")
                break
            except Exception as e:
                bt.logging.error(traceback.format_exc())
                continue

if __name__ == "__main__":
    miner = OptimizedMiner()
    miner.run()