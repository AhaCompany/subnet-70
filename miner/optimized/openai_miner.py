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

# OpenAI client
from openai import OpenAI

# debug
bt.logging.set_trace()

load_dotenv()

class OpenAIMiner:
    def __init__(self):
        self.config = self.get_config()
        self.setup_bittensor_objects()
        self.setup_logging()

        # Load API keys
        self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.openai_api_key:
            bt.logging.error("OPENAI_API_KEY environment variable is not set. Please set it to use this miner.")
            exit(1)
        
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
        # OpenAI client
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        bt.logging.info("OpenAI client initialized")

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
                "openai_miner",
            )
        )
        os.makedirs(config.full_path, exist_ok=True)
        return config

    def setup_logging(self):
        bt.logging(config=self.config, logging_dir=self.config.full_path)
        bt.logging.info(
            f"Running OpenAI miner for subnet: {self.config.netuid} on network: {self.config.subtensor.network} with config:"
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
        1. Search with OpenAI to find relevant sources
        2. Expand context
        3. Filter for relevance
        4. Verify sources
        5. Return optimized evidence
        """
        bt.logging.info(f"{synapse.request_id} | Received Veridex request")
        statement = synapse.statement
        
        # 1. Get diverse results using OpenAI
        raw_results = self.search_with_openai(statement)
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

    def search_with_openai(self, statement: str) -> List[Dict[str, str]]:
        """
        Use OpenAI to find sources that support or contradict the statement
        """
        system_content = """
You are an expert fact-checking assistant tasked with finding evidence for or against statements.

Rules:
1. Return the response **ONLY as a valid JSON array** with the following format:
   [{"url": "https://example.com/page", "snippet": "The exact text from the website that directly addresses the statement"}]
2. ONLY include HTTPS URLs
3. The snippet must be substantial (at least 2-3 sentences) and DIRECTLY address the statement
4. Do not make up information or change the text from sources
5. Prioritize reputable sources (.edu, .gov, major news outlets, academic journals)
6. Return at least 5 sources if possible, with a mix that both support and contradict the statement
7. Do not include any explanations, introductions, or commentary outside the JSON array

Response MUST be returned as a JSON array. If it isn't returned as a JSON array, the response MUST BE EMPTY.
"""

        user_content = f"Find sources that either support or contradict this statement: {statement}"

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",  # Use the most capable model
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.2,  # Lower temperature for focused factual responses
                max_tokens=4000,
                response_format={"type": "json_object"}  # Ensure JSON output
            )
            
            # Extract response text
            raw_text = response.choices[0].message.content.strip()
            
            # Parse JSON
            try:
                data = json.loads(raw_text)
                # Check if the parsed data has a structure that contains our array
                if isinstance(data, dict) and "array" in data:
                    results = data["array"]
                elif isinstance(data, dict) and any(isinstance(data.get(k), list) for k in data):
                    # Find the first list in the data
                    for k, v in data.items():
                        if isinstance(v, list):
                            results = v
                            break
                else:
                    # Assume the top level is our data
                    results = data
                
                # Ensure we have a list
                if not isinstance(results, list):
                    bt.logging.warn(f"Response is not a list: {data}")
                    results = []
                
                return results
            except json.JSONDecodeError:
                bt.logging.error(f"Failed to parse JSON response: {raw_text}")
                return []
                
        except Exception as e:
            bt.logging.error(f"Error calling OpenAI API: {e}")
            return []

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
    miner = OpenAIMiner()
    miner.run()